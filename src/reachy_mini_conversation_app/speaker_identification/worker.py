"""Speaker identification worker for real-time speaker recognition."""

import queue
import logging
import threading
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
from scipy import signal
from numpy.typing import NDArray

from reachy_mini_conversation_app.speaker_identification.embeddings_store import SpeakerEmbeddingsStore


if TYPE_CHECKING:
    from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)

# Target sample rate for ECAPA-TDNN model
TARGET_SAMPLE_RATE = 16000


class SpeakerIDWorker:
    """Worker thread for real-time speaker identification.

    This worker processes audio in a separate thread to avoid blocking the main
    conversation loop. It uses SpeechBrain's ECAPA-TDNN model for speaker embeddings.

    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        threshold: float = 0.25,
        embeddings_path: Path | str = "~/.reachy_mini/speaker_embeddings.npz",
        device: str = "cpu",
        preload_model: bool = True,
    ) -> None:
        """Initialize the speaker identification worker.

        Args:
            model_source: HuggingFace model ID for the speaker encoder.
            threshold: Minimum cosine similarity to consider a speaker match.
            embeddings_path: Path to store registered speaker embeddings.
            device: Device for model inference ('cpu' or 'cuda').
            preload_model: If True, load the model immediately during init.

        """
        self._model_source = model_source
        self._device = device
        self._threshold = threshold

        # Encoder (can be preloaded or lazy-loaded)
        self._encoder: "EncoderClassifier | None" = None
        self._encoder_lock = threading.Lock()

        # Embeddings store
        self._embeddings_store = SpeakerEmbeddingsStore(embeddings_path)

        # Audio processing queue
        self._audio_queue: queue.Queue[tuple[NDArray[np.int16 | np.float32], int]] = queue.Queue(maxsize=10)

        # Current speaker state (thread-safe access)
        self._current_speaker: str | None = None
        self._current_confidence: float = 0.0
        self._state_lock = threading.Lock()

        # Registration mode
        self._registration_name: str | None = None
        self._registration_audio: list[NDArray[np.float32]] = []
        self._registration_lock = threading.Lock()

        # Audio accumulation buffer for identification (need enough audio for embedding)
        self._audio_buffer: list[NDArray[np.float32]] = []
        self._audio_buffer_samples: int = 0
        self._min_samples_for_embedding: int = int(TARGET_SAMPLE_RATE * 1.5)  # 1.5 seconds

        # Worker thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Preload model if requested
        if preload_model:
            self.load_model()

    def load_model(self) -> None:
        """Load the SpeechBrain encoder model.

        Call this method to preload the model at startup instead of lazy loading.
        This avoids delays during the first speaker identification.

        Raises:
            ImportError: If speechbrain is not installed.

        """
        self._load_encoder()

    def _load_encoder(self) -> "EncoderClassifier":
        """Load or return the cached SpeechBrain encoder model.

        Returns:
            The loaded EncoderClassifier model.

        Raises:
            ImportError: If speechbrain is not installed.

        """
        with self._encoder_lock:
            if self._encoder is None:
                try:
                    from speechbrain.inference.speaker import EncoderClassifier

                    logger.info("Loading speaker encoder model: %s", self._model_source)
                    self._encoder = EncoderClassifier.from_hparams(
                        source=self._model_source,
                        savedir=f"~/.cache/speechbrain/{self._model_source.replace('/', '_')}",
                        run_opts={"device": self._device},
                    )
                    logger.info("Speaker encoder loaded successfully")
                except ImportError as e:
                    raise ImportError(
                        "SpeechBrain is required for speaker identification. "
                        "Install with: uv sync --group speaker_id"
                    ) from e
            return self._encoder

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Worker already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._working_loop, daemon=True, name="SpeakerIDWorker")
        self._thread.start()
        logger.info("Speaker identification worker started")

    def stop(self) -> None:
        """Stop the worker thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Speaker identification worker stopped")

    def feed_audio(self, audio: NDArray[np.int16 | np.float32], sample_rate: int) -> None:
        """Add an audio segment to the processing queue.

        Args:
            audio: Audio samples (mono, int16 or float32).
            sample_rate: Sample rate of the audio.

        """
        try:
            self._audio_queue.put_nowait((audio, sample_rate))
        except queue.Full:
            # Drop old audio if queue is full (real-time processing)
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait((audio, sample_rate))
            except queue.Empty:
                pass

    def get_current_speaker(self) -> tuple[str | None, float]:
        """Get the currently identified speaker.

        Returns:
            Tuple of (speaker_name, confidence). Returns (None, 0.0) if unknown.

        """
        with self._state_lock:
            return self._current_speaker, self._current_confidence

    def start_registration(self, name: str) -> None:
        """Start recording audio for speaker registration.

        Args:
            name: Name of the speaker to register.

        """
        with self._registration_lock:
            self._registration_name = name
            self._registration_audio = []
        logger.info("Started registration for speaker: %s", name)

    def finish_registration(self) -> bool:
        """Finish registration and save the speaker embedding.

        Returns:
            True if registration was successful, False otherwise.

        """
        with self._registration_lock:
            if self._registration_name is None or not self._registration_audio:
                logger.warning("No registration in progress or no audio collected")
                return False

            name = self._registration_name
            audio_chunks = self._registration_audio
            self._registration_name = None
            self._registration_audio = []

        # Concatenate audio and compute embedding
        audio = np.concatenate(audio_chunks)
        embedding = self._compute_embedding(audio)

        if embedding is None:
            logger.error("Failed to compute embedding for registration")
            return False

        self._embeddings_store.register(name, embedding)
        logger.info("Successfully registered speaker: %s", name)
        return True

    def cancel_registration(self) -> None:
        """Cancel ongoing registration."""
        with self._registration_lock:
            self._registration_name = None
            self._registration_audio = []
        logger.info("Registration cancelled")

    def is_registering(self) -> bool:
        """Check if registration is in progress."""
        with self._registration_lock:
            return self._registration_name is not None

    def list_speakers(self) -> list[str]:
        """List all registered speakers."""
        return self._embeddings_store.list_speakers()

    def remove_speaker(self, name: str) -> bool:
        """Remove a registered speaker."""
        return self._embeddings_store.remove(name)

    def _working_loop(self) -> None:
        """Process audio from the queue in a loop."""
        while not self._stop_event.is_set():
            try:
                audio, sample_rate = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Convert to float32 if needed
            if audio.dtype == np.int16:
                audio_f32: NDArray[np.float32] = audio.astype(np.float32) / 32768.0
            else:
                audio_f32 = audio.astype(np.float32)

            # Resample to target sample rate
            if sample_rate != TARGET_SAMPLE_RATE:
                audio_f32 = self._resample_audio(audio_f32, sample_rate, TARGET_SAMPLE_RATE)

            # Handle registration mode
            with self._registration_lock:
                if self._registration_name is not None:
                    self._registration_audio.append(audio_f32)
                    continue

            # Accumulate audio in buffer
            self._audio_buffer.append(audio_f32)
            self._audio_buffer_samples += len(audio_f32)

            # Skip if not enough audio accumulated (need at least 1.5 seconds)
            if self._audio_buffer_samples < self._min_samples_for_embedding:
                continue

            # Concatenate accumulated audio
            accumulated_audio = np.concatenate(self._audio_buffer)

            # Keep only the last 3 seconds of audio (sliding window)
            max_samples = int(TARGET_SAMPLE_RATE * 3.0)
            if len(accumulated_audio) > max_samples:
                accumulated_audio = accumulated_audio[-max_samples:]

            # Clear buffer but keep some overlap for continuity
            overlap_samples = int(TARGET_SAMPLE_RATE * 0.5)  # 0.5 second overlap
            self._audio_buffer = [accumulated_audio[-overlap_samples:]]
            self._audio_buffer_samples = overlap_samples

            # Compute embedding and identify speaker
            embedding = self._compute_embedding(accumulated_audio)
            if embedding is not None:
                speaker, confidence = self._embeddings_store.identify(embedding, self._threshold)
                with self._state_lock:
                    self._current_speaker = speaker
                    self._current_confidence = confidence
                logger.debug(
                    "Speaker identification: %s (confidence: %.2f)",
                    speaker if speaker else "unknown",
                    confidence,
                )

    def _resample_audio(
        self,
        audio: NDArray[np.float32],
        from_sr: int,
        to_sr: int = TARGET_SAMPLE_RATE,
    ) -> NDArray[np.float32]:
        """Resample audio to the target sample rate.

        Args:
            audio: Input audio samples.
            from_sr: Source sample rate.
            to_sr: Target sample rate.

        Returns:
            Resampled audio.

        """
        if from_sr == to_sr:
            return audio

        num_samples = int(len(audio) * to_sr / from_sr)
        resampled: NDArray[np.float32] = signal.resample(audio, num_samples).astype(np.float32)
        return resampled

    def _compute_embedding(self, audio: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Compute speaker embedding from audio.

        Args:
            audio: Audio samples (float32, 16kHz).

        Returns:
            Speaker embedding vector, or None if computation failed.

        """
        try:
            import torch

            encoder = self._load_encoder()

            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)

            # Compute embedding
            with torch.no_grad():
                embedding = encoder.encode_batch(audio_tensor)

            # Convert to numpy and squeeze
            result: NDArray[np.float32] = embedding.squeeze().cpu().numpy().astype(np.float32)
            return result
        except Exception as e:
            logger.error("Failed to compute speaker embedding: %s", e)
            return None

    def __del__(self) -> None:
        """Cleanup when the worker is destroyed."""
        self.stop()
