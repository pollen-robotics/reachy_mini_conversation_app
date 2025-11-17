"""Parakeet-MLX streaming ASR provider (Apple Silicon optimized).

IMPORTANT: This is WORK IN PROGRESS and does not really work yet.

"""

from __future__ import annotations
import io
import wave
import asyncio
import logging
import threading
from typing import Any, Optional

import numpy as np
import mlx.core as mx
import numpy.typing as npt

from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)


class ParakeetMLXStreamingASR(StreamingASRProvider):
    """Parakeet-MLX streaming ASR implementation optimized for Apple Silicon.

    This provider uses the native streaming API of parakeet-mlx to provide
    real-time incremental transcription with partial results as audio is recorded.

    Key features:
    - Native RNNT streaming (not simulated with VAD)
    - Optimized for Apple Neural Engine via MLX
    - Separates stable (finalized) vs unstable (draft) tokens
    - ~100-300ms first-word latency
    - Fully local (no internet required)

    Thread Safety:
    - All MLX operations are protected by a lock to prevent memory corruption
    """

    def __init__(
        self,
        model_name: str = "mlx-community/parakeet-tdt-0.6b-v3",
        precision: str = "fp32",
        context_size: tuple[int, int] = (256, 256),
        depth: Optional[int] = None,
    ):
        """Initialize Parakeet-MLX streaming ASR.

        Args:
            model_name: Model ID from HuggingFace (parakeet-tdt-0.6b-v3 recommended)
            precision: Inference precision (fp32 for quality, bf16 for speed)
            context_size: (left, right) attention window in frames for streaming.
                         Larger = better accuracy but more latency.
                         Default (256, 256) ≈ ~1.6s context each direction @ 16kHz
            depth: Number of encoder layers that preserve exact computation across chunks.
                  None = auto (model-dependent), lower = faster streaming

        """
        self.model_name = model_name
        self.precision = precision
        self.context_size = context_size
        self.depth = depth
        self.target_sample_rate = 16000  # Parakeet requires 16kHz

        # Model (loaded on first use)
        self.model: Any = None

        # Streaming state
        self.transcriber_context: Any = None  # Context manager
        self.transcriber: Any = None  # Active transcriber
        self.is_streaming = False

        # Stable vs unstable text tracking
        self.stable_text = ""  # Finalized tokens only
        self.unstable_tail = ""  # Draft tokens after stable part
        self.num_finalized_tokens = 0  # Track how many tokens are finalized

        # Audio tracking
        self.chunk_buffer: list[npt.NDArray[np.float32]] = []
        self.buffer_target_samples = int(self.target_sample_rate * 0.25)  # 250ms buffer
        self.total_buffered_samples = 0
        self.cumulative_samples_sent = 0  # Total samples sent to model

        # Thread safety lock for MLX operations
        self._mlx_lock = threading.Lock()

        # Preload model immediately to avoid first-call delay
        logger.info(f"Loading Parakeet model: {model_name} (precision: {precision})...")
        self._ensure_model()
        logger.info("Parakeet model loaded successfully")

        # Warmup: Run a dummy inference to pre-compile MLX kernels
        self._warmup_model()

    def _ensure_model(self) -> None:
        """Load the Parakeet model."""
        if self.model is None:
            from parakeet_mlx import from_pretrained

            # Convert string precision to MLX dtype
            if self.precision == "fp32":
                dtype = mx.float32
            elif self.precision == "bf16":
                dtype = mx.bfloat16
            elif self.precision == "fp16":
                dtype = mx.float16
            else:
                logger.warning(f"Unknown precision '{self.precision}', using fp32")
                dtype = mx.float32

            self.model = from_pretrained(
                self.model_name,
                dtype=dtype,
            )

    def _warmup_model(self) -> None:
        """Warmup model with dummy inference to pre-compile MLX kernels."""
        import time
        import tempfile

        logger.info("Warming up Parakeet model (pre-compiling MLX kernels)...")
        warmup_start = time.perf_counter()

        try:
            # Create a short dummy audio file (1 second of silence)
            sample_rate = 16000
            duration = 1.0  # 1 second
            num_samples = int(sample_rate * duration)
            silence = np.zeros(num_samples, dtype=np.int16)

            # Write to temp WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(silence.tobytes())

            # Run dummy inference (this compiles MLX kernels)
            _ = self.model.transcribe(temp_file.name)

            # Cleanup
            from pathlib import Path
            Path(temp_file.name).unlink(missing_ok=True)

            warmup_duration = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"Parakeet warmup complete! First inference took {warmup_duration:.0f}ms")

        except Exception as e:
            logger.warning(f"Parakeet warmup failed (non-critical): {e}")

    async def start_stream(self) -> None:
        """Initialize streaming session."""
        try:
            logger.info(f"Starting Parakeet streaming session (context_size={self.context_size})")

            # Create streaming context with proper thread safety
            def _create_stream() -> None:
                with self._mlx_lock:
                    # Use parakeet-mlx's streaming API
                    kwargs = {"context_size": self.context_size}
                    if self.depth is not None:
                        kwargs["depth"] = self.depth

                    # Create context manager
                    self.transcriber_context = self.model.transcribe_stream(**kwargs)
                    # Enter the context
                    self.transcriber = self.transcriber_context.__enter__()

            await asyncio.to_thread(_create_stream)

            # Reset state
            self.stable_text = ""
            self.unstable_tail = ""
            self.num_finalized_tokens = 0
            self.is_streaming = True
            self.chunk_buffer = []
            self.total_buffered_samples = 0
            self.cumulative_samples_sent = 0

            logger.info("Parakeet streaming session started")

        except Exception as e:
            logger.exception(f"Failed to start Parakeet stream: {e}")
            raise

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send audio chunk to streaming transcriber.

        Small chunks are buffered until we have ~250ms of audio,
        then sent to the model. This prevents hallucinations.

        Args:
            audio_chunk: Audio data in WAV format

        """
        if not self.is_streaming or self.transcriber is None:
            logger.warning("Parakeet streaming session not active, skipping audio chunk")
            return

        try:
            # Convert WAV bytes to numpy array
            audio_array_np = self._wav_bytes_to_numpy(audio_chunk)

            if len(audio_array_np) == 0:
                return  # Skip empty chunks

            # Buffer the chunk
            self.chunk_buffer.append(audio_array_np)
            self.total_buffered_samples += len(audio_array_np)

            logger.debug(f"Buffered {len(audio_array_np)} samples (buffer: {self.total_buffered_samples}/{self.buffer_target_samples})")

            # Only send to model when we have enough data (250ms)
            if self.total_buffered_samples >= self.buffer_target_samples:
                # Concatenate buffered chunks
                buffered_audio = np.concatenate(self.chunk_buffer)
                batch_size = len(buffered_audio)

                # Add audio with thread safety
                def _add_audio() -> None:
                    with self._mlx_lock:
                        audio_array_mlx = mx.array(buffered_audio)
                        self.transcriber.add_audio(audio_array_mlx)

                await asyncio.to_thread(_add_audio)

                # Update tracking
                self.cumulative_samples_sent += batch_size
                logger.info(f"✓ Sent {batch_size} samples ({batch_size/self.target_sample_rate*1000:.0f}ms) | Cumulative: {self.cumulative_samples_sent} ({self.cumulative_samples_sent/self.target_sample_rate:.1f}s)")

                # Reset buffer
                self.chunk_buffer = []
                self.total_buffered_samples = 0

        except Exception as e:
            logger.warning(f"Failed to send audio chunk to Parakeet: {e}")

    def get_stable_text(self) -> str:
        """Get only stable (finalized) text.

        This should be used for downstream processing like entity extraction.
        Draft tokens are excluded as they may contain repetitions and errors.

        Returns:
            Stable text (finalized tokens only)

        """
        return self.stable_text

    async def get_partial_transcript(self) -> Optional[str]:
        """Get current partial transcript.

        Returns stable (finalized) + unstable (draft) text.
        Only stable text should be used for downstream processing.

        Returns:
            Current partial transcript (stable + unstable), or None if nothing yet

        """
        if not self.is_streaming or self.transcriber is None:
            return None

        try:
            # Get tokens with thread safety
            def _get_tokens() -> tuple[list[Any], list[Any]]:
                with self._mlx_lock:
                    finalized = self.transcriber.finalized_tokens
                    draft = self.transcriber.draft_tokens
                    return finalized, draft

            finalized_tokens, draft_tokens = await asyncio.to_thread(_get_tokens)

            num_finalized = len(finalized_tokens)
            num_draft = len(draft_tokens)

            # Check if we have NEW finalized tokens
            new_finalized_count = num_finalized - self.num_finalized_tokens

            if new_finalized_count > 0:
                # Build stable text from ALL finalized tokens
                # NOTE: Tokens already contain spacing, so concatenate directly (no join with space)
                self.stable_text = "".join(token.text for token in finalized_tokens)
                self.num_finalized_tokens = num_finalized
                logger.info(f"📌 +{new_finalized_count} finalized tokens | Total finalized: {num_finalized} | Stable: '{self.stable_text}'")

            # Build unstable tail from draft tokens
            if draft_tokens:
                # NOTE: Tokens already contain spacing, so concatenate directly (no join with space)
                draft_text = "".join(token.text for token in draft_tokens)
                # Remove stable prefix if it overlaps
                if self.stable_text and draft_text.startswith(self.stable_text):
                    self.unstable_tail = draft_text[len(self.stable_text):].strip()
                else:
                    self.unstable_tail = draft_text
            else:
                self.unstable_tail = ""

            # Log state
            logger.debug(f"ASR state: {num_finalized} finalized, {num_draft} draft | stable='{self.stable_text}' | tail='{self.unstable_tail}'")

            # Return combined text for display
            if self.stable_text and self.unstable_tail:
                return self.stable_text + " " + self.unstable_tail
            elif self.stable_text:
                return self.stable_text
            elif self.unstable_tail:
                return self.unstable_tail
            else:
                return None

        except Exception as e:
            logger.warning(f"Error getting partial transcript: {e}", exc_info=True)
            return None

    async def end_stream(self) -> str:
        """Finalize stream and get final transcript.

        Forces token finalization and returns only stable text
        with repetition cleanup applied.

        Returns:
            Final clean transcript

        """
        try:
            if self.transcriber is not None:
                # Flush any remaining buffered audio
                if self.chunk_buffer and self.total_buffered_samples > 0:
                    logger.info(f"Flushing final {self.total_buffered_samples} buffered samples")
                    buffered_audio = np.concatenate(self.chunk_buffer)

                    def _add_final_audio() -> None:
                        with self._mlx_lock:
                            audio_array_mlx = mx.array(buffered_audio)
                            self.transcriber.add_audio(audio_array_mlx)

                    await asyncio.to_thread(_add_final_audio)
                    self.cumulative_samples_sent += len(buffered_audio)
                    self.chunk_buffer = []
                    self.total_buffered_samples = 0

                logger.info(f"🎬 End of audio. Total sent: {self.cumulative_samples_sent} samples ({self.cumulative_samples_sent/self.target_sample_rate:.1f}s)")

                # Wait for model to finalize tokens
                # Poll multiple times to let finalization complete
                logger.info("⏳ Waiting for token finalization...")
                max_polls = 5
                for i in range(max_polls):
                    await asyncio.sleep(0.1)  # 100ms between polls

                    # Check for new finalized tokens
                    await self.get_partial_transcript()

                    if self.num_finalized_tokens > 0:
                        logger.info(f"✓ Poll {i+1}/{max_polls}: {self.num_finalized_tokens} tokens finalized")
                        # Keep polling to see if more finalize
                    else:
                        logger.debug(f"Poll {i+1}/{max_polls}: No finalized tokens yet")

                # Get final state
                final_partial = await self.get_partial_transcript()
                logger.debug(f"Final partial transcript: '{final_partial}'")

                # Use ONLY stable text as final transcript
                final_transcript = self.stable_text

                # If we have no finalized tokens but have draft, log warning and use it
                if not final_transcript and self.unstable_tail:
                    logger.warning(f"⚠️ No finalized tokens! Using draft as fallback: '{self.unstable_tail}'")
                    final_transcript = self.unstable_tail

                # Apply repetition cleanup
                final_transcript = self._cleanup_repetitions(final_transcript)

                logger.info(f"📝 FINAL stable_text: '{final_transcript}'")
                if self.unstable_tail:
                    logger.info(f"🗑️ DROPPED unstable_tail: '{self.unstable_tail}'")

                # Exit streaming context with thread safety
                def _exit_stream() -> None:
                    with self._mlx_lock:
                        try:
                            if self.transcriber_context is not None:
                                self.transcriber_context.__exit__(None, None, None)
                        except Exception as e:
                            logger.debug(f"Error exiting stream context: {e}")

                await asyncio.to_thread(_exit_stream)

                # Clean up
                self.transcriber = None
                self.transcriber_context = None
                self.is_streaming = False

                return final_transcript.strip()
            else:
                # No transcriber
                return self.stable_text.strip()

        except Exception as e:
            logger.exception(f"Error ending Parakeet stream: {e}")
            # Return stable text as fallback
            return self._cleanup_repetitions(self.stable_text).strip()

    def _cleanup_repetitions(self, text: str) -> str:
        """Clean up immediate word repetitions in transcript.

        Examples:
            "hello hello hello" → "hello"
            "my name is my name is" → "my name is"
            "well well well" → "well"

        Args:
            text: Raw transcript

        Returns:
            Cleaned transcript

        """
        if not text:
            return text

        # Split into words
        words = text.split()

        # Collapse repeated n-grams (1-4 words)
        for n in range(4, 0, -1):  # Try longer phrases first
            i = 0
            while i < len(words) - n:
                # Get n-gram
                ngram = words[i:i+n]
                ngram_str = " ".join(ngram)

                # Count how many times this n-gram repeats immediately (case-insensitive)
                repeat_count = 1
                j = i + n
                while j + n <= len(words):
                    next_ngram = words[j:j+n]
                    # Case-insensitive comparison
                    if [w.lower() for w in next_ngram] == [w.lower() for w in ngram]:
                        repeat_count += 1
                        j += n
                    else:
                        break

                # If repeated 2+ times, collapse to single occurrence
                if repeat_count >= 2:
                    # Remove the repetitions
                    del words[i+n:i+n*repeat_count]
                    logger.debug(f"Collapsed {repeat_count}x repetition: '{ngram_str}'")

                i += 1

        cleaned = " ".join(words)

        if cleaned != text:
            logger.info(f"🧹 Repetition cleanup: '{text}' → '{cleaned}'")

        return cleaned

    def _wav_bytes_to_numpy(self, audio_bytes: bytes) -> npt.NDArray[np.float32]:
        """Convert WAV file bytes to numpy array.

        Args:
            audio_bytes: WAV file bytes

        Returns:
            Audio as float32 numpy array (16kHz, mono, normalized to [-1, 1])

        """
        try:
            # Parse WAV bytes
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                # Read frames
                frames = wav_file.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 2:  # 16-bit
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio_array = np.frombuffer(frames, dtype=np.int32)
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width} bytes")

                # Convert to float32 and normalize
                audio_float = audio_array.astype(np.float32)
                if sample_width == 2:
                    audio_float /= 32768.0
                elif sample_width == 4:
                    audio_float /= 2147483648.0

                # Convert stereo to mono
                if n_channels == 2:
                    audio_float = audio_float.reshape(-1, 2).mean(axis=1)

                # Resample to 16kHz if needed
                if sample_rate != self.target_sample_rate:
                    audio_float = self._resample_audio(
                        audio_float, sample_rate, self.target_sample_rate
                    )

                return audio_float

        except Exception as e:
            logger.error(f"Failed to convert WAV bytes: {e}")
            return np.array([], dtype=np.float32)

    def _resample_audio(
        self, audio: npt.NDArray[np.float32], from_rate: int, to_rate: int
    ) -> npt.NDArray[np.float32]:
        """Resample audio to target sample rate."""
        if from_rate == to_rate:
            return audio

        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(old_indices, np.arange(len(audio)), audio)
        return resampled.astype(np.float32)
