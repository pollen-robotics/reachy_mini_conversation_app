"""Parakeet-MLX ASR provider (Apple Silicon optimized)."""

from __future__ import annotations
import io
import wave
import asyncio
import logging
from typing import Any, Optional

import numpy as np
import mlx.core as mx
import numpy.typing as npt

from .base import ASRProvider


logger = logging.getLogger(__name__)


class ParakeetMLXASR(ASRProvider):
    """Parakeet-MLX ASR implementation optimized for Apple Silicon."""

    def __init__(
        self,
        model_name: str = "mlx-community/parakeet-tdt-0.6b-v3",
        precision: str = "fp32",
    ):
        """Initialize Parakeet-MLX ASR.

        Args:
            model_name: Model ID from HuggingFace (parakeet-tdt-0.6b-v3 recommended)
            precision: Inference precision (fp32 for quality, bf16 for speed)

        """
        self.model_name = model_name
        self.precision = precision
        self.target_sample_rate = 16000  # Parakeet requires 16kHz
        self.model: Any = None  # Will be set by _ensure_model

        # Preload model immediately to avoid first-call delay
        logger.info(f"Loading Parakeet model: {model_name} (precision: {precision})...")
        self._ensure_model()
        logger.info("Parakeet model loaded successfully")

        # Warmup: Run a dummy inference to pre-compile kernels
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
            logger.info(f"Parakeet warmup complete! First inference took {warmup_duration:.0f}ms (subsequent calls will be ~4-5x faster)")

        except Exception as e:
            logger.warning(f"Parakeet warmup failed (non-critical): {e}")

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio using Parakeet-MLX (batch mode).

        Args:
            audio_bytes: Audio data in WAV format
            language: Language code (currently unused - Parakeet auto-detects)

        Returns:
            Transcribed text

        """
        import tempfile
        from pathlib import Path

        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.debug(f"Transcribing {len(audio_bytes)} bytes of audio with Parakeet-MLX")

        tracker.mark("asr_start", {"bytes": len(audio_bytes)})

        # Create temporary WAV file for Parakeet
        # Parakeet expects a file path, not raw audio data
        temp_file = None
        try:
            # Save audio bytes to temporary file
            tracker.mark("asr_temp_file_write_start")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_bytes)
            temp_file.close()
            tracker.mark("asr_temp_file_write_end")

            temp_path = temp_file.name
            logger.debug(f"Saved audio to temporary file: {temp_path}")

            # Run transcription in thread pool (Parakeet is synchronous)
            def _transcribe_sync() -> str:
                from reachy_mini_conversation_app.cascade.timing import tracker

                tracker.mark("asr_model_inference_start")
                # Transcribe using Parakeet (pass file path)
                result = self.model.transcribe(temp_path)
                tracker.mark("asr_model_inference_end")

                # Extract text from result
                if hasattr(result, "text"):
                    return str(result.text)
                elif isinstance(result, str):
                    return result
                elif isinstance(result, dict) and "text" in result:
                    return str(result["text"])
                else:
                    # Fallback: try to convert to string
                    logger.warning(f"Unexpected result type from Parakeet: {type(result)}")
                    return str(result)

            transcript = await asyncio.to_thread(_transcribe_sync)

            # Clean up transcript
            transcript = transcript.strip()

            logger.info(f"Parakeet transcription successful: '{transcript[:100]}...'")

            tracker.mark("asr_complete", {"transcript_len": len(transcript)})
            return transcript

        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}")
            raise

        finally:
            # Clean up temporary file
            if temp_file is not None:
                try:
                    Path(temp_file.name).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file.name}: {e}")

    def _parse_wav_bytes(self, audio_bytes: bytes) -> npt.NDArray[np.float32]:
        """Parse WAV file bytes into numpy array.

        Args:
            audio_bytes: WAV file bytes

        Returns:
            Audio as float32 numpy array (normalized to [-1, 1])

        """
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()

            # Store detected sample rate for resampling check
            self._detected_sample_rate = sample_rate

            logger.debug(
                f"WAV: {sample_rate}Hz, {n_channels}ch, {sample_width}byte, {n_frames} frames"
            )

            # Read frames
            frames = wav_file.readframes(n_frames)

            # Convert to numpy array based on sample width
            if sample_width == 2:  # 16-bit
                audio_array = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_array = np.frombuffer(frames, dtype=np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")

            # Convert to float32 and normalize to [-1, 1]
            audio_float = audio_array.astype(np.float32)
            if sample_width == 2:
                audio_float /= 32768.0  # int16 max
            elif sample_width == 4:
                audio_float /= 2147483648.0  # int32 max

            # Convert stereo to mono if needed
            if n_channels == 2:
                audio_float = audio_float.reshape(-1, 2).mean(axis=1)
                logger.debug("Converted stereo to mono")

            return audio_float

    def _resample_audio(
        self, audio: npt.NDArray[np.float32], from_rate: int, to_rate: int
    ) -> npt.NDArray[np.float32]:
        """Resample audio to target sample rate.

        Args:
            audio: Audio array
            from_rate: Current sample rate
            to_rate: Target sample rate

        Returns:
            Resampled audio

        """
        if from_rate == to_rate:
            return audio

        logger.debug(f"Resampling audio: {from_rate}Hz → {to_rate}Hz")

        # Simple linear interpolation resampling
        # For production, consider using scipy.signal.resample or librosa
        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)

        # Linear interpolation indices
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(old_indices, np.arange(len(audio)), audio)

        logger.debug(f"Resampled: {len(audio)} → {len(resampled)} samples")
        result: npt.NDArray[np.float32] = resampled.astype(np.float32)
        return result
