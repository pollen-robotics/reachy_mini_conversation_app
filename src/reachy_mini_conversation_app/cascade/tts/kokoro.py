"""Kokoro TTS provider for cascade pipeline (Apple Silicon optimized)."""

from __future__ import annotations
import logging
from typing import List, Optional, AsyncIterator

from .base import TTSProvider
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


class KokoroTTS(TTSProvider):
    """Kokoro-82M TTS implementation optimized for Apple Silicon."""

    def __init__(
        self,
        voice: str = "af_bella",
        lang_code: str = "a",  # 'a' for American English
        repo_id: str = "hexgrad/Kokoro-82M",
    ):
        """Initialize Kokoro TTS.

        Args:
            voice: Voice to use (af_bella, af_sarah, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis)
            lang_code: Language code ('a' for American English, 'b' for British English)
            repo_id: HuggingFace repo ID for model

        """
        self.default_voice = voice
        self.lang_code = lang_code
        self.repo_id = repo_id

        # Load pipeline immediately (not lazy) to avoid loading delay on first synthesis
        from kokoro import KPipeline
        logger.info(f"Loading Kokoro pipeline from {repo_id}...")
        self.pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id=self.repo_id
        )

        # Preload voice to avoid download delay on first synthesis
        logger.info(f"Preloading voice: {voice}...")
        try:
            # Generate a tiny test sentence to trigger voice caching
            result = self.pipeline('.', voice=voice)
            for _ in result:
                break  # Only need first chunk to trigger cache
            logger.info(f"Voice {voice} preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload voice {voice}: {e}")

        logger.info(f"Kokoro TTS initialized with voice: {voice}, lang: {lang_code}")

    def _ensure_pipeline(self) -> None:
        """Lazy load the pipeline on first use."""
        if self.pipeline is None:
            from kokoro import KPipeline

            logger.info(f"Loading Kokoro pipeline from {self.repo_id}...")
            self.pipeline = KPipeline(
                lang_code=self.lang_code,
                repo_id=self.repo_id
            )
            logger.info("Kokoro pipeline loaded successfully")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using Kokoro TTS with streaming.

        Args:
            text: Text to synthesize
            voice: Voice override (uses default if not provided)

        Yields:
            Audio bytes (PCM 16-bit, 24kHz mono)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.default_voice
        logger.info(f"Kokoro TTS: Starting synthesis for '{text[:50]}...'")

        tracker.mark("tts_start", {"text_len": len(text)})

        try:
            # Pipeline is already loaded at init
            # Generate audio (this returns chunks)
            import asyncio

            # Run synthesis in thread pool since Kokoro is synchronous
            def _synthesize_sync() -> List[bytes]:
                import time

                from reachy_mini_conversation_app.cascade.timing import tracker

                chunks = []

                # Track model generation start
                tracker.mark("tts_model_generation_start")
                generation_start = time.perf_counter()

                result = self.pipeline(text, voice=voice_to_use)

                # Track model generation complete (time to first byte equivalent for local models)
                generation_time_ms = (time.perf_counter() - generation_start) * 1000
                tracker.mark("tts_model_generation_complete", {"generation_ms": round(generation_time_ms, 1)})

                # Kokoro returns a Result object with 'audio' attribute
                import numpy as np
                if hasattr(result, 'audio'):
                    audio_data = result.audio
                else:
                    # Fallback: if it's an iterator, collect chunks
                    audio_data = []
                    for chunk in result:
                        if hasattr(chunk, 'audio'):
                            audio_data.append(chunk.audio)
                        else:
                            audio_data.append(chunk)
                    audio_data = np.concatenate(audio_data) if len(audio_data) > 0 else np.array([])

                logger.debug(f"Kokoro audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {audio_data.min():.3f}, max: {audio_data.max():.3f}")

                # Start audio processing
                tracker.mark("tts_processing_start")

                # Check for leading silence (configurable via CASCADE_TTS_TRIM_SILENCE)
                import numpy as np
                original_length = len(audio_data)
                threshold = 0.01  # Consider anything below this as silence
                non_silent = np.where(np.abs(audio_data) > threshold)[0]
                if len(non_silent) > 0:
                    first_sound_sample = non_silent[0]
                    silence_duration_ms = (first_sound_sample / 24000) * 1000
                    if silence_duration_ms > 100:  # More than 100ms silence
                        logger.warning(f"Kokoro TTS: {silence_duration_ms:.0f}ms of leading silence detected (trim_silence={config.CASCADE_TTS_TRIM_SILENCE})")
                        if config.CASCADE_TTS_TRIM_SILENCE:
                            logger.info(f"Kokoro TTS: Trimming from {original_length} to {len(audio_data) - first_sound_sample} samples")
                            audio_data = audio_data[first_sound_sample:]
                            logger.info(f"Kokoro TTS: After trim - new length: {len(audio_data)} samples ({len(audio_data)/24000*1000:.0f}ms)")
                        else:
                            logger.info("Kokoro TTS: Keeping silence (CASCADE_TTS_TRIM_SILENCE=false)")
                    else:
                        logger.debug(f"Kokoro TTS: {silence_duration_ms:.0f}ms leading silence (acceptable)")

                # Convert float32 to int16 PCM
                audio_int16 = (audio_data * 32767).astype(np.int16)

                # Split into chunks for streaming (4096 samples = ~170ms at 24kHz)
                chunk_size = 4096
                for i in range(0, len(audio_int16), chunk_size):
                    chunk = audio_int16[i:i+chunk_size]
                    chunks.append(chunk.tobytes())

                # Mark processing complete
                tracker.mark("tts_processing_end")

                return chunks

            # Run in thread pool
            chunks = await asyncio.to_thread(_synthesize_sync)

            logger.info(f"Kokoro TTS: Generated {len(chunks)} audio chunks")

            # Mark first chunk ready
            tracker.mark("tts_first_chunk_ready")

            # Yield chunks
            for i, chunk in enumerate(chunks):
                if i == 0:
                    logger.info("Kokoro TTS: First chunk ready (can start playback now!)")
                yield chunk

            logger.info(f"Kokoro TTS: Synthesis complete for '{text[:50]}...'")

        except Exception as e:
            logger.error(f"Kokoro TTS synthesis failed: {e}")
            raise
