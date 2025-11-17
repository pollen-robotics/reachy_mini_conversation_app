"""ElevenLabs TTS provider for cascade pipeline."""

from __future__ import annotations
import logging
from typing import List, Optional, AsyncIterator

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from .base import TTSProvider
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS implementation using Eleven Flash v2.5."""

    def __init__(
        self,
        api_key: str,
        voice_id: str = "pNInz6obpgDQGcFmaJgB",  # Adam voice
        model: str = "eleven_flash_v2_5",
        output_format: str = "pcm_24000",  # PCM 24kHz to match other TTS providers
    ):
        """Initialize ElevenLabs TTS.

        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use (default: Adam)
            model: TTS model (eleven_flash_v2_5 for lowest latency, eleven_multilingual_v2 for quality)
            output_format: Audio format (pcm_24000 for 24kHz 16-bit PCM)

        """
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model = model
        self.output_format = output_format
        logger.info(f"Initialized ElevenLabs TTS with model: {model}, voice: {voice_id}, format: {output_format}")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using ElevenLabs TTS API with streaming.

        Args:
            text: Text to synthesize
            voice: Voice ID override (uses default if not provided)

        Yields:
            Audio bytes (PCM 16-bit, 24kHz mono)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.voice_id
        logger.info(f"ElevenLabs TTS: Starting synthesis for '{text[:50]}...'")

        tracker.mark("tts_start", {"text_len": len(text)})

        try:
            # ElevenLabs streaming API
            import time
            import asyncio

            def _synthesize_sync() -> List[bytes]:
                """Run synchronous synthesis (ElevenLabs SDK is sync)."""
                from reachy_mini_conversation_app.cascade.timing import tracker

                tracker.mark("tts_api_request_sending")
                request_start = time.perf_counter()
                # Use convert() to get complete audio
                # Both convert() and stream() return generators - we need to consume them fully
                audio_generator = self.client.text_to_speech.convert(
                    voice_id=voice_to_use,
                    output_format=self.output_format,
                    text=text,
                    model_id=self.model,
                    # Voice settings optimized for low latency
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.0,
                    ),
                )

                # Consume the generator completely to get all audio
                audio_chunks = []
                first_byte = True
                for chunk in audio_generator:
                    if chunk:
                        if first_byte:
                            # Track time to first byte from API
                            ttfb_ms = (time.perf_counter() - request_start) * 1000
                            tracker.mark("tts_api_first_byte", {"ttfb_ms": round(ttfb_ms, 1)})
                            first_byte = False
                        audio_chunks.append(chunk)

                # Combine all chunks into single bytes object
                audio_bytes = b''.join(audio_chunks)
                logger.debug(f"ElevenLabs API: Received {len(audio_chunks)} chunks, {len(audio_bytes)} bytes total")

                tracker.mark("tts_api_complete")

                # Return as a single chunk (will be re-chunked later for streaming)
                return [audio_bytes]

            # Run in thread pool since ElevenLabs SDK is synchronous
            chunks = await asyncio.to_thread(_synthesize_sync)

            import numpy as np

            original_total_bytes = sum(len(c) for c in chunks)
            logger.debug(f"ElevenLabs: Received {original_total_bytes} bytes from API")

            # Convert to numpy array for processing
            tracker.mark("tts_processing_start")
            full_audio = b''.join(chunks)
            audio_array = np.frombuffer(full_audio, dtype=np.int16)

            logger.info(f"ElevenLabs TTS audio: {len(audio_array)} samples ({len(audio_array)/24000:.2f}s), min: {audio_array.min()}, max: {audio_array.max()}")

            # Check for leading silence (if trimming enabled)
            if config.CASCADE_TTS_TRIM_SILENCE:
                threshold = 327  # For int16 PCM (0.01 * 32767)
                non_silent = np.where(np.abs(audio_array) > threshold)[0]
                if len(non_silent) > 0:
                    first_sound_sample = non_silent[0]
                    silence_duration_ms = (first_sound_sample / 24000) * 1000
                    if silence_duration_ms > 100:
                        logger.warning(f"ElevenLabs TTS: {silence_duration_ms:.0f}ms of leading silence detected - trimming")
                        logger.info(f"ElevenLabs TTS: Trimming from {len(audio_array)} to {len(audio_array) - first_sound_sample} samples")
                        audio_array = audio_array[first_sound_sample:]
                        logger.info(f"ElevenLabs TTS: After trim - new length: {len(audio_array)} samples ({len(audio_array)/24000:.2f}s)")
                    else:
                        logger.debug(f"ElevenLabs TTS: {silence_duration_ms:.0f}ms leading silence (acceptable)")
                else:
                    logger.warning("ElevenLabs TTS: No non-silent samples found!")

            # Always re-chunk for streaming playback (4096 samples = ~170ms at 24kHz)
            audio_bytes = audio_array.tobytes()
            chunk_size = 4096 * 2  # 4096 samples * 2 bytes per sample
            chunks = []
            for i in range(0, len(audio_bytes), chunk_size):
                chunks.append(audio_bytes[i:i+chunk_size])

            logger.info(f"ElevenLabs: Final output: {len(chunks)} chunks, {len(audio_bytes)} bytes ({len(audio_array)/24000:.2f}s)")

            tracker.mark("tts_processing_end")
            tracker.mark("tts_first_chunk_ready")

            # Yield chunks for streaming
            for i, chunk in enumerate(chunks):
                if i == 0:
                    logger.info("ElevenLabs TTS: First chunk ready (can start playback now!)")
                yield chunk

            logger.info(f"ElevenLabs TTS: Synthesis complete for '{text[:50]}...'")

        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis failed: {e}")
            raise
