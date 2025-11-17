"""OpenAI TTS implementation."""

from __future__ import annotations
import logging
from typing import Literal, Optional, AsyncIterator

from openai import AsyncOpenAI

from .base import TTSProvider
from reachy_mini_conversation_app.config import config


AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


logger = logging.getLogger(__name__)


class OpenAITTS(TTSProvider):
    """OpenAI TTS API implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: AudioFormat = "pcm",
    ):
        """Initialize OpenAI TTS.

        Args:
            api_key: OpenAI API key
            model: TTS model (tts-1 or tts-1-hd)
                   Note: tts-1 has lower latency than tts-1-hd
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            response_format: Audio format (pcm, mp3, opus, aac, flac, wav)

        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.default_voice = voice
        self.response_format = response_format
        logger.info(f"Initialized OpenAI TTS with model: {model}, voice: {voice} (format: {response_format})")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using OpenAI TTS API with streaming.

        Args:
            text: Text to synthesize
            voice: Voice override (uses default if not provided)

        Yields:
            Audio bytes

        """
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self.default_voice
        logger.info(f"TTS: Starting synthesis for '{text[:50]}...'")

        try:
            # OpenAI TTS streaming - collect all chunks first to check for silence
            all_chunks = []
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_to_use,
                input=text,
                response_format=self.response_format,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=1024):
                    if chunk:
                        all_chunks.append(chunk)

            # Combine all chunks to check for leading silence
            import numpy as np
            full_audio = b''.join(all_chunks)
            audio_array = np.frombuffer(full_audio, dtype=np.int16)

            logger.debug(f"OpenAI TTS audio: {len(audio_array)} samples, min: {audio_array.min()}, max: {audio_array.max()}")

            # Check for leading silence (configurable via CASCADE_TTS_TRIM_SILENCE)
            threshold = 327  # For int16 PCM (0.01 * 32767)
            non_silent = np.where(np.abs(audio_array) > threshold)[0]
            if len(non_silent) > 0:
                first_sound_sample = non_silent[0]
                silence_duration_ms = (first_sound_sample / 24000) * 1000
                if silence_duration_ms > 100:
                    logger.warning(f"OpenAI TTS: {silence_duration_ms:.0f}ms of leading silence detected (trim_silence={config.CASCADE_TTS_TRIM_SILENCE})")
                    if config.CASCADE_TTS_TRIM_SILENCE:
                        logger.info(f"OpenAI TTS: Trimming from {len(audio_array)} to {len(audio_array) - first_sound_sample} samples")
                        audio_array = audio_array[first_sound_sample:]
                        logger.info(f"OpenAI TTS: After trim - new length: {len(audio_array)} samples ({len(audio_array)/24000*1000:.0f}ms)")
                    else:
                        logger.info("OpenAI TTS: Keeping silence (CASCADE_TTS_TRIM_SILENCE=false)")
                else:
                    logger.debug(f"OpenAI TTS: {silence_duration_ms:.0f}ms leading silence (acceptable)")

            # Re-chunk the (potentially trimmed) audio
            trimmed_bytes = audio_array.tobytes()
            chunk_size = 1024
            chunk_count = 0
            for i in range(0, len(trimmed_bytes), chunk_size):
                chunk = trimmed_bytes[i:i+chunk_size]
                if chunk:
                    if chunk_count == 0:
                        logger.info("TTS: First chunk ready (can start playback now!)")
                    chunk_count += 1
                    yield chunk

            logger.info(f"TTS: Synthesis complete - {chunk_count} chunks for '{text[:50]}...'")

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
