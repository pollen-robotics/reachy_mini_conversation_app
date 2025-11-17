"""OpenAI Whisper ASR implementation."""

from __future__ import annotations
import io
import asyncio
import logging
from typing import Optional

from openai import OpenAI

from .base import ASRProvider


logger = logging.getLogger(__name__)


class OpenAIWhisperASR(ASRProvider):
    """OpenAI Whisper API implementation for ASR."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        """Initialize OpenAI Whisper ASR.

        Args:
            api_key: OpenAI API key
            model: Whisper model name (default: whisper-1)

        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI Whisper ASR with model: {model}")

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio using OpenAI Whisper API.

        Args:
            audio_bytes: Audio data in WAV or other supported format
            language: Optional language code for transcription

        Returns:
            Transcribed text

        """
        logger.debug(f"Transcribing {len(audio_bytes)} bytes of audio")

        try:
            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # Whisper API needs a filename

            # Run synchronous OpenAI call in thread pool
            def _transcribe() -> str:
                if language:
                    resp = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=language
                    )
                else:
                    resp = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file
                    )
                return resp.text.strip()

            transcript = await asyncio.to_thread(_transcribe)
            logger.info(f"Transcription successful: {transcript[:100]}...")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
