"""ASR (Automatic Speech Recognition) abstraction for cascade pipeline."""

from __future__ import annotations
import abc
from typing import Optional


class ASRProvider(abc.ABC):
    """Abstract base class for ASR providers."""

    @abc.abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV format preferred)
            language: Optional language code (e.g., 'en', 'fr')

        Returns:
            Transcribed text string

        """
        raise NotImplementedError

