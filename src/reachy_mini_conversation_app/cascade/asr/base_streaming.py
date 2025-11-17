"""Streaming ASR abstraction for cascade pipeline."""

from __future__ import annotations
import abc
from typing import Optional

from .base import ASRProvider


class StreamingASRProvider(ASRProvider):
    """Abstract base class for streaming ASR providers.

    Streaming providers support real-time transcription where audio chunks
    are sent continuously and partial transcripts are received incrementally.
    """

    @abc.abstractmethod
    async def start_stream(self) -> None:
        """Initialize a streaming session.

        This should establish any necessary connections (e.g., WebSocket)
        and prepare the provider to receive audio chunks.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send an audio chunk to the streaming ASR service.

        Args:
            audio_chunk: Audio data in the expected format (typically raw PCM or WAV)

        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_partial_transcript(self) -> Optional[str]:
        """Get the current partial transcript if available.

        Returns:
            Partial transcript string, or None if no transcript is ready yet

        """
        raise NotImplementedError

    @abc.abstractmethod
    async def end_stream(self) -> str:
        """Finalize the streaming session and get the final transcript.

        This should close any connections and return the complete transcription.

        Returns:
            Final complete transcript

        """
        raise NotImplementedError

    # Implement batch transcribe for fallback/compatibility
    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Fallback batch transcription using streaming interface.

        This allows streaming providers to work with code expecting batch ASR.
        """
        await self.start_stream()
        await self.send_audio_chunk(audio_bytes)
        return await self.end_stream()
