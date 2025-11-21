"""Base handler interface for conversation providers."""

import asyncio
from abc import abstractmethod
from typing import Any, Tuple

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from numpy.typing import NDArray


class ConversationHandler(AsyncStreamHandler):
    """Abstract base class for conversation handlers.

    All conversation providers (OpenAI, Gemini, Cascaded) must implement this interface.
    This ensures compatibility with both LocalStream and fastrtc.Stream transport layers.
    """

    @abstractmethod
    async def start_up(self) -> None:
        """Initialize connections and start processing.

        This method should establish connections to the conversation provider
        and start any background tasks needed for processing.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone.

        Args:
            frame: Tuple of (sample_rate, audio_array) from microphone

        """
        raise NotImplementedError

    @abstractmethod
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame or transcript updates.

        Returns:
            - Tuple of (sample_rate, audio_array) for speaker playback
            - AdditionalOutputs containing transcripts or metadata
            - None if no output available

        """
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of all connections.

        This method should close all connections gracefully and clean up resources.
        """
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> "ConversationHandler":
        """Create a copy of the handler for new sessions.

        Returns:
            A new instance of the handler with the same configuration

        """
        raise NotImplementedError
