from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import TypeAlias
from collections.abc import Callable

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from numpy.typing import NDArray

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
HandlerOutput: TypeAlias = AudioFrame | AdditionalOutputs | None
QueueItem: TypeAlias = AudioFrame | AdditionalOutputs


class ConversationHandler(AsyncStreamHandler, ABC):
    """Shared app handler contract for realtime conversation backends."""

    deps: ToolDependencies
    output_queue: asyncio.Queue[QueueItem]
    _clear_queue: Callable[[], None] | None

    @abstractmethod
    def copy(self) -> ConversationHandler:
        """Create a copy of the handler."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def emit(self) -> HandlerOutput:
        """Emit the next output item."""
        ...

    @abstractmethod
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile."""
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Return voices available for the active backend."""
        ...

    @abstractmethod
    def get_current_voice(self) -> str:
        """Return the current voice."""
        ...

    @abstractmethod
    async def change_voice(self, voice: str) -> str:
        """Change the current voice."""
        ...
