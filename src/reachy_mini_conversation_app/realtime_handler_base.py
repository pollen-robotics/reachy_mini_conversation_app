"""Base class for realtime conversation handlers."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

from fastrtc import AsyncStreamHandler, AdditionalOutputs
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class RealtimeHandlerBase(AsyncStreamHandler, ABC):
    """Abstract base class for realtime conversation handlers.
    
    Defines the interface that all realtime providers (OpenAI, Sarvam, etc.) must implement.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        expected_layout: str = "mono",
        output_sample_rate: int = 24000,
        input_sample_rate: int = 24000,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        """Initialize the realtime handler.
        
        Args:
            deps: Tool dependencies for the conversation
            expected_layout: Audio layout (default: mono)
            output_sample_rate: Output audio sample rate
            input_sample_rate: Input audio sample rate
            gradio_mode: Whether running in Gradio mode
            instance_path: Path to instance directory for config persistence
        """
        super().__init__(
            expected_layout=expected_layout,
            output_sample_rate=output_sample_rate,
            input_sample_rate=input_sample_rate,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        self.connection: Any = None
        self._shutdown_requested: bool = False

    @abstractmethod
    def copy(self) -> "RealtimeHandlerBase":
        """Create a copy of the handler."""
        pass

    @abstractmethod
    async def start_up(self) -> None:
        """Start the handler and establish connection to the realtime service."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the handler and clean up resources."""
        pass

    @abstractmethod
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone."""
        pass

    @abstractmethod
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to speaker."""
        pass

    @abstractmethod
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality profile at runtime."""
        pass

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Get list of available voices."""
        pass

    @abstractmethod
    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to the service."""
        pass
