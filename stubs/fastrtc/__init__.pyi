"""Type stubs for fastrtc package."""

from typing import Any, TypeVar
from asyncio import Queue

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")

class AdditionalOutputs:
    """Container for additional outputs from stream handlers."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class AsyncStreamHandler:
    """Base class for async stream handlers.

    Subclasses implement audio streaming with async processing.
    """

    output_sample_rate: int
    output_frame_size: int
    input_sample_rate: int
    input_frame_size: int
    latest_args: tuple[Any, ...]

    def __init__(
        self,
        expected_layout: str = ...,
        output_sample_rate: int = ...,
        input_sample_rate: int = ...,
        output_frame_size: int = ...,
        input_frame_size: int = ...,
    ) -> None: ...
    async def wait_for_args(self) -> None:
        """Wait for arguments to be provided to the handler."""
        ...

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Receive an audio frame for processing.

        Args:
            frame: Tuple of (sample_rate, audio_data)

        """
        ...

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit processed audio or additional outputs.

        Returns:
            Audio frame tuple, AdditionalOutputs, or None.

        """
        ...

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        ...

    async def start_up(self) -> None:
        """Start up the handler."""
        ...

class Stream:
    """Audio stream manager for real-time audio processing."""

    ui: Any  # Gradio Blocks instance

    def __init__(
        self,
        handler: AsyncStreamHandler,
        modality: str = ...,
        mode: str = ...,
        additional_inputs: list[Any] | None = ...,
        additional_outputs: list[Any] | None = ...,
        additional_outputs_handler: Any | None = ...,
        ui_args: dict[str, Any] | None = ...,
    ) -> None: ...

async def wait_for_item(queue: Queue[Any]) -> Any:
    """Wait for an item from a queue.

    Args:
        queue: The asyncio queue to wait on.

    Returns:
        The item from the queue (or None if queue times out).

    """
    ...

def audio_to_float32(audio: NDArray[np.int16]) -> NDArray[np.float32]:
    """Convert int16 audio to float32.

    Args:
        audio: Audio data as int16 array.

    Returns:
        Audio data as float32 array normalized to [-1, 1].

    """
    ...

def audio_to_int16(audio: NDArray[Any]) -> NDArray[np.int16]:
    """Convert audio to int16.

    Args:
        audio: Audio data as array (float32 or int16).

    Returns:
        Audio data as int16 array.

    """
    ...
