import asyncio
from typing import TypeVar, TypeAlias
from collections.abc import Mapping, Callable

import numpy as np
from numpy.typing import NDArray


StreamMessage: TypeAlias = Mapping[str, object]
AudioArray: TypeAlias = NDArray[np.int16] | NDArray[np.float32]

QueueItem = TypeVar("QueueItem")


class AdditionalOutputs:
    """Text or metadata emitted alongside audio frames."""

    def __init__(self, *args: StreamMessage) -> None:
        """Initialize with one or more emitted messages."""
        self.args = args


class AsyncStreamHandler:
    """Minimal async stream handler state used by the local audio loop."""

    def __init__(
        self,
        output_sample_rate: int = 24000,
        input_sample_rate: int = 48000,
    ) -> None:
        """Initialize the audio stream metadata used by conversation handlers."""
        if output_sample_rate % 50 != 0:
            raise ValueError(f"output_sample_rate must be a multiple of 50, got {output_sample_rate}")
        self.output_sample_rate = output_sample_rate
        self.input_sample_rate = input_sample_rate
        self._clear_queue: Callable[[], None] | None = None


async def wait_for_item(queue: asyncio.Queue[QueueItem], timeout: float = 0.1) -> QueueItem | None:
    """Return the next queue item, or None when no item arrives before timeout."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def audio_to_int16(audio: AudioArray) -> NDArray[np.int16]:
    """Convert int16 or float32 audio data to int16 samples."""
    if audio.dtype == np.int16:
        return audio.astype(np.int16, copy=False)
    if audio.dtype == np.float32:
        return (audio * 32767.0).astype(np.int16)
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def audio_to_float32(audio: AudioArray) -> NDArray[np.float32]:
    """Convert int16 or float32 audio data to float32 samples."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.float32:
        return audio.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")
