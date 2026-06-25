import asyncio
import warnings
from typing import Literal, TypeVar, TypeAlias
from collections.abc import Mapping, Callable

import numpy as np
from numpy.typing import NDArray


StreamMessage: TypeAlias = Mapping[str, object]
AudioArray: TypeAlias = NDArray[np.int16] | NDArray[np.float32]
AudioInput: TypeAlias = AudioArray | tuple[int, AudioArray]

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
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int | None = None,
        input_sample_rate: int = 48000,
        fps: int = 30,
    ) -> None:
        """Initialize the audio stream metadata used by conversation handlers."""
        self.expected_layout = expected_layout
        self.output_sample_rate = output_sample_rate
        self.input_sample_rate = input_sample_rate
        self.fps = fps
        self.latest_args: list[object] = []
        self.args_set = asyncio.Event()
        self.channel_set = asyncio.Event()
        self._clear_queue: Callable[[], None] | None = None

        sample_rate_to_frame_size_coef = 50
        if output_sample_rate % sample_rate_to_frame_size_coef != 0:
            raise ValueError(
                f"output_sample_rate must be a multiple of {sample_rate_to_frame_size_coef}, got {output_sample_rate}"
            )

        actual_output_frame_size = output_sample_rate // sample_rate_to_frame_size_coef
        if output_frame_size is not None and output_frame_size != actual_output_frame_size:
            warnings.warn(
                "The output_frame_size parameter is deprecated and will be removed "
                "in a future release. The value passed in will be ignored. "
                f"The actual output frame size is {actual_output_frame_size}, "
                f"corresponding to {1 / sample_rate_to_frame_size_coef:.2f}s "
                f"at output_sample_rate={output_sample_rate}Hz.",
                UserWarning,
                stacklevel=2,
            )
        self.output_frame_size = actual_output_frame_size


async def wait_for_item(queue: asyncio.Queue[QueueItem], timeout: float = 0.1) -> QueueItem | None:
    """Return the next queue item, or None when no item arrives before timeout."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def _unpack_audio(audio: AudioInput) -> AudioArray:
    if isinstance(audio, tuple):
        warnings.warn(
            "Passing a (sample_rate, audio) tuple is deprecated; pass only the audio array.",
            UserWarning,
            stacklevel=2,
        )
        return audio[1]
    return audio


def audio_to_int16(audio: AudioInput) -> NDArray[np.int16]:
    """Convert int16 or float32 audio data to int16 samples."""
    audio_array = _unpack_audio(audio)
    if audio_array.dtype == np.int16:
        return audio_array.astype(np.int16, copy=False)
    if audio_array.dtype == np.float32:
        return (audio_array * 32767.0).astype(np.int16)
    raise TypeError(f"Unsupported audio data type: {audio_array.dtype}")


def audio_to_float32(audio: AudioInput) -> NDArray[np.float32]:
    """Convert int16 or float32 audio data to float32 samples."""
    audio_array = _unpack_audio(audio)
    if audio_array.dtype == np.int16:
        return audio_array.astype(np.float32) / 32768.0
    if audio_array.dtype == np.float32:
        return audio_array.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported audio data type: {audio_array.dtype}")
