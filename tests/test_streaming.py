import asyncio

import numpy as np
import pytest

from reachy_mini_conversation_app.streaming import (
    AdditionalOutputs,
    AsyncStreamHandler,
    wait_for_item,
    audio_to_int16,
    audio_to_float32,
)


def test_additional_outputs_stores_messages() -> None:
    """Additional outputs should keep the emitted role/content payloads."""
    message = {"role": "assistant", "content": "hello"}

    outputs = AdditionalOutputs(message)

    assert outputs.args == (message,)


def test_async_stream_handler_derives_output_frame_size() -> None:
    """Stream handlers should derive 20 ms output frames from sample rate."""
    handler = AsyncStreamHandler(output_sample_rate=24000)

    assert handler.output_frame_size == 480


@pytest.mark.asyncio
async def test_wait_for_item_returns_none_on_timeout() -> None:
    """Queue waits should time out without raising."""
    queue: asyncio.Queue[int] = asyncio.Queue()

    assert await wait_for_item(queue, timeout=0.01) is None


def test_audio_to_int16_scales_float32() -> None:
    """Float audio should scale into the int16 sample range."""
    audio = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(audio_to_int16(audio), np.array([-32767, 0, 32767], dtype=np.int16))


def test_audio_to_float32_scales_int16() -> None:
    """Int16 audio should scale into the float sample range."""
    audio = np.array([-32768, 0, 32767], dtype=np.int16)

    np.testing.assert_allclose(audio_to_float32(audio), np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))


def test_audio_converters_reject_unsupported_dtype() -> None:
    """Unsupported audio dtypes should fail explicitly."""
    audio = np.array([1.0], dtype=np.float64)

    with pytest.raises(TypeError, match="Unsupported audio data type"):
        audio_to_int16(audio)
