import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _make_handler():
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    deps = _make_deps()
    handler = GeminiTTSResponseHandler(deps)
    handler._client = MagicMock()
    return handler


@pytest.mark.asyncio
async def test_conversation_history_accumulates() -> None:
    """Each transcript + response is appended to history in the correct Gemini format."""
    handler = _make_handler()

    async def fake_llm() -> str:
        return "You look like you comb your hair with a pork chop."

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        return b"\x00\x01" * 2400  # 2400 samples of silence

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("tell me a joke")

    assert len(handler._conversation_history) == 2
    assert handler._conversation_history[0]["role"] == "user"
    assert handler._conversation_history[0]["parts"][0]["text"] == "tell me a joke"
    assert handler._conversation_history[1]["role"] == "model"
    assert "pork chop" in handler._conversation_history[1]["parts"][0]["text"]


@pytest.mark.asyncio
async def test_pcm_bytes_are_chunked_and_queued() -> None:
    """TTS PCM output is split into ~2400-sample frames and pushed to output_queue."""
    handler = _make_handler()

    # 7200 samples = 3 frames of 2400
    raw_samples = np.zeros(7200, dtype=np.int16)
    pcm_bytes = raw_samples.tobytes()

    async def fake_llm() -> str:
        return "Hockey puck."

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        return pcm_bytes

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("say something")

    audio_frames = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, tuple):
            audio_frames.append(item)

    assert len(audio_frames) == 3
    sample_rate, samples = audio_frames[0]
    assert sample_rate == 24000
    assert len(samples) == 2400


@pytest.mark.asyncio
async def test_tts_failure_pushes_error_output() -> None:
    """When TTS returns None (all retries exhausted), an error AdditionalOutputs is queued."""
    handler = _make_handler()

    async def fake_llm() -> str:
        return "Beautiful."

    async def failing_tts(text: str, system_instruction: str | None = None) -> None:
        return None  # simulate all retries exhausted

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = failing_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("say something")

    items = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())

    error_items = [i for i in items if isinstance(i, AdditionalOutputs) and "error" in str(i.args).lower()]
    assert len(error_items) >= 1


@pytest.mark.asyncio
async def test_call_tts_with_retry_returns_none_after_all_failures() -> None:
    """_call_tts_with_retry returns None when all 3 attempts raise."""
    handler = _make_handler()

    with patch.object(handler, "_client") as mock_client:
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("500"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await handler._call_tts_with_retry("Hello")

    assert result is None


@pytest.mark.asyncio
async def test_voice_override() -> None:
    """get_current_voice returns the override when set, otherwise the default."""
    from robot_comic.config import GEMINI_TTS_DEFAULT_VOICE
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    handler = GeminiTTSResponseHandler(_make_deps())
    assert handler.get_current_voice() == GEMINI_TTS_DEFAULT_VOICE

    handler._voice_override = "Puck"
    assert handler.get_current_voice() == "Puck"


@pytest.mark.asyncio
async def test_synthetic_status_marker_kept_out_of_history() -> None:
    """Issue #306: if response_text is a status marker, it must not pollute history."""
    handler = _make_handler()

    async def fake_llm() -> str:
        return "[Skipped TTS: empty LLM response]"

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        return b"\x00\x01" * 2400

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("hello")

    # Monitor still saw the marker on the output queue.
    seen: list[str] = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, AdditionalOutputs):
            seen.append(str(item.args))
    assert any("[Skipped TTS:" in m for m in seen)

    # But _conversation_history has no model-role marker.
    model_turns = [t for t in handler._conversation_history if t.get("role") == "model"]
    assert model_turns == []


@pytest.mark.asyncio
async def test_apply_personality_clears_history() -> None:
    """apply_personality resets conversation history."""
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    handler = GeminiTTSResponseHandler(_make_deps())
    handler._conversation_history = [{"role": "user", "parts": [{"text": "hi"}]}]

    with patch("robot_comic.gemini_tts.set_custom_profile"):
        await handler.apply_personality("house_comedian")

    assert handler._conversation_history == []


@pytest.mark.asyncio
async def test_short_pause_tag_inserts_silence_before_sentence() -> None:
    """A sentence carrying [short pause] gets a silence frame burst before its TTS audio."""
    handler = _make_handler()

    raw_samples = np.zeros(2400, dtype=np.int16)
    pcm_bytes = raw_samples.tobytes()
    captured: list[str] = []

    async def fake_llm() -> str:
        return "First line. [short pause] Second line."

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        captured.append(text)
        return pcm_bytes

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("go")

    audio_frames = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, tuple):
            audio_frames.append(item)

    # Two sentences (one PCM frame each) plus silence frames for the [short pause]
    # before the second sentence. Expect more than 2 audio frames total.
    assert len(audio_frames) > 2
    assert captured == ["First line.", "Second line."]


@pytest.mark.asyncio
async def test_tts_call_includes_speed_delivery_cue() -> None:
    """_call_tts_with_retry conveys fast-delivery styling to TTS.

    The default model (`gemini-3.1-flash-tts-preview`) rejects
    `system_instruction`, so the instruction is prepended to the spoken
    contents as a parenthetical cue instead. Validate either path so this
    test stays useful if GEMINI_TTS_MODEL changes.
    """
    handler = _make_handler()

    captured_configs = []
    captured_contents = []

    async def fake_generate(model, contents, config):
        captured_configs.append(config)
        captured_contents.append(contents)
        fake_data = b"\x00" * 4800

        encoded = base64.b64encode(fake_data).decode()
        part = MagicMock()
        part.inline_data.data = encoded
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        return response

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = fake_generate

    result = await handler._call_tts_with_retry("You hockey puck!")

    assert result is not None
    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    contents = captured_contents[0]
    if cfg.system_instruction is not None:
        instruction_text = cfg.system_instruction.lower()
    else:
        # Preview-model path: cue is prepended to contents.
        instruction_text = contents.lower()
    assert "fast" in instruction_text or "brooklyn" in instruction_text or "pace" in instruction_text


# ---------------------------------------------------------------------------
# Phase 5e.6 — leaf guard on GeminiTTSResponseHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_startup_credentials_is_idempotent() -> None:
    """Second call must NOT re-instantiate ``genai.Client``.

    Pre-5e.6, the mixin shell's ``_prepare_startup_credentials``
    (``LocalSTTInputMixin._prepare_startup_credentials``) gated the
    handler's prepare via ``_startup_credentials_ready``. Post-5e.6
    the migrated triple's factory composes a plain
    :class:`GeminiTTSResponseHandler` (no mixin shell), and the LLM
    and TTS adapters each call ``handler._prepare_startup_credentials``
    once during their ``prepare`` lifecycle. Without a per-handler
    guard the second call would leak a fresh ``genai.Client``.

    Pattern mirrors 5e.3 / 5e.4 / 5e.5 leaf-guard tests.
    """
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    handler = GeminiTTSResponseHandler(_make_deps())

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock(name="genai_Client_instance")

        await handler._prepare_startup_credentials()
        first_client = handler._client
        assert first_client is not None
        assert mock_client_cls.call_count == 1

        await handler._prepare_startup_credentials()
        # Same client instance; no SDK-client leak.
        assert handler._client is first_client
        assert mock_client_cls.call_count == 1
