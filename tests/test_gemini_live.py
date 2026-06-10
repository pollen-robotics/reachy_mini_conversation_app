"""Behavior tests for the Gemini Live handler."""

import base64
import asyncio
from types import SimpleNamespace
from typing import Any, Callable, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

import robot_comic.gemini_live as gemini_mod
from robot_comic.gemini_live import GeminiLiveHandler, _strip_tts_delivery_tags
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.tools.tool_constants import ToolState
from robot_comic.tools.background_tool_manager import ToolNotification


def _server_content(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "model_turn": None,
        "turn_complete": None,
        "interrupted": None,
        "grounding_metadata": None,
        "generation_complete": None,
        "input_transcription": None,
        "output_transcription": None,
        "url_context_metadata": None,
        "turn_complete_reason": None,
        "waiting_for_input": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _response(server_content: Any = None, tool_call: Any = None) -> SimpleNamespace:
    return SimpleNamespace(server_content=server_content, tool_call=tool_call)


async def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    # Bumped from 1.0 to 5.0 under test-infra-2 — at `-n auto` the loaded
    # worker can blow a 1 s budget waiting on `_FakeSession`'s background
    # task without indicating any real bug.
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


class _FakeSession:
    def __init__(self, batches: list[list[SimpleNamespace]], stop_event: asyncio.Event) -> None:
        self._batches = list(batches)
        self._stop_event = stop_event
        self.realtime_inputs: list[dict[str, Any]] = []
        self.tool_responses: list[dict[str, Any]] = []

    async def close(self) -> None:
        self._stop_event.set()

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime_inputs.append(kwargs)
        return None

    async def send_tool_response(self, **kwargs: Any) -> None:
        self.tool_responses.append(kwargs)
        return None

    async def receive(self) -> AsyncIterator[SimpleNamespace]:
        if self._batches:
            for response in self._batches.pop(0):
                yield response
            return

        await self._stop_event.wait()
        return
        yield


class _FakeConnectContext:
    def __init__(self, session: _FakeSession):
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _FakeLiveClient:
    def __init__(self, session: _FakeSession) -> None:
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=lambda **_kwargs: _FakeConnectContext(session)))


@pytest.mark.asyncio
async def test_gemini_turn_buffers_transcripts_and_schedules_motion_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini turns should emit one transcript per role and let the wobbler reset after speech."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = False
    head_wobbler = MagicMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None))
    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        head_wobbler=head_wobbler,
    )
    handler = GeminiLiveHandler(deps)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    audio_bytes = b"\x00\x00\x10\x00" * 256
    session = _FakeSession(
        batches=[
            [
                _response(
                    _server_content(
                        input_transcription=SimpleNamespace(text="How's it going, Reachy?"),
                    )
                ),
                _response(
                    _server_content(
                        model_turn=SimpleNamespace(
                            parts=[SimpleNamespace(inline_data=SimpleNamespace(data=audio_bytes))]
                        ),
                    )
                ),
                _response(
                    _server_content(
                        output_transcription=SimpleNamespace(text="Doing"),
                    )
                ),
                _response(
                    _server_content(
                        output_transcription=SimpleNamespace(text=" great."),
                    )
                ),
                _response(
                    _server_content(
                        turn_complete=True,
                    )
                ),
            ]
        ],
        stop_event=handler._stop_event,
    )
    handler.client = _FakeLiveClient(session)

    task = asyncio.create_task(handler._run_live_session())
    await _wait_for(
        lambda: head_wobbler.request_reset_after_current_audio.called and handler.output_queue.qsize() >= 3
    )

    handler._stop_event.set()
    # 5.0 s budget for the same reason as `_wait_for` above.
    await asyncio.wait_for(task, timeout=5.0)

    outputs = []
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())

    transcript_messages = [
        message
        for output in outputs
        if isinstance(output, AdditionalOutputs)
        for message in output.args
        if isinstance(message.get("content"), str)
    ]

    assert transcript_messages == [
        {"role": "user", "content": "How's it going, Reachy?"},
        {"role": "assistant", "content": "Doing great."},
    ]
    assert any(isinstance(output, tuple) for output in outputs), "audio output was not emitted"
    movement_manager.set_listening.assert_has_calls([call(True), call(False)])
    assert movement_manager.set_listening.call_args_list[-1] == call(False)
    head_wobbler.feed.assert_not_called()
    head_wobbler.request_reset_after_current_audio.assert_called_once()
    head_wobbler.reset.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_camera_tool_sends_snapshot_and_returns_json_result() -> None:
    """Camera tool should push the snapshot via realtime video input and return a JSON-safe tool result."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )
    handler = GeminiLiveHandler(deps)
    session = _FakeSession([], handler._stop_event)
    handler.session = session

    await handler._handle_tool_result(
        ToolNotification(
            id="call_camera_1",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"b64_im": base64.b64encode(b"jpeg-bytes").decode("ascii")},
        )
    )

    # Image pushed as a realtime video frame (not embedded in FunctionResponse)
    assert len(session.realtime_inputs) == 1
    blob = session.realtime_inputs[0]["video"]
    assert blob.data == b"jpeg-bytes"
    assert blob.mime_type == "image/jpeg"

    # Tool response is plain JSON (no binary, no parts)
    assert len(session.tool_responses) == 1
    function_response = session.tool_responses[0]["function_responses"][0]
    assert function_response.response == {"status": "image_captured"}
    assert not hasattr(function_response, "parts") or function_response.parts is None

    # Console output is JSON-safe
    outputs = []
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())

    tool_messages = [
        message
        for output in outputs
        if isinstance(output, AdditionalOutputs)
        for message in output.args
        if isinstance(message.get("content"), str)
    ]
    assert tool_messages == [
        {
            "role": "assistant",
            "content": '{"status": "image_captured"}',
            "metadata": {"title": "🛠️ Used tool camera", "status": "done"},
        }
    ]


@pytest.mark.asyncio
async def test_gemini_tool_result_sends_b64_scene_as_video_and_compacts_json() -> None:
    """Profile tools can return b64_scene without wedging the live function response."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
    )
    handler = GeminiLiveHandler(deps)
    session = _FakeSession([], handler._stop_event)
    handler.session = session

    await handler._handle_tool_result(
        ToolNotification(
            id="call_roast_1",
            tool_name="roast",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={
                "b64_scene": base64.b64encode(b"scene-jpeg").decode("ascii"),
                "extraction_prompt": "describe the person",
                "note": "No local vision processor available.",
            },
        )
    )

    assert len(session.realtime_inputs) == 1
    blob = session.realtime_inputs[0]["video"]
    assert blob.data == b"scene-jpeg"
    assert blob.mime_type == "image/jpeg"

    assert len(session.tool_responses) == 1
    function_response = session.tool_responses[0]["function_responses"][0]
    assert function_response.response == {
        "extraction_prompt": "describe the person",
        "note": "No local vision processor available.",
        "image": "sent_as_realtime_video_input",
        "image_source": "b64_scene",
    }
    assert "b64_scene" not in function_response.response

    outputs = []
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())

    tool_messages = [
        message
        for output in outputs
        if isinstance(output, AdditionalOutputs)
        for message in output.args
        if isinstance(message.get("content"), str)
    ]
    assert tool_messages == [
        {
            "role": "assistant",
            "content": (
                '{"extraction_prompt": "describe the person", '
                '"note": "No local vision processor available.", '
                '"image": "sent_as_realtime_video_input", '
                '"image_source": "b64_scene"}'
            ),
            "metadata": {"title": "🛠️ Used tool roast", "status": "done"},
        }
    ]


@pytest.mark.asyncio
async def test_apply_personality_preserves_manual_voice_override(monkeypatch) -> None:
    """Applying a profile should keep a manually selected Gemini voice active."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr("robot_comic.config.set_custom_profile", lambda _profile: None)

    handler = GeminiLiveHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.session = object()
    handler._voice_override = "Orus"
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)

    status = await handler.apply_personality("example")

    assert status == "Applied personality and restarted Gemini session."
    assert handler.get_current_voice() == "Orus"
    restart.assert_awaited_once()


def test_handler_uses_startup_voice_at_startup() -> None:
    """Gemini handler startup should restore a persisted startup voice."""
    handler = GeminiLiveHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Orus",
    )

    assert handler.get_current_voice() == "Orus"


def test_copy_preserves_current_voice_override() -> None:
    """Copied Gemini handlers should keep the current voice override."""
    handler = GeminiLiveHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Orus",
    )
    handler._voice_override = "Zephyr"

    copied_handler = handler.copy()

    assert copied_handler.get_current_voice() == "Zephyr"


def test_gemini_excludes_head_tracking_when_no_head_tracker(monkeypatch) -> None:
    """head_tracking tool must not appear in Gemini session config when head_tracker is not active."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")

    # Monkeypatch get_active_tool_specs on the gemini_mod namespace (where it is
    # bound at import time) so _build_live_config sees the controlled spec list.
    # Patching ct_mod.ALL_TOOL_SPECS instead would not work reliably because
    # get_active_tool_specs calls _initialize_tools() which rebinds ALL_TOOL_SPECS
    # from the real registry, stomping the monkeypatch before it can be read.
    _FAKE_SPECS = [
        {"type": "function", "name": "head_tracking", "description": "head_tracking", "parameters": {}},
        {"type": "function", "name": "fake_tool", "description": "fake_tool", "parameters": {}},
    ]

    def _fake_get_active_tool_specs(deps: ToolDependencies) -> list:
        if not (deps.camera_worker and deps.camera_worker.head_tracker):
            return [s for s in _FAKE_SPECS if s["name"] != "head_tracking"]
        return list(_FAKE_SPECS)

    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", _fake_get_active_tool_specs)

    # case 1: no camera at all, --no-camera flag passed
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=None)
    handler = GeminiLiveHandler(deps)
    live_config = handler._build_live_config()
    tool_names = [fd.name for fd in live_config.tools[0].function_declarations] if live_config.tools else []
    assert "head_tracking" not in tool_names, "case 1 failed: camera_worker=None"
    assert "fake_tool" in tool_names, "case 1 failed: a non-head-tracking tool was unexpectedly excluded"

    # case 2: camera is running but --head-tracker flag was not passed
    camera_worker = MagicMock()
    camera_worker.head_tracker = None
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=camera_worker)
    handler = GeminiLiveHandler(deps)
    live_config = handler._build_live_config()
    tool_names = [fd.name for fd in live_config.tools[0].function_declarations] if live_config.tools else []
    assert "head_tracking" not in tool_names, "case 2 failed: camera_worker.head_tracker=None"
    assert "fake_tool" in tool_names, "case 2 failed: a non-head-tracking tool was unexpectedly excluded"


@pytest.mark.asyncio
async def test_video_task_not_started_when_streaming_disabled(monkeypatch):
    """Video sender task must not start when GEMINI_LIVE_VIDEO_STREAMING=False."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    camera = MagicMock()
    camera.get_latest_frame.return_value = None
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera,
    )

    handler = GeminiLiveHandler(deps)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    stop_event = asyncio.Event()
    session = _FakeSession(batches=[], stop_event=stop_event)
    fake_client = _FakeLiveClient(session)
    handler.client = fake_client

    # Flag off (default)
    import robot_comic.config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "GEMINI_LIVE_VIDEO_STREAMING", False)

    video_sender_calls = []
    original = handler._video_sender_loop

    async def spy_video_loop():
        video_sender_calls.append(True)
        await original()

    monkeypatch.setattr(handler, "_video_sender_loop", spy_video_loop)

    # Set the handler's stop event so the receive loop exits immediately.
    # Setting the session stop_event alone is insufficient — the outer while loop
    # checks handler._stop_event and would spin forever on an empty session.
    handler._stop_event.set()
    await handler._run_live_session()

    assert video_sender_calls == [], "Video sender must not start when flag is False"


def _make_1008_goaway() -> Exception:
    from google.genai.errors import APIError

    response_json = {
        "error": {
            "code": 1008,
            "status": None,
            "message": (
                "Connection aborted because the client failed to close the "
                "connection after receiving a GoAway signal once the session "
                "duration cap was reached"
            ),
        }
    }
    return APIError(1008, response_json, None)


@pytest.mark.asyncio
async def test_start_up_reconnects_on_goaway_1008_without_counting_attempts(monkeypatch) -> None:
    """A GoAway/1008 from a connected session is a scheduled rotation: start_up
    must reconnect in-process even when rotations outnumber max_attempts."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = GeminiLiveHandler(deps)

    call_count = 0

    async def capped_session() -> None:
        nonlocal call_count
        call_count += 1
        if call_count <= 4:  # one more than start_up's historical max_attempts=3
            handler.session = object()  # the session connected before the cap hit
            raise _make_1008_goaway()
        return None  # clean exit (stop requested)

    handler._run_live_session = capped_session  # type: ignore[method-assign]

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", fast_sleep)

    await handler.start_up()  # must not raise

    assert call_count == 5


@pytest.mark.asyncio
async def test_start_up_1008_before_connect_still_counts_as_failure(monkeypatch) -> None:
    """A 1008 raised without the session ever connecting is a real failure and
    must exhaust the bounded retry loop, not reconnect forever."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = GeminiLiveHandler(deps)

    err = _make_1008_goaway()
    call_count = 0

    async def never_connects() -> None:
        nonlocal call_count
        call_count += 1
        raise err  # handler.session stays None: connect itself failed

    handler._run_live_session = never_connects  # type: ignore[method-assign]

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", fast_sleep)

    with pytest.raises(type(err)):
        await handler.start_up()

    assert call_count == 3


@pytest.mark.asyncio
async def test_start_up_treats_rotation_sentinel_as_rotation(monkeypatch) -> None:
    """The proactive GoAway sentinel must also reconnect without counting attempts."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = GeminiLiveHandler(deps)

    call_count = 0

    async def rotating_session() -> None:
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            raise gemini_mod.GeminiSessionRotation("server sent GoAway")
        return None

    handler._run_live_session = rotating_session  # type: ignore[method-assign]

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", fast_sleep)

    await handler.start_up()  # must not raise

    assert call_count == 5


@pytest.mark.asyncio
async def test_go_away_notice_rotates_session_proactively(monkeypatch) -> None:
    """A GoAway notice in the response stream must close the session cleanly
    (raise the rotation sentinel) instead of waiting for the 1008 kill."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda default=None, backend=None: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = GeminiLiveHandler(deps)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    go_away_response = SimpleNamespace(
        server_content=None,
        tool_call=None,
        go_away=SimpleNamespace(time_left="10s"),
    )
    session = _FakeSession(batches=[[go_away_response]], stop_event=handler._stop_event)
    handler.client = _FakeLiveClient(session)

    with pytest.raises(gemini_mod.GeminiSessionRotation):
        await handler._run_live_session()


def test_strip_tts_delivery_tags_removes_section():
    instructions = """\
## IDENTITY
You are a robot.

## GEMINI TTS DELIVERY TAGS
Use [fast] for speed.
- [slow] — drag it out
- [amusement] — love your own jokes

## GUARDRAILS
Be safe.
"""
    result = _strip_tts_delivery_tags(instructions)
    assert "GEMINI TTS DELIVERY TAGS" not in result
    assert "[fast]" not in result
    assert "[amusement]" not in result
    assert "IDENTITY" in result
    assert "GUARDRAILS" in result
    assert "You are a robot." in result
    assert "Be safe." in result


def test_strip_tts_delivery_tags_removes_stray_tags():
    instructions = "Say [fast] this [amusement] line [slow] clearly."
    result = _strip_tts_delivery_tags(instructions)
    assert "[fast]" not in result
    assert "[amusement]" not in result
    assert "[slow]" not in result
    assert "Say" in result
    assert "this" in result


def test_strip_tts_delivery_tags_leaves_unrelated_brackets():
    instructions = "See section [PHYSICAL BEATS] for moves."
    result = _strip_tts_delivery_tags(instructions)
    assert "[PHYSICAL BEATS]" in result


def test_strip_tts_delivery_tags_no_section_is_noop():
    instructions = "## IDENTITY\nYou are a robot.\n"
    result = _strip_tts_delivery_tags(instructions)
    assert result == instructions.strip()
