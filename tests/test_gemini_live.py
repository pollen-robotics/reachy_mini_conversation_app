"""Behavior tests for the Gemini Live handler."""

import base64
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

import reachy_mini_conversation_app.gemini_live as gemini_mod
from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import ToolNotification


def _server_content(**kwargs):
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


def _response(server_content=None, tool_call=None):
    return SimpleNamespace(server_content=server_content, tool_call=tool_call)


async def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


class _FakeSession:
    def __init__(self, batches, stop_event: asyncio.Event):
        self._batches = list(batches)
        self._stop_event = stop_event
        self.realtime_inputs = []
        self.tool_responses = []

    async def close(self) -> None:
        self._stop_event.set()

    async def send_realtime_input(self, **kwargs) -> None:
        self.realtime_inputs.append(kwargs)
        return None

    async def send_tool_response(self, **kwargs) -> None:
        self.tool_responses.append(kwargs)
        return None

    async def receive(self):
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

    async def __aexit__(self, *_args) -> bool:
        return False


class _FakeLiveClient:
    def __init__(self, session: _FakeSession):
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=lambda **_kwargs: _FakeConnectContext(session)))


@pytest.mark.asyncio
async def test_gemini_turn_buffers_transcripts_and_schedules_motion_reset(monkeypatch) -> None:
    """Gemini turns should emit one transcript per role and let the wobbler reset after speech."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda: "Kore")
    monkeypatch.setattr(gemini_mod, "get_tool_specs", lambda: [])

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
    object.__setattr__(handler.tool_manager, "start_up", MagicMock())
    object.__setattr__(handler.tool_manager, "shutdown", AsyncMock())

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
    await asyncio.wait_for(task, timeout=1.0)

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
async def test_apply_personality_preserves_manual_voice_override(monkeypatch) -> None:
    """Applying a profile should keep a manually selected Gemini voice active."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda: "Kore")
    monkeypatch.setattr("reachy_mini_conversation_app.config.set_custom_profile", lambda _profile: None)

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
