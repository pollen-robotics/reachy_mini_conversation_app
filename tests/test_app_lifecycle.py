import urllib.request
from io import BytesIO
from types import SimpleNamespace
from http.client import HTTPMessage, HTTPResponse
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app import daemon_api, app_lifecycle
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def test_initialize_tools_migrates_bundled_spaces_before_loading(monkeypatch) -> None:
    """Startup should migrate bundled tools before resolving the active registry."""
    calls: list[str] = []
    monkeypatch.setattr(app_lifecycle, "migrate_legacy_profiles", lambda _path: calls.append("profiles"))
    monkeypatch.setattr(app_lifecycle, "migrate_bundled_tool_spaces", lambda _path: calls.append("spaces"))
    monkeypatch.setattr(app_lifecycle, "initialize_tools", lambda **_kwargs: calls.append("tools"))

    assert app_lifecycle.initialize_tools_with_default_fallback(None, MagicMock()) is None
    assert calls == ["profiles", "spaces", "tools"]


def test_request_stop_current_app_posts_to_daemon(monkeypatch) -> None:
    """The app stop request should call the connected Reachy daemon endpoint."""

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://192.168.1.42:8000/api/apps/stop-current-app"
        assert request.get_method() == "POST"
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert app_lifecycle.request_stop_current_app(robot, MagicMock())


def test_wake_up_if_sleeping_enables_motors_before_wake_up() -> None:
    """Startup should enable sleeping motors before playing the wake-up movement."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    assert robot.method_calls == [
        call.get_current_head_pose(),
        call.enable_motors(),
        call.wake_up(),
    ]


def test_wake_up_if_sleeping_skips_non_sleep_head_pose() -> None:
    """Startup should leave an already-awake robot alone."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


def test_run_go_to_sleep_tool_uses_runtime_callback() -> None:
    """Synchronous lifecycle paths should enter through the go_to_sleep tool."""
    expected = {"status": "sleeping"}
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert result == expected
    go_to_sleep.assert_called_once_with()


def test_request_stop_current_app_returns_false_on_urlerror(monkeypatch) -> None:
    """A daemon that is unreachable is reported as a failed stop, not an exception."""

    def fake_urlopen(request, timeout):
        raise daemon_api.urllib.error.URLError("connection refused")

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert not app_lifecycle.request_stop_current_app(robot, MagicMock())


def test_request_stop_current_app_treats_already_stopping_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The daemon's stopped-or-stopping response is idempotent success."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> None:
        raise daemon_api.urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs=HTTPMessage(),
            fp=BytesIO(b'{"detail":"No app is currently running"}'),
        )

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))
    logger = MagicMock()

    assert app_lifecycle.request_stop_current_app(robot, logger)
    logger.error.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (400, b'{"detail":"Stop cleanup failed"}'),
        (400, b""),
        (400, b"<html>Bad Request</html>"),
        (400, b'{"detail":["No app is currently running"]}'),
        (409, b'{"detail":"No app is currently running"}'),
        (500, b'{"detail":"No app is currently running"}'),
    ],
)
def test_request_stop_current_app_returns_false_on_other_httperror(
    monkeypatch: pytest.MonkeyPatch, status_code: int, body: bytes
) -> None:
    """Only the verified daemon response is accepted without an error log."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> None:
        raise daemon_api.urllib.error.HTTPError(
            request.full_url,
            status_code,
            "Request failed",
            hdrs=HTTPMessage(),
            fp=BytesIO(body),
        )

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))
    logger = MagicMock()

    assert not app_lifecycle.request_stop_current_app(robot, logger)
    logger.error.assert_called_once()


def test_request_stop_current_app_returns_false_on_truncated_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated error body remains a logged failure so local shutdown can continue."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> None:
        connection = MagicMock()
        connection.makefile.return_value = BytesIO(
            b'HTTP/1.1 400 Bad Request\r\nContent-Length: 100\r\n\r\n{"detail":'
        )
        response = HTTPResponse(connection)
        response.begin()
        raise daemon_api.urllib.error.HTTPError(request.full_url, 400, "Bad Request", HTTPMessage(), response)

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))
    logger = MagicMock()

    assert not app_lifecycle.request_stop_current_app(robot, logger)
    logger.error.assert_called_once()
    error = logger.error.call_args.args[1]
    assert isinstance(error, daemon_api.DaemonApiError)
    assert error.status_code == 400
    assert error.detail is None


def test_wake_up_if_sleeping_handles_pose_read_failure() -> None:
    """A robot that cannot report its pose skips the wake-up movement."""
    robot = MagicMock()
    robot.get_current_head_pose.side_effect = RuntimeError("no telemetry")

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_reports_wake_up_failure() -> None:
    """A wake-up movement that fails is reported without raising."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    robot.wake_up.side_effect = RuntimeError("motor fault")

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_called_once_with()


def test_wake_up_if_sleeping_ignores_malformed_pose() -> None:
    """A pose that cannot be coerced to a float array is treated as not-sleeping."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = "not-a-pose"

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_ignores_wrong_shape_pose() -> None:
    """A pose that is not 4x4 is treated as not-sleeping."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(3)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())
    robot.wake_up.assert_not_called()


def test_run_go_to_sleep_tool_reports_tool_failure(monkeypatch) -> None:
    """A crash inside the go_to_sleep tool is surfaced as an error dict, not raised."""

    class _BoomTool:
        async def __call__(self, deps: object) -> dict[str, object]:
            raise RuntimeError("boom")

    monkeypatch.setattr(app_lifecycle, "GoToSleep", _BoomTool)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert "error" in result
    assert "boom" in str(result["error"])
