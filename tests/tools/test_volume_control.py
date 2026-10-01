from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import volume_control as volume_control_module
from reachy_mini_conversation_app.daemon_api import DaemonApiError
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _fake_request(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> MagicMock:
    """Replace the tool module's daemon_request with a mock and return it."""
    request = MagicMock(**kwargs)
    monkeypatch.setattr(volume_control_module, "daemon_request", request)
    return request


@pytest.mark.asyncio
async def test_volume_control_reads_current_level_without_level_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting 'level' should GET the current volume instead of setting one."""
    request = _fake_request(monkeypatch, return_value={"volume": 42})

    result = await volume_control_module.VolumeControl()(_deps())

    assert result == {"device": "speaker", "volume": 42}
    assert request.call_args.args[1] == "/api/volume/current"
    assert "method" not in request.call_args.kwargs


@pytest.mark.asyncio
async def test_volume_control_sets_microphone_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """A level on the microphone device should POST to the microphone endpoint."""
    request = _fake_request(monkeypatch, return_value={"volume": 70})

    result = await volume_control_module.VolumeControl()(_deps(), device="microphone", level=70)

    assert result == {"device": "microphone", "volume": 70}
    assert request.call_args.args[1] == "/api/volume/microphone/set"
    assert request.call_args.kwargs["method"] == "POST"
    assert request.call_args.kwargs["payload"] == {"volume": 70}


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [-1, 101, "loud"])
async def test_volume_control_rejects_invalid_levels(monkeypatch: pytest.MonkeyPatch, level: object) -> None:
    """Out-of-range or non-numeric levels must never reach the daemon."""
    request = _fake_request(monkeypatch)

    result = await volume_control_module.VolumeControl()(_deps(), level=level)

    assert "error" in result
    request.assert_not_called()


@pytest.mark.asyncio
async def test_volume_control_rejects_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown device must not be turned into an arbitrary daemon path."""
    request = _fake_request(monkeypatch)

    result = await volume_control_module.VolumeControl()(_deps(), device="../daemon/stop")

    assert "error" in result
    request.assert_not_called()


@pytest.mark.asyncio
async def test_volume_control_reports_daemon_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon error should surface as a tool error, not an exception."""
    _fake_request(monkeypatch, side_effect=DaemonApiError("boom"))

    result = await volume_control_module.VolumeControl()(_deps())

    assert result == {"error": "boom"}
