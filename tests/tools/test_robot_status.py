from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import robot_status as robot_status_module
from reachy_mini_conversation_app.daemon_api import DaemonApiError
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _fake_get(monkeypatch: pytest.MonkeyPatch, responses: dict[str, object]) -> list[str]:
    """Route _get by path, recording the call order. Values may be exceptions to raise."""
    calls: list[str] = []

    async def fake(deps: object, path: str, timeout_s: float | None = None) -> object:
        calls.append(path)
        value = responses[path]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(robot_status_module, "_get", fake)
    return calls


@pytest.mark.asyncio
async def test_name_reads_the_daemon_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """The name comes off the daemon status, not a second endpoint."""
    calls = _fake_get(monkeypatch, {robot_status_module._DAEMON_STATUS_PATH: {"robot_name": "Michel"}})

    result = await robot_status_module.RobotStatus()(_deps(), topic="name")

    assert result == {"topic": "name", "robot_name": "Michel"}
    assert calls == [robot_status_module._DAEMON_STATUS_PATH]


@pytest.mark.asyncio
async def test_apps_returns_names_skipping_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed apps are reduced to names; the daemon leaves descriptions empty."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._INSTALLED_APPS_PATH: [
                {"name": "dance_app", "source_kind": "installed"},
                {"name": "another_app"},
                {"source_kind": "installed"},
            ]
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="apps")

    assert result == {"topic": "apps", "apps": ["dance_app", "another_app"]}


@pytest.mark.asyncio
async def test_apps_rejects_unexpected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-list response must not be passed through as an app list."""
    _fake_get(monkeypatch, {robot_status_module._INSTALLED_APPS_PATH: {"apps": []}})

    result = await robot_status_module.RobotStatus()(_deps(), topic="apps")

    assert "error" in result


@pytest.mark.asyncio
async def test_unknown_topic_is_rejected() -> None:
    """An unrecognised topic must not reach the daemon."""
    result = await robot_status_module.RobotStatus()(_deps(), topic="battery")

    assert "error" in result


@pytest.mark.asyncio
async def test_software_reports_version_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """The software topic combines the daemon version with the update check."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._DAEMON_STATUS_PATH: {"version": "1.11.0rc1", "state": "running"},
            robot_status_module._UPDATE_PATH: {
                "update": {"reachy_mini": {"is_available": False, "available_version": "1.10.0"}}
            },
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="software")

    assert result == {
        "topic": "software",
        "version": "1.11.0rc1",
        "update_available": False,
        "published_version": "1.10.0",
    }


@pytest.mark.asyncio
async def test_software_survives_missing_update_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Lite robot has no /update route, but the version must still be reported."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._DAEMON_STATUS_PATH: {"version": "1.11.0rc1", "state": "running"},
            robot_status_module._UPDATE_PATH: DaemonApiError("nope", status_code=404),
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="software")

    assert result["version"] == "1.11.0rc1"
    assert result["update_check"] == "unavailable"


@pytest.mark.asyncio
async def test_wifi_reports_ip_and_count_not_saved_ssids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The saved-network list must not be handed to the model verbatim."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._DAEMON_STATUS_PATH: {"wlan_ip": "192.168.0.90"},
            robot_status_module._WIFI_PATH: {
                "mode": "wlan",
                "known_networks": ["HomeNet", "Office", "Phone"],
                "connected_network": "HomeNet",
            },
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="wifi")

    assert result == {
        "topic": "wifi",
        "ip_address": "192.168.0.90",
        "mode": "wlan",
        "connected_network": "HomeNet",
    }
    assert "Office" not in str(result)


@pytest.mark.asyncio
async def test_wifi_still_reports_ip_without_wifi_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Lite robot has no /wifi route, but it still has an address."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._DAEMON_STATUS_PATH: {"wlan_ip": "192.168.0.90"},
            robot_status_module._WIFI_PATH: DaemonApiError("nope", status_code=404),
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="wifi")

    assert result == {"topic": "wifi", "ip_address": "192.168.0.90", "wifi_details": "unavailable"}


@pytest.mark.asyncio
async def test_wifi_missing_route_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 on the always-present daemon status is a real failure, not a Lite robot."""
    _fake_get(monkeypatch, {robot_status_module._DAEMON_STATUS_PATH: DaemonApiError("nope", status_code=404)})

    result = await robot_status_module.RobotStatus()(_deps(), topic="wifi")

    assert result == {"topic": "wifi", "error": "wifi status is not available on this robot"}


@pytest.mark.asyncio
async def test_account_reports_only_login_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The account topic must never carry a token, even if the daemon sends one."""
    _fake_get(
        monkeypatch,
        {robot_status_module._HF_STATUS_PATH: {"is_logged_in": True, "username": "someone", "token": "hf_secret"}},
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="account")

    assert result == {"topic": "account", "is_logged_in": True, "username": "someone"}
    assert "hf_secret" not in str(result)


@pytest.mark.asyncio
async def test_imu_derives_tilt_from_gravity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real resting reading: the IMU says 28.6 deg, present_head_pose says 28.4 deg."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._IMU_PATH: {
                "accelerometer": [-4.6931825, 0.3735866, 8.6508662],
                "gyroscope": [0.0018642, -0.0063915, 0.0005326],
                "temperature": 45.5,
            }
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="imu")

    # ax < 0 is nose-down, which is how a motors-disabled head rests.
    assert result["tilted"] == "forward"
    assert result["tilt_forward_deg"] == 28.5
    assert result["moving"] is False
    assert result["temperature_c"] == 45.5


@pytest.mark.asyncio
async def test_imu_matches_a_hand_tilted_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured with the head held to the right; present_head_pose read roll +0.760 rad."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._IMU_PATH: {
                "accelerometer": [-2.02, 6.83, 6.71],
                "gyroscope": [0.0, 0.0, 0.0],
                "temperature": 49.5,
            }
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="imu")

    assert result["tilted"] == "forward and right"
    # 43.5 deg by kinematics; gravity and the Stewart model agree to about a degree.
    assert result["tilt_right_deg"] == pytest.approx(44.35, abs=0.15)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accelerometer", "expected"),
    [
        ([0.0, 0.0, 9.81], "level"),
        # X is forward, so -ax is a nose-down tilt and +ax is tilted back.
        ([-4.9, 0.0, 8.5], "forward"),
        ([4.9, 0.0, 8.5], "back"),
        # Y is left, so +ay means gravity has moved toward the left side: tilted right.
        # Measured on hardware: a head tilted right by hand read accel_y +6.83.
        ([0.0, 4.9, 8.5], "right"),
        ([0.0, -4.9, 8.5], "left"),
        ([-4.9, 4.9, 6.9], "forward and right"),
        # Just inside the level tolerance on both axes.
        ([-0.6, 0.6, 9.7], "level"),
    ],
)
async def test_imu_names_the_tilt_direction(
    monkeypatch: pytest.MonkeyPatch, accelerometer: list[float], expected: str
) -> None:
    """Each axis sign must map to the word a person would use."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._IMU_PATH: {
                "accelerometer": accelerometer,
                "gyroscope": [0.0, 0.0, 0.0],
                "temperature": 40.0,
            }
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="imu")

    assert result["tilted"] == expected


@pytest.mark.asyncio
async def test_imu_flags_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spinning gyro should read as moving."""
    _fake_get(
        monkeypatch,
        {
            robot_status_module._IMU_PATH: {
                "accelerometer": [0.0, 0.0, 9.81],
                "gyroscope": [0.0, 0.0, 1.2],
                "temperature": 40.0,
            }
        },
    )

    result = await robot_status_module.RobotStatus()(_deps(), topic="imu")

    assert result["moving"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reading",
    [None, {"accelerometer": [0.0, 0.0], "gyroscope": [0.0, 0.0, 0.0]}, {"accelerometer": [0.0, 0.0, 0.0]}],
)
async def test_imu_handles_absent_or_malformed_reading(monkeypatch: pytest.MonkeyPatch, reading: object) -> None:
    """Lite robots and simulation return null or short vectors; never divide by zero."""
    _fake_get(monkeypatch, {robot_status_module._IMU_PATH: reading})

    result = await robot_status_module.RobotStatus()(_deps(), topic="imu")

    assert result == {"topic": "imu", "error": "this robot has no IMU reading available"}
