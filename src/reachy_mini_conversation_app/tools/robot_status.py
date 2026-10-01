import math
import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.daemon_api import DEFAULT_TIMEOUT_S, DaemonApiError, daemon_request
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# /wifi and /update are mounted on the daemon root, not under /api, and only on the
# wireless version -- they answer 404 on a Lite robot.
_DAEMON_STATUS_PATH = "/api/daemon/status"
_UPDATE_PATH = "/update/available"
_WIFI_PATH = "/wifi/status"
_HF_STATUS_PATH = "/api/hf-auth/status"
_IMU_PATH = "/api/state/imu"
_INSTALLED_APPS_PATH = "/api/apps/list-available/installed"

# Checking PyPI for a newer release goes over the internet. So does the HF sign-in
# check: the daemon's /hf-auth/status calls whoami() from an async route, so a cold
# lookup both runs slow and stalls the daemon loop. Listing apps shells out to the
# apps venv on a 5 s daemon-side budget. The plain local reads stay on the default.
_NETWORK_TIMEOUT_S = 10.0

# Gyroscope magnitude, rad/s, above which the head is considered moving. At rest the
# BMI088 reads ~0.007 rad/s, so this sits well clear of the noise floor.
_MOVING_GYRO_RAD_S = 0.15

# Below this the head is called "level" rather than named a direction. Tune if a robot
# rests visibly off-centre: a disabled head droops nose-down well past this.
_LEVEL_TOLERANCE_DEG = 5.0


async def _get(deps: ToolDependencies, path: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Any:
    """Read a daemon endpoint off the event loop."""
    return await asyncio.to_thread(daemon_request, deps.reachy_mini, path, timeout_s=timeout_s)


async def _name(deps: ToolDependencies) -> dict[str, Any]:
    """Report the robot's configured display name."""
    status = await _get(deps, _DAEMON_STATUS_PATH)
    return {"robot_name": status.get("robot_name") if isinstance(status, dict) else None}


async def _apps(deps: ToolDependencies) -> dict[str, Any]:
    """Report the apps installed on the robot."""
    response = await _get(deps, _INSTALLED_APPS_PATH, timeout_s=_NETWORK_TIMEOUT_S)
    if not isinstance(response, list):
        return {"error": "daemon returned an unexpected app list"}
    # The daemon leaves AppInfo.description empty for installed apps, so only the name is useful.
    return {"apps": [app["name"] for app in response if isinstance(app, dict) and app.get("name")]}


async def _software(deps: ToolDependencies) -> dict[str, Any]:
    """Report the running version and whether a newer release is published."""
    status = await _get(deps, _DAEMON_STATUS_PATH)
    result: dict[str, Any] = {"version": status.get("version") if isinstance(status, dict) else None}

    try:
        update = await _get(deps, _UPDATE_PATH, timeout_s=_NETWORK_TIMEOUT_S)
    except DaemonApiError as e:
        # Lite robots have no update route; a network failure is also not fatal here.
        result["update_check"] = "unavailable" if e.status_code == 404 else f"failed: {e}"
        return result

    entry = update.get("update", {}).get("reachy_mini", {}) if isinstance(update, dict) else {}
    result["update_available"] = entry.get("is_available")
    # The published version can be *older* than the installed one on a pre-release build.
    result["published_version"] = entry.get("available_version")
    return result


async def _wifi(deps: ToolDependencies) -> dict[str, Any]:
    """Report the LAN address, the Wi-Fi mode and the network the robot is on."""
    # The IP lives on the daemon status, which every robot has; /wifi/status is
    # wireless-only, so read the address first and degrade from there.
    status = await _get(deps, _DAEMON_STATUS_PATH)
    result: dict[str, Any] = {"ip_address": status.get("wlan_ip") if isinstance(status, dict) else None}

    try:
        wifi = await _get(deps, _WIFI_PATH)
    except DaemonApiError as e:
        result["wifi_details"] = "unavailable" if e.status_code == 404 else f"failed: {e}"
        return result

    if not isinstance(wifi, dict):
        return {**result, "error": "daemon returned an unexpected wifi status"}

    # known_networks is every saved SSID; the connected one is all the model needs.
    result["mode"] = wifi.get("mode")
    result["connected_network"] = wifi.get("connected_network")
    return result


async def _account(deps: ToolDependencies) -> dict[str, Any]:
    """Report the Hugging Face sign-in state (never the token)."""
    status = await _get(deps, _HF_STATUS_PATH, timeout_s=_NETWORK_TIMEOUT_S)
    if not isinstance(status, dict):
        return {"error": "daemon returned an unexpected auth status"}
    return {"is_logged_in": status.get("is_logged_in"), "username": status.get("username")}


async def _imu(deps: ToolDependencies) -> dict[str, Any]:
    """Report which way the head is tilted, whether it is moving, and its temperature."""
    reading = await _get(deps, _IMU_PATH)
    if not isinstance(reading, dict):
        return {"error": "this robot has no IMU reading available"}

    accelerometer = reading.get("accelerometer") or []
    gyroscope = reading.get("gyroscope") or []
    if len(accelerometer) != 3 or len(gyroscope) != 3:
        return {"error": "this robot has no IMU reading available"}

    # At rest the accelerometer reads gravity as an "up" vector in the head frame,
    # which is X forward / Y left / Z up (see look_at.py: straight_head_vector is
    # +X). So ax = -g*sin(pitch) and ay = +g*sin(roll). Both signs were checked on a
    # wireless unit against present_head_pose: resting nose-down read 28.6 deg here
    # vs 28.4 by kinematics, and a head tilted right by hand read +44.4 vs +43.5.
    gravity_magnitude = math.sqrt(sum(axis * axis for axis in accelerometer))
    if gravity_magnitude == 0:
        return {"error": "this robot has no IMU reading available"}

    forward_deg = math.degrees(math.asin(max(-1.0, min(1.0, -accelerometer[0] / gravity_magnitude))))
    right_deg = math.degrees(math.asin(max(-1.0, min(1.0, accelerometer[1] / gravity_magnitude))))

    directions = []
    if abs(forward_deg) >= _LEVEL_TOLERANCE_DEG:
        directions.append("forward" if forward_deg > 0 else "back")
    if abs(right_deg) >= _LEVEL_TOLERANCE_DEG:
        directions.append("right" if right_deg > 0 else "left")

    rotation_rate = math.sqrt(sum(axis * axis for axis in gyroscope))
    return {
        "tilted": " and ".join(directions) or "level",
        "tilt_forward_deg": round(forward_deg, 1),
        "tilt_right_deg": round(right_deg, 1),
        "moving": rotation_rate > _MOVING_GYRO_RAD_S,
        "temperature_c": reading.get("temperature"),
    }


_TOPICS = {
    "name": _name,
    "software": _software,
    "wifi": _wifi,
    "account": _account,
    "imu": _imu,
    "apps": _apps,
}


class RobotStatus(Tool):
    """Read one of the robot's own status topics from the daemon."""

    name = "robot_status"
    description = (
        "Read Reachy's own status. Topics: 'name' for the robot's configured name, 'software' for the running "
        "version and whether an update is available, 'wifi' for the IP address and the network it is connected "
        "to, 'account' for the Hugging Face sign-in state, 'imu' for which way the head is tilted, whether it "
        "is moving, and its temperature, 'apps' for the apps installed on the robot."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": list(_TOPICS),
                "description": "Which status to read.",
            },
        },
        "required": ["topic"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Read the requested status topic."""
        topic = str(kwargs.get("topic") or "")
        handler = _TOPICS.get(topic)
        if handler is None:
            return {"error": f"unknown topic {topic!r}"}

        logger.info("Tool call: robot_status topic=%s", topic)
        try:
            result = await handler(deps)
        except DaemonApiError as e:
            logger.error("robot_status(%s) failed: %s", topic, e)
            if e.status_code == 404:
                result = {"error": f"{topic} status is not available on this robot"}
            else:
                result = {"error": str(e)}

        return {"topic": topic, **result}
