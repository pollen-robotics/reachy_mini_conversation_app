"""Helpers for app startup and shutdown lifecycle behavior."""

import asyncio
import logging
import urllib.error
import urllib.request

import numpy as np
import numpy.typing as npt

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini.utils.interpolation import distance_between_poses
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep


_STOP_CURRENT_APP_PATH = "/api/apps/stop-current-app"
_STOP_CURRENT_APP_TIMEOUT_S = 2.0
_SLEEP_HEAD_TRANSLATION_TOLERANCE_M = 0.05
_SLEEP_HEAD_ROTATION_TOLERANCE_RAD = 0.35


def request_stop_current_app(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Request the Reachy Mini daemon to stop the current app."""
    stop_current_app_url = f"http://{robot.client.host}:{robot.client.port}{_STOP_CURRENT_APP_PATH}"
    request = urllib.request.Request(stop_current_app_url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_STOP_CURRENT_APP_TIMEOUT_S) as response:
            response.read()
    except urllib.error.URLError as e:
        logger.error("Failed to request current app stop via %s: %s", stop_current_app_url, e)
        return False

    logger.info("Requested current app stop via %s", stop_current_app_url)
    return True


def _is_sleep_head_pose(head_pose: npt.ArrayLike) -> bool:
    try:
        current_head_pose: npt.NDArray[np.float64] = np.asarray(head_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return False

    if current_head_pose.shape != (4, 4):
        return False

    pose_distances = distance_between_poses(current_head_pose, SLEEP_HEAD_POSE)
    translation_distance = float(pose_distances[0])
    rotation_angle = float(pose_distances[1])
    return (
        translation_distance <= _SLEEP_HEAD_TRANSLATION_TOLERANCE_M
        and rotation_angle <= _SLEEP_HEAD_ROTATION_TOLERANCE_RAD
    )


def wake_up_if_sleeping(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Run the SDK wake-up movement when Reachy starts from the sleep pose."""
    try:
        head_pose = robot.get_current_head_pose()
    except Exception as e:
        logger.warning("Could not read robot pose before startup wake-up check: %s", e)
        return False

    if not _is_sleep_head_pose(head_pose):
        return False

    logger.info("Robot is in sleep pose; running wake-up movement.")
    try:
        robot.enable_motors()
        robot.wake_up()
    except Exception as e:
        logger.error("Failed to run wake-up movement: %s", e)
        return False
    return True


def run_go_to_sleep_tool(deps: ToolDependencies, logger: logging.Logger) -> dict[str, object]:
    """Run the shared go_to_sleep tool from synchronous shutdown paths."""
    try:
        return asyncio.run(GoToSleep()(deps))
    except Exception as e:
        logger.error("Failed to run go_to_sleep tool during shutdown: %s", e)
        return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
