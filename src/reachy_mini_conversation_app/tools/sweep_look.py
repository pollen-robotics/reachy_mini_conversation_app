import logging
from typing import Any

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)


class SweepLook(Tool):
    """Sweep the head left and right, then return it to center."""

    name = "sweep_look"
    description = (
        "Sweep head from left to right while rotating the body, pausing at each extreme, then return to center"
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Queue one complete left-to-right sweep."""
        logger.info("Tool call: sweep_look")
        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]
        current_antennas = (antenna_joints[0], antenna_joints[1])
        max_angle = 0.9 * np.pi
        transition_duration = 3.0
        hold_duration = 1.0

        left_head_pose = create_head_pose(0, 0, 0, 0, 0, max_angle, degrees=False)
        center_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)
        right_head_pose = create_head_pose(0, 0, 0, 0, 0, -max_angle, degrees=False)
        moves = [
            GotoQueueMove(
                target_head_pose=left_head_pose,
                start_head_pose=current_head_pose,
                target_antennas=current_antennas,
                start_antennas=current_antennas,
                target_body_yaw=current_body_yaw + max_angle,
                start_body_yaw=current_body_yaw,
                duration=transition_duration,
            ),
            GotoQueueMove(
                target_head_pose=left_head_pose,
                start_head_pose=left_head_pose,
                target_antennas=current_antennas,
                start_antennas=current_antennas,
                target_body_yaw=current_body_yaw + max_angle,
                start_body_yaw=current_body_yaw + max_angle,
                duration=hold_duration,
            ),
            GotoQueueMove(
                target_head_pose=center_head_pose,
                start_head_pose=left_head_pose,
                target_antennas=current_antennas,
                start_antennas=current_antennas,
                target_body_yaw=current_body_yaw,
                start_body_yaw=current_body_yaw + max_angle,
                duration=transition_duration,
            ),
            GotoQueueMove(
                target_head_pose=right_head_pose,
                start_head_pose=center_head_pose,
                target_antennas=current_antennas,
                start_antennas=current_antennas,
                target_body_yaw=current_body_yaw - max_angle,
                start_body_yaw=current_body_yaw,
                duration=transition_duration,
            ),
            GotoQueueMove(
                target_head_pose=right_head_pose,
                start_head_pose=right_head_pose,
                target_antennas=current_antennas,
                start_antennas=current_antennas,
                target_body_yaw=current_body_yaw - max_angle,
                start_body_yaw=current_body_yaw - max_angle,
                duration=hold_duration,
            ),
            GotoQueueMove(
                target_head_pose=center_head_pose,
                start_head_pose=right_head_pose,
                target_antennas=current_antennas,
                start_antennas=current_antennas,
                target_body_yaw=current_body_yaw,
                start_body_yaw=current_body_yaw - max_angle,
                duration=transition_duration,
            ),
        ]
        for move in moves:
            deps.movement_manager.queue_move(move)

        total_duration = transition_duration * 4 + hold_duration * 2
        deps.movement_manager.set_moving_state(total_duration)
        return {"status": f"sweeping look left-right-center, total {total_duration:.1f}s"}
