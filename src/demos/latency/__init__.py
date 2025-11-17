"""Simple demo for latency benchmarking with a custom tool."""

import logging
from typing import Any, Dict

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)


instructions = """
You are a witty robot named Reachy Mini. You answer very concisely and sarcastically, maximum 2 sentences.
**IMPORTANT**: For every answer you give, you ALWAYS start by using the `look_around` tool to perform a quick look around while you answer.
For every user input, you MUST call the `look_around` tool ONCE and answer the question.
You ONLY speak in English.
"""


# Create a custom tool
class LookAround(Tool):
    """Example of custom tool call."""

    name = "look_around"
    description = "Performs a quick look around."
    parameters_schema = {
        "type": "object",
        "properties": {
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute sweep look: left -> hold -> right -> hold -> center."""
        logger.info("Tool call: look_around")

        # Clear any existing moves
        deps.movement_manager.clear_move_queue()

        # Get current state
        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()

        # Extract body_yaw from head joints (first element of the 7 head joint positions)
        current_body_yaw = head_joints[0]
        current_antenna1 = antenna_joints[0]
        current_antenna2 = antenna_joints[1]

        # Define sweep parameters
        max_angle = 0.5 * 3.1415  # Maximum rotation angle (radians)
        transition_duration = 0.25  # Time to move between positions
        hold_duration = 0.1  # Time to hold at each extreme

        # Move 1: Sweep to the left (positive yaw for both body and head)
        left_head_pose = create_head_pose(0, 0, 0, 0, 0, max_angle, degrees=False)
        move_to_left = GotoQueueMove(
            target_head_pose=left_head_pose,
            start_head_pose=current_head_pose,
            target_antennas=(current_antenna1, current_antenna2),
            start_antennas=(current_antenna1, current_antenna2),
            target_body_yaw=current_body_yaw + max_angle,
            start_body_yaw=current_body_yaw,
            duration=transition_duration,
        )

        # Move 2: Hold at left position
        hold_left = GotoQueueMove(
            target_head_pose=left_head_pose,
            start_head_pose=left_head_pose,
            target_antennas=(current_antenna1, current_antenna2),
            start_antennas=(current_antenna1, current_antenna2),
            target_body_yaw=current_body_yaw + max_angle,
            start_body_yaw=current_body_yaw + max_angle,
            duration=hold_duration,
        )

        # Move 3: Return to center from left (to avoid crossing pi/-pi boundary)
        center_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)
        return_to_center_from_left = GotoQueueMove(
            target_head_pose=center_head_pose,
            start_head_pose=left_head_pose,
            target_antennas=(current_antenna1, current_antenna2),
            start_antennas=(current_antenna1, current_antenna2),
            target_body_yaw=current_body_yaw,
            start_body_yaw=current_body_yaw + max_angle,
            duration=transition_duration,
        )

        # Move 4: Sweep to the right (negative yaw for both body and head)
        right_head_pose = create_head_pose(0, 0, 0, 0, 0, -max_angle, degrees=False)
        move_to_right = GotoQueueMove(
            target_head_pose=right_head_pose,
            start_head_pose=center_head_pose,
            target_antennas=(current_antenna1, current_antenna2),
            start_antennas=(current_antenna1, current_antenna2),
            target_body_yaw=current_body_yaw - max_angle,
            start_body_yaw=current_body_yaw,
            duration=transition_duration,
        )

        # Move 5: Hold at right position
        hold_right = GotoQueueMove(
            target_head_pose=right_head_pose,
            start_head_pose=right_head_pose,
            target_antennas=(current_antenna1, current_antenna2),
            start_antennas=(current_antenna1, current_antenna2),
            target_body_yaw=current_body_yaw - max_angle,
            start_body_yaw=current_body_yaw - max_angle,
            duration=hold_duration,
        )

        # Move 6: Return to center from right
        return_to_center_final = GotoQueueMove(
            target_head_pose=center_head_pose,
            start_head_pose=right_head_pose,
            target_antennas=(0, 0),  # Reset antennas to neutral
            start_antennas=(current_antenna1, current_antenna2),
            target_body_yaw=current_body_yaw,  # Return to original body yaw
            start_body_yaw=current_body_yaw - max_angle,
            duration=transition_duration,
        )

        # Queue all moves in sequence
        deps.movement_manager.queue_move(move_to_left)
        deps.movement_manager.queue_move(hold_left)
        deps.movement_manager.queue_move(return_to_center_from_left)
        deps.movement_manager.queue_move(move_to_right)
        deps.movement_manager.queue_move(hold_right)
        deps.movement_manager.queue_move(return_to_center_final)

        total_duration = transition_duration * 4 + hold_duration * 2
        deps.movement_manager.set_moving_state(total_duration)

        return {"status": f"Looking around for {total_duration} seconds."}
