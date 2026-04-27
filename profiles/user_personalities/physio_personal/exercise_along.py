import logging
from typing import Any, Dict

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

logger = logging.getLogger(__name__)


class ExerciseAlong(Tool):
    """Robot exercises along with the user - head nods and bobs while antennas pump."""

    name = "exercise_along"
    description = (
        "Perform an energetic exercising animation: head moves up and down and "
        "side to side while antennas pump in and out, as if the robot is exercising "
        "along with the user. Use this while the user is doing their exercise."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute exercise animation: nod up/down, bob side to side, pump antennas."""
        logger.info("Tool call: exercise_along")

        deps.movement_manager.clear_move_queue()

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()

        current_body_yaw = head_joints[0]
        current_antenna1 = antenna_joints[0]
        current_antenna2 = antenna_joints[1]

        move_duration = 0.4   # fast, energetic moves
        hold_duration = 0.15  # short hold at each extreme

        # Angles in radians
        nod_up    = create_head_pose(pitch=20,  degrees=True)
        nod_down  = create_head_pose(pitch=-20, degrees=True)
        tilt_left  = create_head_pose(roll=20,  degrees=True)
        tilt_right = create_head_pose(roll=-20, degrees=True)
        center    = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)

        antennas_up   = (np.deg2rad(40),  np.deg2rad(40))   # both raised
        antennas_down = (np.deg2rad(-10), np.deg2rad(-10))  # both lowered
        antennas_mid  = (current_antenna1, current_antenna2)

        def make_move(target_head, start_head, target_ant, start_ant, duration):
            return GotoQueueMove(
                target_head_pose=target_head,
                start_head_pose=start_head,
                target_antennas=target_ant,
                start_antennas=start_ant,
                target_body_yaw=current_body_yaw,
                start_body_yaw=current_body_yaw,
                duration=duration,
            )

        moves = [
            # Nod up — antennas pump up
            make_move(nod_up,    current_head_pose, antennas_up,   antennas_mid,  move_duration),
            make_move(nod_up,    nod_up,            antennas_up,   antennas_up,   hold_duration),
            # Nod down — antennas pump down
            make_move(nod_down,  nod_up,            antennas_down, antennas_up,   move_duration),
            make_move(nod_down,  nod_down,          antennas_down, antennas_down, hold_duration),
            # Back to center
            make_move(center,    nod_down,          antennas_mid,  antennas_down, move_duration),

            # Tilt left — antennas splay out
            make_move(tilt_left,  center,     (np.deg2rad(40), np.deg2rad(-10)),
                                              antennas_mid,  move_duration),
            make_move(tilt_left,  tilt_left,  (np.deg2rad(40), np.deg2rad(-10)),
                                              (np.deg2rad(40), np.deg2rad(-10)), hold_duration),
            # Tilt right
            make_move(tilt_right, tilt_left,  (np.deg2rad(-10), np.deg2rad(40)),
                                              (np.deg2rad(40),  np.deg2rad(-10)), move_duration),
            make_move(tilt_right, tilt_right, (np.deg2rad(-10), np.deg2rad(40)),
                                              (np.deg2rad(-10), np.deg2rad(40)), hold_duration),
            # Return to center
            make_move(center, tilt_right, antennas_mid,
                              (np.deg2rad(-10), np.deg2rad(40)), move_duration),

            # Second nod cycle — faster, more intense
            make_move(nod_up,   center,   antennas_up,   antennas_mid,  move_duration * 0.7),
            make_move(nod_down, nod_up,   antennas_down, antennas_up,   move_duration * 0.7),
            make_move(nod_up,   nod_down, antennas_up,   antennas_down, move_duration * 0.7),
            make_move(nod_down, nod_up,   antennas_down, antennas_up,   move_duration * 0.7),
            # Settle back to center
            make_move(center, nod_down, antennas_mid, antennas_down, move_duration),
        ]

        for move in moves:
            deps.movement_manager.queue_move(move)

        total_duration = (
            move_duration * 11 +
            move_duration * 0.7 * 4 +
            hold_duration * 4
        )
        deps.movement_manager.set_moving_state(total_duration)

        return {"status": f"exercising along for {total_duration:.1f}s"}