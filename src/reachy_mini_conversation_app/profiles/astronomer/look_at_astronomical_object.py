import logging
from typing import Any, Dict

import numpy as np

from reachy_mini.utils import create_head_pose
from .astronomer import find_celestial_angles
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)


class LookAtAstronomicalObject(Tool):
    """Look at a specific point in the sky given an object name."""

    name = "look_at_astronomical_object"
    description = "Look at the sky in the direction of an astronomical object (stars, planets, moon, sun, galaxies, nebulae, clusters) to point it to the user."
    parameters_schema = {
        "type": "object",
        "properties": {
            "astronomical_object": {
                "type": "string",
                "description": "The name of the object to look at.",
            }
        },
        "required": ["astronomical_object"],
    }


    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute look at sky tool with hardcoded test parameters."""

        astronomical_object = kwargs.get("astronomical_object")
        result = find_celestial_angles(astronomical_object)

        if result['found']:
            logger.info(f" 🔭 Found: {result['object_name']} ({result['type']})")
            logger.info(f"    Match score: {result['match_score']}%")
            logger.info(f"    Azimuth: {result['azimuth']:.2f}°")
            logger.info(f"    Altitude: {result['altitude']:.2f}°")

            if result['altitude'] > 0:
                sky_orientation(deps, result['altitude'], result['azimuth'])
                status = f"Looking in the direction of {astronomical_object} at azimuth {result['azimuth']}° and altitude {result['altitude']}°"
                return {"status": status}
            else:
                DEFAULT_BELOW_HORIZON = -5
                sky_orientation(deps, DEFAULT_BELOW_HORIZON, result['azimuth'])
                return {"status": f"Looking in the direction of {astronomical_object} at azimuth {result['azimuth']}° but it's currently below the horizon."}
        else:
            return {"status": "Object is unknown to me for now."}





def sky_orientation(deps, altitude_deg, azimuth_deg):
    """Look at sky point: rotate to azimuth, then tilt to altitude."""
    # Safety limits for head pitch (altitude angle)
    MAX_ALTITUDE_DEG = 35.0  # Maximum upward tilt in degrees
    MIN_ALTITUDE_DEG = -35.0  # Maximum downward tilt in degrees

    # Safe pre-rotation position: lift and tilt head to avoid collisions during rotation
    SAFE_PREROTATION_Z_LIFT_M = 0.02  # Lift head 2cm vertically before rotating

    # Clamp altitude to safe range
    altitude_deg_clamped = max(MIN_ALTITUDE_DEG, min(MAX_ALTITUDE_DEG, altitude_deg))
    if altitude_deg != altitude_deg_clamped:
        logger.warning(
            "Altitude %.1f° clamped to %.1f° (safe range: %.1f° to %.1f°)",
            altitude_deg,
            altitude_deg_clamped,
            MIN_ALTITUDE_DEG,
            MAX_ALTITUDE_DEG,
        )

    # Convert azimuth in the -180° +180° interval
    azimuth_deg = azimuth_deg - 360.0 if azimuth_deg > 180.0 else azimuth_deg

    # Convert to radians
    azimuth_rad = np.deg2rad(azimuth_deg)
    altitude_rad = np.deg2rad(altitude_deg_clamped)

    logger.info("Tool call: look_at_astronomical_object azimuth=%.1f° altitude=%.1f°", azimuth_deg, altitude_deg_clamped)

    # Get current state
    current_head_pose = deps.reachy_mini.get_current_head_pose()
    head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()

    current_body_yaw = head_joints[0]
    current_antenna1 = antenna_joints[0]
    current_antenna2 = antenna_joints[1]

    # Define timing
    pre_rotation_duration = 0.5  # Time to raise head before rotation
    rotation_duration = 1.0  # Time to rotate body to azimuth
    tilt_duration = 1.0  # Time to tilt to final altitude
    hold_duration = 5.0  # Time to hold at target

    # Move 0: Raise head to safe position (lift z + pitch up, neutral yaw) before rotating
    # This prevents collisions during body rotation
    safe_pre_pose = create_head_pose(
        x=0,
        y=0,
        z=SAFE_PREROTATION_Z_LIFT_M,  # Lift head vertically for extra clearance
        roll=0,
        pitch=0,  # Tilt head up to safe position
        yaw=0,  # Keep yaw neutral (will follow body)
        degrees=False,
    )

    raise_head = GotoQueueMove(
        target_head_pose=safe_pre_pose,
        start_head_pose=current_head_pose,
        target_antennas=(current_antenna1, current_antenna2),
        start_antennas=(current_antenna1, current_antenna2),
        target_body_yaw=current_body_yaw,  # Don't rotate body yet
        start_body_yaw=current_body_yaw,
        duration=pre_rotation_duration,
    )

    # Move 1: Rotate body AND head yaw to target azimuth together
    # Head pose is in world coordinates, so we need to rotate head yaw along with body yaw
    safe_pre_pose_at_azimuth = create_head_pose(
        x=0,
        y=0,
        z=SAFE_PREROTATION_Z_LIFT_M,  # Keep z-lift
        roll=0,
        pitch=0,  # Keep same pitch as safe_pre_pose
        yaw= - azimuth_rad,  # Rotate head yaw to match body rotation
        degrees=False,
    )

    rotate_to_azimuth = GotoQueueMove(
        target_head_pose=safe_pre_pose_at_azimuth,  # Head rotates with body
        start_head_pose=safe_pre_pose,  # Start from safe position (yaw=0)
        target_antennas=(current_antenna1, current_antenna2),
        start_antennas=(current_antenna1, current_antenna2),
        target_body_yaw= - azimuth_rad,  # Rotate body to azimuth
        start_body_yaw=current_body_yaw,
        duration=rotation_duration,
    )

    # Move 2: Lower head and tilt to final altitude while at target azimuth
    # Return to neutral z position along with final pitch
    target_head_pose = create_head_pose(
        x=0,
        y=0,
        z=0,  # Return to neutral vertical position
        roll=0,
        pitch=-altitude_rad,  # Negate so that positive angles are towards the sky
        yaw= - azimuth_rad,  # Head yaw follows body azimuth
        degrees=False,
    )

    tilt_to_altitude = GotoQueueMove(
        target_head_pose=target_head_pose,
        start_head_pose=safe_pre_pose_at_azimuth,  # Start from rotated safe position
        target_antennas=(current_antenna1, current_antenna2),
        start_antennas=(current_antenna1, current_antenna2),
        target_body_yaw= - azimuth_rad,  # Stay at azimuth
        start_body_yaw= - azimuth_rad,
        duration=tilt_duration,
    )

    # Move 3: Hold at target position
    hold_at_target = GotoQueueMove(
        target_head_pose=target_head_pose,
        start_head_pose=target_head_pose,
        target_antennas=(current_antenna1, current_antenna2),
        start_antennas=(current_antenna1, current_antenna2),
        target_body_yaw= - azimuth_rad,
        start_body_yaw= - azimuth_rad,
        duration=hold_duration,
    )

    # Move 4: Return to original position smoothly
    return_duration = 2.0  # Time to return to original position
    return_to_original = GotoQueueMove(
        target_head_pose=current_head_pose,  # Back to original head pose
        start_head_pose=target_head_pose,  # From final target pose
        target_antennas=(current_antenna1, current_antenna2),  # Back to original antennas
        start_antennas=(current_antenna1, current_antenna2),
        target_body_yaw=current_body_yaw,  # Back to original body yaw
        start_body_yaw= - azimuth_rad,  # From azimuth
        duration=return_duration,
    )

    # Queue all moves in sequence
    total_duration = 0
    deps.movement_manager.queue_move(raise_head)
    total_duration += raise_head.duration
    deps.movement_manager.queue_move(rotate_to_azimuth)
    total_duration += rotation_duration
    deps.movement_manager.queue_move(tilt_to_altitude)
    total_duration += tilt_duration
    deps.movement_manager.queue_move(hold_at_target)
    total_duration += hold_duration
    deps.movement_manager.queue_move(return_to_original)
    total_duration += return_duration

    # Mark as moving
    deps.movement_manager.set_moving_state(total_duration)

    return {
        "status": f"looking at sky: azimuth={azimuth_deg:.1f}°, altitude={altitude_deg_clamped:.1f}°, duration={total_duration:.1f}s"
    }
