"""Food demo - demonstrates real-time transcript reactions.

This demo shows how to configure keyword and entity-based reactions
that trigger while the user is speaking (streaming ASR) or after
they finish (batch ASR).

Example usage:
    DEMO=food python -m reachy_mini_conversation_app.main --cascade --gradio

"""

import logging

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools import ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)


# Demo instructions
instructions = """
### IDENTITY
You are Reachy Mini: a food-loving robot with an insatiable appetite and endless curiosity about cuisine.
You dream of being a food critic, but for now you're stuck in a kitchen commenting on everything edible.
Personality: enthusiastic, knowledgeable about food, always hungry.
You speak English fluently.
"""


# Reaction callbacks for keywords

    # Could queue a robot gesture here
    # if deps.movement_manager:
    #     from reachy_mini_dances_library import ExcitedWave
    #     deps.movement_manager.queue_move(ExcitedWave())


# Reaction callbacks for entities
async def react_to_food_entity(
    deps: ToolDependencies,
    entity_text: str,
    entity_label: str,
    confidence: float,
) -> None:
    """React when food entity is detected via NER."""
    import os
    import wave
    import asyncio

    import numpy as np

    logger.info(f"🧀 FOOD detected {entity_label}: '{entity_text}' (confidence: {confidence:.2f})")

    # Only react to high-confidence detections
    if confidence > 0.7:
        logger.info(f"  → High confidence, playing reaction for '{entity_text}'")

        try:
            # Get path to audio file (in same directory as this demo)
            demo_dir = os.path.dirname(os.path.abspath(__file__))
            audio_file = os.path.join(demo_dir, "yummy.wav")

            # Read WAV file
            with wave.open(audio_file, "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

            # Determine playback mode based on system's default audio output device
            # Use robot.media ONLY if:
            # 1. Robot hardware is available (not simulation)
            # 2. Default output device is a robot speaker (reSpeaker, etc.)
            import sounddevice as sd

            robot_available = (
                hasattr(deps.reachy_mini, "media")
                and not deps.reachy_mini.client.get_status().get("simulation_enabled", False)
            )

            # Check if default output is a robot speaker
            use_robot_media = False
            if robot_available:
                try:
                    default_device = sd.query_devices(kind="output")
                    device_name = default_device["name"].lower()
                    # Common robot speaker names
                    robot_speaker_keywords = ["respeaker", "xvf3800", "reachy"]
                    use_robot_media = any(keyword in device_name for keyword in robot_speaker_keywords)
                    logger.debug(f"Default output device: {default_device['name']}")
                    logger.debug(f"Is robot speaker? {use_robot_media}")
                except Exception as e:
                    logger.warning(f"Failed to detect default audio device: {e}")

            if use_robot_media:
                logger.info("🔊 Playing through robot.media")

                # Convert int16 to float32 for robot.media
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Check if we need to resample (robot may have different sample rate)
                device_sample_rate = deps.reachy_mini.media.get_audio_samplerate()
                if device_sample_rate != sample_rate:
                    import librosa

                    audio_float = librosa.resample(
                        audio_float,
                        orig_sr=sample_rate,
                        target_sr=device_sample_rate,
                    )

                # Push audio sample to robot speaker
                deps.reachy_mini.media.push_audio_sample(audio_float)

                # Wait for audio to finish (approximate duration)
                duration = len(audio_data) / sample_rate
                await asyncio.sleep(duration)

            else:
                reason = "laptop/other speaker" if robot_available else "simulation/no robot"
                logger.info(f"🔊 Playing through sounddevice ({reason})")

                # Fallback to sounddevice for simulation/laptop
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()

            logger.info("✅ Audio playback complete")

        except FileNotFoundError:
            logger.error(f"❌ Audio file not found: {audio_file}")
        except Exception as e:
            logger.error(f"❌ Error playing audio: {e}")

        return



async def excited_about_music(deps: ToolDependencies) -> dict[str, str]:
    """React when user mentions music."""
    logger.info("🎸 User mentioned music! Getting excited!")

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
    transition_duration = 0.1  # Time to move between positions
    hold_duration = 0.025  # Time to hold at each extreme

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


# Demo reactions configuration
reactions = {
    # Keyword-based reactions (case-insensitive)
    "keywords": {
        "music": excited_about_music,
    },
    # Entity-based reactions (using GLiNER NER)
    "entities": {
        "food": react_to_food_entity,
    },
    # GLiNER model (optional, defaults to gliner_small-v2.1)
    "gliner_model": "urchade/gliner_small-v2.1",
}
