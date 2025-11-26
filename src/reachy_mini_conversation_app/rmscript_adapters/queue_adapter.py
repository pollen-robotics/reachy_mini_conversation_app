"""Queue-based execution adapter for rmscript."""

import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from rmscript.adapters import ExecutionContext
from rmscript.ir import IRAction, IRWaitAction, IRPictureAction, IRPlaySoundAction

from reachy_mini_conversation_app.rmscript_adapters.queue_moves import (
    GotoQueueMove,
    SoundQueueMove,
    PictureQueueMove,
)
from reachy_mini_conversation_app.rmscript_adapters.sound_player import (
    find_sound_file,
    get_sound_duration,
)


logger = logging.getLogger(__name__)


@dataclass
class QueueAdapterContext:
    """Extended context for queue-based execution.

    Includes all ExecutionContext fields plus robot-specific dependencies.
    """

    script_name: str
    script_description: str
    reachy_mini: Any
    movement_manager: Any
    source_file_path: Optional[str] = None
    camera_worker: Any = None


class QueueExecutionAdapter:
    """Executes rmscript IR via MovementManager queue.

    Converts IR actions into queue moves for smooth, sequential robot execution.
    """

    def execute(self, ir: List[IRAction | IRWaitAction | IRPictureAction | IRPlaySoundAction], context: QueueAdapterContext) -> Dict[str, Any]:
        """Execute IR as queued moves through the movement manager.

        Args:
            ir: List of IR actions to execute
            context: Execution context with robot, movement manager, camera worker

        Returns:
            Dictionary with execution results:
            - status: Status message
            - total_duration: Total duration of queued moves
            - b64_im: Base64-encoded image (if single picture)
            - pictures: List of base64 images (if multiple pictures)
            - picture_count: Number of pictures (if multiple)
            - _picture_moves: Internal reference to picture moves (non-JSON)

        """
        mini = context.reachy_mini
        movement_manager = context.movement_manager

        # Get initial state from robot
        current_head_pose = mini.get_current_head_pose()
        head_joints, antenna_joints = mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]
        current_antennas = (antenna_joints[0], antenna_joints[1])

        total_duration = 0.0
        move_count = 0
        picture_moves = []  # Track picture moves to collect results later

        for action in ir:
            if isinstance(action, IRWaitAction):
                # Handle wait by creating a hold move (same start and target)
                hold_move = GotoQueueMove(
                    target_head_pose=current_head_pose,
                    start_head_pose=current_head_pose,
                    target_antennas=current_antennas,
                    start_antennas=current_antennas,
                    target_body_yaw=current_body_yaw,
                    start_body_yaw=current_body_yaw,
                    duration=action.duration,
                )
                movement_manager.queue_move(hold_move)
                total_duration += action.duration
                move_count += 1
                continue

            if isinstance(action, IRPictureAction):
                # Handle picture by queueing a PictureQueueMove
                picture_move = PictureQueueMove(
                    camera_worker=context.camera_worker,
                    duration=0.01,  # Instant move
                )
                movement_manager.queue_move(picture_move)
                picture_moves.append(picture_move)  # Track for later collection
                total_duration += picture_move.duration
                move_count += 1
                continue

            if isinstance(action, IRPlaySoundAction):
                # Handle sound playback by queueing a SoundQueueMove

                # Determine search paths for sound files
                search_paths = []

                # 1. Directory containing the .rmscript file (highest priority)
                if context.source_file_path:
                    script_dir = Path(context.source_file_path).parent
                    search_paths.append(script_dir)

                # 2. Current working directory
                search_paths.append(Path.cwd())

                # 3. sounds/ subdirectory
                search_paths.append(Path.cwd() / "sounds")

                # Find the sound file
                sound_file = find_sound_file(action.sound_name, search_paths)

                if sound_file:
                    # Determine duration
                    if action.loop:
                        # Loop mode: use specified duration (or default 10s)
                        sound_duration = action.duration if action.duration is not None else 10.0
                    elif action.duration is not None:
                        # Explicit duration specified (for play with duration)
                        sound_duration = action.duration
                    elif action.blocking:
                        # Blocking mode: get actual sound duration
                        sound_duration = get_sound_duration(sound_file)
                    else:
                        # Async mode: instant (0.0)
                        sound_duration = 0.0

                    # Create a SoundQueueMove and add to queue
                    sound_move = SoundQueueMove(
                        sound_file_path=str(sound_file),
                        duration=sound_duration,
                        blocking=action.blocking,
                        loop=action.loop
                    )
                    movement_manager.queue_move(sound_move)

                    # Add to total duration and move count
                    total_duration += sound_move.duration
                    move_count += 1
                else:
                    logger.warning(f"Sound file not found: {action.sound_name}")

                continue

            if isinstance(action, IRAction):
                # Determine target state (use current if not specified)
                target_head_pose = (
                    action.head_pose if action.head_pose is not None else current_head_pose
                )
                target_antennas = (
                    (action.antennas[0], action.antennas[1])
                    if action.antennas is not None
                    else current_antennas
                )
                target_body_yaw = (
                    action.body_yaw if action.body_yaw is not None else current_body_yaw
                )

                # Create the move
                move = GotoQueueMove(
                    target_head_pose=target_head_pose,  # type: ignore[arg-type]
                    start_head_pose=current_head_pose,
                    target_antennas=target_antennas,
                    start_antennas=current_antennas,
                    target_body_yaw=target_body_yaw,
                    start_body_yaw=current_body_yaw,
                    duration=action.duration,
                )

                movement_manager.queue_move(move)
                total_duration += action.duration
                move_count += 1

                # Update current state for next iteration
                current_head_pose = target_head_pose
                current_antennas = target_antennas
                current_body_yaw = target_body_yaw

        # Set the moving state with total duration
        movement_manager.set_moving_state(total_duration)

        result = {
            "status": f"Queued {move_count} moves from '{context.script_name}'",
            "total_duration": f"{total_duration:.1f}s",
        }

        # If there are pictures, wait for them to be captured and add to result
        if picture_moves:
            # Wait for movements to complete so pictures are captured
            logger.info(f"Waiting {total_duration:.1f}s for movements and picture capture...")
            time.sleep(total_duration + 0.5)  # Add buffer for capture completion

            # Collect pictures
            pictures = []
            for pm in picture_moves:
                if pm.picture_base64 is not None:
                    pictures.append(pm.picture_base64)
                    if pm.saved_path:
                        logger.info(f"Picture captured: {pm.saved_path}")
                else:
                    logger.warning("Picture capture failed")

            # Format result based on number of pictures
            if len(pictures) == 1:
                # Single picture: use Camera tool format for LLM compatibility
                result["b64_im"] = pictures[0]
                logger.info("Returning single picture in Camera-compatible format (b64_im)")
            elif len(pictures) > 1:
                # Multiple pictures: return as array
                result["pictures"] = pictures  # type: ignore[assignment]
                result["picture_count"] = len(pictures)  # type: ignore[assignment]
                logger.info(f"Returning {len(pictures)} pictures in array format")
            else:
                # No pictures captured successfully
                result["error"] = "Picture capture failed"
                logger.error("All picture captures failed")

            # Store picture_moves separately for non-JSON uses (like run_rmscript.py)
            result["_picture_moves"] = picture_moves  # type: ignore[assignment]

        return result
