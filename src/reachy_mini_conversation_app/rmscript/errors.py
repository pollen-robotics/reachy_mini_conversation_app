"""Error handling and compilation result types for ReachyMiniScript."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Callable, Optional
from dataclasses import field, dataclass

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@dataclass
class CompilationError:
    """Represents a compilation error or warning."""

    line: int
    column: int = 0
    message: str = ""
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        """Format error message for display."""
        icon = "❌" if self.severity == "error" else "⚠️ "
        return f"{icon} Line {self.line}: {self.message}"


@dataclass
class Action:
    """Resolved action - all defaults applied, ready to execute."""

    head_pose: Optional[npt.NDArray[np.float64]] = None  # 4x4 matrix
    antennas: Optional[List[float]] = None  # [right, left] in radians
    body_yaw: Optional[float] = None  # radians
    duration: float = 1.0
    interpolation: str = "minjerk"

    # Metadata for debugging
    source_line: int = 0
    original_text: str = ""


@dataclass
class WaitAction:
    """Wait/pause action."""

    duration: float
    source_line: int = 0
    original_text: str = ""


@dataclass
class PictureAction:
    """Take a picture action."""

    source_line: int = 0
    original_text: str = ""


@dataclass
class PlaySoundAction:
    """Play a sound action."""

    sound_name: str  # Name of the sound file (without extension)
    blocking: bool = False  # True = wait for sound to finish, False = play in background
    source_line: int = 0
    original_text: str = ""


@dataclass
class CompiledTool:
    """Result of ReachyMiniScript compilation."""

    # Tool metadata (for LLM)
    name: str
    description: str  # LLM uses this to decide when to call the tool

    # Execution
    executable: Optional[Callable[..., Any]] = None  # The compiled function

    # Compilation results
    success: bool = False
    errors: List[CompilationError] = field(default_factory=list)
    warnings: List[CompilationError] = field(default_factory=list)

    # Debugging/inspection
    source_code: str = ""
    source_file_path: Optional[str] = None  # Path to the .rmscript file (for sound file discovery)
    ir: List[Action | WaitAction | PictureAction | PlaySoundAction] = field(
        default_factory=list
    )  # Intermediate representation

    def execute(self, mini: "ReachyMini") -> None:  # type: ignore # noqa: F821
        """Execute the compiled behavior on the robot."""
        if not self.success:
            error_messages = "\n".join(str(e) for e in self.errors)
            raise RuntimeError(
                f"Cannot execute tool '{self.name}' with compilation errors:\n{error_messages}"
            )
        if self.executable is None:
            raise RuntimeError(f"Tool '{self.name}' has no executable function")
        self.executable(mini)

    def execute_queued(self, deps: "ToolDependencies") -> Dict[str, Any]:
        """Execute as queued moves through the movement manager.

        This converts the IR actions into GotoQueueMove objects and queues them
        for smooth, sequential execution.
        """
        if not self.success:
            error_messages = "\n".join(str(e) for e in self.errors)
            return {
                "error": f"Cannot execute tool '{self.name}' with compilation errors: {error_messages}"
            }

        # Import here to avoid circular dependency
        from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
        from reachy_mini_conversation_app.rmscript.sound_player import SoundQueueMove
        from reachy_mini_conversation_app.rmscript.picture_capture import PictureQueueMove

        mini = deps.reachy_mini
        movement_manager = deps.movement_manager

        # Get initial state from robot
        current_head_pose = mini.get_current_head_pose()
        head_joints, antenna_joints = mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]
        current_antennas = (antenna_joints[0], antenna_joints[1])

        total_duration = 0.0
        move_count = 0
        picture_moves = []  # Track picture moves to collect results later

        for action in self.ir:
            if isinstance(action, WaitAction):
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

            if isinstance(action, PictureAction):
                # Handle picture by queueing a PictureQueueMove
                picture_move = PictureQueueMove(
                    camera_worker=deps.camera_worker,
                    duration=0.01,  # Instant move
                )
                movement_manager.queue_move(picture_move)
                picture_moves.append(picture_move)  # Track for later collection
                total_duration += picture_move.duration
                move_count += 1
                continue

            if isinstance(action, PlaySoundAction):
                # Handle sound playback by queueing a SoundQueueMove
                from pathlib import Path

                from reachy_mini_conversation_app.rmscript.sound_player import (
                    find_sound_file,
                    get_sound_duration,
                )

                # Determine search paths for sound files
                search_paths = []

                # 1. Directory containing the .rmscript file (highest priority)
                if self.source_file_path:
                    script_dir = Path(self.source_file_path).parent
                    search_paths.append(script_dir)

                # 2. Current working directory
                search_paths.append(Path.cwd())

                # 3. sounds/ subdirectory
                search_paths.append(Path.cwd() / "sounds")

                # Find the sound file
                sound_file = find_sound_file(action.sound_name, search_paths)

                if sound_file:
                    # Get duration for blocking mode
                    sound_duration = get_sound_duration(sound_file) if action.blocking else 0.0

                    # Create a SoundQueueMove and add to queue
                    sound_move = SoundQueueMove(
                        sound_file_path=str(sound_file),
                        duration=sound_duration,
                        blocking=action.blocking
                    )
                    movement_manager.queue_move(sound_move)

                    # Add to total duration and move count
                    total_duration += sound_move.duration
                    move_count += 1
                else:
                    import logging
                    logging.warning(f"Sound file not found: {action.sound_name}")

                continue

            if isinstance(action, Action):
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
            "status": f"Queued {move_count} moves from '{self.name}'",
            "total_duration": f"{total_duration:.1f}s",
        }

        # If there are pictures, wait for them to be captured and add to result
        if picture_moves:
            import time

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

    def to_python_code(self) -> str:
        """Generate equivalent Python code (for learning/debugging)."""
        if not self.success:
            return f"# Compilation failed\n# {len(self.errors)} error(s)\n"

        lines = [
            "import time",
            "import numpy as np",
            "from reachy_mini.utils import create_head_pose",
            "",
            f'def {self.name}(mini):',
            f'    """{self.description}"""',
        ]

        for action in self.ir:
            if isinstance(action, WaitAction):
                lines.append(f"    time.sleep({action.duration})")
            elif isinstance(action, Action):
                params = []
                if action.head_pose is not None:
                    # Extract euler angles for display
                    from scipy.spatial.transform import Rotation as R

                    rotation = R.from_matrix(action.head_pose[:3, :3])
                    roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
                    position = action.head_pose[:3, 3] * 1000  # to mm

                    pose_params = []
                    if abs(position[0]) > 0.001:
                        pose_params.append(f"x={position[0]:.1f}")
                    if abs(position[1]) > 0.001:
                        pose_params.append(f"y={position[1]:.1f}")
                    if abs(position[2]) > 0.001:
                        pose_params.append(f"z={position[2]:.1f}")
                    if abs(roll) > 0.1:
                        pose_params.append(f"roll={roll:.1f}")
                    if abs(pitch) > 0.1:
                        pose_params.append(f"pitch={pitch:.1f}")
                    if abs(yaw) > 0.1:
                        pose_params.append(f"yaw={yaw:.1f}")

                    if pose_params:
                        pose_str = ", ".join(pose_params)
                        params.append(
                            f"head=create_head_pose({pose_str}, mm=True, degrees=True)"
                        )
                    else:
                        params.append("head=create_head_pose()")

                if action.antennas is not None:
                    right_deg = np.rad2deg(action.antennas[0])
                    left_deg = np.rad2deg(action.antennas[1])
                    params.append(
                        f"antennas=np.deg2rad([{right_deg:.1f}, {left_deg:.1f}])"
                    )

                if action.body_yaw is not None:
                    body_deg = np.rad2deg(action.body_yaw)
                    params.append(f"body_yaw=np.deg2rad({body_deg:.1f})")

                params.append(f"duration={action.duration}")

                if action.interpolation != "minjerk":
                    params.append(f'method="{action.interpolation}"')

                params_str = ", ".join(params)
                lines.append(f"    mini.goto_target({params_str})")

        return "\n".join(lines)

    def print_errors(self) -> None:
        """Print all errors and warnings to console."""
        for error in self.errors:
            print(str(error))
        for warning in self.warnings:
            print(str(warning))
