from __future__ import annotations
import abc
import json
import asyncio
import inspect
import logging
from typing import Any, Dict, Literal, Optional
from dataclasses import dataclass

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


logger = logging.getLogger(__name__)

# Initialize dance and emotion libraries
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    from reachy_mini_conversation_demo.dance_emotion_moves import (
        GotoQueueMove,
        DanceQueueMove,
        EmotionQueueMove,
    )

    # Initialize recorded moves for emotions
    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    DANCE_AVAILABLE = True
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dance/emotion libraries not available: {e}")
    AVAILABLE_MOVES = {}
    RECORDED_MOVES = None
    DANCE_AVAILABLE = False
    EMOTION_AVAILABLE = False


def get_concrete_subclasses(base):
    """Recursively find all concrete (non-abstract) subclasses of a base class."""
    result = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        # recurse into subclasses
        result.extend(get_concrete_subclasses(cls))
    return result


# Types & state
Direction = Literal["left", "right", "up", "down", "front"]


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any  # MovementManager from moves.py
    # Optional deps
    camera_worker: Optional[Any] = None  # CameraWorker for frame buffering
    vision_manager: Optional[Any] = None
    head_wobbler: Optional[Any] = None  # HeadWobbler for audio-reactive motion
    motion_duration_s: float = 1.0


# Tool base class
class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError


# Concrete tools


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = "Move your head in a given direction: left, right, up, down or front."
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: dict[str, tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Move head in a given direction."""
        direction: Direction = kwargs.get("direction")
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        # Use new movement manager
        try:
            movement_manager = deps.movement_manager

            # Get current state for interpolation
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            _, current_antennas = deps.reachy_mini.get_current_joint_positions()

            # Create goto move
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),  # Reset antennas to default
                start_antennas=(
                    current_antennas[0],
                    current_antennas[1],
                ),  # Skip body_yaw
                target_body_yaw=0,  # Reset body yaw
                start_body_yaw=current_antennas[0],  # body_yaw is first in joint positions
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed")
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer (like main_works.py)
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )
        else:
            # Return base64 encoded image like main_works.py camera tool
            import base64

            import cv2

            temp_path = "/tmp/camera_frame.jpg"
            cv2.imwrite(temp_path, frame)
            with open(temp_path, "rb") as f:
                b64_encoded = base64.b64encode(f.read()).decode("utf-8")
            return {"b64_im": b64_encoded}


class HeadTracking(Tool):
    """Toggle head tracking state."""

    name = "head_tracking"
    description = "Toggle head tracking state."
    parameters_schema = {
        "type": "object",
        "properties": {"start": {"type": "boolean"}},
        "required": ["start"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Enable or disable head tracking."""
        enable = bool(kwargs.get("start"))

        # Update camera worker head tracking state
        if deps.camera_worker is not None:
            deps.camera_worker.set_head_tracking_enabled(enable)

        status = "started" if enable else "stopped"
        logger.info("Tool call: head_tracking %s", status)
        return {"status": f"head tracking {status}"}



class Dance(Tool):
    """Play a named or random dance move once (or repeat). Non-blocking."""

    name = "dance"
    description = "Play a named or random dance move once (or repeat). Non-blocking."
    parameters_schema = {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": """Name of the move; use 'random' or omit for random.
                                    Here is a list of the available moves:
                                        simple_nod: A simple, continuous up-and-down nodding motion.
                                        head_tilt_roll: A continuous side-to-side head roll (ear to shoulder).
                                        side_to_side_sway: A smooth, side-to-side sway of the entire head.
                                        dizzy_spin: A circular 'dizzy' head motion combining roll and pitch.
                                        stumble_and_recover: A simulated stumble and recovery with multiple axis movements. Good vibes
                                        headbanger_combo: A strong head nod combined with a vertical bounce.
                                        interwoven_spirals: A complex spiral motion using three axes at different frequencies.
                                        sharp_side_tilt: A sharp, quick side-to-side tilt using a triangle waveform.
                                        side_peekaboo: A multi-stage peekaboo performance, hiding and peeking to each side.
                                        yeah_nod: An emphatic two-part yeah nod using transient motions.
                                        uh_huh_tilt: A combined roll-and-pitch uh-huh gesture of agreement.
                                        neck_recoil: A quick, transient backward recoil of the neck.
                                        chin_lead: A forward motion led by the chin, combining translation and pitch.
                                        groovy_sway_and_roll: A side-to-side sway combined with a corresponding roll for a groovy effect.
                                        chicken_peck: A sharp, forward, chicken-like pecking motion.
                                        side_glance_flick: A quick glance to the side that holds, then returns.
                                        polyrhythm_combo: A 3-beat sway and a 2-beat nod create a polyrhythmic feel.
                                        grid_snap: A robotic, grid-snapping motion using square waveforms.
                                        pendulum_swing: A simple, smooth pendulum-like swing using a roll motion.
                                        jackson_square: Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint.
                """,
            },
            "repeat": {
                "type": "integer",
                "description": "How many times to repeat the move (default 1).",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Play a named or random dance move once (or repeat). Non-blocking."""
        if not DANCE_AVAILABLE:
            return {"error": "Dance system not available"}

        move_name = kwargs.get("move", None)
        repeat = int(kwargs.get("repeat", 1))

        logger.info("Tool call: dance move=%s repeat=%d", move_name, repeat)

        if not move_name or move_name == "random":
            import random

            move_name = random.choice(list(AVAILABLE_MOVES.keys()))

        if move_name not in AVAILABLE_MOVES:
            return {"error": f"Unknown dance move '{move_name}'. Available: {list(AVAILABLE_MOVES.keys())}"}

        # Add dance moves to queue
        movement_manager = deps.movement_manager
        for _ in range(repeat):
            dance_move = DanceQueueMove(move_name)
            movement_manager.queue_move(dance_move)

        return {"status": "queued", "move": move_name, "repeat": repeat}


class StopDance(Tool):
    """Stop the current dance move."""

    name = "stop_dance"
    description = "Stop the current dance move"
    parameters_schema = {
        "type": "object",
        "properties": {
            "dummy": {
                "type": "boolean",
                "description": "dummy boolean, set it to true",
            }
        },
        "required": ["dummy"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Stop the current dance move."""
        logger.info("Tool call: stop_dance")
        movement_manager = deps.movement_manager
        movement_manager.clear_move_queue()
        return {"status": "stopped dance and cleared queue"}


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"

class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": f"""Name of the emotion to play.
                                    Here is a list of the available emotions:
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")
        if not emotion_name:
            return {"error": "Emotion name is required"}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Check if emotion exists
        try:
            emotion_names = RECORDED_MOVES.list_moves()
            if emotion_name not in emotion_names:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {str(e)}"}


class StopEmotion(Tool):
    """Stop the current emotion."""

    name = "stop_emotion"
    description = "Stop the current emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "dummy": {
                "type": "boolean",
                "description": "dummy boolean, set it to true",
            }
        },
        "required": ["dummy"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Stop the current emotion."""
        logger.info("Tool call: stop_emotion")
        movement_manager = deps.movement_manager
        movement_manager.clear_move_queue()
        return {"status": "stopped emotion and cleared queue"}


class DoNothing(Tool):
    """Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill."""

    name = "do_nothing"
    description = "Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional reason for doing nothing (e.g., 'contemplating existence', 'saving energy', 'being mysterious')",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Do nothing - stay still and silent."""
        reason = kwargs.get("reason", "just chilling")
        logger.info("Tool call: do_nothing reason=%s", reason)
        return {"status": "doing nothing", "reason": reason}


# Registry & specs (dynamic)

# List of available tool classes
ALL_TOOLS: Dict[str, Tool] = {cls.name: cls() for cls in get_concrete_subclasses(Tool)}
ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]


# Dispatcher
def _safe_load_obj(args_json: str) -> dict[str, Any]:
    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def dispatch_tool_call(tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
    """Dispatch a tool call by name with JSON args and dependencies."""
    tool = ALL_TOOLS.get(tool_name)

    if not tool:
        return {"error": f"unknown tool: {tool_name}"}

    args = _safe_load_obj(args_json)
    try:
        return await tool(deps, **args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", tool_name, msg)
        return {"error": msg}
