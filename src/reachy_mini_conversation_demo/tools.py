from __future__ import annotations

import abc
import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import cv2
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from reachy_mini_conversation_demo.movement import MovementManager
from reachy_mini_conversation_demo.vision.processors import VisionManager

logger = logging.getLogger(__name__)


def all_concrete_subclasses(base):
    result = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        # recurse into subclasses
        result.extend(all_concrete_subclasses(cls))
    return result


# Types & state
Direction = Literal["left", "right", "up", "down", "front"]


@dataclass
class ToolDependencies:
    """External dependencies injected into tools"""

    reachy_mini: ReachyMini
    create_head_pose: Any
    movement_manager: MovementManager
    # Optional deps
    camera: Optional[cv2.VideoCapture] = None
    vision_manager: Optional[VisionManager] = None
    camera_retry_attempts: int = 5
    camera_retry_delay_s: float = 0.10
    vision_timeout_s: float = 8.0
    motion_duration_s: float = 1.0


# Helpers
def _read_frame(
    cap: cv2.VideoCapture, attempts: int = 5, delay_s: float = 0.1
) -> np.ndarray:
    """Read a frame from the camera with retries."""
    trials, frame, ret = 0, None, False
    while trials < attempts and not ret:
        ret, frame = cap.read()
        trials += 1
        if not ret and trials < attempts:
            time.sleep(delay_s)
    if not ret or frame is None:
        logger.error("Failed to capture image from camera after %d attempts", attempts)
        raise RuntimeError("Failed to capture image from camera.")
    return frame


def _execute_motion(deps: ToolDependencies, target: Any) -> Dict[str, Any]:
    """Apply motion to reachy_mini and update movement_manager state."""
    movement_manager = deps.movement_manager
    movement_manager.moving_start = time.monotonic()
    movement_manager.moving_for = deps.motion_duration_s
    movement_manager.current_head_pose = target
    try:
        deps.reachy_mini.goto_target(target, duration=deps.motion_duration_s)
    except Exception as e:
        logger.exception("motion failed")
        return {"error": f"motion failed: {type(e).__name__}: {e}"}

    return {"status": "ok"}


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
    name = "move_head"
    description = "Move my head in a given direction."
    parameters_schema = {
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
        direction: Direction = kwargs.get("direction")
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        result = _execute_motion(deps, target)
        if "error" in result:
            return result

        return {"status": f"Now looking {direction}"}


class Camera(Tool):
    name = "camera"
    description = "Take a picture and ask a question about it."
    parameters_schema = {
        "properties": {
            "question": {
                "type": "string",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Capture a frame
        try:
            frame = await asyncio.to_thread(_read_frame, deps.camera)
        except Exception as e:
            logger.exception("camera: failed to capture image")
            return {"error": f"camera capture failed: {type(e).__name__}: {e}"}

        result = await asyncio.to_thread(
            deps.vision_manager.processor.process_image, frame, image_query
        )
        if isinstance(result, dict) and "error" in result:
            return result
        return (
            {"image_description": result}
            if isinstance(result, str)
            else {"error": "vision returned non-string"}
        )


# Registry & specs (dynamic)

# List of available tool classes
ALL_TOOLS: Dict[str, Tool] = {cls.name: cls() for cls in all_concrete_subclasses(Tool)}
ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]


# Dispatcher
def _safe_load_obj(args_json: str) -> dict[str, Any]:
    try:
        obj = json.loads(args_json or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def dispatch_tool_call(
    tool_name: str, args_json: str, deps: ToolDependencies
) -> Dict[str, Any]:
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
