from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
import inspect

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import cv2
import numpy as np

from reachy_mini_conversation_demo.vision import VisionManager
from reachy_mini_conversation_demo.movement import MovementManager

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
    """
    Base abstraction for tools used in function-calling.

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
        direction: Direction = kwargs.get("direction")
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        result = _execute_motion(deps, target)
        if "error" in result:
            return result

        return {"status": f"looking {direction}"}


class Camera(Tool):
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


class HeadTracking(Tool):
    name = "head_tracking"
    description = "Toggle head tracking state."
    parameters_schema = {
        "type": "object",
        "properties": {"start": {"type": "boolean"}},
        "required": ["start"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        enable = bool(kwargs.get("start"))
        movement_manager = deps.movement_manager
        movement_manager.is_head_tracking_on = enable
        status = "started" if enable else "stopped"
        logger.info("Tool call: head_tracking %s", status)
        return {"status": f"head tracking {status}"}


class DescribeCurrentScene(Tool):
    name = "describe_current_scene"
    description = "Get a detailed description of the current scene."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        logger.info("Tool call: describe_current_scene")

        result = await deps.vision_manager.process_current_frame(
            "Describe what you currently see in detail, focusing on people, objects, and activities."
        )

        if isinstance(result, dict) and "error" in result:
            return result
        return result


class GetSceneContext(Tool):
    name = "get_scene_context"
    description = (
        "Get the most recent automatic scene description for conversational context."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        logger.info("Tool call: get_scene_context")
        vision_manager = deps.vision_manager
        if not vision_manager:
            return {"error": "Vision processing not available"}

        try:
            description = await deps.vision_manager.get_current_description()

            if not description:
                return {
                    "context": "No scene description available yet",
                    "note": "Vision processing may still be initializing",
                }
            return {
                "context": description,
                "note": "This comes from periodic automatic analysis",
            }
        except Exception as e:
            logger.exception("Failed to get scene context")
            return {"error": f"Scene context failed: {type(e).__name__}: {e}"}


class AnalyzeSceneFor(Tool):
    name = "analyze_scene_for"
    description = "Analyze the current scene for a specific purpose."
    parameters_schema = {
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "enum": [
                    "safety",
                    "people",
                    "objects",
                    "activity",
                    "navigation",
                    "general",
                ],
                "default": "general",
            }
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        purpose = (kwargs.get("purpose") or "general").lower()
        logger.info("Tool call: analyze_scene_for purpose=%s", purpose)

        prompts = {
            "safety": "Look for safety concerns, obstacles, or hazards.",
            "people": "Describe people, their positions and actions.",
            "objects": "Identify and describe main visible objects.",
            "activity": "Describe ongoing activities or actions.",
            "navigation": "Describe the space for navigation: obstacles, pathways, layout.",
            "general": "Give a general description of the scene including people, objects, and activities.",
        }
        prompt = prompts.get(purpose, prompts["general"])

        result = await deps.vision_manager.process_current_frame(prompt)

        if isinstance(result, dict) and "error" in result:
            return result

        if not isinstance(result, dict):
            return {"error": "vision returned non-dict"}

        result["analysis_purpose"] = purpose
        return result


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
