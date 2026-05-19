"""Roast target extractor — scene scan, head orient, close-up capture, structured extraction."""

from __future__ import annotations
import re
import base64
import asyncio
import logging
from typing import Any, Dict

from robot_comic.tools.move_head import MoveHead
from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

SCENE_SCAN_PROMPT = (
    "Is there a person visible in this image? "
    "If yes, reply: 'PERSON: <position>' where position is one of: left, center, right. "
    "If no person is visible, reply: 'NO PERSON'."
)

EXTRACTION_PROMPT = (
    "Describe this person for a comedy roast. Be specific and a little uncharitable. "
    "Reply EXACTLY in this format with no extra text:\n"
    "hair: <description>\n"
    "clothing: <description>\n"
    "build: <description>\n"
    "expression: <description>\n"
    "standout: <the single most notable or ridiculous thing about them>\n"
    "energy: <how they seem — nervous, confident, bored, etc.>"
)

_DIRECTION_MAP: Dict[str, str] = {
    "left": "left",
    "right": "right",
    "center": "front",
}


class Roast(Tool):
    """Locate a person, orient toward them, and return labelled roast targets."""

    name = "roast"
    description = (
        "Scan the scene for a person, aim the head toward them, and return labelled roast targets: "
        "hair, clothing, build, expression, standout, energy. Call this once at conversation open."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def _describe(self, deps: ToolDependencies, frame: Any, prompt: str) -> str | None:
        if deps.vision_processor is None:
            return None
        result = await asyncio.to_thread(deps.vision_processor.process_image, frame, prompt)
        return result if isinstance(result, str) else None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Two-phase capture: scene scan → head orient → close-up → structured extraction."""
        logger.info("Tool call: roast")

        if deps.camera_worker is None:
            return {"error": "Camera worker not available"}

        # Phase 1: Wide scene scan
        wide_frame = deps.camera_worker.get_latest_frame()
        if wide_frame is None:
            return {"error": "No frame available from camera worker"}

        if deps.vision_processor is None:
            # No local vision — return b64 frames for the realtime backend to interpret
            # Deferred import: av (via camera_frame_encoding) costs ~0.5–0.8 s at
            # module load time on the Pi. Only this branch needs it, so import here.
            from robot_comic.camera_frame_encoding import encode_bgr_frame_as_jpeg  # noqa: PLC0415

            jpeg = encode_bgr_frame_as_jpeg(wide_frame)
            return {
                "b64_scene": base64.b64encode(jpeg).decode("utf-8"),
                "extraction_prompt": EXTRACTION_PROMPT,
                "note": (
                    "No local vision processor available. "
                    "Describe the person in the image using these fields: "
                    "hair, clothing, build, expression, standout, energy."
                ),
            }

        scan_result = await self._describe(deps, wide_frame, SCENE_SCAN_PROMPT)
        if scan_result is None:
            return {"error": "Vision processor returned no result during scene scan"}

        # Phase 2: Check for person and determine direction
        if "NO PERSON" in scan_result.upper():
            return {"no_subject": True}

        _dir_match = re.search(r"PERSON:\s*(\w+)", scan_result, re.IGNORECASE)
        raw_dir = _dir_match.group(1).lower() if _dir_match else "center"
        direction = _DIRECTION_MAP.get(raw_dir, "front")

        # Phase 3: Orient head toward person
        move_result = await MoveHead()(deps, direction=direction)
        if "error" in move_result:
            logger.warning("move_head failed during roast: %s", move_result["error"])

        await asyncio.sleep(deps.motion_duration_s + 0.2)

        # Phase 4: Close-up capture and structured extraction
        close_frame = deps.camera_worker.get_latest_frame()
        if close_frame is None:
            return {"error": "No frame available after head movement"}

        extraction = await self._describe(deps, close_frame, EXTRACTION_PROMPT)
        if extraction is None:
            return {"error": "Vision processor returned no result during extraction"}

        return _parse_extraction(extraction)


def _parse_extraction(text: str) -> Dict[str, Any]:
    """Parse the labelled extraction response into a structured dict."""
    fields = ["hair", "clothing", "build", "expression", "standout", "energy"]
    result: Dict[str, Any] = {}
    for field in fields:
        pattern = rf"(?i){re.escape(field)}:\s*(.+?)(?=\n\w[\w ]*:|$)"
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        result[field] = match.group(1).strip() if match else "unknown"
    return result
