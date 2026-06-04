import base64
import asyncio
import logging
from typing import Any, Dict
from pathlib import Path

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg


logger = logging.getLogger(__name__)
_SOUNDS_DIR = Path(__file__).resolve().parents[1] / "sounds"
_CAMERA_SNAPSHOT_SOUND = "camera_snapshot.wav"


async def _play_snapshot_sound(deps: ToolDependencies) -> None:
    """Play a short shutter sound when a camera frame is captured."""
    media = getattr(getattr(deps, "reachy_mini", None), "media", None)
    play_sound = getattr(media, "play_sound", None)
    if not callable(play_sound):
        logger.debug("camera: media.play_sound is unavailable; skipping snapshot sound")
        return

    path = _SOUNDS_DIR / _CAMERA_SNAPSHOT_SOUND
    try:
        await asyncio.to_thread(play_sound, str(path))
    except Exception:
        logger.warning("camera: failed to play snapshot sound", exc_info=True)


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

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        await _play_snapshot_sound(deps)

        if deps.vision_processor is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_processor.process_image,
                frame,
                question,
            )
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
