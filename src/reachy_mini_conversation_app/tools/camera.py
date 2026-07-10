import base64
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


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

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
        if jpeg_bytes is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}

        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
