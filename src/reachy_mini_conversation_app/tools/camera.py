import base64
import asyncio
import logging
from typing import Any, Dict

import cv2

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

        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        if deps.vision_processor is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_processor.process_image, frame, question,
            )
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        # Encode image directly to JPEG bytes without writing to file
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        b64_encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return {"b64_im": b64_encoded}
