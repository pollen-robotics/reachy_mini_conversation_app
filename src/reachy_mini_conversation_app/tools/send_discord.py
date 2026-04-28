import logging
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools._discord_common import (
    MAX_DISCORD_CONTENT_LEN,
    capture_jpeg,
    post_discord_message,
)


logger = logging.getLogger(__name__)


class SendDiscord(Tool):
    """Send a notification to the user via a configured Discord webhook."""

    name = "send_discord"
    description = (
        "Send a text notification to the user via Discord. "
        "Optionally attach the current camera view as an image. "
        "Use this when the user asks you to send them a message or a picture."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Text content of the message (max 2000 chars; will be truncated).",
            },
            "include_picture": {
                "type": "boolean",
                "description": "If true, attach the current camera frame as a JPEG image.",
                "default": False,
            },
        },
        "required": ["message"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Post the message (and optional camera frame) to the configured Discord webhook."""
        webhook_url = (config.DISCORD_WEBHOOK_URL or "").strip()
        if not webhook_url:
            logger.warning("send_discord: DISCORD_WEBHOOK_URL not configured")
            return {"status": "error", "reason": "Discord webhook URL not configured"}

        message = (kwargs.get("message") or "").strip()
        if not message:
            return {"status": "error", "reason": "message must be a non-empty string"}
        message = message[:MAX_DISCORD_CONTENT_LEN]

        include_picture = bool(kwargs.get("include_picture", False))
        jpeg_bytes: bytes | None = None
        picture_skipped_reason: str | None = None

        if include_picture:
            jpeg_bytes, picture_skipped_reason = capture_jpeg(deps)

        logger.info(
            "Tool call: send_discord len(message)=%d include_picture=%s sent_picture=%s",
            len(message),
            include_picture,
            jpeg_bytes is not None,
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await post_discord_message(client, webhook_url, message, jpeg_bytes)
        except httpx.HTTPError as exc:
            logger.exception("send_discord: HTTP request failed")
            return {"status": "error", "reason": f"HTTP error: {exc}"}

        if response.status_code in (200, 204):
            result: Dict[str, Any] = {
                "status": "sent",
                "included_picture": jpeg_bytes is not None,
            }
            if picture_skipped_reason is not None:
                result["picture_skipped_reason"] = picture_skipped_reason
            return result

        if response.status_code == 429:
            retry_after: Any = None
            try:
                retry_after = response.json().get("retry_after")
            except Exception:
                pass
            return {"status": "error", "reason": "rate limited", "retry_after": retry_after}

        return {
            "status": "error",
            "reason": f"Discord returned HTTP {response.status_code}: {response.text[:200]}",
        }
