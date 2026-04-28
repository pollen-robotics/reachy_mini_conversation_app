import logging
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools._discord_common import (
    DISCORD_API_BASE,
    MAX_DISCORD_CONTENT_LEN,
    capture_jpeg,
    bot_auth_header,
    post_discord_message,
)


logger = logging.getLogger(__name__)


class SendDiscordDM(Tool):
    """Send a direct message to the user via a configured Discord bot."""

    name = "send_discord_dm"
    description = (
        "Send a private direct message to the user on Discord via a bot. "
        "Optionally attach the current camera view as an image. "
        "Use this when the user asks you to DM them on Discord."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Text content of the DM (max 2000 chars; will be truncated).",
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
        """Open (or reuse) a DM channel with the configured user and post the message."""
        bot_token = (config.DISCORD_BOT_TOKEN or "").strip()
        user_id = (config.DISCORD_USER_ID or "").strip()
        if not bot_token or not user_id:
            logger.warning("send_discord_dm: DISCORD_BOT_TOKEN and/or DISCORD_USER_ID not configured")
            return {"status": "error", "reason": "Discord DM not configured"}

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
            "Tool call: send_discord_dm len(message)=%d include_picture=%s sent_picture=%s",
            len(message),
            include_picture,
            jpeg_bytes is not None,
        )

        headers = bot_auth_header(bot_token)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Step 1: open/get DM channel (idempotent — returns existing if any).
                dm_resp = await client.post(
                    f"{DISCORD_API_BASE}/users/@me/channels",
                    json={"recipient_id": user_id},
                    headers=headers,
                )
                if dm_resp.status_code != 200:
                    return _interpret_dm_open_error(dm_resp)

                try:
                    channel_id = dm_resp.json()["id"]
                except (KeyError, ValueError):
                    return {"status": "error", "reason": "malformed DM channel response"}

                # Step 2: post the message into that DM channel.
                msg_resp = await post_discord_message(
                    client,
                    f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                    message,
                    jpeg_bytes,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            logger.exception("send_discord_dm: HTTP request failed")
            return {"status": "error", "reason": f"HTTP error: {exc}"}

        if msg_resp.status_code == 200:
            result: Dict[str, Any] = {
                "status": "sent",
                "included_picture": jpeg_bytes is not None,
            }
            if picture_skipped_reason is not None:
                result["picture_skipped_reason"] = picture_skipped_reason
            return result

        return _interpret_message_error(msg_resp)


def _interpret_dm_open_error(response: httpx.Response) -> Dict[str, Any]:
    """Translate a failed POST /users/@me/channels response to a tool error."""
    if response.status_code == 401:
        return {"status": "error", "reason": "invalid Discord bot token"}
    if response.status_code == 403:
        return {
            "status": "error",
            "reason": "bot is not allowed to DM this user (not sharing a server, or user has DMs closed)",
        }
    if response.status_code == 404:
        return {"status": "error", "reason": "Discord user ID not found"}
    return {
        "status": "error",
        "reason": f"Failed to open DM channel: HTTP {response.status_code}: {response.text[:200]}",
    }


def _interpret_message_error(response: httpx.Response) -> Dict[str, Any]:
    """Translate a failed POST /channels/.../messages response to a tool error."""
    if response.status_code == 429:
        retry_after: Any = None
        try:
            retry_after = response.json().get("retry_after")
        except Exception:
            pass
        return {"status": "error", "reason": "rate limited", "retry_after": retry_after}
    if response.status_code == 403:
        return {"status": "error", "reason": "bot is not allowed to DM this user"}
    return {
        "status": "error",
        "reason": f"Discord returned HTTP {response.status_code}: {response.text[:200]}",
    }
