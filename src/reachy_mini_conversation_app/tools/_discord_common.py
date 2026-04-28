"""Shared helpers for the Discord tools (webhook channel post and bot DM)."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

import httpx

from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


MAX_DISCORD_CONTENT_LEN = 2000
DISCORD_API_BASE = "https://discord.com/api/v10"


def bot_auth_header(token: str) -> dict[str, str]:
    """Build the Authorization header for Discord bot API calls."""
    return {"Authorization": f"Bot {token}"}


def capture_jpeg(deps: ToolDependencies) -> tuple[bytes | None, str | None]:
    """Grab the latest camera frame and JPEG-encode it; return (bytes, skip_reason)."""
    if deps.camera_worker is None:
        return None, "camera worker not available"
    frame = deps.camera_worker.get_latest_frame()
    if frame is None:
        return None, "no camera frame available"
    try:
        return encode_bgr_frame_as_jpeg(frame), None
    except RuntimeError as exc:
        return None, f"failed to encode frame as JPEG: {exc}"


async def post_discord_message(
    client: httpx.AsyncClient,
    url: str,
    message: str,
    jpeg_bytes: bytes | None,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """POST a Discord message (text + optional JPEG) to the given endpoint.

    Webhooks and bot `channels/{id}/messages` accept the same payload shape,
    so callers just pass the URL (and an `Authorization: Bot …` header for
    the bot path).
    """
    if jpeg_bytes is not None:
        files = {"files[0]": ("reachy_view.jpg", jpeg_bytes, "image/jpeg")}
        data = {"payload_json": json.dumps({"content": message})}
        return await client.post(url, data=data, files=files, headers=headers)
    return await client.post(url, json={"content": message}, headers=headers)
