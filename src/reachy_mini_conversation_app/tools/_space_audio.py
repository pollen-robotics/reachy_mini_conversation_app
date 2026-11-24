"""Call audio-generating Hugging Face Spaces as the signed-in user and keep their audio on disk."""

import re
import time
import base64
import asyncio
import logging
import tempfile
from typing import Any
from pathlib import Path

import httpx

from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    build_namespaced_tool_name,
)
from reachy_mini_conversation_app.tool_spaces import current_hf_token, explain_space_failure
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

AUDIO_DIRECTORY = Path(tempfile.gettempdir()) / "reachy_mini_audio"
_ALIAS = "space"
# Quiet this long first: shorter gaps are the pause between the user's turn and the robot's reply.
QUIET_BEFORE_PLAYING_S = 1.5
_MAX_WAIT_FOR_QUIET_S = 60.0
_QUIET_POLL_S = 0.1


async def wait_for_quiet(deps: ToolDependencies) -> None:
    """Wait until neither the robot nor the user has spoken for a moment; play anyway after a minute."""
    if deps.conversation_busy is None:
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _MAX_WAIT_FOR_QUIET_S
    quiet_since: float | None = None
    while loop.time() < deadline:
        now = loop.time()
        if deps.conversation_busy():
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= QUIET_BEFORE_PLAYING_S:
            return
        await asyncio.sleep(_QUIET_POLL_S)
    logger.warning("The conversation stayed busy for %.0f s; playing anyway", _MAX_WAIT_FOR_QUIET_S)


async def call_space_tool(
    slug: str,
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout_s: float,
    max_event_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Call one tool of an MCP Space as the signed-in user.

    Returns the tool result, or ``{"error", ...}`` naming why the Space failed.
    """
    token = await asyncio.to_thread(current_hf_token)
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=_ALIAS,
            url=mcp_url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            tool_timeout_s=timeout_s,
            max_event_bytes=max_event_bytes,
        )
    )
    try:
        result = await client.call_tool(build_namespaced_tool_name(_ALIAS, tool_name), arguments)
    except McpClientError as exc:
        detail = f"{type(exc).__name__}: {exc}"
    else:
        if result.get("status") != "error":
            return result
        detail = result.get("text") or f"{slug} returned an error"
    logger.error("Space %s failed: %s", slug, detail)
    return await asyncio.to_thread(explain_space_failure, slug, detail) or {"error": detail}


def save_audio(audio: str, name: str) -> Path:
    """Write audio given as a data URI or a URL to the audio directory; blocking."""
    if audio.startswith("data:"):
        header, encoded = audio.split(",", 1)
        data = base64.b64decode(encoded)
        suffix = "." + header.split(";")[0].split("/")[-1]
    else:
        response = httpx.get(audio, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        data = response.content
        suffix = Path(audio.split("?")[0]).suffix or ".wav"
    AUDIO_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "audio"
    # Nanoseconds: two lines saved in the same second must not share a file.
    path = AUDIO_DIRECTORY / f"{stem}-{time.time_ns()}{suffix}"
    path.write_bytes(data)
    return path
