"""Background worker that polls a Discord DM and forwards content to the realtime session.

Text and supported image attachments become `input_text` / `input_image` content
parts on a new user conversation item. Other attachment types are ignored and
the sender gets a brief "text and images only" reply back.
"""

import base64
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools._discord_common import DISCORD_API_BASE, bot_auth_header


if TYPE_CHECKING:
    from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler


logger = logging.getLogger(__name__)


_SUPPORTED_IMAGE_CTYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_UNSUPPORTED_REPLY = "I can only handle text and images for now."
_INJECT_TIMEOUT_S = 15.0


class DiscordInboundWorker:
    """Polls the configured Discord DM and injects new messages into the realtime session."""

    def __init__(self, realtime_handler: "OpenaiRealtimeHandler") -> None:
        """Initialize."""
        self._handler = realtime_handler
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._dm_channel_id: str | None = None
        self._last_seen_id: str | None = None

    def start(self) -> None:
        """Start the polling thread if bot credentials are configured."""
        bot_token = (config.DISCORD_BOT_TOKEN or "").strip()
        user_id = (config.DISCORD_USER_ID or "").strip()
        if not bot_token or not user_id:
            logger.info("discord inbound: bot token / user id missing, worker not started")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="discord-inbound")
        self._thread.start()
        logger.info(
            "discord inbound worker started (poll every %.1fs)",
            config.DISCORD_INBOUND_POLL_SECONDS,
        )

    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.debug("discord inbound worker stopped")

    def _run_loop(self) -> None:
        """Run the polling loop until stopped."""
        with httpx.Client(timeout=15.0) as client:
            while not self._stop_event.is_set():
                try:
                    self._tick(client)
                except Exception as e:
                    logger.warning("discord inbound tick failed: %s", e)
                poll_s = max(1.0, float(config.DISCORD_INBOUND_POLL_SECONDS))
                self._stop_event.wait(timeout=poll_s)

    def _tick(self, client: httpx.Client) -> None:
        """Poll for new DM messages and forward them."""
        bot_token = (config.DISCORD_BOT_TOKEN or "").strip()
        user_id = (config.DISCORD_USER_ID or "").strip()
        if not bot_token or not user_id:
            return
        headers = bot_auth_header(bot_token)

        if self._dm_channel_id is None and not self._open_dm_channel(client, headers, user_id):
            return

        params: dict[str, Any] = {"limit": 10}
        if self._last_seen_id:
            params["after"] = self._last_seen_id
        resp = client.get(
            f"{DISCORD_API_BASE}/channels/{self._dm_channel_id}/messages",
            params=params,
            headers=headers,
        )
        if resp.status_code == 401:
            logger.error("discord inbound: invalid bot token, stopping worker")
            self._stop_event.set()
            return
        if resp.status_code != 200:
            logger.warning("discord inbound: fetch failed HTTP %d: %s", resp.status_code, resp.text[:200])
            return

        try:
            messages = resp.json()
        except Exception:
            logger.warning("discord inbound: malformed messages response")
            return
        if not isinstance(messages, list):
            return

        # Discord returns newest-first; process oldest-first so last_seen_id advances monotonically.
        messages.sort(key=lambda m: int(m.get("id", "0")))

        # On the first successful poll (no baseline yet) skip history — only react to new messages.
        if self._last_seen_id is None and messages:
            self._last_seen_id = messages[-1]["id"]
            return

        for msg in messages:
            if (msg.get("author") or {}).get("bot"):
                self._last_seen_id = msg["id"]
                continue
            self._process_message(client, headers, msg)
            self._last_seen_id = msg["id"]

    def _open_dm_channel(self, client: httpx.Client, headers: dict[str, str], user_id: str) -> bool:
        """Open (idempotent) a DM channel with the configured user; cache its id."""
        resp = client.post(
            f"{DISCORD_API_BASE}/users/@me/channels",
            json={"recipient_id": user_id},
            headers=headers,
        )
        if resp.status_code != 200:
            logger.warning(
                "discord inbound: open DM failed HTTP %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            return False
        try:
            self._dm_channel_id = resp.json()["id"]
        except Exception:
            logger.warning("discord inbound: malformed DM channel response")
            return False
        logger.info("discord inbound: DM channel opened (id=%s)", self._dm_channel_id)
        return True

    def _process_message(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        msg: dict[str, Any],
    ) -> None:
        """Build realtime content parts from a Discord message and inject them."""
        parts: list[dict[str, Any]] = []
        unsupported = False

        text = (msg.get("content") or "").strip()
        if text:
            parts.append({"type": "input_text", "text": text})

        for attachment in msg.get("attachments") or []:
            content_type = (attachment.get("content_type") or "").lower()
            if content_type in _SUPPORTED_IMAGE_CTYPES:
                try:
                    data_url = self._download_as_data_url(client, attachment, content_type)
                except Exception as e:
                    logger.warning(
                        "discord inbound: failed to download %s: %s",
                        attachment.get("url"),
                        e,
                    )
                    continue
                parts.append({"type": "input_image", "image_url": data_url})
            else:
                unsupported = True

        if parts:
            logger.info(
                "discord inbound: forwarding message %s (%d parts)",
                msg.get("id"),
                len(parts),
            )
            if not self._inject_parts(parts):
                logger.warning("discord inbound: inject failed for message %s", msg.get("id"))

        if unsupported and self._dm_channel_id is not None:
            try:
                client.post(
                    f"{DISCORD_API_BASE}/channels/{self._dm_channel_id}/messages",
                    json={"content": _UNSUPPORTED_REPLY},
                    headers=headers,
                )
            except Exception as e:
                logger.warning("discord inbound: failed to post 'unsupported' reply: %s", e)

    def _download_as_data_url(
        self,
        client: httpx.Client,
        attachment: dict[str, Any],
        content_type: str,
    ) -> str:
        """Download an attachment and encode it as a data URL for realtime input_image."""
        url = attachment["url"]
        resp = client.get(url, timeout=30.0)
        resp.raise_for_status()
        b64 = base64.b64encode(resp.content).decode("ascii")
        return f"data:{content_type};base64,{b64}"

    def _inject_parts(self, parts: list[dict[str, Any]]) -> bool:
        """Schedule inject_user_content on the handler's event loop and wait for the result."""
        loop = self._handler._realtime_loop
        if loop is None:
            logger.warning("discord inbound: handler has no event loop yet, dropping")
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._handler.inject_user_content(parts),
            loop,
        )
        try:
            return bool(future.result(timeout=_INJECT_TIMEOUT_S))
        except Exception as e:
            logger.warning("discord inbound: inject coroutine raised: %s", e)
            return False
