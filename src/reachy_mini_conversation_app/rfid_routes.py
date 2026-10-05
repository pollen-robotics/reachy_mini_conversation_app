"""NFC accessory reader: background polling and its JSON-RPC control surface.

Each accessory carries a personality token (see :mod:`personality_tag`); placing
one on the reader applies that personality, removing it reverts to the one this
instance starts as — the personality set as default, or the built-in one.
A blank accessory starts the "give it a personality" conversation, at the end of
which :mod:`tools.create_accessory_personality` writes the new token onto it.

The serial link is owned by the Reachy Mini daemon; this app is an HTTP client
of the daemon's ``/api/nfc`` endpoints.

The reader is polled by a background thread owned by this module, not by the web
UI: swapping personalities is robot behaviour and has to keep working with no
browser attached. Clients learn about tag changes from the ``rfid.tag``
notification and drive writes through the ``rfid.*`` methods.
"""

import time
import asyncio
import logging
import threading
import webbrowser
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_conversation_app.config import LOCKED_PROFILE
from reachy_mini_conversation_app.personality import list_personalities
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME
from reachy_mini_conversation_app.personality_tag import to_tag_token, from_tag_token
from reachy_mini_conversation_app.tools.core_tools import set_accessory_personality_tool_available
from reachy_mini_conversation_app.nfc_daemon_client import (
    NfcTagSnapshot,
    NfcDaemonClient,
    describe_write_error,
)
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.4
# A board that is not plugged in will not answer any sooner for being asked
# 2.5 times a second: while the reader reports itself disconnected, polling it
# at the tag-detection rate only fills the daemon log. Hot-plug is still
# picked up at this slower rate, just up to two seconds later.
DISCONNECTED_POLL_INTERVAL_S = 2.0
# A tag briefly reading NO_TAG (a hand moving it, a marginal antenna position)
# must not re-trigger the blank-tag conversation as soon as it reads again.
BLANK_TAG_COOLDOWN_S = 3.0
TRANSITION_MOVE_DATASET = "cdeplanne/local-dataset"
TRANSITION_MOVE_NAME = "switch-personnality-5"
WRITE_MOVE_DATASET = "glannuzel/local-dataset"
WRITE_MOVE_NAME = "write-tag-6"
# The sound is started slightly before the movement is queued so the two line up.
WRITE_SOUND_LEAD_S = 0.15
ADD_ON_STORE_URL = "https://store.pollen-robotics.com/collections/reachy-mini"

HandlerGetter = Callable[[], "HuggingFaceRealtimeHandler"]
LoopGetter = Callable[[], asyncio.AbstractEventLoop | None]
PersonalityObserver = Callable[[str | None], None]
DefaultPersonalityGetter = Callable[[], str | None]


def accessory_personality_on_reader(timeout: float = 2.0) -> str | None:
    """Return the personality named by an accessory already on the reader, or None.

    Read once before the first handler is built, so the app starts as that
    personality instead of applying it as a change right after launch. Daemon
    failures read as an absent tag, so a reader that is down cannot hold up the launch.
    """
    client = NfcDaemonClient(timeout=timeout)
    try:
        if not client.get_status().get("driver_available"):
            return None
        return _tag_personality(client.get_tag())
    finally:
        client.close()


def _tag_personality(tag: NfcTagSnapshot) -> str | None:
    """Return the personality a tag names, if this robot has it."""
    if not tag.present or tag.blank or not tag.content:
        return None
    personality = from_tag_token(tag.content)
    return personality if personality in list_personalities() else None


def _load_move_dataset(repo_id: str) -> RecordedMoves | None:
    """Load a recorded-move dataset, or None when it cannot be fetched."""
    try:
        return RecordedMoves(repo_id)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("[RFID] move dataset %s unavailable: %s", repo_id, exc)
        return None


class RfidController:
    """Owns the NFC daemon client, the polling loop and the tag state machine."""

    def __init__(
        self,
        get_handler: HandlerGetter,
        get_loop: LoopGetter,
        robot: "ReachyMini",
        rpc: JsonRpcServer | None = None,
        on_personality_applied: PersonalityObserver | None = None,
        get_default_personality: DefaultPersonalityGetter | None = None,
        initial_personality: str | None = None,
    ) -> None:
        """Build a controller; call :meth:`start` to begin polling.

        ``get_default_personality`` returns the personality to fall back to when
        the accessory is removed (the one set as default). ``initial_personality``
        is the one the app started as from an accessory already on the reader, so
        that removing it is seen as a change.
        """
        self._client = NfcDaemonClient()
        self._get_handler = get_handler
        self._get_loop = get_loop
        self._robot = robot
        self._rpc = rpc
        # An accessory swaps the personality without any client asking, so the
        # badges would otherwise keep showing the one it replaced.
        self._on_personality_applied = on_personality_applied
        self._get_default_personality = get_default_personality

        self._transition_moves = _load_move_dataset(TRANSITION_MOVE_DATASET)
        self._write_moves = _load_move_dataset(WRITE_MOVE_DATASET)

        self._apply_lock = threading.Lock()

        self._current_personality: str | None = initial_personality
        self._blank_tag_active = False
        self._blank_tag_cooldown_until = 0.0
        self._delayed_switch_future: Any = None
        self._inject_move_future: Any = None
        self._previous_tag: NfcTagSnapshot | None = None
        self._last_broadcast: dict[str, Any] | None = None
        self._last_status: dict[str, Any] | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def client(self) -> NfcDaemonClient:
        """The underlying NFC daemon client."""
        return self._client

    def start(self) -> None:
        """Start the background polling thread, which lives as long as the app."""
        threading.Thread(target=self._poll_loop, name="rfid-poll", daemon=True).start()
        logger.info("[RFID] polling thread started")

    def _poll_loop(self) -> None:
        while True:
            connected = False
            try:
                connected = bool(self.poll_once().get("connected"))
            except Exception:
                logger.exception("[RFID] polling iteration failed")
            # A failed iteration backs off too: it means the daemon is not
            # answering, which is no more urgent than a missing board.
            time.sleep(POLL_INTERVAL_S if connected else DISCONNECTED_POLL_INTERVAL_S)

    # ── snapshot helpers ─────────────────────────────────────────────────────

    def connection_status(self) -> dict[str, Any]:
        """Return the reader's connection state as the UI shows it.

        Also offers or withdraws the accessory-personality tool as the reader comes and goes.
        """
        status = self._client.get_status()
        set_accessory_personality_tool_available(bool(status.get("driver_available", False)))
        summary = {
            "connected": status.get("connected", False),
            # False when switched off in the control app: "not connected" then says
            # nothing about whether the add-on is fitted. Older daemons omit it.
            "enabled": status.get("enabled") is not False,
            "port": status.get("port"),
            "driver_available": status.get("driver_available", False),
            "error": status.get("error"),
        }
        self._last_status = summary
        return summary

    def last_status(self) -> dict[str, Any]:
        """Return the most recent reader status without contacting the daemon.

        The polling thread refreshes it continuously; callers on the request
        path (conversation.status) must not block on an HTTP round trip.
        """
        if self._last_status is not None:
            return dict(self._last_status)
        return self.connection_status()

    def accessory_view(self, tag: NfcTagSnapshot | None) -> dict[str, Any]:
        """Describe, for the panel, the accessory on the reader (``tag`` is None without a reader)."""
        if tag is None:
            return {"state": "unavailable", "personality": None, "content": None}
        if not tag.present:
            return {"state": "none", "personality": None, "content": None}
        if tag.blank or not tag.content:
            return {"state": "blank", "personality": None, "content": None}
        personality = _tag_personality(tag)
        if personality is not None:
            return {"state": "known", "personality": personality, "content": tag.content}
        # Carries something, but nothing this robot can act on: an old opaque
        # code, a deleted profile, or a tag written elsewhere.
        return {"state": "unknown", "personality": None, "content": tag.content}

    def snapshot(self) -> dict[str, Any]:
        """Return the full reader + accessory state, without advancing the machine."""
        status = self.connection_status()
        tag = self._client.get_tag() if status["connected"] else None
        return {**status, "accessory": self.accessory_view(tag)}

    def _announce_personality(self, profile: str | None) -> None:
        """Tell the app a personality was applied by the reader, not by a client."""
        if self._on_personality_applied is None:
            return
        try:
            self._on_personality_applied(profile)
        except Exception as exc:
            logger.warning("[RFID] failed to announce the applied personality: %s", exc)

    def _broadcast(self, payload: dict[str, Any]) -> None:
        """Push an rfid.tag notification when the visible state changed.

        Skipped until the conversation loop is up: broadcast_threadsafe has no
        loop to schedule on before then, and would strand a coroutine.
        """
        if self._rpc is None or payload == self._last_broadcast:
            return
        loop = self._get_loop()
        if loop is None or not loop.is_running():
            return
        self._last_broadcast = payload
        try:
            self._rpc.broadcast_threadsafe("rfid.tag", payload)
        except Exception as exc:
            logger.debug("[RFID] broadcast failed: %s", exc)

    # ── actions ──────────────────────────────────────────────────────────────

    def link_tag(self, personality: str) -> dict[str, Any]:
        """Link the accessory currently on the reader to ``personality``.

        The token is derived from the personality, so a tag already carrying it
        needs no write.
        """
        personality = (personality or "").strip()
        if not personality:
            raise JsonRpcError("Choose a personality to link.", reason="no_personality")
        tag = self._client.get_tag()
        if not tag.present:
            raise JsonRpcError("Place an accessory on the reader first.", reason="no_tag")
        code = to_tag_token(personality)
        if code is None:
            raise JsonRpcError(
                "This personality cannot be written to an accessory.",
                reason="not_writable",
            )
        if tag.content == code:
            logger.info("[RFID] Tag already carries %r", personality)
            return {"ok": True, "code": code, "personality": personality, "written": False}
        logger.info("[RFID] Writing %r to tag for personality %r", code, personality)
        success, detail = self._client.write_tag(code)
        if not success:
            logger.warning("[RFID] Write failed for %r: %s", code, detail)
            raise JsonRpcError(
                describe_write_error(detail),
                reason="write_failed",
                data={"detail": describe_write_error(detail), "code": detail},
            )
        logger.info("[RFID] Tag written with %r", code)
        return {"ok": True, "code": code, "personality": personality, "written": True}

    def erase_tag(self, full: bool = False) -> dict[str, Any]:
        """Make the accessory on the reader blank again.

        ``full`` also zeroes the whole user memory — the only way to remove a
        payload that is not NDEF. It writes one page at a time, so it takes a
        few seconds on an NTAG215.
        """
        success, result = self._client.erase_tag(full=full)
        if not success:
            logger.warning("[RFID] Erase failed: %s", result)
            raise JsonRpcError(
                describe_write_error(result),
                reason="erase_failed",
                data={"detail": describe_write_error(result), "code": result},
            )
        return {"ok": True, "message": "Tag erased"}

    # ── state machine ────────────────────────────────────────────────────────

    def poll_once(self) -> dict[str, Any]:
        """Read the reader once, act on every transition since the last read, and return the snapshot."""
        status = self.connection_status()
        if not status["connected"]:
            self._previous_tag = None
            payload = {**status, "accessory": self.accessory_view(None)}
            self._broadcast(payload)
            return payload

        tag = self._client.get_tag()
        previous = self._previous_tag
        self._previous_tag = tag

        write_results = self._client.drain_write_results()
        if self._apply_lock.acquire(blocking=False):
            try:
                self._process(write_results, previous, tag)
            finally:
                self._apply_lock.release()

        payload = {**status, "accessory": self.accessory_view(tag)}
        self._broadcast(payload)
        return payload

    def _process(
        self,
        write_results: list[tuple[bool, str]],
        previous: NfcTagSnapshot | None,
        tag: NfcTagSnapshot,
    ) -> None:
        """Act on finished writes first, then on what changed on the reader since the last read."""
        handler = self._get_handler()
        for success, code in write_results:
            self._on_write_result(handler, success, code)
        # No previous read (first poll, or the reader just came back): nothing to compare.
        if previous is None:
            return
        if previous.present and not tag.present:
            self._on_tag_removed(handler)
        elif tag.present and not previous.present:
            self._on_tag_read(handler, "" if tag.blank else tag.content or "")
        elif tag.present:
            if tag.blank and not previous.blank:
                self._on_tag_read(handler, "")
            elif not tag.blank and tag.content and tag.content != previous.content:
                self._on_tag_read(handler, tag.content)

    def _queue_move(self, handler: "HuggingFaceRealtimeHandler", moves: RecordedMoves | None, name: str) -> None:
        """Queue a recorded emotion move, doing nothing when its dataset is missing."""
        if moves is None:
            return
        try:
            handler.deps.movement_manager.queue_move(EmotionQueueMove(name, moves))
        except (KeyError, ValueError, RuntimeError) as exc:
            logger.warning("[RFID] >>> move %r failed: %s", name, exc)

    def _run_on_loop(self, coroutine: Any, description: str, timeout: float = 10.0) -> bool:
        """Run a handler coroutine on the conversation loop and wait for it."""
        loop = self._get_loop()
        if loop is None:
            coroutine.close()
            return False
        try:
            asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout=timeout)
            return True
        except Exception as exc:
            logger.warning("[RFID] >>> %s FAILED: %s", description, exc)
            return False

    def _on_tag_removed(self, handler: "HuggingFaceRealtimeHandler") -> None:
        was_blank = self._blank_tag_active
        self._blank_tag_active = False
        self._blank_tag_cooldown_until = time.monotonic() + BLANK_TAG_COOLDOWN_S
        handler.deps.blank_tag_present = False

        for future_name in ("_inject_move_future", "_delayed_switch_future"):
            future = getattr(self, future_name)
            if future is not None:
                future.cancel()
                setattr(self, future_name, None)

        self._run_on_loop(handler.abort_nfc_collection(), "abort_nfc_collection", timeout=5.0)

        if was_blank:
            logger.info("[RFID] >>> blank tag removed (blank_tag_present cleared)")
            handler.deps.pending_nfc_write = None

        if self._current_personality is None:
            return

        profile = self._default_personality()
        if profile == self._current_personality:
            # The accessory carried the default personality: no restart needed.
            logger.info("[RFID] >>> NO_TAG received — already on the default personality")
            self._current_personality = None
            return

        logger.info("[RFID] >>> NO_TAG received — reverting to %s", profile or DEFAULT_PROFILE_NAME)
        self._queue_move(handler, self._transition_moves, TRANSITION_MOVE_NAME)
        if not self._run_on_loop(handler.apply_personality(profile), "default revert"):
            return
        self._current_personality = None
        self._announce_personality(profile)
        logger.info("[RFID] >>> default personality applied OK")

    def _default_personality(self) -> str | None:
        """Return the personality to fall back to with no accessory on the reader.

        None means the built-in default, also used when the saved one has since been deleted.
        """
        if self._get_default_personality is None:
            return None
        try:
            profile = self._get_default_personality()
        except (OSError, ValueError) as exc:
            logger.warning("[RFID] could not read the default personality: %s", exc)
            return None
        if profile is None or profile not in list_personalities():
            return None
        return profile

    def _on_tag_read(self, handler: "HuggingFaceRealtimeHandler", code: str) -> None:
        """Handle a tag arriving or changing; an empty ``code`` means a blank tag."""
        if not code:
            self._on_blank_tag(handler)
            return

        handler.deps.blank_tag_present = False
        personality = from_tag_token(code)
        if personality is None or personality not in list_personalities():
            logger.info("[RFID] >>> tag %r names no personality on this robot, keeping current one", code)
            return
        if personality == self._current_personality:
            logger.debug("[RFID] >>> same personality %r, skipping", personality)
            return

        if code in handler.deps.recently_written_codes:
            handler.deps.recently_written_codes.discard(code)
            self._current_personality = personality
            logger.info("[RFID] >>> newly written tag %r — delaying personality switch", code)
            self._schedule_delayed_switch(handler, personality)
            return

        logger.info("[RFID] >>> applying personality %r for code %r", personality, code)
        self._queue_move(handler, self._transition_moves, TRANSITION_MOVE_NAME)
        if not self._run_on_loop(handler.apply_personality(personality), "apply"):
            return
        self._current_personality = personality
        self._announce_personality(personality)
        logger.info("[RFID] >>> personality applied OK")

    def _on_blank_tag(self, handler: "HuggingFaceRealtimeHandler") -> None:
        handler.deps.blank_tag_present = True
        pending = handler.deps.pending_nfc_write
        if pending is not None:
            self._blank_tag_active = True
            handler.deps.pending_nfc_write = None
            logger.info("[RFID] >>> blank tag with pending write — writing code %r", pending["code"])
            handler.deps.recently_written_codes.add(pending["code"])
            self._client.write_tag_in_background(pending["code"])
            self._run_on_loop(
                handler.cancel_nfc_retry(),
                "cancel_nfc_retry",
            )
            return
        if self._blank_tag_active or time.monotonic() < self._blank_tag_cooldown_until:
            return
        self._blank_tag_active = True
        logger.info("[RFID] >>> blank tag detected — injecting event to LLM")
        self._run_on_loop(handler.inject_blank_nfc_tag(), "blank tag inject")

    def _schedule_delayed_switch(self, handler: "HuggingFaceRealtimeHandler", personality: str) -> None:
        """Switch personality only once the welcome speech has finished playing."""
        loop = self._get_loop()
        if loop is None:
            return
        if self._delayed_switch_future is not None:
            self._delayed_switch_future.cancel()

        async def _delayed_switch() -> None:
            try:
                try:
                    await asyncio.wait_for(handler._nfc_speech_done_event.wait(), timeout=12.0)
                except asyncio.TimeoutError:
                    logger.warning("[RFID] >>> NFC speech done event timed out, switching anyway")
                await self._wait_for_welcome_speech(handler)
                self._queue_move(handler, self._transition_moves, TRANSITION_MOVE_NAME)
                await handler.apply_personality(personality)
                self._announce_personality(personality)
                logger.info("[RFID] >>> delayed personality switch to %r done", personality)
            except asyncio.CancelledError:
                logger.info("[RFID] >>> delayed personality switch cancelled (tag removed)")
                self._current_personality = None
            except Exception as exc:
                logger.warning("[RFID] >>> delayed personality switch FAILED: %s", exc)
            finally:
                handler._nfc_transition = False
                handler._nfc_speech_done_event.set()
                self._delayed_switch_future = None

        self._delayed_switch_future = asyncio.run_coroutine_threadsafe(_delayed_switch(), loop)

    @staticmethod
    async def _wait_for_welcome_speech(handler: "HuggingFaceRealtimeHandler") -> None:
        """Sleep until the tracked welcome-speech audio has had time to play out."""
        start = handler._nfc_speech_start_time
        samples = handler._nfc_speech_samples
        sample_rate = handler.SAMPLE_RATE
        if start is not None and samples > 0 and sample_rate > 0:
            speech_duration = samples / sample_rate
            remaining = start + speech_duration + 0.8 - asyncio.get_running_loop().time()
            logger.info(
                "[RFID] >>> welcome speech: %.2fs, waiting %.2fs more before switch",
                speech_duration,
                max(0.0, remaining),
            )
            if remaining > 0:
                await asyncio.sleep(remaining)
            return

        logger.warning("[RFID] >>> no speech audio tracked, falling back to drain+sleep")

        async def _drain() -> None:
            while not handler.output_queue.empty():
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(_drain(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("[RFID] >>> audio queue drain timed out")
        await asyncio.sleep(1.0)

    def _on_write_result(self, handler: "HuggingFaceRealtimeHandler", success: bool, code: str) -> None:
        logger.info("[RFID] >>> write result: %s", code)
        loop = self._get_loop()
        if loop is None:
            return
        if success:
            self._run_on_loop(handler.stop_current_speech(), "stop_current_speech", timeout=6.0)

        move_duration = 0.0
        if success:
            move_duration = self._play_write_move(handler)

        if move_duration <= 0.0:
            detail = "" if success else describe_write_error(code)
            self._run_on_loop(handler.inject_nfc_write_result(success, detail), "write result inject")
            return

        # Arm the speech gate before the move so the welcome speech that follows
        # is measured from its own first audio sample.
        def _arm_gate() -> None:
            handler._nfc_speech_done_event.clear()
            handler._nfc_speech_start_time = None
            handler._nfc_speech_samples = 0

        loop.call_soon_threadsafe(_arm_gate)

        async def _inject_after_move() -> None:
            try:
                await asyncio.sleep(move_duration)
                await handler.inject_nfc_write_result(True)
            finally:
                self._inject_move_future = None

        self._inject_move_future = asyncio.run_coroutine_threadsafe(_inject_after_move(), loop)

    def _play_write_move(self, handler: "HuggingFaceRealtimeHandler") -> float:
        """Start the write-tag sound and movement; return the movement duration."""
        if self._write_moves is None or handler.deps.movement_manager is None:
            return 0.0
        try:
            move = EmotionQueueMove(WRITE_MOVE_NAME, self._write_moves)
            sound_path = getattr(getattr(move, "emotion_move", None), "sound_path", None)
            if sound_path is not None:
                self._robot.media.play_sound(str(sound_path))
                logger.info(
                    "[RFID] >>> %s sound started (%.0fms lead): %s",
                    WRITE_MOVE_NAME,
                    WRITE_SOUND_LEAD_S * 1000,
                    sound_path,
                )
                time.sleep(WRITE_SOUND_LEAD_S)
            else:
                logger.warning("[RFID] >>> %s: no sound_path found on emotion_move", WRITE_MOVE_NAME)
            handler.deps.movement_manager.queue_move(move)
            duration = float(move.duration)
            logger.info("[RFID] >>> %s queued (%.2fs), speech delayed", WRITE_MOVE_NAME, duration)
            return duration
        except (KeyError, ValueError, RuntimeError, OSError) as exc:
            logger.warning("[RFID] >>> %s move failed: %s", WRITE_MOVE_NAME, exc)
            return 0.0


def register_rfid_methods(rpc: JsonRpcServer, controller: RfidController) -> None:
    """Register the rfid.* JSON-RPC methods against ``controller``."""

    async def _status(_params: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(controller.snapshot)

    async def _link_tag(params: dict[str, Any]) -> dict[str, Any]:
        if LOCKED_PROFILE is not None:
            raise JsonRpcError("Personality editing is locked.", reason="profile_locked")
        personality = params.get("personality")
        if not isinstance(personality, str):
            raise JsonRpcError("Choose a personality to link.", reason="no_personality")
        return await asyncio.to_thread(controller.link_tag, personality)

    async def _erase(params: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(controller.erase_tag, bool(params.get("full", False)))

    async def _open_add_on_store(_params: dict[str, Any]) -> dict[str, Any]:
        # The panel runs in the control app's webview, which drops target="_blank"
        # and window.open, so the browser has to be raised by a process with a
        # screen. This app's own is the only one it can ask; on a robot running
        # headless there is none, and the caller falls back to showing the URL.
        try:
            opened = await asyncio.to_thread(webbrowser.open, ADD_ON_STORE_URL)
        except webbrowser.Error as exc:
            logger.info("No browser on this host to open the add-on store: %s", exc)
            opened = False
        return {"opened": opened, "url": ADD_ON_STORE_URL}

    rpc.register("rfid.status", _status)
    rpc.register("rfid.link_tag", _link_tag)
    rpc.register("rfid.erase", _erase)
    rpc.register("rfid.open_add_on_store", _open_add_on_store)
