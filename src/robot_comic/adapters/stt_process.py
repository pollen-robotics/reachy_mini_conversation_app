"""Run local Moonshine STT in a dedicated subprocess (#540).

Process-isolation boundary for the on-robot local STT path. The
Moonshine streaming recognizer historically ran in-process with the
app's asyncio audio loop, so a crash inside the ONNX runtime or a
silent ``state=idle`` stall (see #314) could wedge the whole
conversation. This module pushes Moonshine into a child process and
exposes an :class:`~robot_comic.backends.STTBackend`-shaped proxy in the
parent. A crash or stall in the child no longer touches the audio loop:
captured frames are dropped with a logged warning while a watchdog kills
and respawns the worker independently.

Design mirrors the YOLO head-tracker subprocess
(``robot_comic.vision.head_tracking.yolo_process``): a length-prefixed
pickle protocol over the child's stdin/stdout, a background reader
thread in the parent, ``atexit`` cleanup, and a PYTHONPATH bootstrap so
the editable ``src/`` layout is importable in the child.

The protocols differ in one respect. The head tracker is strictly
request/response (one frame in, one result out). STT is *streaming and
asynchronous*: many audio frames flow in, and transcript events
(``started`` / ``partial`` / ``completed`` / ``error``) flow back at the
recognizer's own cadence with no 1:1 correspondence to frames. So the
parent → child channel here is fire-and-forget ``feed`` messages, and
the child → parent channel carries unsolicited ``event`` messages plus a
periodic ``heartbeat`` the parent's watchdog uses to detect a stall.

## Wire protocol

Parent → child (length-prefixed pickle tuples):

* ``("feed", sample_rate: int, samples: np.ndarray[int16])`` — one
  captured audio frame.
* ``("close", None)`` — graceful shutdown request.

Child → parent:

* ``("ready", None)`` — model loaded, accepting audio.
* ``("event", kind: str, text: str)`` — a transcript event;
  ``kind`` is one of ``started`` / ``partial`` / ``completed`` /
  ``error``.
* ``("heartbeat", None)`` — periodic liveness ping for the watchdog.
* ``("error", repr: str)`` — fatal startup error (model load failed).

Heavy imports (numpy is unavoidable for the frame dtype, but
``moonshine_voice`` / scipy) stay deferred to the child process and to
inside functions, preserving the parent's import budget.
"""

from __future__ import annotations
import os
import sys
import time
import queue
import atexit
import pickle
import struct
import asyncio
import logging
import threading
import subprocess
import importlib.util
from typing import IO, Any, Callable, Awaitable
from pathlib import Path

import numpy as np

from robot_comic.backends import (
    AudioFrame,
    TranscriptCallback,
    SpeechStartedCallback,
)


logger = logging.getLogger(__name__)

# Worker-side event kinds (must match adapters.moonshine_listener constants;
# duplicated here so the parent doesn't import the listener module, which
# would pull scipy into the parent's import graph).
EVENT_STARTED = "started"
EVENT_PARTIAL = "partial"
EVENT_COMPLETED = "completed"
EVENT_ERROR = "error"

_HEADER_STRUCT = struct.Struct("!I")

# Defaults — overridable per-instance for tests and tuning.
_DEFAULT_START_TIMEOUT = 60.0  # Moonshine cold load is ~20s; allow generous headroom.
_DEFAULT_STALL_TIMEOUT = 30.0  # No heartbeat/event within this window → respawn.
_DEFAULT_HEARTBEAT_INTERVAL = 2.0
_DEFAULT_RESTART_BACKOFF = 1.0
_SHUTDOWN_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Framed pickle helpers (shared shape with yolo_process).
# ---------------------------------------------------------------------------


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    """Read exactly ``size`` bytes or raise ``EOFError``."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("Unexpected EOF while reading STT message")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_message(stream: IO[bytes], payload: object, lock: threading.Lock | None = None) -> None:
    """Serialize and write a single length-prefixed message."""
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def _write() -> None:
        stream.write(_HEADER_STRUCT.pack(len(data)))
        stream.write(data)
        stream.flush()

    if lock is not None:
        with lock:
            _write()
    else:
        _write()


def _receive_message(stream: IO[bytes]) -> object:
    """Read and deserialize a single length-prefixed message."""
    header = _read_exact(stream, _HEADER_STRUCT.size)
    (size,) = _HEADER_STRUCT.unpack(header)
    data = _read_exact(stream, size)
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# Worker (child process) entry point
# ---------------------------------------------------------------------------


def _worker_main() -> int:
    """Run the STT worker protocol in the child process.

    Loads the standalone :class:`MoonshineListener`, streams audio frames
    from stdin into it, and forwards transcript events + periodic
    heartbeats back over stdout. Any startup failure is reported as a
    single ``("error", repr)`` message so the parent can surface it.
    """
    # The recognizer must not write to the protocol stream; redirect any
    # stray prints to stderr (mirrors yolo_process).
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    send_lock = threading.Lock()

    # Deferred heavy imports — kept out of the parent process entirely.
    import asyncio as _asyncio

    from robot_comic.adapters.moonshine_listener import MoonshineListener

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    listener = MoonshineListener()

    async def _on_event(kind: str, text: str) -> None:
        try:
            _send_message(protocol_out, ("event", kind, text), send_lock)
        except Exception:  # pragma: no cover - pipe broke; parent is gone
            pass

    try:
        loop.run_until_complete(listener.start(_on_event))
        _send_message(protocol_out, ("ready", None), send_lock)
    except Exception as exc:
        _send_message(protocol_out, ("error", repr(exc)), send_lock)
        return 1

    # Periodic heartbeat thread so the parent watchdog can distinguish a
    # stalled-but-alive worker from a busy one.
    stop_heartbeat = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop_heartbeat.wait(_DEFAULT_HEARTBEAT_INTERVAL):
            try:
                _send_message(protocol_out, ("heartbeat", None), send_lock)
            except Exception:  # pragma: no cover - pipe broke
                return

    hb_thread = threading.Thread(target=_heartbeat_loop, name="stt-worker-heartbeat", daemon=True)
    hb_thread.start()

    try:
        while True:
            try:
                message = _receive_message(sys.stdin.buffer)
            except (EOFError, KeyboardInterrupt):
                return 0

            if not isinstance(message, tuple) or not message or not isinstance(message[0], str):
                continue
            command = message[0]
            if command == "close":
                return 0
            if command == "feed" and len(message) == 3:
                sample_rate = message[1]
                samples = message[2]
                if not isinstance(sample_rate, int) or not isinstance(samples, np.ndarray):
                    continue
                try:
                    loop.run_until_complete(listener.feed_audio(sample_rate, samples))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("STT worker feed_audio raised: %s", exc)
    finally:
        stop_heartbeat.set()
        try:
            loop.run_until_complete(listener.stop())
        except Exception:  # pragma: no cover - best-effort
            pass
        loop.close()


# ---------------------------------------------------------------------------
# Parent-side reader thread
# ---------------------------------------------------------------------------


def _reader_loop(stream: IO[bytes], messages: queue.Queue[tuple[str, object | None]]) -> None:
    """Read messages from the child and forward them onto a queue."""
    try:
        while True:
            payload = _receive_message(stream)
            messages.put(("message", payload))
    except EOFError:
        messages.put(("eof", None))
    except Exception as exc:  # pragma: no cover - defensive
        messages.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# Parent-side proxy adapter (STTBackend)
# ---------------------------------------------------------------------------

EventCallback = Callable[[str, str], Awaitable[None]]


class SubprocessSTTAdapter:
    """STTBackend proxy that runs Moonshine in an isolated child process.

    Satisfies the :class:`~robot_comic.backends.STTBackend` Protocol.
    Spawns a worker on :meth:`start`, forwards audio via :meth:`feed_audio`
    (fire-and-forget), routes transcript events to the registered
    callbacks from a background reader thread, and runs a stall watchdog
    that kills + respawns the worker if it stops sending events /
    heartbeats. All failure modes degrade to dropped frames + a logged
    warning rather than propagating into the caller's audio loop.
    """

    def __init__(
        self,
        *,
        start_timeout: float = _DEFAULT_START_TIMEOUT,
        stall_timeout: float = _DEFAULT_STALL_TIMEOUT,
        heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
        restart_backoff: float = _DEFAULT_RESTART_BACKOFF,
        should_drop_frame: Callable[[], bool] | None = None,
    ) -> None:
        """Configure timeouts and the optional echo-guard drop predicate.

        ``should_drop_frame`` mirrors :class:`MoonshineSTTAdapter` standalone
        mode: when it returns truthy the parent drops the frame before it
        crosses the process boundary (saves a pickle + pipe write while the
        robot is speaking).
        """
        self._start_timeout = start_timeout
        self._stall_timeout = stall_timeout
        self._heartbeat_interval = heartbeat_interval
        self._restart_backoff = restart_backoff
        self._should_drop_frame = should_drop_frame

        self._on_completed: TranscriptCallback | None = None
        self._on_partial: TranscriptCallback | None = None
        self._on_speech_started: SpeechStartedCallback | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None
        self._stdout: IO[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._messages: queue.Queue[tuple[str, object | None]] = queue.Queue()
        self._send_lock = threading.Lock()

        self._watchdog: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._closed = False
        self._last_alive_at = 0.0
        self.restart_count = 0

    # -- STTBackend interface ------------------------------------------------

    async def start(
        self,
        on_completed: TranscriptCallback,
        on_partial: TranscriptCallback | None = None,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> None:
        """Bind callbacks, spawn the worker, and launch reader + watchdog."""
        self._on_completed = on_completed
        self._on_partial = on_partial
        self._on_speech_started = on_speech_started
        self._loop = asyncio.get_running_loop()

        await asyncio.to_thread(self._spawn_and_wait_ready)

        atexit.register(self._close_sync)
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="stt-watchdog", daemon=True)
        self._watchdog.start()

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Forward one captured frame to the worker (fire-and-forget).

        Never raises: a dead/respawning worker or a broken pipe is logged
        at debug and the frame is dropped, exactly as a real-world
        utterance during a restart window would be lost.
        """
        if self._should_drop_frame is not None and self._should_drop_frame():
            return

        with self._state_lock:
            process = self._process
            stdin = self._stdin
        if self._closed or process is None or stdin is None or process.poll() is not None:
            return

        samples = frame.samples
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.int16)
        try:
            _send_message(stdin, ("feed", int(frame.sample_rate), samples), self._send_lock)
        except Exception as exc:
            logger.debug("Dropping STT frame; worker pipe unavailable: %s", exc)

    async def stop(self) -> None:
        """Stop the watchdog and tear down the worker."""
        await asyncio.to_thread(self._close_sync)
        self._on_completed = None
        self._on_partial = None
        self._on_speech_started = None

    # -- worker lifecycle ----------------------------------------------------

    def _spawn_and_wait_ready(self) -> None:
        """Spawn the child and block until it reports ``ready`` (or fails)."""
        module_path = "robot_comic.adapters.stt_process"
        env = os.environ.copy()
        package_spec = importlib.util.find_spec("robot_comic")
        package_locations = None if package_spec is None else package_spec.submodule_search_locations
        if not package_locations:
            raise RuntimeError("Unable to determine the robot_comic package path")
        project_src = Path(next(iter(package_locations))).resolve().parent
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(project_src) if not existing_pythonpath else f"{project_src}{os.pathsep}{existing_pythonpath}"
        )
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            [sys.executable, "-m", module_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            env=env,
            start_new_session=True,
        )
        if process.stdin is None or process.stdout is None:
            try:
                process.kill()
            except Exception:
                pass
            raise RuntimeError("Failed to create pipes for STT worker process")

        # Drain any leftover messages from a previous worker so the new
        # reader's queue starts clean.
        self._messages = queue.Queue()
        reader = threading.Thread(
            target=_reader_loop,
            args=(process.stdout, self._messages),
            daemon=True,
            name="stt-reader",
        )
        reader.start()

        # Wait for ready / error.
        deadline = time.monotonic() + self._start_timeout
        ready = False
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                event, payload = self._messages.get(timeout=max(0.01, min(remaining, 0.5)))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if event == "eof":
                break
            if event == "error":
                break
            if event != "message":
                continue
            if isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "ready":
                ready = True
                break
            if isinstance(payload, tuple) and payload and payload[0] == "error":
                logger.error("STT worker reported a startup error: %s", payload[-1])
                break
            # Any other message before ready: ignore, keep waiting.

        if not ready:
            try:
                process.kill()
            except Exception:
                pass
            raise RuntimeError("STT worker failed to become ready")

        with self._state_lock:
            # stop()/_close_sync may have run while we waited out the worker's
            # cold load (~20s for Moonshine). Storing the fresh process after
            # shutdown would orphan it — nothing would ever reap it.
            shutting_down = self._stop.is_set() or self._closed
            if not shutting_down:
                self._process = process
                self._stdin = process.stdin
                self._stdout = process.stdout
                self._reader = reader
                self._last_alive_at = time.monotonic()
        if shutting_down:
            try:
                process.kill()
                process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except Exception:
                pass
            raise RuntimeError("STT adapter is shutting down; discarded fresh worker")

        # Spawn the event-dispatch thread that drains the message queue and
        # schedules callbacks onto the asyncio loop. One per worker; it
        # exits when it sees eof/error for the worker it was started with.
        dispatcher = threading.Thread(
            target=self._dispatch_loop,
            args=(process,),
            name="stt-dispatch",
            daemon=True,
        )
        dispatcher.start()

    def _dispatch_loop(self, process: subprocess.Popen[bytes]) -> None:
        """Drain the message queue for ``process`` and route events.

        Runs until the worker emits eof/error (or shutdown is requested).
        Updates ``_last_alive_at`` on every message so the watchdog can
        detect a stall.
        """
        while not self._stop.is_set():
            try:
                event, payload = self._messages.get(timeout=0.5)
            except queue.Empty:
                # Only the current worker's dispatcher should keep spinning.
                with self._state_lock:
                    if self._process is not process:
                        return
                continue

            self._last_alive_at = time.monotonic()

            if event in ("eof", "error"):
                if event == "error":
                    logger.warning("STT worker reader failed: %s", payload)
                with self._state_lock:
                    current = self._process is process
                if current and not self._stop.is_set() and not self._closed:
                    logger.warning("STT worker exited unexpectedly; watchdog will respawn")
                return
            if event != "message":
                continue
            if not (isinstance(payload, tuple) and payload and isinstance(payload[0], str)):
                continue
            kind = payload[0]
            if kind == "heartbeat":
                continue
            if kind == "event" and len(payload) == 3:
                self._route_event(str(payload[1]), str(payload[2]))
            elif kind == "error":
                logger.warning("STT worker error: %s", payload[-1])

    def _route_event(self, kind: str, text: str) -> None:
        """Schedule the matching transcript callback on the asyncio loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        if kind == EVENT_COMPLETED:
            cb: Any = self._on_completed
            args: tuple[Any, ...] = (text,)
        elif kind == EVENT_PARTIAL:
            if self._on_partial is None or not text:
                return
            cb = self._on_partial
            args = (text,)
        elif kind == EVENT_STARTED:
            if self._on_speech_started is None:
                return
            cb = self._on_speech_started
            args = ()
        else:
            # EVENT_ERROR and unknowns are dropped (telemetry counts them
            # worker-side, mirroring the standalone listener).
            return

        if cb is None:
            return

        def _fire() -> None:
            try:
                asyncio.create_task(cb(*args))
            except Exception:  # pragma: no cover - defensive
                logger.exception("STT %s callback scheduling failed", kind)

        loop.call_soon_threadsafe(_fire)

    def _watchdog_loop(self) -> None:
        """Kill + respawn the worker if it stalls or dies.

        A stall is "no message (event or heartbeat) for longer than
        ``stall_timeout``". A death is the process exiting. Either way the
        watchdog tears the worker down and spawns a fresh one, after a
        short backoff, without disturbing the parent's audio loop.
        """
        while not self._stop.wait(min(self._heartbeat_interval, self._stall_timeout) / 2):
            if self._closed:
                return
            with self._state_lock:
                process = self._process
            if process is None:
                continue

            stalled = (time.monotonic() - self._last_alive_at) > self._stall_timeout
            dead = process.poll() is not None
            if not (stalled or dead):
                continue

            reason = "stalled" if stalled and not dead else "died"
            logger.warning("STT worker %s; respawning (restart #%d)", reason, self.restart_count + 1)
            self._kill_process(process)
            if self._stop.wait(self._restart_backoff):
                return
            try:
                self._spawn_and_wait_ready()
                self.restart_count += 1
            except Exception as exc:
                logger.error("STT worker respawn failed: %s", exc)
                # Leave _process None; next watchdog tick retries after backoff.
                with self._state_lock:
                    self._process = None
                if self._stop.wait(self._restart_backoff):
                    return

    def _kill_process(self, process: subprocess.Popen[bytes]) -> None:
        """Forcibly terminate a worker process."""
        with self._state_lock:
            if self._process is process:
                self._process = None
                self._stdin = None
                self._stdout = None
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception:  # pragma: no cover - best-effort
            pass

    def _close_sync(self) -> None:
        """Stop the watchdog and tear down the current worker (idempotent)."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            stdin = self._stdin
            stdout = self._stdout
            self._process = None
            self._stdin = None
            self._stdout = None
        self._stop.set()

        if process is not None:
            try:
                if process.poll() is None and stdin is not None:
                    _send_message(stdin, ("close", None), self._send_lock)
            except Exception:
                pass
            try:
                if stdin is not None:
                    stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Force-terminating STT worker process")
                try:
                    process.terminate()
                    process.wait(timeout=_SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except Exception:  # pragma: no cover
                    pass

        # Close the parent-side read pipe so the FD doesn't dangle across
        # force-kill paths (mirrors the yolo subprocess teardown).
        try:
            if stdout is not None:
                stdout.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self._close_sync()
        except Exception:  # pragma: no cover
            pass


def main() -> int:
    """CLI entry point for the STT worker subprocess."""
    if len(sys.argv) != 1:
        print("usage: python -m robot_comic.adapters.stt_process", file=sys.stderr)
        return 2
    return _worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
