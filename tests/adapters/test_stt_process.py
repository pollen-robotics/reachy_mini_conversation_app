"""Tests for the out-of-process Moonshine STT worker + proxy adapter (#540).

The proxy isolates the local STT path in a child process so a crash or
stall in Moonshine can no longer take down the app's audio loop. These
tests patch the child process with a tiny fake worker that speaks the
same length-prefixed pickle protocol, so they run anywhere (Windows
included) without the optional ``moonshine_voice`` dependency.
"""

from __future__ import annotations
import os
import sys
import asyncio
import subprocess
import importlib.util
from typing import Any
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from robot_comic.backends import AudioFrame, STTBackend
from robot_comic.adapters.stt_process import SubprocessSTTAdapter


@pytest.fixture(autouse=True)
def _restore_event_loop_for_worker() -> Any:
    """Leave a fresh current event loop set after each ``asyncio.run`` test.

    These tests drive the proxy via ``asyncio.run()``, which on Python 3.12
    sets the thread's current loop to ``None`` on close. Legacy handler tests
    sharing the same xdist worker construct handlers outside a running loop
    and rely on the deprecated ``asyncio.get_event_loop()`` auto-create — a
    closed/absent loop makes them raise ``RuntimeError``. Restoring a usable
    loop keeps this module from surfacing that pre-existing latent flake in
    unrelated tests scheduled afterward on the same worker.
    """
    yield
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


# Shared protocol preamble injected into every fake worker so the test
# scripts only have to express their behaviour loop.
_WORKER_PREAMBLE = """
import pickle
import struct
import sys
import time
import threading

HEADER = struct.Struct("!I")
_send_lock = threading.Lock()


def _read_exact(size):
    data = bytearray()
    while len(data) < size:
        chunk = sys.stdin.buffer.read(size - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


def _receive_message():
    (size,) = HEADER.unpack(_read_exact(HEADER.size))
    return pickle.loads(_read_exact(size))


def _send_message(payload):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    with _send_lock:
        sys.stdout.buffer.write(HEADER.pack(len(data)))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
"""


def _patch_fake_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker_body: str,
    popen_kwargs: dict[str, Any] | None = None,
) -> None:
    """Patch the STT subprocess with a test worker script."""
    worker_script = tmp_path / "fake_stt_worker.py"
    worker_script.write_text(
        dedent(_WORKER_PREAMBLE) + "\n" + dedent(worker_body),
        encoding="utf-8",
    )

    real_popen: Any = subprocess.Popen

    def _spawn_fake_worker(*args: object, **kwargs: Any) -> Any:
        if popen_kwargs is not None:
            popen_kwargs.update(kwargs)
        return real_popen([sys.executable, str(worker_script)], **kwargs)

    monkeypatch.setattr(
        "robot_comic.adapters.stt_process.subprocess.Popen",
        _spawn_fake_worker,
    )


def test_adapter_satisfies_stt_backend_protocol() -> None:
    """The proxy must structurally satisfy the STTBackend Protocol."""
    adapter = SubprocessSTTAdapter()
    assert isinstance(adapter, STTBackend)


def test_completed_event_fires_on_completed_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``completed`` event from the worker should invoke on_completed."""
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        """
        _send_message(("ready", None))
        while True:
            try:
                message = _receive_message()
            except EOFError:
                raise SystemExit(0)
            kind = message[0]
            if kind == "close":
                raise SystemExit(0)
            if kind == "feed":
                # Echo a completed transcript on the first frame.
                _send_message(("event", "completed", "hello world"))
        """,
    )

    async def _run() -> None:
        adapter = SubprocessSTTAdapter(start_timeout=10.0)
        completed: list[str] = []

        async def _on_completed(text: str) -> None:
            completed.append(text)

        await adapter.start(_on_completed)
        try:
            frame = AudioFrame(samples=np.zeros(160, dtype=np.int16), sample_rate=16000)
            await adapter.feed_audio(frame)
            # Wait for the reader thread to surface the event.
            for _ in range(100):
                if completed:
                    break
                await asyncio.sleep(0.02)
            assert completed == ["hello world"]
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_partial_and_speech_started_events_route_to_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """started/partial events route to their optional callbacks."""
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        """
        _send_message(("ready", None))
        while True:
            try:
                message = _receive_message()
            except EOFError:
                raise SystemExit(0)
            if message[0] == "close":
                raise SystemExit(0)
            if message[0] == "feed":
                _send_message(("event", "started", ""))
                _send_message(("event", "partial", "hel"))
                _send_message(("event", "completed", "hello"))
        """,
    )

    async def _run() -> None:
        adapter = SubprocessSTTAdapter(start_timeout=10.0)
        started = 0
        partials: list[str] = []
        completed: list[str] = []

        async def _on_completed(text: str) -> None:
            completed.append(text)

        async def _on_partial(text: str) -> None:
            partials.append(text)

        async def _on_started() -> None:
            nonlocal started
            started += 1

        await adapter.start(_on_completed, on_partial=_on_partial, on_speech_started=_on_started)
        try:
            await adapter.feed_audio(AudioFrame(samples=np.zeros(160, dtype=np.int16), sample_rate=16000))
            for _ in range(100):
                if completed:
                    break
                await asyncio.sleep(0.02)
            assert started == 1
            assert partials == ["hel"]
            assert completed == ["hello"]
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_worker_crash_does_not_raise_into_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the worker dies mid-session, feed_audio must degrade, not raise."""
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        """
        _send_message(("ready", None))
        # Exit immediately after announcing readiness, simulating a crash.
        raise SystemExit(1)
        """,
    )

    async def _run() -> None:
        adapter = SubprocessSTTAdapter(start_timeout=10.0, restart_backoff=0.05)

        async def _on_completed(text: str) -> None:  # pragma: no cover - never fires
            raise AssertionError("should not complete")

        await adapter.start(_on_completed)
        try:
            # Give the reader thread a moment to observe EOF.
            await asyncio.sleep(0.2)
            # Must not raise even though the worker is gone.
            await adapter.feed_audio(AudioFrame(samples=np.zeros(160, dtype=np.int16), sample_rate=16000))
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_stall_watchdog_restarts_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that stalls (stops heartbeating) is killed and respawned."""
    # The worker records each launch by appending to a shared file; the
    # first launch goes silent after ready (no heartbeats), the watchdog
    # should kill+respawn, and the second launch behaves normally.
    marker = tmp_path / "launches.txt"
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        f"""
        marker = {str(marker)!r}
        with open(marker, "a") as fh:
            fh.write("x")
        with open(marker) as fh:
            launch_index = len(fh.read())

        _send_message(("ready", None))

        if launch_index == 1:
            # First launch: go completely silent (simulated stall).
            while True:
                time.sleep(0.5)
        else:
            # Subsequent launch: heartbeat normally and answer feeds.
            def _hb():
                while True:
                    _send_message(("heartbeat", None))
                    time.sleep(0.05)
            threading.Thread(target=_hb, daemon=True).start()
            while True:
                try:
                    message = _receive_message()
                except EOFError:
                    raise SystemExit(0)
                if message[0] == "close":
                    raise SystemExit(0)
                if message[0] == "feed":
                    _send_message(("event", "completed", "recovered"))
        """,
    )

    async def _run() -> None:
        adapter = SubprocessSTTAdapter(
            start_timeout=10.0,
            stall_timeout=0.3,
            heartbeat_interval=0.05,
            restart_backoff=0.05,
        )
        completed: list[str] = []

        async def _on_completed(text: str) -> None:
            completed.append(text)

        await adapter.start(_on_completed)
        try:
            # Wait long enough for the watchdog to notice the stall and respawn.
            for _ in range(200):
                if adapter.restart_count >= 1:
                    break
                await asyncio.sleep(0.02)
            assert adapter.restart_count >= 1
            # The respawned worker should now answer feeds.
            for _ in range(50):
                await adapter.feed_audio(AudioFrame(samples=np.zeros(160, dtype=np.int16), sample_rate=16000))
                if completed:
                    break
                await asyncio.sleep(0.05)
            assert completed and completed[0] == "recovered"
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_bootstrap_adds_src_parent_to_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subprocess bootstrap should prepend the src directory to PYTHONPATH."""
    popen_kwargs: dict[str, Any] = {}
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        """
        _send_message(("ready", None))
        while True:
            try:
                message = _receive_message()
            except EOFError:
                raise SystemExit(0)
            if message[0] == "close":
                raise SystemExit(0)
        """,
        popen_kwargs=popen_kwargs,
    )

    async def _run() -> None:
        adapter = SubprocessSTTAdapter(start_timeout=10.0)

        async def _noop(_text: str) -> None:
            return None

        await adapter.start(_noop)
        try:
            env = popen_kwargs["env"]
            assert isinstance(env, dict)
            pythonpath = env["PYTHONPATH"]
            package_spec = importlib.util.find_spec("robot_comic")
            locations = None if package_spec is None else package_spec.submodule_search_locations
            assert locations
            expected = str(Path(next(iter(locations))).resolve().parent)
            assert pythonpath.split(os.pathsep)[0] == expected
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_spawn_discards_fresh_worker_when_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop() racing a respawn must not orphan the freshly spawned worker.

    Review finding (#550): _close_sync can run while the watchdog is inside
    _spawn_and_wait_ready (Moonshine cold load is ~20s); the new child must
    be killed and the spawn aborted, not stored after shutdown.
    """
    worker_script = tmp_path / "fake_stt_worker.py"
    worker_script.write_text(
        dedent(_WORKER_PREAMBLE)
        + dedent(
            """
            _send_message(("ready", None))
            while True:
                try:
                    _receive_message()
                except EOFError:
                    raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    spawned: list[Any] = []
    real_popen: Any = subprocess.Popen

    def _spawn_and_record(*_args: object, **kwargs: Any) -> Any:
        process = real_popen([sys.executable, str(worker_script)], **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr("robot_comic.adapters.stt_process.subprocess.Popen", _spawn_and_record)

    adapter = SubprocessSTTAdapter(start_timeout=10.0)
    adapter._stop.set()  # simulate stop()/shutdown arriving mid-respawn

    with pytest.raises(RuntimeError, match="shutting down"):
        adapter._spawn_and_wait_ready()

    assert adapter._process is None
    assert spawned, "the fake worker should have been spawned"
    spawned[0].wait(timeout=10)
    assert spawned[0].poll() is not None, "fresh worker must be terminated, not orphaned"


def test_stop_closes_stdout_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop() must close the parent-side stdout pipe (review finding, #550)."""
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        """
        _send_message(("ready", None))
        while True:
            try:
                message = _receive_message()
            except EOFError:
                raise SystemExit(0)
            if message[0] == "close":
                raise SystemExit(0)
        """,
    )

    async def _run() -> None:
        adapter = SubprocessSTTAdapter(start_timeout=10.0)

        async def _noop(_text: str) -> None:
            return None

        await adapter.start(_noop)
        stdout_pipe = adapter._stdout
        assert stdout_pipe is not None
        await adapter.stop()
        assert stdout_pipe.closed, "parent-side stdout pipe must be closed on stop"

    asyncio.run(_run())
