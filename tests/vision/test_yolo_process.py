"""Tests for the YOLO tracking process."""

import sys
import time
import subprocess
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import reachy_mini_conversation_app.vision.head_tracking.yolo_process as head_tracker_module
from reachy_mini_conversation_app.vision.head_tracking.yolo_process import YoloHeadTrackerProcess


def _patch_fake_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, worker_body: str) -> None:
    """Patch the tracker subprocess with a test worker script."""
    worker_script = tmp_path / "fake_head_tracker_worker.py"
    worker_script.write_text(
        dedent(
            """
            import pickle
            import struct
            import sys
            import time

            import numpy as np

            HEADER = struct.Struct("!I")


            def _read_exact(size: int) -> bytes:
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


            def _send_message(payload) -> None:
                data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                sys.stdout.buffer.write(HEADER.pack(len(data)))
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            """
        )
        + "\n"
        + dedent(worker_body),
        encoding="utf-8",
    )

    real_popen = subprocess.Popen

    def _spawn_fake_worker(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        return real_popen([sys.executable, str(worker_script)], **kwargs)

    monkeypatch.setattr(head_tracker_module.subprocess, "Popen", _spawn_fake_worker)


def test_head_tracker_discards_stale_reply_after_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Late replies from timed-out requests should not leak into the next frame."""
    _patch_fake_worker(
        monkeypatch,
        tmp_path,
        """
        _send_message(("ready", None))
        call_count = 0

        while True:
            try:
                message = _receive_message()
            except EOFError:
                raise SystemExit(0)

            if message[0] == "close":
                raise SystemExit(0)

            request_id = message[1]
            call_count += 1
            if call_count == 1:
                time.sleep(0.05)

            value = float(call_count)
            _send_message(("result", request_id, (np.array([value, value], dtype=np.float32), value)))
        """,
    )

    tracker = YoloHeadTrackerProcess(request_timeout=0.01)
    try:
        frame = np.zeros((12, 20, 3), dtype=np.uint8)

        eye_center, roll = tracker.get_head_position(frame)
        assert eye_center is None
        assert roll is None

        time.sleep(0.08)

        eye_center, roll = tracker.get_head_position(frame)
        assert eye_center is not None
        assert np.allclose(eye_center, np.array([2.0, 2.0], dtype=np.float32))
        assert roll == 2.0
    finally:
        tracker.close()


def test_head_tracker_accepts_numpy_floating_roll_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proxy should accept NumPy floating roll values from backend implementations."""
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

            request_id = message[1]
            _send_message(
                (
                    "result",
                    request_id,
                    (np.array([0.25, -0.5], dtype=np.float32), np.float64(0.75)),
                )
            )
        """,
    )

    tracker = YoloHeadTrackerProcess()
    try:
        eye_center, roll = tracker.get_head_position(np.zeros((12, 20, 3), dtype=np.uint8))
        assert eye_center is not None
        assert np.allclose(eye_center, np.array([0.25, -0.5], dtype=np.float32))
        assert roll == pytest.approx(0.75)
    finally:
        tracker.close()
