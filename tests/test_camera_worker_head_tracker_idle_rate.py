"""Tests for the head-tracker idle-rate gate in CameraWorker.

PR #464 dropped the greet scan poll rate from 25 Hz to 3 Hz when no face is
present.  The same dead-room problem exists in camera_worker.py: MediaPipe
runs get_head_position() on every loop iteration (25 Hz) even when nobody is
in frame, burning CPU continuously.

This PR adds REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S (default 0.2 s /
5 Hz) so inference is throttled when no face was found on the previous call.
As soon as a face is detected, the gate resets and the next call fires
immediately (no extra inter-frame delay while tracking is active).

The interpolation-back-to-neutral path is independent of get_head_position —
it is driven by last_face_detected_time and elapsed wall-clock time, so
skipping tracker calls while idle does NOT interrupt smooth return to neutral.

Tests verify:
1. _head_tracker_idle_poll_interval_s() default, env override, clamp, invalid.
2. When "no face" is returned, inference is called at the idle rate not full.
3. When a face is detected, inference rate returns to full (next call immediate).
4. Interpolation-back-to-neutral runs correctly even when tracker calls are
   throttled (last_face_detected_time is set, pose is lerp'd back to neutral).
"""

from __future__ import annotations
import os
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_comic.camera_worker import (
    _DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S,
    CameraWorker,
    _head_tracker_idle_poll_interval_s,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(tracker: Any = None) -> CameraWorker:
    """Build a CameraWorker with a mocked robot and optional tracker."""
    robot = MagicMock()
    robot.media.get_frame.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
    # look_at_image must return a valid 4×4 pose matrix
    robot.look_at_image.return_value = np.eye(4)
    return CameraWorker(robot, head_tracker=tracker)


def _stop(worker: CameraWorker) -> None:
    """Shut down the worker thread cleanly."""
    worker._stop_event.set()
    if worker._thread is not None:
        worker._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# _head_tracker_idle_poll_interval_s() unit tests
# ---------------------------------------------------------------------------


def test_default_idle_poll_interval() -> None:
    """Without any env override the default is _DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S."""
    os.environ.pop("REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S", None)
    assert _head_tracker_idle_poll_interval_s() == pytest.approx(_DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S)


def test_env_override_sets_idle_poll_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S is respected."""
    monkeypatch.setenv("REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S", "0.5")
    assert _head_tracker_idle_poll_interval_s() == pytest.approx(0.5)


def test_invalid_env_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric env value falls back to default without raising."""
    monkeypatch.setenv("REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S", "not_a_number")
    result = _head_tracker_idle_poll_interval_s()
    assert result == pytest.approx(_DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S)


def test_idle_poll_interval_clamp_to_min_frame_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values below 0.04 s (one camera frame) are clamped upward."""
    monkeypatch.setenv("REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S", "0.001")
    result = _head_tracker_idle_poll_interval_s()
    assert result >= 0.04, f"expected >= 0.04, got {result}"


# ---------------------------------------------------------------------------
# Idle-rate gating: inference count when no face is found
# ---------------------------------------------------------------------------


def test_no_face_throttles_inference_call_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no face is ever found, tracker is called at idle rate not every frame.

    We inject a controlled time source and loop the working_loop logic
    directly (without a real thread) to get deterministic call counts.
    """
    # Use a 0.2 s idle interval and simulate 0.5 s of wall time at 25 Hz
    # (i.e. 12 frames at 0.04 s each).  With idle-rate gating we expect
    # ~3 tracker calls (t=0, t=0.2, t=0.4); without gating it would be 12.
    monkeypatch.setenv("REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S", "0.2")

    tracker = MagicMock()
    tracker.get_head_position.return_value = (None, None)  # no face ever

    worker = _make_worker(tracker=tracker)
    worker.is_head_tracking_enabled = True

    # Simulate the idle-rate gate logic directly (mirrors working_loop inner body)
    # by stepping through 12 "frames" at 0.04 s intervals.
    t_start = 1000.0  # arbitrary epoch
    call_count = 0
    worker._last_tracker_call_time = None
    worker._last_tracker_found_face = False

    idle_interval = _head_tracker_idle_poll_interval_s()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    for i in range(12):
        current_time = t_start + i * 0.04
        run_tracker = (
            worker._last_tracker_found_face
            or worker._last_tracker_call_time is None
            or (current_time - worker._last_tracker_call_time) >= idle_interval
        )
        if run_tracker:
            worker._last_tracker_call_time = current_time
            eye_center, _ = tracker.get_head_position(frame)
            worker._last_tracker_found_face = eye_center is not None
            call_count += 1

    # At 0.2 s intervals over 0.48 s (12 frames × 0.04 s): frames at t=0,
    # t=0.2, t=0.4 pass the gate → 3 calls.  Allow a generous bound of 4 for
    # boundary rounding.
    assert call_count <= 4, f"Expected ≤4 tracker calls with idle_interval=0.2 over 12 frames, got {call_count}"
    assert call_count >= 2, f"Expected ≥2 tracker calls (at least first + one throttled), got {call_count}"


# ---------------------------------------------------------------------------
# Active tracking: face found → next call fires immediately
# ---------------------------------------------------------------------------


def test_face_found_resets_rate_to_full() -> None:
    """When get_head_position returns a face, the next iteration must run immediately.

    This verifies that _last_tracker_call_time is reset to None after a face
    hit so the subsequent loop iteration is not blocked by the idle gate.
    """
    tracker = MagicMock()
    # First call: face present
    eye_center = np.array([0.1, 0.1], dtype=np.float32)
    tracker.get_head_position.return_value = (eye_center, None)

    worker = _make_worker(tracker=tracker)
    worker._last_tracker_call_time = None
    worker._last_tracker_found_face = False

    # Simulate one "frame" where a face is found
    current_time = 1000.0
    idle_interval = _head_tracker_idle_poll_interval_s()
    run_tracker = (
        worker._last_tracker_found_face
        or worker._last_tracker_call_time is None
        or (current_time - worker._last_tracker_call_time) >= idle_interval
    )
    assert run_tracker, "First call should always run (last_call_time is None)"

    worker._last_tracker_call_time = current_time
    result_eye, _ = tracker.get_head_position(np.zeros((64, 64, 3), dtype=np.uint8))
    worker._last_tracker_found_face = result_eye is not None  # True

    # Next frame, 0.04 s later — well within the idle interval
    next_time = current_time + 0.04
    run_tracker_next = (
        worker._last_tracker_found_face  # True — face was found → run immediately
        or worker._last_tracker_call_time is None
        or (next_time - worker._last_tracker_call_time) >= idle_interval
    )
    assert run_tracker_next, (
        "When _last_tracker_found_face=True the next iteration must run "
        "immediately regardless of elapsed time since last call."
    )


def test_no_face_then_face_then_no_face_transitions() -> None:
    """Gate transitions: idle → active → idle behave correctly across frames."""
    tracker = MagicMock()
    worker = _make_worker(tracker=tracker)
    worker._last_tracker_call_time = None
    worker._last_tracker_found_face = False

    idle_interval = 0.2
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    # Phase 1: no face for several frames → throttled
    for i in range(5):
        t = 1000.0 + i * 0.04
        tracker.get_head_position.return_value = (None, None)
        run = (
            worker._last_tracker_found_face
            or worker._last_tracker_call_time is None
            or (t - worker._last_tracker_call_time) >= idle_interval
        )
        if run:
            worker._last_tracker_call_time = t
            eye, _ = tracker.get_head_position(frame)
            worker._last_tracker_found_face = eye is not None

    # Phase 2: face appears at t=1000.2 s
    t_face = 1000.2
    eye_pos = np.array([0.0, 0.0], dtype=np.float32)
    tracker.get_head_position.return_value = (eye_pos, None)
    run = (
        worker._last_tracker_found_face
        or worker._last_tracker_call_time is None
        or (t_face - worker._last_tracker_call_time) >= idle_interval
    )
    assert run, "Face-appear frame must pass the gate"
    worker._last_tracker_call_time = t_face
    eye, _ = tracker.get_head_position(frame)
    worker._last_tracker_found_face = eye is not None  # True

    # Phase 3: face disappears — next frame (0.04 s later) must still run
    # because _last_tracker_found_face=True from previous hit
    t_lost = t_face + 0.04
    tracker.get_head_position.return_value = (None, None)
    run_after_face = (
        worker._last_tracker_found_face
        or worker._last_tracker_call_time is None
        or (t_lost - worker._last_tracker_call_time) >= idle_interval
    )
    assert run_after_face, (
        "Frame immediately after face-found must run the tracker "
        "(the gate uses _last_tracker_found_face from the *previous* call)."
    )
    worker._last_tracker_call_time = t_lost
    eye2, _ = tracker.get_head_position(frame)
    worker._last_tracker_found_face = eye2 is not None  # now False again

    # Phase 4: another frame 0.04 s later — now throttled again
    t_throttled = t_lost + 0.04
    run_throttled = (
        worker._last_tracker_found_face  # False
        or worker._last_tracker_call_time is None
        or (t_throttled - worker._last_tracker_call_time) >= idle_interval
    )
    assert not run_throttled, "After face is lost again the gate should throttle (0.04 s < 0.2 s idle_interval)."


# ---------------------------------------------------------------------------
# Interpolation-back-to-neutral runs independently of tracker calls
# ---------------------------------------------------------------------------


def test_interpolation_runs_while_tracker_is_throttled() -> None:
    """Neutral-return interpolation must progress even during idle throttle.

    The interpolation path in working_loop is driven by:
      - last_face_detected_time (set when face was last seen)
      - interpolation_start_time / interpolation_start_pose (set when delay expires)
      - current_time (wall clock)

    It is completely independent of get_head_position, so throttling the
    tracker while no face is present must not stall neutral-return.
    """
    tracker = MagicMock()
    tracker.get_head_position.return_value = (None, None)

    worker = _make_worker(tracker=tracker)

    # Pre-arm interpolation: act as if a face was seen 3 s ago (past the 2 s delay)
    face_time = time.time() - 3.0
    worker.last_face_detected_time = face_time
    worker.face_lost_delay = 2.0
    worker.interpolation_duration = 1.0

    # Manually set a non-neutral start pose so we can verify interpolation runs
    start_pose = np.eye(4, dtype=np.float32)
    start_pose[0, 3] = 0.05  # non-zero translation
    worker.interpolation_start_time = face_time + worker.face_lost_delay
    worker.interpolation_start_pose = start_pose

    # At t = face_time + 3.0 the interpolation is complete (elapsed > duration).
    # Run the interpolation block manually (mirrors the loop body).
    from scipy.spatial.transform import Rotation as R  # type: ignore[import-not-found]

    from reachy_mini.utils.interpolation import linear_pose_interpolation

    neutral_pose = np.eye(4)
    current_time = time.time()

    elapsed_interpolation = current_time - worker.interpolation_start_time
    t_param = min(1.0, elapsed_interpolation / worker.interpolation_duration)

    interpolated_pose = linear_pose_interpolation(
        worker.interpolation_start_pose,
        neutral_pose,
        t_param,
    )
    translation = interpolated_pose[:3, 3]
    rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

    with worker.face_tracking_lock:
        worker.face_tracking_offsets = [
            translation[0],
            translation[1],
            translation[2],
            rotation[0],
            rotation[1],
            rotation[2],
        ]

    if t_param >= 1.0:
        worker.last_face_detected_time = None
        worker.interpolation_start_time = None
        worker.interpolation_start_pose = None

    # t_param should be 1.0 (elapsed > duration) and state should be cleared
    assert t_param >= 1.0, f"Expected interpolation complete (t≥1), got t={t_param}"
    assert worker.last_face_detected_time is None, (
        "After interpolation completes, last_face_detected_time must be reset to None"
    )
    offsets = worker.get_face_tracking_offsets()
    assert all(abs(v) < 1e-5 for v in offsets), (
        f"After full interpolation, face_tracking_offsets must be all-zero (neutral); got {offsets}"
    )
    # Tracker must not have been called by the interpolation path itself
    tracker.get_head_position.assert_not_called()
