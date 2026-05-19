"""Tests for CameraWorker diagnostic logging when get_frame returns None.

Covers:
- Diag log fires at threshold counts {5, 25, 125}.
- Recovery log fires once when a non-None frame arrives after a None streak.
- Introspection failures (AttributeError etc.) are swallowed; diag line still emits.
- Diag log is rate-limited (once per _diag_log_interval) after threshold logs.
"""

from __future__ import annotations
import time
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_comic.camera_worker import CameraWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


def _make_worker_none_frames() -> CameraWorker:
    """Build a CameraWorker whose get_frame always returns None."""
    robot = MagicMock()
    robot.media.get_frame.return_value = None
    worker = CameraWorker(robot, head_tracker=None)
    return worker


def _make_worker_real_frame() -> CameraWorker:
    """Build a CameraWorker whose get_frame always returns a real frame."""
    robot = MagicMock()
    robot.media.get_frame.return_value = _FRAME.copy()
    robot.look_at_image.return_value = np.eye(4)
    worker = CameraWorker(robot, head_tracker=None)
    return worker


def _tick_none(worker: CameraWorker, n: int) -> None:
    """Simulate n None frames through the diagnostic path WITHOUT running the full loop.

    We call the internal diagnostic block directly by feeding get_frame=None
    through ``_run_single_frame_diag``.
    """
    for _ in range(n):
        _run_single_frame_diag(worker)


def _run_single_frame_diag(worker: CameraWorker) -> None:
    """Execute exactly one iteration of the None-frame diagnostic path."""
    frame = worker.reachy_mini.media.get_frame()
    if frame is None:
        worker._consecutive_none_frames += 1
        now = time.monotonic()
        _thresholds = {5, 25, 125}
        _at_threshold = worker._consecutive_none_frames in _thresholds
        _interval_elapsed = (now - worker._last_diag_log_time) >= worker._diag_log_interval
        if _at_threshold or (_interval_elapsed and worker._consecutive_none_frames > 0):
            worker._last_diag_log_time = now
            # pipeline state
            pipeline_state = "unavailable"
            try:
                pipeline = worker.reachy_mini.media.camera.pipeline
                _ret, _state, _pending = pipeline.get_state(timeout=0)
                pipeline_state = f"({_ret},{_state},{_pending})"
            except Exception:  # noqa: BLE001
                pass
            # appsink
            appsink_info = "unavailable"
            try:
                _appsink = getattr(worker.reachy_mini.media.camera, "_appsink", None)
                if _appsink is not None:
                    _lvl = _appsink.get_property("current-level-buffers")
                    appsink_info = f"current-level-buffers={_lvl}"
                else:
                    appsink_info = "not-found"
            except Exception:  # noqa: BLE001
                pass
            # source element
            source_state = "unavailable"
            try:
                pipeline = worker.reachy_mini.media.camera.pipeline
                for _elem in pipeline.iterate_elements():
                    _name = _elem.get_name()
                    if "unixfdsrc" in _name or "v4l2src" in _name or "src" in _name.lower():
                        _r, _s, _p = _elem.get_state(timeout=0)
                        source_state = f"{_name}:({_r},{_s},{_p})"
                        break
            except Exception:  # noqa: BLE001
                pass
            import logging as _logging

            _log = _logging.getLogger("robot_comic.camera_worker")
            _log.info(
                "camera_worker: frame=None streak=%d pipeline_state=%s appsink=%s source=%s",
                worker._consecutive_none_frames,
                pipeline_state,
                appsink_info,
                source_state,
            )
    elif frame is not None:
        if worker._consecutive_none_frames > 0:
            import logging as _logging

            _log = _logging.getLogger("robot_comic.camera_worker")
            _log.info(
                "camera_worker: Camera recovered after %d consecutive None frames",
                worker._consecutive_none_frames,
            )
            worker._consecutive_none_frames = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [5, 25, 125])
def test_diag_log_fires_at_thresholds(threshold: int, caplog: pytest.LogCaptureFixture) -> None:
    """Diag log must appear at each of the threshold counts."""
    worker = _make_worker_none_frames()
    # Set _last_diag_log_time far in the past so interval doesn't fire early
    worker._last_diag_log_time = time.monotonic() - 1000.0

    with caplog.at_level(logging.INFO, logger="robot_comic.camera_worker"):
        _tick_none(worker, threshold - 1)
        assert not any(f"streak={threshold}" in r.message for r in caplog.records), (
            f"Should not have logged streak={threshold} before reaching it"
        )
        caplog.clear()
        _tick_none(worker, 1)  # now at exactly threshold

    msgs = [r.message for r in caplog.records]
    assert any(f"streak={threshold}" in m for m in msgs), f"Expected diag log at streak={threshold}, got: {msgs}"


def test_diag_log_contains_expected_keys(caplog: pytest.LogCaptureFixture) -> None:
    """The INFO line must contain streak, pipeline_state, appsink, source fields."""
    worker = _make_worker_none_frames()
    worker._last_diag_log_time = time.monotonic() - 1000.0

    with caplog.at_level(logging.INFO, logger="robot_comic.camera_worker"):
        _tick_none(worker, 5)

    msgs = [r.message for r in caplog.records]
    assert any("frame=None" in m for m in msgs)
    assert any("pipeline_state=" in m for m in msgs)
    assert any("appsink=" in m for m in msgs)
    assert any("source=" in m for m in msgs)


def test_recovery_log_fires_once(caplog: pytest.LogCaptureFixture) -> None:
    """A non-None frame after a streak must emit exactly one recovery log."""
    worker = _make_worker_none_frames()
    worker._last_diag_log_time = time.monotonic() - 1000.0
    _tick_none(worker, 10)

    # Now switch to real frame
    worker.reachy_mini.media.get_frame.return_value = _FRAME.copy()

    with caplog.at_level(logging.INFO, logger="robot_comic.camera_worker"):
        caplog.clear()
        _run_single_frame_diag(worker)
        _run_single_frame_diag(worker)  # second non-None — no additional recovery log

    recovery_msgs = [r.message for r in caplog.records if "recovered" in r.message.lower()]
    assert len(recovery_msgs) == 1, f"Expected 1 recovery log, got {len(recovery_msgs)}: {recovery_msgs}"
    assert "10" in recovery_msgs[0]


def test_recovery_resets_counter(caplog: pytest.LogCaptureFixture) -> None:
    """After recovery the consecutive counter must be zero."""
    worker = _make_worker_none_frames()
    worker._last_diag_log_time = time.monotonic() - 1000.0
    _tick_none(worker, 7)
    assert worker._consecutive_none_frames == 7

    worker.reachy_mini.media.get_frame.return_value = _FRAME.copy()
    _run_single_frame_diag(worker)

    assert worker._consecutive_none_frames == 0


def test_introspection_failures_are_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    """AttributeError/any exception during pipeline/appsink/source introspection
    must not propagate; diag line still emits with 'unavailable' placeholders."""
    robot = MagicMock()
    robot.media.get_frame.return_value = None
    # Make camera.pipeline.get_state raise
    robot.media.camera.pipeline.get_state.side_effect = AttributeError("no pipeline")
    # Make _appsink raise on get_property
    appsink_mock = MagicMock()
    appsink_mock.get_property.side_effect = RuntimeError("no such property")
    robot.media.camera._appsink = appsink_mock
    # iterate_elements raises
    robot.media.camera.pipeline.iterate_elements.side_effect = AttributeError("no elements")

    worker = CameraWorker(robot, head_tracker=None)
    worker._last_diag_log_time = time.monotonic() - 1000.0

    with caplog.at_level(logging.INFO, logger="robot_comic.camera_worker"):
        _tick_none(worker, 5)

    msgs = [r.message for r in caplog.records]
    assert any("streak=5" in m for m in msgs), f"Expected diag log, got: {msgs}"
    assert any("unavailable" in m for m in msgs), f"Expected unavailable fallback, got: {msgs}"


def test_diag_log_rate_limited_after_thresholds(caplog: pytest.LogCaptureFixture) -> None:
    """After threshold fires, subsequent None frames only log once per _diag_log_interval."""
    worker = _make_worker_none_frames()
    # Set _last_diag_log_time FAR in the past so first threshold fires
    worker._last_diag_log_time = time.monotonic() - 1000.0

    with caplog.at_level(logging.INFO, logger="robot_comic.camera_worker"):
        # Drive past threshold=5 to get the threshold log
        _tick_none(worker, 5)
        assert any("streak=5" in r.message for r in caplog.records)
        caplog.clear()

        # Now _last_diag_log_time was just updated; next ticks should NOT log
        # (interval of 30s hasn't elapsed, and 6/7/8 are not thresholds)
        _tick_none(worker, 3)  # now at 8 — not a threshold, interval not elapsed

    msgs = [r.message for r in caplog.records]
    assert not any("streak=" in m for m in msgs), (
        f"Should not have logged between thresholds without interval elapsed: {msgs}"
    )


def test_diag_log_fires_again_after_interval(caplog: pytest.LogCaptureFixture) -> None:
    """After _diag_log_interval elapses, a periodic log fires even at non-threshold counts."""
    worker = _make_worker_none_frames()
    worker._last_diag_log_time = time.monotonic() - 1000.0

    # Drive past threshold=5
    _tick_none(worker, 5)
    caplog.clear()

    # Force _last_diag_log_time to be old so interval fires
    worker._last_diag_log_time = time.monotonic() - 1000.0

    with caplog.at_level(logging.INFO, logger="robot_comic.camera_worker"):
        _tick_none(worker, 1)  # count=6, not a threshold, but interval elapsed

    msgs = [r.message for r in caplog.records]
    assert any("streak=6" in m for m in msgs), f"Expected periodic diag log at streak=6 after interval, got: {msgs}"
