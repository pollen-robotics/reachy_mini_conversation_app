"""Unit tests for the diagnostic logging added to greet._scan (#289)."""

from __future__ import annotations
import re
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import robot_comic.tools.greet as greet_mod
from robot_comic.tools.greet import Greet
from robot_comic.tools.core_tools import ToolDependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deps(camera_worker: object) -> ToolDependencies:
    """Build a minimal ToolDependencies with a fake camera_worker."""
    robot = SimpleNamespace()
    movement_manager = MagicMock()
    return ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        motion_duration_s=0.0,
    )


def _blank_frame() -> "np.ndarray[Any, np.dtype[np.uint8]]":
    """Return a small black BGR frame with known stats (shape, mean=0)."""
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _grey_frame(value: int = 128) -> "np.ndarray[Any, np.dtype[np.uint8]]":
    """Return a uniform grey frame with mean=value (non-zero, non-saturated)."""
    return np.full((48, 64, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests — DEBUG per-iteration logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_emits_debug_frame_stats_when_frame_present(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DEBUG log per iteration must include shape, dtype, mean, min, max."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)

    camera = MagicMock()
    camera.get_latest_frame.return_value = _grey_frame(128)
    deps = _make_deps(camera)

    caplog.set_level(logging.DEBUG, logger="robot_comic.tools.greet")
    # Face found immediately so scan returns on first poll iteration.
    with patch.object(
        greet_mod,
        "_detect_face_with_scores",
        return_value=(True, [0.91]),
    ):
        result = await Greet()._scan(deps)

    assert result == {"face_detected": True}

    debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    # Frame-stats debug line must mention shape AND mean AND dtype.
    frame_stat_lines = [m for m in debug_msgs if "frame shape=" in m and "mean=" in m]
    assert frame_stat_lines, f"No frame-stats DEBUG log found in: {debug_msgs}"
    assert "(48, 64, 3)" in frame_stat_lines[0]
    assert "uint8" in frame_stat_lines[0]
    assert "mean=128.00" in frame_stat_lines[0]
    assert "min=128.00" in frame_stat_lines[0]
    assert "max=128.00" in frame_stat_lines[0]


@pytest.mark.asyncio
async def test_scan_emits_debug_detection_list_even_when_empty(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DEBUG log must report the detection count + scores even when empty."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.0")
    monkeypatch.setenv("REACHY_MINI_GREET_SWEEP_DISABLED", "1")

    camera = MagicMock()
    camera.get_latest_frame.return_value = _blank_frame()
    deps = _make_deps(camera)

    caplog.set_level(logging.DEBUG, logger="robot_comic.tools.greet")
    with patch.object(
        greet_mod,
        "_detect_face_with_scores",
        return_value=(False, []),
    ):
        result = await Greet()._scan(deps)

    # Sweep-disabled path returns an enriched ``no_subject`` dict with a
    # ``retry_hint`` etc.; only the ``no_subject`` flag is load-bearing for
    # this diagnostic test.
    assert result.get("no_subject") is True
    debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    detection_lines = [m for m in debug_msgs if "mediapipe detections" in m]
    assert detection_lines, f"No detection DEBUG log found in: {debug_msgs}"
    assert "count=0" in detection_lines[0]
    assert "scores=[]" in detection_lines[0]


# ---------------------------------------------------------------------------
# Tests — INFO summary on no_subject
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_emits_info_summary_on_no_subject(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """INFO summary on no_subject must carry the four diagnostic fields."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.0")
    monkeypatch.setenv("REACHY_MINI_GREET_SWEEP_DISABLED", "1")

    camera = MagicMock()
    camera.get_latest_frame.return_value = _grey_frame(73)
    deps = _make_deps(camera)

    caplog.set_level(logging.DEBUG, logger="robot_comic.tools.greet")
    with patch.object(
        greet_mod,
        "_detect_face_with_scores",
        return_value=(False, [0.12, 0.05]),
    ):
        result = await Greet()._scan(deps)

    # Sweep-disabled path: only assert the ``no_subject`` flag — the rest
    # of the dict is asserted in tests/tools/test_greet.py.
    assert result.get("no_subject") is True
    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    summary_lines = [m for m in info_msgs if "no_subject" in m and "frame_shape=" in m]
    assert summary_lines, f"No no_subject INFO summary found in: {info_msgs}"
    summary = summary_lines[0]
    assert "frame_shape=(48, 64, 3)" in summary
    assert "mean=73" in summary
    assert "detection_count=2" in summary
    # Highest confidence reported as the max of the supplied scores.
    assert re.search(r"highest_confidence=0\.120", summary), summary


@pytest.mark.asyncio
async def test_scan_info_summary_after_sweep_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No-subject INFO summary must also fire from the sweep path."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.0")

    camera = MagicMock()
    camera.get_latest_frame.return_value = _blank_frame()
    deps = _make_deps(camera)

    class _NoopMoveHead:
        async def __call__(self, deps: ToolDependencies, **kwargs: object) -> dict[str, Any]:
            return {}

    caplog.set_level(logging.INFO, logger="robot_comic.tools.greet")
    with (
        patch.object(
            greet_mod,
            "_detect_face_with_scores",
            return_value=(False, []),
        ),
        patch("robot_comic.tools.greet.MoveHead", return_value=_NoopMoveHead()),
    ):
        result = await Greet()._scan(deps)

    assert result == {"no_subject": True}
    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    summary_lines = [m for m in info_msgs if "no_subject" in m and "frame_shape=" in m]
    assert summary_lines, f"No no_subject INFO summary found in: {info_msgs}"
    summary = summary_lines[0]
    assert "frame_shape=(48, 64, 3)" in summary
    assert "mean=0.00" in summary
    assert "detection_count=0" in summary
    assert "highest_confidence=0.000" in summary


# ---------------------------------------------------------------------------
# Tests — None-frame paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_persistent_none_frames_log_and_fall_through(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Persistent None frames must trigger the poll-loop DEBUG log and fall through.

    Pre-fix (2026-05-19): an initial-guard early return on a None first frame
    caused greet to bail before the poll loop, the LLM to retry 8 times, and
    turns to finish without TTS (silent robot). Post-fix the poll loop drains
    and logs from its own None branch.
    """
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.0")
    monkeypatch.setenv("REACHY_MINI_GREET_SWEEP_DISABLED", "1")

    camera = MagicMock()
    camera.get_latest_frame.return_value = None
    deps = _make_deps(camera)

    caplog.set_level(logging.DEBUG, logger="robot_comic.tools.greet")
    result = await Greet()._scan(deps)

    # New contract: no fast-error, fall through to no_subject.
    assert result.get("error") != "No frame available"
    assert result.get("no_subject") is True
    debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    none_lines = [m for m in debug_msgs if "returned None" in m and "during poll" in m]
    assert none_lines, f"No poll-None DEBUG log found in: {debug_msgs}"


@pytest.mark.asyncio
async def test_scan_logs_when_poll_frame_is_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a polling iteration sees None, log a DEBUG line and keep polling."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.05")
    monkeypatch.setenv("REACHY_MINI_GREET_SWEEP_DISABLED", "1")

    call_count = {"n": 0}

    def _frames() -> Any:
        # First call: guard sees a valid frame; subsequent polls: None.
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _blank_frame()
        return None

    camera = MagicMock()
    camera.get_latest_frame.side_effect = _frames
    deps = _make_deps(camera)

    caplog.set_level(logging.DEBUG, logger="robot_comic.tools.greet")
    result = await Greet()._scan(deps)

    # Sweep-disabled path: only assert the ``no_subject`` flag.
    assert result.get("no_subject") is True
    debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    poll_none_lines = [m for m in debug_msgs if "returned None" in m and "during poll" in m]
    assert poll_none_lines, f"No poll-None DEBUG log found in: {debug_msgs}"
