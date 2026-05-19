"""Unit tests for the greet tool scan action — specifically the tracker-latch race fix."""

from __future__ import annotations
from types import SimpleNamespace
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
        reachy_mini=robot,  # type: ignore[arg-type]
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        motion_duration_s=0.0,
    )


def _blank_frame() -> "np.ndarray":  # type: ignore[type-arg]
    """Return a small black BGR frame that should NOT trigger face detection."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


class _FrameSource:
    """Fake camera_worker that returns None for ``blank_count`` polls then a frame.

    Used to exercise the lazy-camera race: get_latest_frame() returns None
    on early polls (camera just started, gstreamer hasn't latched a frame
    yet) and then yields a frame once the pipeline catches up. The poll
    loop in _scan must tolerate this.

    To avoid a real MediaPipe inference we patch `_detect_face` in the tests.
    """

    def __init__(self, blank_count: int) -> None:
        # blank_count = how many polls return None before a face
        self._blank_count = blank_count
        self._calls = 0

    def get_latest_frame(self) -> "np.ndarray | None":  # type: ignore[type-arg]
        self._calls += 1
        if self._calls <= self._blank_count:
            return None
        return _blank_frame()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_returns_face_when_tracker_latches_after_initial_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race scenario: first frame is None (camera not ready), face appears shortly after.

    The scan should wait, detect the face on a later poll, and return
    face_detected=True rather than falling through to no_subject.
    """
    # Force MP_AVAILABLE so we exercise the real polling path
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)

    # First 2 calls to get_latest_frame() return None; third returns a frame
    camera = _FrameSource(blank_count=2)
    deps = _make_deps(camera)

    # Patch _detect_face_with_scores to always return True once a frame is present
    with patch.object(greet_mod, "_detect_face_with_scores", return_value=(True, [0.9])):
        result = await Greet()._scan(deps)

    assert result == {"face_detected": True}, f"Unexpected result: {result}"
    # Must have polled at least 3 times (2 blank + 1 hit on the third call).
    # With the lazy-camera guard removed, all calls happen inside the poll loop.
    assert camera._calls >= 3


@pytest.mark.asyncio
async def test_scan_waits_up_to_configured_window_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No face appears during the entire wait window → scan must not hang forever.

    We shorten REACHY_MINI_GREET_SCAN_WAIT_S to 0.15 s so the test is fast,
    then verify the tool returns no_subject (after the sweep is also skipped
    by patching MoveHead).
    """
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.15")

    camera = MagicMock()
    camera.get_latest_frame.return_value = _blank_frame()
    deps = _make_deps(camera)

    # face never detected, sweep also finds nothing
    with (
        patch.object(greet_mod, "_detect_face_with_scores", return_value=(False, [])),
        patch("robot_comic.tools.greet.MoveHead") as MockMoveHead,
    ):
        # MoveHead()(deps, direction=...) must be awaitable
        mock_instance = AsyncMock_MoveHead()
        MockMoveHead.return_value = mock_instance
        result = await Greet()._scan(deps)

    assert result == {"no_subject": True}, f"Unexpected result: {result}"

    # Ensure we didn't poll for much longer than the configured window
    # (wall-clock check not feasible in unit tests, but at least verify result)


class AsyncMock_MoveHead:
    """Minimal async callable that simulates MoveHead() returning success."""

    async def __call__(self, deps: ToolDependencies, **kwargs: object) -> dict:  # type: ignore[override]
        """Return an empty success dict."""
        return {}


@pytest.mark.asyncio
async def test_scan_returns_face_on_very_first_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast path: face visible immediately → should return face_detected=True right away."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)

    camera = MagicMock()
    camera.get_latest_frame.return_value = _blank_frame()
    deps = _make_deps(camera)

    with patch.object(greet_mod, "_detect_face_with_scores", return_value=(True, [0.9])):
        result = await Greet()._scan(deps)

    assert result == {"face_detected": True}
    # With no initial-guard pre-poll, the poll loop's first iteration both
    # fetches the frame and hits the face on the first attempt. Total = 1.
    assert camera.get_latest_frame.call_count == 1


@pytest.mark.asyncio
async def test_scan_no_camera_returns_error() -> None:
    """Scan with camera_worker=None must return an error dict, not raise."""
    deps = _make_deps(camera_worker=None)
    result = await Greet()._scan(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_scan_no_frame_on_first_call_then_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for 2026-05-19 lazy-camera race: first get_latest_frame() is None.

    Pre-fix, an initial-guard early return on a None first frame caused greet
    to bail in <3 ms before the poll loop ran; the LLM retried up to 8 times
    and turns finished without TTS (silent robot). Post-fix the poll loop
    drains and falls through to no_subject — never to the fast-error path.
    """
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.0")
    monkeypatch.setenv("REACHY_MINI_GREET_SWEEP_DISABLED", "1")

    camera = MagicMock()
    camera.get_latest_frame.return_value = None
    deps = _make_deps(camera)

    result = await Greet()._scan(deps)
    # New contract: persistent None falls through, no fast-error.
    assert result.get("error") != "No frame available"
    assert result.get("no_subject") is True
