"""Tests for the greet scan MediaPipe poll-rate cap (Fix B).

The diagnostic showed MediaPipe inference firing at camera frame rate (~25 Hz)
during the face-scan wait window when no face was present, burning ~5% extra
CPU on the Pi 5.  The fix adds REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S
(default 0.3 s) so inference only fires ~3 Hz when the room is empty.

These tests verify:
1. The _scan_poll_interval_s() helper honours the env var and defaults to
   _DEFAULT_SCAN_POLL_INTERVAL_S (0.3 s).
2. Inside the scan poll loop, asyncio.sleep() is called with the configured
   interval, so inference cannot fire faster than 1 / poll_interval Hz.
3. The REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S=0.1 override is respected.
"""

from __future__ import annotations
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import robot_comic.tools.greet as greet_mod
from robot_comic.tools.greet import _DEFAULT_SCAN_POLL_INTERVAL_S, Greet, _scan_poll_interval_s
from robot_comic.tools.core_tools import ToolDependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deps(camera_worker: Any) -> ToolDependencies:
    """Build a minimal ToolDependencies with a fake camera_worker."""
    from types import SimpleNamespace

    robot = SimpleNamespace()
    movement_manager = MagicMock()
    return ToolDependencies(
        reachy_mini=robot,  # type: ignore[arg-type]
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        motion_duration_s=0.0,
    )


def _blank_frame() -> np.ndarray:  # type: ignore[type-arg]
    """Return a small black BGR frame."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _scan_poll_interval_s() unit tests
# ---------------------------------------------------------------------------


def test_default_poll_interval_is_300ms() -> None:
    """Without any env override, poll interval must be _DEFAULT_SCAN_POLL_INTERVAL_S."""
    import os

    os.environ.pop("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S", None)
    assert _scan_poll_interval_s() == pytest.approx(_DEFAULT_SCAN_POLL_INTERVAL_S)


def test_env_override_changes_poll_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S sets the poll interval."""
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S", "0.15")
    assert _scan_poll_interval_s() == pytest.approx(0.15)


def test_invalid_env_override_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric env value should fall back to the default without raising."""
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S", "not_a_number")
    result = _scan_poll_interval_s()
    assert result == pytest.approx(_DEFAULT_SCAN_POLL_INTERVAL_S)


def test_poll_interval_clamp_prevents_busy_spin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values below 0.05 s are clamped to 0.05 s to prevent a busy loop."""
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S", "0.001")
    result = _scan_poll_interval_s()
    assert result >= 0.05, f"expected >= 0.05, got {result}"


# ---------------------------------------------------------------------------
# Integration: asyncio.sleep receives the configured poll interval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_loop_uses_configured_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan loop passes poll_interval_s to asyncio.sleep, not the old constant."""
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    # Short scan window so the test doesn't linger; face is never found.
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.05")
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S", "0.12")

    camera = MagicMock()
    camera.get_latest_frame.return_value = _blank_frame()
    deps = _make_deps(camera)

    sleep_durations: list[float] = []

    async def _capture_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    # Patch at the greet module level so the async poll loop uses our mock.
    # The scan window (0.05 s) is shorter than the poll interval (0.12 s), so
    # the poll loop should exit at the deadline without ever reaching the sweep
    # — no MoveHead mock needed.
    with (
        patch.object(greet_mod, "_detect_face_with_scores", return_value=(False, [])),
        patch("robot_comic.tools.greet.asyncio") as mock_asyncio,
        # SWEEP_DISABLED so we never reach MoveHead even on timing edge cases.
        patch.dict("os.environ", {"REACHY_MINI_GREET_SWEEP_DISABLED": "1"}),
    ):
        mock_asyncio.sleep = _capture_sleep  # replace asyncio.sleep inside greet
        await Greet()._scan(deps)

    # All recorded sleeps must use 0.12 s (the env override), not the old 0.1 s.
    if sleep_durations:
        for dur in sleep_durations:
            assert dur == pytest.approx(0.12), (
                f"Expected sleep(0.12) but got sleep({dur}); poll interval env override was not respected."
            )


@pytest.mark.asyncio
async def test_scan_inference_count_respects_poll_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference call count is bounded by 1 / poll_interval within the scan window.

    With a 0.5 s scan window and a 0.2 s poll interval we expect at most
    ceil(0.5 / 0.2) + 1 = 4 inference calls.  Without the cap the old 0.1 s
    default would allow up to ~6, and full-frame-rate would allow ~12.
    """
    monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True)
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_WAIT_S", "0.5")
    monkeypatch.setenv("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S", "0.2")
    # Disable sweep so the test ends promptly.
    monkeypatch.setenv("REACHY_MINI_GREET_SWEEP_DISABLED", "1")

    camera = MagicMock()
    camera.get_latest_frame.return_value = _blank_frame()
    deps = _make_deps(camera)

    inference_calls = 0

    def _count_inference(frame: Any):  # type: ignore[type-arg]
        nonlocal inference_calls
        inference_calls += 1
        return (False, [])

    with patch.object(greet_mod, "_detect_face_with_scores", side_effect=_count_inference):
        await Greet()._scan(deps)

    # At 0.2 s interval over 0.5 s: at most 3 loop iterations (t=0, t=0.2, t=0.4)
    # plus the initial guard check frame; give generous headroom for timing jitter.
    max_expected = 6
    assert inference_calls <= max_expected, (
        f"inference fired {inference_calls} times; expected <= {max_expected} "
        f"with a 0.2 s poll interval over a 0.5 s window"
    )
