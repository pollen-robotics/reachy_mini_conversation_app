"""Tests for MediapipeHeadTracker lazy initialisation (#472 PR-2).

The mediapipe graph build (vision.HeadTracker()) is expensive (~0.5-1 s on
the Pi).  It must be deferred until the first call to get_head_position(),
not triggered at MediapipeHeadTracker.__init__ time.
"""

from __future__ import annotations
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_bgr_frame(h: int = 4, w: int = 4) -> np.ndarray:  # type: ignore[type-arg]
    """Return a tiny blank BGR uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_fake_vision_module(mock_tracker_cls: Any) -> tuple[ModuleType, ModuleType]:
    """Build fake reachy_mini_toolbox and reachy_mini_toolbox.vision modules."""
    fake_vision = ModuleType("reachy_mini_toolbox.vision")
    fake_vision.HeadTracker = mock_tracker_cls  # type: ignore[attr-defined]
    fake_toolbox = ModuleType("reachy_mini_toolbox")
    fake_toolbox.vision = fake_vision  # type: ignore[attr-defined]
    return fake_toolbox, fake_vision


# ---------------------------------------------------------------------------
# Lazy-init tests
# ---------------------------------------------------------------------------


def test_mediapipe_tracker_not_constructed_at_init() -> None:
    """vision.HeadTracker() must NOT be called when MediapipeHeadTracker is instantiated."""
    mock_tracker_cls = MagicMock()
    fake_toolbox, fake_vision = _make_fake_vision_module(mock_tracker_cls)

    # Remove any cached module state so we get a fresh import.
    for key in list(sys.modules.keys()):
        if "robot_comic.vision.head_tracking.mediapipe" in key:
            del sys.modules[key]

    with patch.dict(
        sys.modules,
        {
            "reachy_mini_toolbox": fake_toolbox,
            "reachy_mini_toolbox.vision": fake_vision,
        },
    ):
        import importlib

        import robot_comic.vision.head_tracking.mediapipe as _mod

        importlib.reload(_mod)

        _mod.MediapipeHeadTracker()

    # HeadTracker constructor must not have been called yet.
    assert mock_tracker_cls.call_count == 0, (
        f"vision.HeadTracker() was called {mock_tracker_cls.call_count} time(s) at __init__ "
        "— construction must be deferred to first use"
    )


def test_mediapipe_tracker_constructed_on_first_get_head_position() -> None:
    """vision.HeadTracker() must be called exactly once on the first get_head_position()."""
    mock_instance = MagicMock()
    mock_instance.get_head_position.return_value = (None, None)
    mock_tracker_cls = MagicMock(return_value=mock_instance)
    fake_toolbox, fake_vision = _make_fake_vision_module(mock_tracker_cls)

    for key in list(sys.modules.keys()):
        if "robot_comic.vision.head_tracking.mediapipe" in key:
            del sys.modules[key]

    with patch.dict(
        sys.modules,
        {
            "reachy_mini_toolbox": fake_toolbox,
            "reachy_mini_toolbox.vision": fake_vision,
        },
    ):
        import importlib

        import robot_comic.vision.head_tracking.mediapipe as _mod

        importlib.reload(_mod)

        tracker = _mod.MediapipeHeadTracker()

        # Not yet constructed.
        assert mock_tracker_cls.call_count == 0

        # First call — must construct.
        tracker.get_head_position(_blank_bgr_frame())
        assert mock_tracker_cls.call_count == 1, (
            "vision.HeadTracker() should be called exactly once on first get_head_position()"
        )

        # Second call — must NOT reconstruct.
        tracker.get_head_position(_blank_bgr_frame())
        assert mock_tracker_cls.call_count == 1, (
            "vision.HeadTracker() must not be called again on subsequent get_head_position() calls"
        )


def test_mediapipe_tracker_ensure_tracker_idempotent() -> None:
    """_ensure_tracker() called multiple times must construct the tracker only once."""
    mock_instance = MagicMock()
    mock_tracker_cls = MagicMock(return_value=mock_instance)
    fake_toolbox, fake_vision = _make_fake_vision_module(mock_tracker_cls)

    for key in list(sys.modules.keys()):
        if "robot_comic.vision.head_tracking.mediapipe" in key:
            del sys.modules[key]

    with patch.dict(
        sys.modules,
        {
            "reachy_mini_toolbox": fake_toolbox,
            "reachy_mini_toolbox.vision": fake_vision,
        },
    ):
        import importlib

        import robot_comic.vision.head_tracking.mediapipe as _mod

        importlib.reload(_mod)

        tracker = _mod.MediapipeHeadTracker()
        tracker._ensure_tracker()
        tracker._ensure_tracker()
        tracker._ensure_tracker()

    assert mock_tracker_cls.call_count == 1


def test_mediapipe_tracker_respects_pre_set_tracker_monkeypatch() -> None:
    """Tests that monkeypatch _tracker directly must be honoured by _ensure_tracker()."""
    mock_instance = MagicMock()
    mock_tracker_cls = MagicMock(return_value=mock_instance)
    fake_toolbox, fake_vision = _make_fake_vision_module(mock_tracker_cls)

    for key in list(sys.modules.keys()):
        if "robot_comic.vision.head_tracking.mediapipe" in key:
            del sys.modules[key]

    with patch.dict(
        sys.modules,
        {
            "reachy_mini_toolbox": fake_toolbox,
            "reachy_mini_toolbox.vision": fake_vision,
        },
    ):
        import importlib

        import robot_comic.vision.head_tracking.mediapipe as _mod

        importlib.reload(_mod)

        tracker = _mod.MediapipeHeadTracker()

        # Simulate a test monkeypatching _tracker directly (pre-existing contract).
        pre_set_mock = MagicMock()
        tracker._tracker = pre_set_mock

        returned = tracker._ensure_tracker()

    # _ensure_tracker must return the pre-set mock without constructing a new one.
    assert returned is pre_set_mock
    assert mock_tracker_cls.call_count == 0, (
        "_ensure_tracker() must not reconstruct the tracker when _tracker is already set"
    )
