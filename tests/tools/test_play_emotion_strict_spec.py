"""Tests for the play_emotion strict-spec contract (issue #484).

Verifies that:
- ``spec()`` populates the ``emotion`` enum from the post-denylist catalog.
- An unknown emotion name is rejected with a WARNING log and a nearest-match
  hint in the error string (via difflib.get_close_matches).
- A valid emotion still queues the move and returns the success status shape.
"""

from __future__ import annotations
import sys
import logging
from unittest.mock import MagicMock

import pytest

import robot_comic.config as config_module
import robot_comic.tools.play_emotion as pe
from robot_comic.tools.core_tools import ToolDependencies


# ---------------------------------------------------------------------------
# Helpers (mirrors test_play_emotion_denylist.py)
# ---------------------------------------------------------------------------


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _patch_catalog(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    """Inject a known catalog into play_emotion without touching the cache file."""
    monkeypatch.setattr(pe, "_CATALOG_NAMES", list(names))
    monkeypatch.setattr(pe, "_CATALOG_DESCRIPTIONS", {n: f"desc-{n}" for n in names})
    monkeypatch.setattr(pe, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(pe, "RECORDED_MOVES", MagicMock())


# ---------------------------------------------------------------------------
# 1) Catalog-driven enum: spec() enum must equal the post-denylist catalog
# ---------------------------------------------------------------------------


def test_spec_enum_is_sourced_from_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``emotion`` enum exposed via spec() is exactly _safe_emotion_names()."""
    _patch_catalog(monkeypatch, ["smile1", "frown2", "wave", "boredom1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert enum_values == pe._safe_emotion_names()
    assert enum_values == ["smile1", "frown2", "wave"]


# ---------------------------------------------------------------------------
# 2) Unknown emotions are rejected, log WARNING, and suggest the nearest valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_emotion_is_rejected_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown emotion returns an error with a 'Did you mean' suggestion
    and emits a WARNING log naming the bad input.
    """
    _patch_catalog(monkeypatch, ["smile1", "frown2"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "")
    config_module.refresh_runtime_config_from_env()

    caplog.set_level(logging.WARNING, logger="robot_comic.tools.play_emotion")

    result = await pe.PlayEmotion()(_make_deps(), emotion="thinking1")

    # (a) Error return shape
    assert "error" in result
    # (b) Did-you-mean hint is present
    assert "Did you mean" in result["error"]
    # (c) WARNING log fired with the bad name
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("unknown emotion" in r.getMessage() and "thinking1" in r.getMessage() for r in warning_records), (
        f"Expected a WARNING mentioning 'unknown emotion' and 'thinking1'. Got: {[r.getMessage() for r in warning_records]}"
    )


# ---------------------------------------------------------------------------
# 3) Valid emotion still queues and returns the success status shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_emotion_queues_and_returns_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known emotion calls queue_move once and returns {status, emotion}."""
    _patch_catalog(monkeypatch, ["smile1", "frown2"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "")
    config_module.refresh_runtime_config_from_env()

    # Mock EmotionQueueMove via the import-site path (matches the denylist test)
    fake_deqm = MagicMock()
    fake_deqm.EmotionQueueMove = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "robot_comic.dance_emotion_moves", fake_deqm)

    movement_manager = MagicMock()
    movement_manager.speed_factor = 1.0
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    result = await pe.PlayEmotion()(deps, emotion="smile1")

    assert result == {"status": "queued", "emotion": "smile1"}
    movement_manager.queue_move.assert_called_once()
