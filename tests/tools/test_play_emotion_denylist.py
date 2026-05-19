"""Tests for the play_emotion chassis-safety denylist (issue #480).

Verifies that:
- The default denylist covers boredom1, boredom2, sleep1.
- Env override (empty, custom, whitespace-tolerant) works via monkeypatch.
- Calling play_emotion with a denylisted emotion returns a structured error.
- The tool spec returned by spec() excludes denylisted emotions.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

import robot_comic.config as config_module
import robot_comic.tools.play_emotion as pe
from robot_comic.config import PLAY_EMOTION_DENYLIST_DEFAULT, _parse_play_emotion_denylist
from robot_comic.tools.core_tools import ToolDependencies


# ---------------------------------------------------------------------------
# Helpers
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
# Default denylist values
# ---------------------------------------------------------------------------


def test_default_denylist_covers_boredom1() -> None:
    denylist = _parse_play_emotion_denylist(None)
    assert "boredom1" in denylist


def test_default_denylist_covers_boredom2() -> None:
    denylist = _parse_play_emotion_denylist(None)
    assert "boredom2" in denylist


def test_default_denylist_covers_sleep1() -> None:
    denylist = _parse_play_emotion_denylist(None)
    assert "sleep1" in denylist


def test_default_denylist_contains_exactly_three_entries() -> None:
    denylist = _parse_play_emotion_denylist(None)
    assert len(denylist) == 3


def test_config_play_emotion_denylist_default_value() -> None:
    """Config.PLAY_EMOTION_DENYLIST matches the module-level default string."""
    expected = frozenset(n.strip() for n in PLAY_EMOTION_DENYLIST_DEFAULT.split(",") if n.strip())
    assert config_module.config.PLAY_EMOTION_DENYLIST == expected


# ---------------------------------------------------------------------------
# Env override via _parse_play_emotion_denylist
# ---------------------------------------------------------------------------


def test_env_override_empty_string_clears_denylist() -> None:
    denylist = _parse_play_emotion_denylist("")
    assert denylist == frozenset()


def test_env_override_custom_list() -> None:
    denylist = _parse_play_emotion_denylist("angry1,scared2")
    assert denylist == frozenset({"angry1", "scared2"})


def test_env_override_whitespace_tolerant() -> None:
    denylist = _parse_play_emotion_denylist("  boredom1 , sleep1 , wave1  ")
    assert "boredom1" in denylist
    assert "sleep1" in denylist
    assert "wave1" in denylist
    assert len(denylist) == 3


def test_refresh_runtime_config_picks_up_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """refresh_runtime_config_from_env re-reads the env var."""
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "custom1,custom2")
    config_module.refresh_runtime_config_from_env()
    assert config_module.config.PLAY_EMOTION_DENYLIST == frozenset({"custom1", "custom2"})


def test_refresh_runtime_config_empty_env_clears_denylist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "")
    config_module.refresh_runtime_config_from_env()
    assert config_module.config.PLAY_EMOTION_DENYLIST == frozenset()


# ---------------------------------------------------------------------------
# __call__ returns a structured error for denylisted emotions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_boredom1_returns_denylist_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "wave", "boredom1", "boredom2", "sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,boredom2,sleep1")
    config_module.refresh_runtime_config_from_env()

    result = await pe.PlayEmotion()(_make_deps(), emotion="boredom1")

    assert "error" in result
    assert "denylisted" in result["error"]
    assert "boredom1" in result["error"]


@pytest.mark.asyncio
async def test_call_boredom2_returns_denylist_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "boredom2", "sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,boredom2,sleep1")
    config_module.refresh_runtime_config_from_env()

    result = await pe.PlayEmotion()(_make_deps(), emotion="boredom2")

    assert "error" in result
    assert "denylisted" in result["error"]


@pytest.mark.asyncio
async def test_call_sleep1_returns_denylist_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,boredom2,sleep1")
    config_module.refresh_runtime_config_from_env()

    result = await pe.PlayEmotion()(_make_deps(), emotion="sleep1")

    assert "error" in result
    assert "denylisted" in result["error"]


@pytest.mark.asyncio
async def test_call_allowed_emotion_queues_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-denylisted emotions still queue normally."""
    _patch_catalog(monkeypatch, ["smile1", "wave", "boredom1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1")
    config_module.refresh_runtime_config_from_env()

    # Mock the EmotionQueueMove to avoid importing dance_emotion_moves
    monkeypatch.setattr(
        "robot_comic.tools.play_emotion.EmotionQueueMove",
        MagicMock(return_value=MagicMock()),
        raising=False,
    )
    # Patch the import at the call site
    import sys

    fake_deqm = MagicMock()
    fake_deqm.EmotionQueueMove = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "robot_comic.dance_emotion_moves", fake_deqm)

    result = await pe.PlayEmotion()(_make_deps(), emotion="smile1")

    assert result.get("status") == "queued"
    assert result.get("emotion") == "smile1"


@pytest.mark.asyncio
async def test_call_denylist_error_has_chassis_safety_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error message mentions chassis safety so the LLM gets corrective signal."""
    _patch_catalog(monkeypatch, ["sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "sleep1")
    config_module.refresh_runtime_config_from_env()

    result = await pe.PlayEmotion()(_make_deps(), emotion="sleep1")

    assert "chassis safety" in result["error"]
    assert "Pick a different emotion" in result["error"]


# ---------------------------------------------------------------------------
# spec() excludes denylisted emotions from the LLM tool spec enum
# ---------------------------------------------------------------------------


def test_spec_excludes_boredom1_from_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "wave", "boredom1", "boredom2", "sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,boredom2,sleep1")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert "boredom1" not in enum_values


def test_spec_excludes_boredom2_from_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "boredom2", "sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,boredom2,sleep1")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert "boredom2" not in enum_values


def test_spec_excludes_sleep1_from_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "sleep1", "wave"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,boredom2,sleep1")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert "sleep1" not in enum_values


def test_spec_keeps_non_denylisted_emotions(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["smile1", "wave", "boredom1", "sleep1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "boredom1,sleep1")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert "smile1" in enum_values
    assert "wave" in enum_values


def test_spec_empty_denylist_exposes_all_emotions(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["boredom1", "boredom2", "sleep1", "smile1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert set(enum_values) == {"boredom1", "boredom2", "sleep1", "smile1"}


def test_spec_custom_denylist_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, ["angry1", "scared2", "smile1"])
    monkeypatch.setenv("REACHY_MINI_PLAY_EMOTION_DENYLIST", "angry1,scared2")
    config_module.refresh_runtime_config_from_env()

    spec = pe.PlayEmotion().spec()
    enum_values: list[str] = spec["parameters"]["properties"]["emotion"]["enum"]

    assert "angry1" not in enum_values
    assert "scared2" not in enum_values
    assert "smile1" in enum_values
