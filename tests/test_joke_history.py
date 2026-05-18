"""Tests for joke_history module."""

from __future__ import annotations
import os
import json
import math
from typing import Any
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from robot_comic.joke_history import (
    _DECAY_TAU_DAYS,
    JokeHistory,
    last_sentence_of,
    default_history_path,
)


# ---------------------------------------------------------------------------
# last_sentence_of
# ---------------------------------------------------------------------------


def test_last_sentence_single_terminator() -> None:
    assert last_sentence_of("Hello. World!") == "World!"


def test_last_sentence_no_terminator() -> None:
    assert last_sentence_of("Just one") == "Just one"


def test_last_sentence_empty() -> None:
    assert last_sentence_of("") == ""


def test_last_sentence_whitespace_only() -> None:
    assert last_sentence_of("   ") == ""


def test_last_sentence_multiple_terminators() -> None:
    result = last_sentence_of("First. Second? Third!")
    assert result == "Third!"


def test_last_sentence_strips_leading_whitespace() -> None:
    assert last_sentence_of("  Hello.  World!  ") == "World!"


# ---------------------------------------------------------------------------
# JokeHistory.load
# ---------------------------------------------------------------------------


def test_load_missing_file(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "nonexistent.json")
    assert history.load() == []


def test_load_empty_entries(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("[]", encoding="utf-8")
    history = JokeHistory(path)
    assert history._entries == []


def test_load_existing_entries(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    entries = [{"ts": "2024-01-01T00:00:00+00:00", "punchline": "Nice tie!", "topic": "fashion"}]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    assert len(history._entries) == 1
    assert history._entries[0]["punchline"] == "Nice tie!"


def test_load_corrupt_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("not json{{{{", encoding="utf-8")
    history = JokeHistory(path)
    assert history._entries == []


def test_load_wrong_type_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    history = JokeHistory(path)
    assert history._entries == []


# ---------------------------------------------------------------------------
# JokeHistory.add / truncation
# ---------------------------------------------------------------------------


def test_add_appends_entry(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json", max_entries=50)
    history.add("You're so ugly, mirrors refuse to look at you!")
    assert len(history._entries) == 1
    assert history._entries[0]["punchline"] == "You're so ugly, mirrors refuse to look at you!"
    assert history._entries[0]["topic"] == ""


def test_add_with_topic(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("Great haircut!", topic="appearance")
    assert history._entries[0]["topic"] == "appearance"


def test_add_with_persona(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("Hockey puck!", topic="sports", persona="house_comedian")
    assert history._entries[0]["persona"] == "house_comedian"


def test_add_truncates_beyond_max(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json", max_entries=3)
    for i in range(5):
        history.add(f"joke {i}")
    assert len(history._entries) == 3
    # Only the last 3 should remain
    punchlines = [e["punchline"] for e in history._entries]
    assert punchlines == ["joke 2", "joke 3", "joke 4"]


def test_add_empty_punchline_skipped(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("")
    history.add("   ")
    assert history._entries == []


def test_add_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = JokeHistory(path)
    history.add("Saved joke!")
    # Read back raw
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["punchline"] == "Saved joke!"


# ---------------------------------------------------------------------------
# JokeHistory.save (atomic write)
# ---------------------------------------------------------------------------


def test_save_writes_correct_content(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = JokeHistory(path)
    entries = [{"ts": "2024-01-01T00:00:00+00:00", "punchline": "atomic!", "topic": ""}]
    history.save(entries)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == entries


def test_save_is_atomic(tmp_path: Path) -> None:
    """Verify atomic write: replace is called, not a direct open-write."""
    path = tmp_path / "history.json"
    history = JokeHistory(path)
    replace_calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def _mock_replace(src: str, dst: str) -> None:
        replace_calls.append((src, dst))
        original_replace(src, dst)

    with patch("os.replace", side_effect=_mock_replace):
        history.save([{"ts": "x", "punchline": "test", "topic": ""}])

    assert len(replace_calls) == 1
    _, dst = replace_calls[0]
    assert Path(dst) == path


# ---------------------------------------------------------------------------
# JokeHistory.recent
# ---------------------------------------------------------------------------


def test_recent_returns_last_n(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    for i in range(15):
        history.add(f"joke {i}")
    recent = history.recent(n=5)
    assert len(recent) == 5
    punchlines = [e["punchline"] for e in recent]
    assert punchlines == ["joke 10", "joke 11", "joke 12", "joke 13", "joke 14"]


def test_recent_returns_all_when_fewer_than_n(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("only one")
    recent = history.recent(n=10)
    assert len(recent) == 1


def test_recent_empty_history(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    assert history.recent() == []


def test_recent_preserves_order(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("first")
    history.add("second")
    history.add("third")
    recent = history.recent(n=3)
    punchlines = [e["punchline"] for e in recent]
    assert punchlines == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# JokeHistory.format_for_prompt
# ---------------------------------------------------------------------------


def test_format_for_prompt_empty_returns_empty_string(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    assert history.format_for_prompt() == ""


def test_format_for_prompt_nonempty_contains_bullets(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("Nice tie!")
    history.add("Your hair is thinning faster than my patience.")
    result = history.format_for_prompt()
    assert result != ""
    assert "## RECENT JOKES (DO NOT REPEAT)" in result
    assert "- Nice tie!" in result
    assert "- Your hair is thinning faster than my patience." in result


def test_format_for_prompt_respects_n(tmp_path: Path) -> None:
    # Build entries with controlled timestamps so that jokes 7, 8, 9 are
    # definitively the most-recent (highest weight) and will be selected by
    # format_for_prompt's weight-descending sort.  Jokes 0-6 are spaced 1 hour
    # apart going back in time, giving them strictly lower weight.
    path = tmp_path / "history.json"
    now = datetime.now(timezone.utc)
    entries = [
        {"ts": (now - timedelta(hours=10 - i)).isoformat(), "punchline": f"joke {i}", "topic": "", "persona": ""}
        for i in range(10)
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt(n=3)
    # Only the last 3 (most-recent by timestamp = highest weight) should appear
    assert "joke 7" in result
    assert "joke 8" in result
    assert "joke 9" in result
    assert "joke 0" not in result


def test_format_for_prompt_one_bullet_per_entry(tmp_path: Path) -> None:
    history = JokeHistory(tmp_path / "history.json")
    history.add("alpha")
    history.add("beta")
    result = history.format_for_prompt()
    assert result.count("- alpha") == 1
    assert result.count("- beta") == 1


# ---------------------------------------------------------------------------
# default_history_path
# ---------------------------------------------------------------------------


def test_default_history_path_returns_path_under_home(tmp_path: Path) -> None:
    with patch("robot_comic.joke_history._DEFAULT_HISTORY_DIR", tmp_path / ".robot-comic"):
        path = default_history_path()
    assert path.name == "joke-history.json"
    assert (tmp_path / ".robot-comic").is_dir()


# ---------------------------------------------------------------------------
# Cross-persona dedup in format_for_prompt
# ---------------------------------------------------------------------------


def _make_entry(punchline: str, topic: str = "", persona: str = "", age_hours: float = 0.0) -> dict:
    """Helper: build a history entry dict with controlled timestamp."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    return {"ts": ts, "punchline": punchline, "topic": topic, "persona": persona}


def test_format_for_prompt_cross_persona(tmp_path: Path) -> None:
    """format_for_prompt must include entries from multiple personas."""
    path = tmp_path / "history.json"
    entries = [
        _make_entry("Hockey puck!", topic="sports", persona="house_comedian", age_hours=1),
        _make_entry("Why so serious?", topic="mood", persona="joker", age_hours=2),
        _make_entry("You're the best audience I've had all week!", persona="carlin", age_hours=3),
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt()
    assert "[house_comedian] Hockey puck!" in result
    assert "[joker] Why so serious?" in result
    assert "You're the best audience" in result


def test_format_for_prompt_cross_persona_persona_prefix(tmp_path: Path) -> None:
    """Entries with a persona should be prefixed with [persona] in the output."""
    path = tmp_path / "history.json"
    entries = [_make_entry("Nice tie!", persona="house_comedian", age_hours=0.1)]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt()
    assert "- [house_comedian] Nice tie!" in result


def test_format_for_prompt_no_persona_prefix_when_empty(tmp_path: Path) -> None:
    """Entries with no persona should not have a prefix in the output."""
    path = tmp_path / "history.json"
    entries = [_make_entry("Timeless joke.", persona="", age_hours=0.1)]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt()
    # Should appear without prefix brackets
    assert "- Timeless joke." in result
    assert "[" not in result.split("Timeless")[0].split("\n")[-1]


def test_format_for_prompt_topic_suffix(tmp_path: Path) -> None:
    """Entries with a topic should include (topic: ...) suffix."""
    path = tmp_path / "history.json"
    entries = [_make_entry("Hockey puck!", topic="sports", persona="house_comedian", age_hours=0.1)]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt()
    assert "(topic: sports)" in result


# ---------------------------------------------------------------------------
# Time-decay weighting in format_for_prompt
# ---------------------------------------------------------------------------


def test_format_for_prompt_time_decay_excludes_old_entries(tmp_path: Path) -> None:
    """An entry older than ~17 days (weight < 0.1 at τ=7d) must be excluded."""
    path = tmp_path / "history.json"
    # At τ=7 days, age=14 days → weight = exp(-14/7) ≈ 0.135 (above threshold).
    # At age=17.03 days → weight = exp(-17.03/7) ≈ 0.0999 (just below 0.1).
    # Use 20 days to be safely below the threshold.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    entries = [
        {"ts": old_ts, "punchline": "Stale joke from 3 weeks ago.", "topic": "", "persona": ""},
        _make_entry("Fresh joke.", age_hours=1),
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt(min_weight_threshold=0.1)
    assert "Stale joke" not in result
    assert "Fresh joke." in result


def test_format_for_prompt_time_decay_includes_recent_entries(tmp_path: Path) -> None:
    """An entry 14 days old (weight ≈ 0.135) must be included with default threshold=0.1."""
    path = tmp_path / "history.json"
    ts_14d = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    entries = [{"ts": ts_14d, "punchline": "Two-week-old zinger.", "topic": "", "persona": ""}]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt(min_weight_threshold=0.1)
    # exp(-14/7) ≈ 0.135 > 0.1 — should be included.
    assert "Two-week-old zinger." in result


def test_format_for_prompt_time_decay_weight_formula(tmp_path: Path) -> None:
    """Verify the decay weight formula: exp(-age_days / tau)."""
    age_days = 7.0
    expected_weight = math.exp(-age_days / _DECAY_TAU_DAYS)
    # At exactly one τ (7 days) weight should be ~0.368
    assert abs(expected_weight - math.exp(-1.0)) < 1e-9


def test_format_for_prompt_decay_n_limits_output(tmp_path: Path) -> None:
    """format_for_prompt(n=2) returns at most 2 entries even if more qualify."""
    path = tmp_path / "history.json"
    entries = [_make_entry(f"joke {i}", age_hours=i) for i in range(5)]
    path.write_text(json.dumps(entries), encoding="utf-8")
    history = JokeHistory(path)
    result = history.format_for_prompt(n=2)
    # Count bullet lines
    bullets = [line for line in result.splitlines() if line.startswith("- ")]
    assert len(bullets) == 2


# ---------------------------------------------------------------------------
# record_joke_history — Lifecycle Hook #4 (#337)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Stand-in for the config singleton used by record_joke_history."""

    JOKE_HISTORY_ENABLED: bool = True
    REACHY_MINI_CUSTOM_PROFILE: str = "rickles"


def _install_fake_config(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> _FakeConfig:
    """Install a fake config so record_joke_history reads predictable values.

    The helper imports ``config`` locally to avoid a module-load cycle, so
    we have to patch the source module attribute rather than the helper's
    reference.
    """
    from robot_comic import config as cfg_mod

    fake = _FakeConfig()
    for k, v in overrides.items():
        setattr(fake, k, v)
    monkeypatch.setattr(cfg_mod, "config", fake)
    return fake


class _RecordingAdd:
    """Replacement for JokeHistory(...).add — records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, *, path: Path) -> "_RecordingJokeHistory":
        return _RecordingJokeHistory(self.calls)


class _RecordingJokeHistory:
    """Stand-in JokeHistory whose ``add`` records to a shared list."""

    def __init__(self, calls: list[tuple[str, str, str]]) -> None:
        self._calls = calls

    def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
        self._calls.append((punchline, topic, persona))


@pytest.mark.asyncio
async def test_record_joke_history_disabled_skips_extraction_and_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``JOKE_HISTORY_ENABLED=False`` the helper is a no-op."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch, JOKE_HISTORY_ENABLED=False)

    extract_calls: list[str] = []

    async def _no_extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        extract_calls.append(text)
        return {"punchline": "ignored", "topic": "x"}

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _no_extract)

    add_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(mod, "JokeHistory", _RecordingAdd())

    await mod.record_joke_history("That's the joke!")

    assert extract_calls == []
    assert add_calls == []


@pytest.mark.asyncio
async def test_record_joke_history_empty_text_skips_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty / whitespace-only text never triggers extraction."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch)

    extract_calls: list[str] = []

    async def _no_extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        extract_calls.append(text)
        return None

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _no_extract)

    await mod.record_joke_history("")
    await mod.record_joke_history("   ")

    assert extract_calls == []


@pytest.mark.asyncio
async def test_record_joke_history_success_path_calls_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty punchline goes through ``JokeHistory.add`` with correct args."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch, REACHY_MINI_CUSTOM_PROFILE="rickles")

    async def _extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        return {"punchline": "You look like a hockey puck.", "topic": "appearance"}

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _extract)

    add_calls: list[tuple[str, str, str]] = []

    class _FakeJH:
        def __init__(self, path: Path) -> None:
            self._path = path

        def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
            add_calls.append((punchline, topic, persona))

    monkeypatch.setattr(mod, "JokeHistory", _FakeJH)

    await mod.record_joke_history("Long assistant text ending in a punchline.")

    assert add_calls == [("You look like a hockey puck.", "appearance", "rickles")]


@pytest.mark.asyncio
async def test_record_joke_history_none_punchline_skips_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When extraction reports the text is setup/banter, no add fires."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch)

    async def _extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        return {"punchline": None, "topic": ""}

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _extract)

    add_calls: list[tuple[str, str, str]] = []

    class _FakeJH:
        def __init__(self, path: Path) -> None:
            pass

        def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
            add_calls.append((punchline, topic, persona))

    monkeypatch.setattr(mod, "JokeHistory", _FakeJH)

    await mod.record_joke_history("Setup line with no punch.")

    assert add_calls == []


@pytest.mark.asyncio
async def test_record_joke_history_extraction_exception_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction raising must not propagate — joke history is best-effort."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch)

    async def _broken_extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        raise RuntimeError("upstream blew up")

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _broken_extract)

    # If it raises, this test fails — pytest catches the exception out of band.
    await mod.record_joke_history("anything")


@pytest.mark.asyncio
async def test_record_joke_history_add_exception_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JokeHistory.add raising must not propagate."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch)

    async def _extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        return {"punchline": "boom", "topic": ""}

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _extract)

    class _BoomJH:
        def __init__(self, path: Path) -> None:
            pass

        def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
            raise OSError("disk full")

    monkeypatch.setattr(mod, "JokeHistory", _BoomJH)

    await mod.record_joke_history("anything")


@pytest.mark.asyncio
async def test_record_joke_history_persona_defaults_to_config_custom_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When persona arg is omitted, helper reads ``config.REACHY_MINI_CUSTOM_PROFILE``."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch, REACHY_MINI_CUSTOM_PROFILE="hicks")

    async def _extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        return {"punchline": "p", "topic": "t"}

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _extract)

    add_calls: list[tuple[str, str, str]] = []

    class _FakeJH:
        def __init__(self, path: Path) -> None:
            pass

        def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
            add_calls.append((punchline, topic, persona))

    monkeypatch.setattr(mod, "JokeHistory", _FakeJH)

    await mod.record_joke_history("text")

    assert add_calls == [("p", "t", "hicks")]


@pytest.mark.asyncio
async def test_record_joke_history_explicit_persona_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit persona kwarg wins over the config default."""
    from robot_comic import joke_history as mod

    _install_fake_config(monkeypatch, REACHY_MINI_CUSTOM_PROFILE="default")

    async def _extract(text: str, http: Any, *, llama_url: str = "") -> dict[str, Any] | None:
        return {"punchline": "p", "topic": ""}

    monkeypatch.setattr(mod, "extract_punchline_via_llm", _extract)

    add_calls: list[tuple[str, str, str]] = []

    class _FakeJH:
        def __init__(self, path: Path) -> None:
            pass

        def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
            add_calls.append((punchline, topic, persona))

    monkeypatch.setattr(mod, "JokeHistory", _FakeJH)

    await mod.record_joke_history("text", persona="explicit")

    assert add_calls == [("p", "", "explicit")]
