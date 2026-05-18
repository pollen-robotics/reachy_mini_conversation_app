"""Tests for the welcome-gate state machine."""

from __future__ import annotations
from pathlib import Path

import pytest

from robot_comic.welcome_gate import (
    GateState,
    WelcomeGate,
    _levenshtein,
    make_gate_for_profile,
    _names_from_profile_dir,
)


# ---------------------------------------------------------------------------
# _levenshtein helpers
# ---------------------------------------------------------------------------


def test_levenshtein_identical() -> None:
    assert _levenshtein("rickles", "rickles") == 0


def test_levenshtein_insertion() -> None:
    assert _levenshtein("rickles", "rickless") == 1


def test_levenshtein_deletion() -> None:
    assert _levenshtein("rickles", "rckles") == 1


def test_levenshtein_substitution() -> None:
    assert _levenshtein("rickles", "ricklez") == 1


def test_levenshtein_empty_strings() -> None:
    assert _levenshtein("", "") == 0
    assert _levenshtein("abc", "") == 3
    assert _levenshtein("", "abc") == 3


# ---------------------------------------------------------------------------
# WelcomeGate basic states
# ---------------------------------------------------------------------------


def test_gate_starts_in_waiting() -> None:
    gate = WelcomeGate(["rickles"])
    assert gate.state is GateState.WAITING


def test_consider_returns_false_and_stays_waiting_when_name_absent() -> None:
    gate = WelcomeGate(["rickles"])
    result = gate.consider("hello there")
    assert result is False
    assert gate.state is GateState.WAITING


def test_consider_returns_true_and_transitions_to_gated_when_name_present() -> None:
    gate = WelcomeGate(["rickles"])
    result = gate.consider("Hey Rickles, how are you?")
    assert result is True
    assert gate.state is GateState.GATED


def test_consider_returns_true_when_already_gated() -> None:
    gate = WelcomeGate(["rickles"])
    gate.consider("rickles")
    assert gate.state is GateState.GATED
    # Second call must still return True without errors.
    assert gate.consider("something else entirely") is True


def test_reset_returns_to_waiting() -> None:
    gate = WelcomeGate(["rickles"])
    gate.consider("hey rickles")
    assert gate.state is GateState.GATED
    gate.reset()
    assert gate.state is GateState.WAITING


# ---------------------------------------------------------------------------
# Exact substring matching
# ---------------------------------------------------------------------------


def test_exact_name_in_transcript() -> None:
    gate = WelcomeGate(["rickles"])
    assert gate.consider("rickles") is True


def test_exact_multi_word_name_in_transcript() -> None:
    gate = WelcomeGate(["don rickles"])
    assert gate.consider("I want to talk to Don Rickles please") is True


def test_name_not_in_transcript() -> None:
    gate = WelcomeGate(["rickles"])
    assert gate.consider("hello world") is False


# ---------------------------------------------------------------------------
# Fuzzy matching — single-word names
# ---------------------------------------------------------------------------


def test_fuzzy_one_char_typo_accepted() -> None:
    """'rickless' is distance 1 from 'rickles' — should open the gate."""
    gate = WelcomeGate(["rickles"], threshold=2)
    assert gate.consider("hey rickless, what's up?") is True


def test_fuzzy_two_char_typo_accepted_at_threshold_2() -> None:
    """'rikless' (distance 2 from 'rickles') should open the gate.

    Trailing punctuation ('!') is stripped before the distance is computed,
    so 'rikless!' becomes 'rikless' → distance 2 → accepted.
    """
    gate = WelcomeGate(["rickles"], threshold=2)
    assert gate.consider("rikless!") is True


def test_fuzzy_at_threshold_boundary() -> None:
    """A word whose edit distance is exactly the threshold is accepted."""
    gate = WelcomeGate(["rickles"], threshold=2)
    # 'riklss' has Levenshtein distance 2 from 'rickles' — accepted.
    assert gate.consider("riklss") is True


def test_fuzzy_beyond_threshold_rejected() -> None:
    """A word whose edit distance exceeds the threshold is rejected."""
    gate = WelcomeGate(["rickles"], threshold=2)
    # 'rklss' — distance 3 from 'rickles' (remove i, c→nothing, less→ss) — rejected.
    assert gate.consider("rklss") is False


def test_fuzzy_threshold_zero_exact_word_match() -> None:
    """At threshold=0, only exact word or substring matches open the gate."""
    gate = WelcomeGate(["rickles"], threshold=0)
    # 'rickles' is a substring of 'rickless', so the fast path triggers.
    assert gate.consider("rickless") is True
    # Completely different word — no match.
    gate2 = WelcomeGate(["rickles"], threshold=0)
    assert gate2.consider("george") is False


# ---------------------------------------------------------------------------
# Fuzzy matching — multi-word names
# ---------------------------------------------------------------------------


def test_fuzzy_multi_word_name_typo() -> None:
    """'don riklles' has distance 2 from 'don rickles' — accepted."""
    gate = WelcomeGate(["don rickles"], threshold=2)
    assert gate.consider("give me don riklles") is True


# ---------------------------------------------------------------------------
# Multiple names
# ---------------------------------------------------------------------------


def test_multiple_names_first_matches() -> None:
    gate = WelcomeGate(["rickles", "don rickles"])
    assert gate.consider("rickles") is True


def test_multiple_names_second_matches() -> None:
    gate = WelcomeGate(["carlin", "don rickles"])
    assert gate.consider("don rickles") is True


def test_multiple_names_none_match() -> None:
    gate = WelcomeGate(["rickles", "carlin"])
    assert gate.consider("hello") is False


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


def test_case_insensitive_match() -> None:
    gate = WelcomeGate(["Rickles"])
    assert gate.consider("RICKLES") is True


def test_case_insensitive_mixed() -> None:
    gate = WelcomeGate(["Don Rickles"])
    assert gate.consider("hey DON RICKLES") is True


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_empty_names_raises() -> None:
    with pytest.raises(ValueError, match="persona_names must not be empty"):
        WelcomeGate([])


def test_whitespace_only_names_raises() -> None:
    with pytest.raises(ValueError, match="persona_names must not be empty"):
        WelcomeGate(["  ", ""])


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def test_names_from_profile_dir_single_word(tmp_path: Path) -> None:
    profile_dir = tmp_path / "carlin"
    profile_dir.mkdir()
    names = _names_from_profile_dir(profile_dir)
    assert "carlin" in names


def test_names_from_profile_dir_multi_word(tmp_path: Path) -> None:
    profile_dir = tmp_path / "house_comedian"
    profile_dir.mkdir()
    names = _names_from_profile_dir(profile_dir)
    # Should include the last word and the full humanised name.
    assert "comedian" in names
    assert "house comedian" in names


def test_names_from_profile_dir_wake_names_file_overrides(tmp_path: Path) -> None:
    profile_dir = tmp_path / "house_comedian"
    profile_dir.mkdir()
    (profile_dir / "wake_names.txt").write_text("hey robot\ncomic\n", encoding="utf-8")
    names = _names_from_profile_dir(profile_dir)
    assert names == ["hey robot", "comic"]


def test_names_from_profile_dir_empty_wake_names_falls_back(tmp_path: Path) -> None:
    profile_dir = tmp_path / "house_comedian"
    profile_dir.mkdir()
    # Empty file → fall back to stem derivation.
    (profile_dir / "wake_names.txt").write_text("\n  \n", encoding="utf-8")
    names = _names_from_profile_dir(profile_dir)
    assert "comedian" in names


def test_make_gate_for_profile(tmp_path: Path) -> None:
    profile_dir = tmp_path / "house_comedian"
    profile_dir.mkdir()
    gate = make_gate_for_profile(profile_dir)
    assert isinstance(gate, WelcomeGate)
    assert gate.state is GateState.WAITING
    # Gate should open for "comedian" (derived from profile dir name).
    assert gate.consider("hey comedian") is True


def test_make_gate_for_profile_with_wake_names_file(tmp_path: Path) -> None:
    profile_dir = tmp_path / "mystery_persona"
    profile_dir.mkdir()
    (profile_dir / "wake_names.txt").write_text("oracle\nmystery\n", encoding="utf-8")
    gate = make_gate_for_profile(profile_dir)
    assert gate.consider("hey oracle") is True
