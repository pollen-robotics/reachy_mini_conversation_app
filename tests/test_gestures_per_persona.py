"""Tests for per-persona gesture-beat mappings.

All tests are pure-Python: no robot hardware, no SDK I/O, no network.
"""

from __future__ import annotations
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[1].resolve()
PROFILES_DIR = PROJECT_ROOT / "profiles"

# The personas shipped in this repository.
PERSONAS = ["house_comedian"]

# All canonical gesture names from gestures.py
_CANONICAL_GESTURES = frozenset({"shrug", "nod_yes", "nod_no", "point_left", "point_right", "scan", "lean_in"})


# ---------------------------------------------------------------------------
# beats.py — constants
# ---------------------------------------------------------------------------


class TestBeatsModule:
    """beats.py exports the expected beat name constants."""

    def test_all_beats_frozenset_is_not_empty(self) -> None:
        from robot_comic.gestures.beats import ALL_BEATS

        assert len(ALL_BEATS) >= 12

    def test_expected_beat_names_present(self) -> None:
        from robot_comic.gestures.beats import (
            ALL_BEATS,
            BEAT_DEFEAT,
            BEAT_SWAGGER,
            BEAT_SURPRISE,
            BEAT_AGREEMENT,
            BEAT_DISMISSAL,
            BEAT_REFLECTION,
            BEAT_DISAPPROVAL,
            BEAT_VULNERABILITY,
            BEAT_PUNCHLINE_DROP,
            BEAT_ACKNOWLEDGEMENT,
            BEAT_PUNCHLINE_SETUP,
            BEAT_CHARACTER_SWITCH,
        )

        expected = {
            BEAT_DISAPPROVAL,
            BEAT_AGREEMENT,
            BEAT_REFLECTION,
            BEAT_CHARACTER_SWITCH,
            BEAT_PUNCHLINE_SETUP,
            BEAT_PUNCHLINE_DROP,
            BEAT_DISMISSAL,
            BEAT_ACKNOWLEDGEMENT,
            BEAT_SURPRISE,
            BEAT_DEFEAT,
            BEAT_SWAGGER,
            BEAT_VULNERABILITY,
        }
        assert expected.issubset(ALL_BEATS)

    def test_beat_constant_values_are_strings(self) -> None:
        from robot_comic.gestures import beats

        for attr in dir(beats):
            if attr.startswith("BEAT_"):
                val = getattr(beats, attr)
                assert isinstance(val, str), f"{attr} should be a str"


# ---------------------------------------------------------------------------
# load_persona_beats
# ---------------------------------------------------------------------------


class TestLoadPersonaBeats:
    """load_persona_beats reads gestures.txt correctly."""

    def test_load_house_comedian_returns_expected_map(self) -> None:
        from robot_comic.gestures.registry import load_persona_beats

        beats = load_persona_beats(PROFILES_DIR / "house_comedian")
        assert isinstance(beats, dict)
        assert len(beats) == 12
        assert beats["punchline_setup"] == "lean_in"
        assert beats["punchline_drop"] == "shrug"
        assert beats["dismissal"] == "point_left"
        assert beats["swagger"] == "point_right"

    def test_absent_file_returns_empty_dict(self, tmp_path: Path) -> None:
        from robot_comic.gestures.registry import load_persona_beats

        result = load_persona_beats(tmp_path / "nonexistent_persona")
        assert result == {}

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        from robot_comic.gestures.registry import load_persona_beats

        result = load_persona_beats(tmp_path)
        assert result == {}

    def test_comments_and_blank_lines_ignored(self, tmp_path: Path) -> None:
        from robot_comic.gestures.registry import load_persona_beats

        gestures_file = tmp_path / "gestures.txt"
        gestures_file.write_text(
            "# this is a comment\n\ndisapproval=shrug\n\n# another comment\nagreement=nod_yes\n",
            encoding="utf-8",
        )
        result = load_persona_beats(tmp_path)
        assert result == {"disapproval": "shrug", "agreement": "nod_yes"}

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        from robot_comic.gestures.registry import load_persona_beats

        gestures_file = tmp_path / "gestures.txt"
        gestures_file.write_text(
            "disapproval=shrug\nnot_a_valid_line\nagreement=nod_yes\n",
            encoding="utf-8",
        )
        result = load_persona_beats(tmp_path)
        # Only valid lines returned; malformed line skipped
        assert result == {"disapproval": "shrug", "agreement": "nod_yes"}


# ---------------------------------------------------------------------------
# GestureRegistry.resolve_for_persona
# ---------------------------------------------------------------------------


class TestResolveForPersona:
    """resolve_for_persona maps beats to canonical gestures."""

    def _house_comedian_beats(self) -> dict:
        from robot_comic.gestures.registry import load_persona_beats

        return load_persona_beats(PROFILES_DIR / "house_comedian")

    def test_punchline_setup_maps_to_lean_in(self) -> None:
        from robot_comic.gestures import registry

        beats = self._house_comedian_beats()
        result = registry.resolve_for_persona("punchline_setup", beats)
        assert result == "lean_in"

    def test_punchline_drop_maps_to_shrug(self) -> None:
        from robot_comic.gestures import registry

        beats = self._house_comedian_beats()
        result = registry.resolve_for_persona("punchline_drop", beats)
        assert result == "shrug"

    def test_dismissal_maps_to_point_left(self) -> None:
        from robot_comic.gestures import registry

        beats = self._house_comedian_beats()
        result = registry.resolve_for_persona("dismissal", beats)
        assert result == "point_left"

    def test_unknown_beat_raises_key_error(self) -> None:
        from robot_comic.gestures import registry

        beats = self._house_comedian_beats()
        with pytest.raises(KeyError) as exc_info:
            registry.resolve_for_persona("nonexistent_beat", beats)
        # The error message must list available beats
        assert "nonexistent_beat" in str(exc_info.value)
        # Should mention at least one known beat
        assert "disapproval" in str(exc_info.value)

    def test_unknown_beat_error_message_lists_available_beats(self) -> None:
        from robot_comic.gestures import registry

        beats_map = {"disapproval": "shrug", "agreement": "nod_yes"}
        with pytest.raises(KeyError) as exc_info:
            registry.resolve_for_persona("swagger", beats_map)
        msg = str(exc_info.value)
        assert "swagger" in msg
        assert "disapproval" in msg
        assert "agreement" in msg

    def test_resolve_returns_canonical_gesture_string(self) -> None:
        from robot_comic.gestures import registry
        from robot_comic.gestures.registry import load_persona_beats

        for persona in PERSONAS:
            persona_beats = load_persona_beats(PROFILES_DIR / persona)
            for beat, gesture in persona_beats.items():
                result = registry.resolve_for_persona(beat, persona_beats)
                assert isinstance(result, str)
                assert result == gesture


# ---------------------------------------------------------------------------
# File existence — all personas have gestures.txt
# ---------------------------------------------------------------------------


class TestGesturesFileExistence:
    """Every persona must have a gestures.txt file."""

    @pytest.mark.parametrize("persona", PERSONAS)
    def test_gestures_txt_exists(self, persona: str) -> None:
        gestures_file = PROFILES_DIR / persona / "gestures.txt"
        assert gestures_file.exists(), f"Missing gestures.txt for persona {persona!r} at {gestures_file}"

    @pytest.mark.parametrize("persona", PERSONAS)
    def test_gestures_txt_is_not_empty(self, persona: str) -> None:
        gestures_file = PROFILES_DIR / persona / "gestures.txt"
        content = gestures_file.read_text(encoding="utf-8")
        # Strip comments and blank lines
        data_lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        assert len(data_lines) > 0, f"gestures.txt for {persona!r} has no data lines"


# ---------------------------------------------------------------------------
# All mappings use only canonical gesture names
# ---------------------------------------------------------------------------


class TestMappingsUseCanonicalGestures:
    """Every value in every persona's gestures.txt must be a canonical gesture name."""

    @pytest.mark.parametrize("persona", PERSONAS)
    def test_all_gesture_values_are_canonical(self, persona: str) -> None:
        from robot_comic.gestures.registry import load_persona_beats

        persona_beats = load_persona_beats(PROFILES_DIR / persona)
        for beat, gesture in persona_beats.items():
            assert gesture in _CANONICAL_GESTURES, (
                f"Persona {persona!r} maps beat {beat!r} → {gesture!r}, "
                f"which is not a canonical gesture. "
                f"Valid names: {sorted(_CANONICAL_GESTURES)}"
            )


# ---------------------------------------------------------------------------
# house_comedian covers all canonical beats
# ---------------------------------------------------------------------------


class TestHouseComedianCoverage:
    """house_comedian gestures.txt covers all 12 standard beats."""

    def test_covers_all_standard_beats(self) -> None:
        from robot_comic.gestures import beats as beats_mod
        from robot_comic.gestures.registry import load_persona_beats

        persona_beats = load_persona_beats(PROFILES_DIR / "house_comedian")
        standard_beats = {getattr(beats_mod, attr) for attr in dir(beats_mod) if attr.startswith("BEAT_")}
        missing = standard_beats - persona_beats.keys()
        assert not missing, f"house_comedian is missing beat mappings: {sorted(missing)}"

    def test_uses_multiple_gesture_types(self) -> None:
        """house_comedian should use more than one unique gesture — it is not all-shrug."""
        from robot_comic.gestures.registry import load_persona_beats

        persona_beats = load_persona_beats(PROFILES_DIR / "house_comedian")
        unique_gestures = set(persona_beats.values())
        assert len(unique_gestures) >= 4, (
            f"house_comedian only uses {len(unique_gestures)} unique gesture(s); expected at least 4"
        )
