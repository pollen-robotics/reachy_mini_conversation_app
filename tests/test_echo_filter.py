"""Tests for the shared self-echo detection helpers (#527).

The three phrases are verbatim from the 2026-06-10 live session where the
robot (multilingual riffs) heard itself through its own mic and Gemini
transcribed the echo as user input — resetting the riff backoff each time.
"""

from __future__ import annotations

from robot_comic.echo_filter import (
    is_echo_of_recent,
    fragment_echo_ratio,
    normalize_for_echo_check,
)


# Recent assistant riffs (what the robot actually said), as the handler
# would have buffered them.
RIFF_ES = (
    "Hablemos de trabajo, amigos. ¿Qué es lo que más te gusta de tu trabajo? "
    "Yo solo cuento chistes y me pagan en electricidad."
)
RIFF_EN = (
    "This hotel, I tell ya. You know, I feel like I should just call the "
    "front desk and ask for a room with fewer existential crises."
)
RIFF_DE = "Mögen Sie Eier? Und wie braten Sie die am liebsten? Ich persönlich bevorzuge sie binär."

# What Gemini transcribed back as "user" input in the live session.
ECHO_ES = "¿Qué es lo que más te gusta de tu trabajo?"
ECHO_EN = "you know I feel like I should just call the…"
ECHO_DE = "Mögen Sie Eier? Und wie braten Sie die am liebsten?"


def _recent() -> list[str]:
    return [normalize_for_echo_check(t) for t in (RIFF_ES, RIFF_EN, RIFF_DE)]


def test_normalize_collapses_punctuation_and_case() -> None:
    assert normalize_for_echo_check("  Mögen Sie… EIER?! ") == "mögen sie eier"


def test_captured_phrases_are_flagged_as_echo() -> None:
    recent = _recent()
    for echo in (ECHO_ES, ECHO_EN, ECHO_DE):
        assert is_echo_of_recent(echo, recent, threshold=0.85), echo


def test_genuine_user_input_is_not_echo() -> None:
    recent = _recent()
    for genuine in (
        "How tall is the Eiffel Tower?",
        "Tell me a joke about robots and cheese.",
        "¿Dónde está la biblioteca más cercana?",
    ):
        assert not is_echo_of_recent(genuine, recent, threshold=0.85), genuine


def test_short_utterances_never_flag() -> None:
    # "ja" appears verbatim inside the German riff; ultra-short input must
    # still pass through (a real user saying "ok"/"ja" must reset pacing).
    assert not is_echo_of_recent("ja", _recent(), threshold=0.85)
    assert not is_echo_of_recent("ok", _recent(), threshold=0.85)


def test_partial_prefix_chunk_is_flagged() -> None:
    # Input transcription arrives in chunks; an early prefix of the echo is
    # itself a fragment of the riff and must already match.
    assert is_echo_of_recent("You know, I feel like I should", _recent(), threshold=0.85)


def test_empty_history_is_never_echo() -> None:
    assert not is_echo_of_recent(ECHO_EN, [], threshold=0.85)


def test_fragment_ratio_exact_substring_is_one() -> None:
    needle = normalize_for_echo_check(ECHO_ES)
    candidate = normalize_for_echo_check(RIFF_ES)
    assert fragment_echo_ratio(needle, candidate) == 1.0


def test_fragment_ratio_handles_minor_word_errors() -> None:
    needle = normalize_for_echo_check("you know I feel like I could just call the")
    candidate = normalize_for_echo_check(RIFF_EN)
    assert fragment_echo_ratio(needle, candidate) >= 0.85


def test_fragment_ratio_low_for_unrelated_text() -> None:
    needle = normalize_for_echo_check("completely unrelated sentence about gardening tools")
    candidate = normalize_for_echo_check(RIFF_EN)
    assert fragment_echo_ratio(needle, candidate) < 0.6
