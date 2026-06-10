import logging
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import play_emotion as play_emotion_module
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.play_emotion import (
    EMOTION_INTENTS,
    PlayEmotion,
    resolve_emotion_name,
    random_curated_emotion,
)


AVAILABLE_EMOTIONS = [
    "cheerful1",
    "confused1",
    "no1",
    "no_sad1",
    "no_excited1",
    "resigned1",
    "understanding2",
    "yes_sad1",
]


def test_play_emotion_schema_uses_compact_intents() -> None:
    """Expose compact intents instead of the full recorded-move catalog."""
    emotion_schema = PlayEmotion.parameters_schema["properties"]["emotion"]

    assert emotion_schema["enum"] == list(EMOTION_INTENTS)
    assert "no_sad" in emotion_schema["enum"]
    assert "no_excited" in emotion_schema["enum"]
    assert "no_firm" in emotion_schema["enum"]
    assert "yes_understanding" in emotion_schema["enum"]
    assert "no_confused" not in emotion_schema["enum"]
    assert "oops" not in emotion_schema["enum"]
    assert "yes_sad" not in emotion_schema["enum"]
    assert "yes_proud" not in emotion_schema["enum"]
    assert "loving1" not in emotion_schema["enum"]
    assert "Available emotions" not in emotion_schema["description"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("no_sad1", "no_sad1"),
        ("sad no", "no_sad1"),
        ("no_excited", "no_excited1"),
        ("yes_understanding", "understanding2"),
    ],
)
def test_resolve_emotion_name_accepts_ids_intents_and_yes_no_phrases(requested: str, expected: str) -> None:
    """Resolve exact IDs, compact intents, and exposed yes/no phrase variants."""
    assert resolve_emotion_name(requested, AVAILABLE_EMOTIONS) == expected


def test_resolve_emotion_name_returns_none_for_random_or_unknown() -> None:
    """Let the caller choose a random fallback when there is no resolved match."""
    assert resolve_emotion_name("random", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("contento", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("totally mysterious mood", AVAILABLE_EMOTIONS) is None


@pytest.mark.parametrize(
    "removed_intent",
    [
        "confused no",
        "curious",
        "inquiring",
        "lost",
        "no_confused",
        "oops",
        "proud",
        "uncomfortable",
        "yes proud",
        "yes sad",
        "yes_proud",
        "yes_sad",
    ],
)
def test_resolve_emotion_name_does_not_accept_removed_substitute_intents(removed_intent: str) -> None:
    """Removed intents should not resolve through unrelated substitute moves."""
    assert resolve_emotion_name(removed_intent, AVAILABLE_EMOTIONS) is None


@pytest.mark.parametrize(
    ("intent", "poor_options"),
    [
        ("excited", ["success2"]),
        ("grateful", ["helpful1", "loving1"]),
        ("happy", ["loving1"]),
        ("lonely", ["sad1"]),
        ("no", ["no_sad1", "no_excited1"]),
        ("no_excited", ["no1"]),
        ("no_sad", ["downcast1"]),
        ("uncertain", ["resigned1"]),
        ("yes_understanding", ["yes1"]),
    ],
)
def test_resolve_emotion_name_does_not_use_weak_fallbacks(intent: str, poor_options: list[str]) -> None:
    """Do not use loosely related moves when a precise move is unavailable."""
    assert resolve_emotion_name(intent, poor_options) is None


@pytest.mark.parametrize("bad_move", ["cheerful1", "oops1", "oops2", "reprimand3", "understanding1", "yes_sad1"])
def test_resolve_emotion_name_does_not_accept_bad_exact_moves(bad_move: str) -> None:
    """Bad-quality recorded move IDs should not bypass the curated resolver."""
    assert resolve_emotion_name(bad_move, [*AVAILABLE_EMOTIONS, bad_move]) is None


@pytest.mark.parametrize(
    "ambiguous_move",
    [
        "contempt1",
        "curious1",
        "dance1",
        "furious1",
        "helpful2",
        "impatient1",
        "incomprehensible2",
        "inquiring1",
        "lost1",
        "proud2",
        "proud3",
        "tired1",
        "uncomfortable1",
        "welcoming1",
    ],
)
def test_resolve_emotion_name_does_not_accept_redundant_ambiguous_exact_moves(ambiguous_move: str) -> None:
    """OK ambiguous moves should be skipped when clear or excellent alternatives exist."""
    assert resolve_emotion_name(ambiguous_move, [*AVAILABLE_EMOTIONS, ambiguous_move]) is None


def test_random_curated_emotion_uses_curated_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random fallback should avoid non-curated moves when curated options exist."""
    choices_seen: list[str] = []

    def fake_choice(choices: list[str]) -> str:
        choices_seen.extend(choices)
        return choices[0]

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    assert random_curated_emotion(["cheerful1", "yes_sad1", "confused1"]) == "confused1"
    assert choices_seen == ["confused1"]


def test_random_curated_emotion_falls_back_when_no_curated_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should still return an available move if the curated pool is unavailable."""
    choices_seen: list[str] = []

    def fake_choice(choices: list[str]) -> str:
        choices_seen.extend(choices)
        return choices[0]

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    assert random_curated_emotion(["cheerful1"]) == "cheerful1"
    assert choices_seen == ["cheerful1"]


@pytest.mark.asyncio
async def test_play_emotion_queues_resolved_emotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should queue the resolved recorded-move ID."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name
            self.recorded_moves = recorded_moves

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "RECORDED_MOVES", FakeRecordedMoves())
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    result = await PlayEmotion()(deps, emotion="sad no")

    assert result == {"status": "queued", "emotion": "no_sad1"}
    queued_move = movement_manager.queue_move.call_args.args[0]
    assert queued_move.emotion_name == "no_sad1"


@pytest.mark.asyncio
async def test_play_emotion_queues_random_for_unknown_emotion(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown explicit values should fall back to a random recorded emotion."""

    class FakeRecordedMoves:
        def list_moves(self) -> list[str]:
            return AVAILABLE_EMOTIONS

    class FakeEmotionQueueMove:
        def __init__(self, emotion_name: str, recorded_moves: FakeRecordedMoves) -> None:
            self.emotion_name = emotion_name
            self.recorded_moves = recorded_moves

    monkeypatch.setattr(play_emotion_module, "EMOTION_AVAILABLE", True)
    monkeypatch.setattr(play_emotion_module, "RECORDED_MOVES", FakeRecordedMoves())
    monkeypatch.setattr(play_emotion_module, "EmotionQueueMove", FakeEmotionQueueMove)

    def fake_choice(emotion_names: list[str]) -> str:
        assert "cheerful1" not in emotion_names
        assert "yes_sad1" not in emotion_names
        return "confused1"

    monkeypatch.setattr(play_emotion_module.random, "choice", fake_choice)

    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)

    with caplog.at_level(logging.INFO, logger=play_emotion_module.logger.name):
        result = await PlayEmotion()(deps, emotion="contento")

    assert result == {"status": "queued", "emotion": "confused1"}
    assert "play_emotion: 'contento' did not resolve; using random curated" in caplog.text
    queued_move = movement_manager.queue_move.call_args.args[0]
    assert queued_move.emotion_name == "confused1"
