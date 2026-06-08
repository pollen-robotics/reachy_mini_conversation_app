from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import play_emotion as play_emotion_module
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.play_emotion import EMOTION_INTENTS, PlayEmotion, resolve_emotion_name


AVAILABLE_EMOTIONS = [
    "cheerful1",
    "confused1",
    "no1",
    "no_sad1",
    "no_excited1",
    "understanding2",
    "yes_sad1",
]


def test_play_emotion_schema_uses_compact_intents() -> None:
    """Expose compact intents instead of the full recorded-move catalog."""
    emotion_schema = PlayEmotion.parameters_schema["properties"]["emotion"]

    assert emotion_schema["enum"] == list(EMOTION_INTENTS)
    assert "no_sad" in emotion_schema["enum"]
    assert "no_confused" in emotion_schema["enum"]
    assert "no_excited" in emotion_schema["enum"]
    assert "yes_sad" in emotion_schema["enum"]
    assert "loving1" not in emotion_schema["enum"]
    assert "Available emotions" not in emotion_schema["description"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("no_sad1", "no_sad1"),
        ("sad no", "no_sad1"),
        ("confused no", "confused1"),
        ("no_excited", "no_excited1"),
        ("contento", "cheerful1"),
        ("yes sad", "yes_sad1"),
        ("understood", "understanding2"),
    ],
)
def test_resolve_emotion_name_accepts_ids_intents_and_aliases(requested: str, expected: str) -> None:
    """Resolve exact IDs, nuanced intents, phrases, and multilingual aliases."""
    assert resolve_emotion_name(requested, AVAILABLE_EMOTIONS) == expected


def test_resolve_emotion_name_returns_none_for_random_or_unknown() -> None:
    """Let the caller choose a random fallback when no confident match exists."""
    assert resolve_emotion_name("random", AVAILABLE_EMOTIONS) is None
    assert resolve_emotion_name("totally mysterious mood", AVAILABLE_EMOTIONS) is None


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
