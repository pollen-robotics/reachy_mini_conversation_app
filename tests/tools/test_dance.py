from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import dance as dance_module
from reachy_mini_conversation_app.tools.dance import Dance, get_available_dances_and_descriptions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class _FakeDanceQueueMove:
    def __init__(self, move_name: str) -> None:
        self.move_name = move_name


def _install_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dance_module, "DANCE_AVAILABLE", True)
    monkeypatch.setattr(dance_module, "AVAILABLE_MOVES", {"wave": (None, None, {"description": "waves"})})
    monkeypatch.setattr(dance_module, "DanceQueueMove", _FakeDanceQueueMove)


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def test_descriptions_lists_available_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The formatter renders each move name with its description."""
    _install_moves(monkeypatch)
    assert get_available_dances_and_descriptions() == "wave: waves\n"


def test_descriptions_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The formatter degrades gracefully when the dance library is missing."""
    monkeypatch.setattr(dance_module, "DANCE_AVAILABLE", False)
    assert get_available_dances_and_descriptions() == "Moves not available."


@pytest.mark.asyncio
async def test_dance_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool errors cleanly when the dance system is unavailable."""
    monkeypatch.setattr(dance_module, "DANCE_AVAILABLE", False)
    result = await Dance()(_deps())
    assert result == {"error": "Dance system not available"}


@pytest.mark.asyncio
async def test_dance_unknown_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown move name is rejected."""
    _install_moves(monkeypatch)
    result = await Dance()(_deps(), move="moonwalk")
    assert "error" in result


@pytest.mark.asyncio
async def test_dance_queues_named_move_with_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    """A named move is queued once per repeat."""
    _install_moves(monkeypatch)
    deps = _deps()
    result = await Dance()(deps, move="wave", repeat=3)
    assert result == {"status": "queued", "move": "wave", "repeat": 3}
    assert deps.movement_manager.queue_move.call_count == 3


@pytest.mark.asyncio
async def test_dance_picks_random_when_no_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting a move name selects one at random."""
    _install_moves(monkeypatch)
    monkeypatch.setattr(dance_module.random, "choice", lambda moves: "wave")
    deps = _deps()
    result = await Dance()(deps)
    assert result["move"] == "wave"
