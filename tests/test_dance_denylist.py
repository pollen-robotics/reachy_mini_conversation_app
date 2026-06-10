"""Tests for the dance chassis-safety denylist (#542).

Mirrors the play_emotion denylist (#480): denylisted dances disappear from
the LLM tool spec enum and are refused at call time, and the random pick
never selects one. The #536 bonk sweep motivated this — dances previously
had no avoidance mechanism at all.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from robot_comic.tools import dance as dance_mod
from robot_comic.config import config, _parse_dance_denylist
from robot_comic.tools.dance import Dance


_FAKE_MOVES = {
    "grid_snap": (None, None, {"description": "sharp grid snaps"}),
    "simple_nod": (None, None, {"description": "a gentle nod"}),
    "yeah_nod": (None, None, {"description": "an enthusiastic nod"}),
}


@pytest.fixture()
def dance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dance_mod, "AVAILABLE_MOVES", _FAKE_MOVES)
    monkeypatch.setattr(dance_mod, "DANCE_AVAILABLE", True)
    monkeypatch.setattr(dance_mod, "DanceQueueMove", MagicMock())
    monkeypatch.setattr(config, "DANCE_DENYLIST", frozenset({"grid_snap"}))


def test_parse_dance_denylist() -> None:
    assert _parse_dance_denylist(None) == frozenset()
    assert _parse_dance_denylist("") == frozenset()
    assert _parse_dance_denylist("grid_snap, chicken_peck") == frozenset({"grid_snap", "chicken_peck"})


def test_spec_excludes_denylisted_moves(dance_env: None) -> None:
    spec = Dance().spec()
    enum = spec["parameters"]["properties"]["move"]["enum"]
    assert "grid_snap" not in enum
    assert set(enum) == {"simple_nod", "yeah_nod"}
    assert "grid_snap" not in spec["parameters"]["properties"]["move"]["description"]


@pytest.mark.asyncio
async def test_call_refuses_denylisted_move(dance_env: None) -> None:
    deps = MagicMock()
    result = await Dance()(deps, move="grid_snap")
    assert "denylisted" in result["error"]
    deps.movement_manager.queue_move.assert_not_called()


@pytest.mark.asyncio
async def test_call_queues_safe_move(dance_env: None) -> None:
    deps = MagicMock()
    result = await Dance()(deps, move="simple_nod", repeat=2)
    assert result == {"status": "queued", "move": "simple_nod", "repeat": 2}
    assert deps.movement_manager.queue_move.call_count == 2


@pytest.mark.asyncio
async def test_random_pick_avoids_denylisted(dance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    deps = MagicMock()
    picked: list[list[str]] = []

    def _spy_choice(seq: list[str]) -> str:
        picked.append(list(seq))
        return seq[0]

    monkeypatch.setattr(dance_mod.random, "choice", _spy_choice)
    result = await Dance()(deps)
    assert result["status"] == "queued"
    assert picked == [["simple_nod", "yeah_nod"]]


@pytest.mark.asyncio
async def test_empty_denylist_keeps_all_moves(dance_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DANCE_DENYLIST", frozenset())
    enum = Dance().spec()["parameters"]["properties"]["move"]["enum"]
    assert set(enum) == set(_FAKE_MOVES)
