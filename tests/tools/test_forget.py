from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import forget as forget_module
from reachy_mini_conversation_app.memory import MemoryFact, ForgetMemoryResult
from reachy_mini_conversation_app.tools.forget import Forget
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _fact(text: str, fact_id: str = "1") -> MemoryFact:
    return MemoryFact(id=fact_id, text=text, created_at=0)


@pytest.mark.asyncio
async def test_forget_rejects_empty_query() -> None:
    """A blank query is rejected before touching the store."""
    result = await Forget()(_deps(), query="  ")
    assert result == {"error": "query must be a non-empty string"}


@pytest.mark.asyncio
async def test_forget_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """When nothing matches the query the tool reports it without removing anything."""
    monkeypatch.setattr(
        forget_module, "forget_memory_fact", lambda path, query: ForgetMemoryResult(removed=None, candidates=())
    )
    result = await Forget()(_deps(), query="pizza")
    assert "error" in result


@pytest.mark.asyncio
async def test_forget_removes_and_lists_other_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful removal returns the removed fact and any other matches."""
    removed = _fact("likes pizza", "a")
    other = _fact("likes pizza too", "b")
    monkeypatch.setattr(
        forget_module,
        "forget_memory_fact",
        lambda path, query: ForgetMemoryResult(removed=removed, candidates=(removed, other)),
    )
    result = await Forget()(_deps(), query="pizza")
    assert result["removed"] == "likes pizza"
    assert result["memory_id"] == "a"
    assert result["other_matches"] == ["likes pizza too"]
