"""Tests for the live-conversation memory tools."""

from pathlib import Path
from dataclasses import dataclass

import pytest

from reachy_mini_conversation_app.tools.recall_memory import RecallMemory
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager
from reachy_mini_conversation_app.tools.recall_memories import RecallMemories


@dataclass
class _FakeDeps:
    memory_manager: MemoryManager | None


@pytest.fixture
def deps(tmp_path: Path) -> _FakeDeps:
    """Build a ToolDependencies-like object backed by a real MemoryManager."""
    return _FakeDeps(memory_manager=MemoryManager(tmp_path / "data"))


def _mid(slug: str, hex3: str = "abc", date: str = "2026-04-17") -> str:
    return f"{date}_{slug}_{hex3}"


class TestRecallMemory:
    """Verify recall_memory returns bundled target + related memories."""

    @pytest.mark.asyncio
    async def test_returns_memory(self, deps: _FakeDeps) -> None:
        """Happy path: existing memory with no related_to."""
        assert deps.memory_manager is not None
        mid = _mid("chess")
        deps.memory_manager.write_memory(
            mid,
            "Loves Queen's Gambit.",
            kind="preference",
            tags=["chess"],
        )
        result = await RecallMemory()(deps, id=mid)
        assert result["memory"]["id"] == mid
        assert "Queen" in result["memory"]["body"]
        assert result["related"] == []

    @pytest.mark.asyncio
    async def test_bundles_related(self, deps: _FakeDeps) -> None:
        """Every memory referenced in related_to must be returned."""
        assert deps.memory_manager is not None
        main = _mid("chess-openings", "111")
        neighbour = _mid("chess-match", "222")
        deps.memory_manager.write_memory(neighbour, "Lost to Jean.", kind="event", tags=["chess"])
        deps.memory_manager.write_memory(
            main,
            "Prefers Queen's Gambit.",
            kind="preference",
            tags=["chess", "openings"],
            related_to=[neighbour],
        )
        result = await RecallMemory()(deps, id=main)
        assert [m["id"] for m in result["related"]] == [neighbour]

    @pytest.mark.asyncio
    async def test_missing_returns_error(self, deps: _FakeDeps) -> None:
        """Unknown ID returns an error plus a sample of known IDs."""
        assert deps.memory_manager is not None
        deps.memory_manager.write_memory(_mid("known"), "body", kind="fact", tags=["foo"])
        result = await RecallMemory()(deps, id=_mid("unknown", "fff"))
        assert "error" in result
        assert _mid("known") in result["known_ids_sample"]

    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        """Missing memory_manager is reported as memory_disabled."""
        deps = _FakeDeps(memory_manager=None)
        result = await RecallMemory()(deps, id="anything")
        assert result == {"status": "memory_disabled"}

    @pytest.mark.asyncio
    async def test_hides_created_and_surfaces_conversation_dates(self, deps: _FakeDeps) -> None:
        """The model must never see `created`; it sees the conversation dates instead."""
        assert deps.memory_manager is not None
        mid = _mid("chess")
        deps.memory_manager.write_memory(
            mid,
            "Loves Queen's Gambit.",
            kind="preference",
            tags=["chess"],
            sources=["2026-04-17_14-37.log", "2026-05-05_09-29.log"],
        )
        result = await RecallMemory()(deps, id=mid)
        assert "created" not in result["memory"]["frontmatter"]
        assert result["memory"]["dates_discussed"] == ["2026-04-17", "2026-05-05"]


class TestRecallMemories:
    """Verify recall_memories filters by tag and/or conversation-date range."""

    @pytest.mark.asyncio
    async def test_filters_by_tag_and_limits(self, deps: _FakeDeps) -> None:
        """Returns only memories matching `tag`, bounded by `limit`."""
        assert deps.memory_manager is not None
        for idx in range(3):
            deps.memory_manager.write_memory(
                _mid(f"chess{idx}", hex3=f"a{idx:02d}"),
                f"chess memory {idx}",
                kind="preference",
                tags=["chess"],
            )
        deps.memory_manager.write_memory(_mid("cooking"), "cooking", kind="preference", tags=["cooking"])
        result = await RecallMemories()(deps, tag="chess", limit=2)
        assert result["returned"] == 2
        assert result["total_matches"] == 3
        for entry in result["memories"]:
            assert "chess" in entry["frontmatter"]["tags"]

    @pytest.mark.asyncio
    async def test_filters_by_date_range(self, deps: _FakeDeps) -> None:
        """Only memories whose conversation date falls in the range are returned."""
        assert deps.memory_manager is not None
        deps.memory_manager.write_memory(
            _mid("old", "111"), "old", kind="event", tags=["x"], sources=["2026-04-10_10-00.log"]
        )
        deps.memory_manager.write_memory(
            _mid("mid", "222"), "mid", kind="event", tags=["x"], sources=["2026-04-20_10-00.log"]
        )
        deps.memory_manager.write_memory(
            _mid("new", "333"), "new", kind="event", tags=["x"], sources=["2026-05-01_10-00.log"]
        )
        result = await RecallMemories()(deps, date_from="2026-04-15", date_to="2026-04-25")
        assert result["returned"] == 1
        assert result["memories"][0]["id"] == _mid("mid", "222")

    @pytest.mark.asyncio
    async def test_single_day_matches_any_source(self, deps: _FakeDeps) -> None:
        """A multi-day memory matches a single day if any of its sources is that day."""
        assert deps.memory_manager is not None
        deps.memory_manager.write_memory(
            _mid("span", "abc"),
            "spans days",
            kind="event",
            tags=["x"],
            sources=["2026-04-17_14-37.log", "2026-05-05_09-29.log"],
        )
        result = await RecallMemories()(deps, date_from="2026-04-17", date_to="2026-04-17")
        assert result["returned"] == 1

    @pytest.mark.asyncio
    async def test_tag_and_date_combined(self, deps: _FakeDeps) -> None:
        """Tag and date filters apply together (AND)."""
        assert deps.memory_manager is not None
        deps.memory_manager.write_memory(
            _mid("chess", "111"), "c", kind="event", tags=["chess"], sources=["2026-04-20_10-00.log"]
        )
        deps.memory_manager.write_memory(
            _mid("cooking", "222"), "k", kind="event", tags=["cooking"], sources=["2026-04-20_10-00.log"]
        )
        result = await RecallMemories()(deps, tag="chess", date_from="2026-04-20", date_to="2026-04-20")
        assert result["returned"] == 1
        assert result["memories"][0]["id"] == _mid("chess", "111")

    @pytest.mark.asyncio
    async def test_no_filter_is_error(self, deps: _FakeDeps) -> None:
        """Calling with no tag and no dates is an error."""
        result = await RecallMemories()(deps)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unmatched_tag_returns_empty(self, deps: _FakeDeps) -> None:
        """A tag with no matches returns an empty bundle (not an error)."""
        result = await RecallMemories()(deps, tag="missing")
        assert result["returned"] == 0
        assert result["memories"] == []
