"""Tests for the background DreamScheduler.

The scheduler runs the dreamer on a daemon thread, so each test injects a stub
dreamer via ``dreamer_factory`` and waits on the ``on_finish`` callback (which
the scheduler guarantees to fire) instead of sleeping.
"""

from __future__ import annotations
import threading
from pathlib import Path

import pytest

from reachy_mini_conversation_app.memory.dreamer import DreamLogStats
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager
from reachy_mini_conversation_app.memory.dream_scheduler import DreamSummary, DreamScheduler


@pytest.fixture
def manager_with_pending(tmp_path: Path) -> MemoryManager:
    """Return a manager with one closed log waiting in pending/."""
    mgr = MemoryManager(tmp_path / "data")
    (mgr.pending_logs_dir / "2026-04-14_09-15.log").write_text("hello\n", encoding="utf-8")
    return mgr


class _StubDreamer:
    """Stand-in for Dreamer: returns canned stats or raises."""

    def __init__(self, stats: list[DreamLogStats] | None = None, boom: bool = False) -> None:
        self._stats = stats or []
        self._boom = boom

    def run(self) -> list[DreamLogStats]:
        if self._boom:
            raise RuntimeError("dream blew up")
        return self._stats


def _make_scheduler(manager: MemoryManager, dreamer: _StubDreamer) -> tuple[DreamScheduler, dict]:
    """Build a scheduler around a stub dreamer, recording callback order."""
    events: dict = {"order": [], "summary": None, "done": threading.Event()}

    def on_start() -> None:
        events["order"].append("start")

    def on_finish(summary: DreamSummary) -> None:
        events["order"].append("finish")
        events["summary"] = summary
        events["done"].set()

    scheduler = DreamScheduler(
        manager,
        model="fake",
        api_key="fake",
        on_start=on_start,
        on_finish=on_finish,
        dreamer_factory=lambda: dreamer,  # type: ignore[arg-type, return-value]
    )
    return scheduler, events


def test_start_fires_callbacks_in_order(manager_with_pending: MemoryManager) -> None:
    """on_start fires before the run, on_finish after, with a real summary."""
    stats = [DreamLogStats(filename="2026-04-14_09-15.log", created=2, updated=1)]
    scheduler, events = _make_scheduler(manager_with_pending, _StubDreamer(stats))

    assert scheduler.start() is True
    assert events["done"].wait(timeout=5.0)

    assert events["order"] == ["start", "finish"]
    summary = events["summary"]
    assert isinstance(summary, DreamSummary)
    assert summary.logs_processed == 1
    assert summary.created == 2
    assert summary.updated == 1
    assert summary.errored is False


def test_on_finish_fires_even_when_dream_raises(manager_with_pending: MemoryManager) -> None:
    """A crashing dreamer must still fire on_finish (errored summary), never propagate."""
    scheduler, events = _make_scheduler(manager_with_pending, _StubDreamer(boom=True))

    assert scheduler.start() is True
    assert events["done"].wait(timeout=5.0)

    assert events["order"] == ["start", "finish"]
    assert events["summary"].errored is True


def test_no_pending_logs_skips_thread(tmp_path: Path) -> None:
    """With nothing to dream about, start() is a no-op and no callback fires."""
    mgr = MemoryManager(tmp_path / "data")
    scheduler, events = _make_scheduler(mgr, _StubDreamer())

    assert scheduler.start() is False
    assert events["order"] == []
    assert scheduler.is_running() is False


def test_summary_from_stats_folds_multiple_logs() -> None:
    """DreamSummary aggregates created/updated counts and flags any error."""
    stats = [
        DreamLogStats(filename="a.log", created=3, updated=0),
        DreamLogStats(filename="b.log", created=1, updated=2, errors=["boom"]),
    ]
    summary = DreamSummary.from_stats(stats)
    assert summary.logs_processed == 2
    assert summary.created == 4
    assert summary.updated == 2
    assert summary.errored is True
