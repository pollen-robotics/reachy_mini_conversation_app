"""Real-LLM integration test for the Dreamer.

Skipped unless ``OPENAI_API_KEY`` is exported. Uses a real model to check
that the dreamer's prompt + tool set actually produce a sensible memory
file end-to-end. Costs a handful of cents per run.

Run explicitly with::

    pytest tests/memory/test_dreamer_integration.py -m integration
"""

from __future__ import annotations
import os
from pathlib import Path

import pytest

from reachy_mini_conversation_app.memory.dreamer import Dreamer
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager


DEFAULT_TEST_MODEL = "gpt-5.4"


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping real-LLM integration test",
)


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    """Seed a fresh store with a single simple log the dreamer should distil."""
    mgr = MemoryManager(tmp_path / "data")
    log_path = mgr.pending_logs_dir / "2026-04-14_09-15.log"
    log_path.write_text(
        "--- session 2026-04-14 09:15 UTC ---\n\n"
        "09:15:12 user: Hi Reachy, my name is Rémi.\n"
        "09:15:14 assistant: Nice to meet you, Rémi!\n"
        "09:15:20 user: I love chess, and my favourite opening is the Queen's Gambit.\n"
        "09:15:25 assistant: Great choice!\n"
        "09:15:34 user: By the way, I'm French and I speak English fluently.\n"
        "09:15:40 assistant: Understood.\n",
        encoding="utf-8",
    )
    return mgr


def test_real_dream_pass_creates_memories(manager: MemoryManager) -> None:
    """End-to-end: the dreamer produces at least one memory from a simple log."""
    model = os.getenv("MEMORY_DREAMER_MODEL") or DEFAULT_TEST_MODEL
    dreamer = Dreamer(manager, model=model)

    stats_list = dreamer.run()
    assert len(stats_list) == 1
    stats = stats_list[0]
    assert stats.created >= 1, f"Dreamer created nothing. Stats: {stats.one_line()}"
    assert not stats.errors, f"Dreamer reported errors: {stats.errors}"

    memory_files = list(manager.memories_dir.glob("*.md"))
    assert memory_files, "No memory file written on disk."

    # Index must have been rebuilt and mention at least one memory ID.
    index_text = manager.active_memory_path.read_text(encoding="utf-8")
    assert "Memory index" in index_text
    assert any(f.stem in index_text for f in memory_files), (
        "Rebuilt index does not reference any of the memories on disk."
    )

    # The source log must have moved to processed/.
    assert not (manager.pending_logs_dir / "2026-04-14_09-15.log").exists()
    assert (manager._processed_logs_dir / "2026-04-14_09-15.log").is_file()
