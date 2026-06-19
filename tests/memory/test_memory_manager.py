"""Tests for MemoryManager (new dreaming-based storage layout)."""

from pathlib import Path

import pytest

from reachy_mini_conversation_app.memory.index_renderer import (
    render_index,
    rebuild_index,
)
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Return a temporary data directory."""
    return tmp_path / "data"


@pytest.fixture
def manager(data_dir: Path) -> MemoryManager:
    """Create a fresh MemoryManager in a temp directory."""
    return MemoryManager(data_dir)


def _memory_id(slug: str = "demo", date: str = "2026-04-17", hex3: str = "abc") -> str:
    return f"{date}_{slug}_{hex3}"


class TestLayout:
    """Verify directory structure on init."""

    def test_creates_all_directories(self, manager: MemoryManager, data_dir: Path) -> None:
        """Ensure every expected subdir exists after init."""
        mem = data_dir / "memory"
        assert mem.is_dir()
        assert (mem / "memories").is_dir()
        assert (mem / "logs" / "pending").is_dir()
        assert (mem / "logs" / "processed").is_dir()

    def test_session_log_written_under_pending(self, manager: MemoryManager, data_dir: Path) -> None:
        """The session log path must be reserved under logs/pending/ from the start.

        The file itself is created lazily on the first append, so a boot with
        no conversation leaves no stub behind.
        """
        pending = data_dir / "memory" / "logs" / "pending"
        assert manager.session_log_path is not None
        assert manager.session_log_path.parent == pending
        assert not manager.session_log_path.exists()  # lazy: not yet written

        manager.log_turn("user", "trigger lazy creation")
        assert manager.session_log_path.exists()


class TestMigration:
    """Verify legacy -> new layout migration."""

    def test_moves_top_level_logs_into_pending(self, data_dir: Path) -> None:
        """Old top-level logs/*.log must be moved to logs/pending/."""
        old_logs = data_dir / "memory" / "logs"
        old_logs.mkdir(parents=True)
        (old_logs / "2026-01-01_10-00.log").write_text("legacy session", encoding="utf-8")
        (old_logs / "2026-01-02_12-00.log").write_text("another session", encoding="utf-8")

        MemoryManager(data_dir)

        pending = old_logs / "pending"
        assert (pending / "2026-01-01_10-00.log").is_file()
        assert (pending / "2026-01-02_12-00.log").is_file()
        assert not (old_logs / "2026-01-01_10-00.log").exists()

    def test_preserves_existing_active_memory(self, data_dir: Path) -> None:
        """A pre-existing active_memory.md must survive construction (it's the prompt-injected index)."""
        mem_dir = data_dir / "memory"
        mem_dir.mkdir(parents=True)
        index = mem_dir / "active_memory.md"
        index.write_text("# Memory index\n\nsome content", encoding="utf-8")

        MemoryManager(data_dir)

        assert index.exists()
        assert index.read_text(encoding="utf-8") == "# Memory index\n\nsome content"

    def test_rebuilds_missing_index_when_memories_exist(self, data_dir: Path) -> None:
        """A missing index is rebuilt at init when memories exist, so injection needn't wait for a dream."""
        memories_dir = data_dir / "memory" / "memories"
        memories_dir.mkdir(parents=True)
        (memories_dir / "2026-04-20_demo_a1b.md").write_text(
            "---\n"
            "id: 2026-04-20_demo_a1b\n"
            "created: 2026-04-20T10:00:00Z\n"
            "kind: fact\n"
            "tags: [demo]\n"
            "pinned: true\n"
            "---\n\n"
            "A demo memory.\n",
            encoding="utf-8",
        )

        mgr = MemoryManager(data_dir)

        assert mgr.active_memory_path.exists()
        assert mgr.get_memory_block() != ""

    def test_no_index_rebuild_without_memories(self, data_dir: Path) -> None:
        """A fresh install with no memories must not fabricate an index."""
        mgr = MemoryManager(data_dir)
        assert not mgr.active_memory_path.exists()

    def test_removes_legacy_archive(self, data_dir: Path) -> None:
        """A legacy archive/ directory must be removed."""
        archive = data_dir / "memory" / "archive"
        archive.mkdir(parents=True)
        (archive / "something.txt").write_text("junk", encoding="utf-8")

        MemoryManager(data_dir)

        assert not archive.exists()


class TestSessionLog:
    """Verify live log write behaviour."""

    def test_log_turn(self, manager: MemoryManager) -> None:
        """Round-trip a user turn through the session log."""
        manager.log_turn("user", "Hello there!")
        content = manager.session_log_path.read_text()  # type: ignore[union-attr]
        assert "user: Hello there!" in content

    def test_log_tool_call(self, manager: MemoryManager) -> None:
        """Round-trip a tool invocation through the session log."""
        manager.log_tool_call("dance", args={"move": "spin"}, result={"status": "queued"})
        content = manager.session_log_path.read_text()  # type: ignore[union-attr]
        assert 'tool: dance({"move": "spin"})' in content

    def test_read_current_session_log(self, manager: MemoryManager) -> None:
        """Return the full session log as a string."""
        manager.log_turn("user", "Hi")
        text = manager.read_current_session_log()
        assert "user: Hi" in text

    def test_new_session_rotates(self, manager: MemoryManager) -> None:
        """Rotating the session reserves a fresh pending log path."""
        first = manager.session_log_path
        manager.log_turn("user", "first session")  # force lazy create
        manager.new_session()
        second = manager.session_log_path
        manager.log_turn("user", "second session")  # force lazy create
        assert first != second
        assert first.exists() and second.exists()  # type: ignore[union-attr]

    def test_list_pending_excludes_active(self, manager: MemoryManager) -> None:
        """The live log must be filtered out of pending listings by default."""
        # create an older pending log
        old = manager.pending_logs_dir / "2025-01-01_00-00.log"
        old.write_text("legacy", encoding="utf-8")

        pending = manager.list_pending_logs()
        assert "2025-01-01_00-00.log" in pending
        assert manager.session_log_path.name not in pending  # type: ignore[union-attr]


class TestMemoryFiles:
    """Verify memory file creation, update, and listing."""

    def test_write_and_read(self, manager: MemoryManager) -> None:
        """Round-trip a memory through write + read."""
        mid = _memory_id("chess-openings", "2026-04-17", "a3f")
        manager.write_memory(
            mid,
            "User prefers Queen's Gambit.",
            kind="preference",
            tags=["chess", "openings"],
            sources=["2026-04-14_09-15.log"],
        )
        mem = manager.read_memory(mid)
        assert mem["id"] == mid
        assert mem["frontmatter"]["kind"] == "preference"
        assert mem["frontmatter"]["tags"] == ["chess", "openings"]
        assert "Queen's Gambit" in mem["body"]

    def test_invalid_id_format_rejected(self, manager: MemoryManager) -> None:
        """Memory IDs must match the expected format."""
        with pytest.raises(ValueError):
            manager.write_memory(
                "bad id",
                "body",
                kind="fact",
                tags=["foo"],
            )

    def test_invalid_kind_rejected(self, manager: MemoryManager) -> None:
        """Kind must be a member of the closed taxonomy."""
        with pytest.raises(ValueError):
            manager.write_memory(
                _memory_id(),
                "body",
                kind="misc",  # not in ALLOWED_KINDS
                tags=["foo"],
            )

    def test_overwrite_refused(self, manager: MemoryManager) -> None:
        """write_memory must refuse to overwrite an existing file."""
        mid = _memory_id()
        manager.write_memory(mid, "first", kind="fact", tags=["foo"])
        with pytest.raises(FileExistsError):
            manager.write_memory(mid, "second", kind="fact", tags=["foo"])

    def test_update_changes_body(self, manager: MemoryManager) -> None:
        """update_memory can rewrite the body while preserving frontmatter."""
        mid = _memory_id()
        manager.write_memory(mid, "first body", kind="fact", tags=["foo"])
        manager.update_memory(mid, body="second body")
        mem = manager.read_memory(mid)
        assert mem["body"].strip() == "second body"

    def test_update_merges_frontmatter(self, manager: MemoryManager) -> None:
        """update_memory merges frontmatter updates."""
        mid = _memory_id()
        manager.write_memory(mid, "body", kind="fact", tags=["foo"])
        manager.update_memory(mid, frontmatter_updates={"pinned": True, "tags": ["foo", "bar"]})
        mem = manager.read_memory(mid)
        assert mem["frontmatter"]["pinned"] is True
        assert mem["frontmatter"]["tags"] == ["foo", "bar"]
        assert mem["frontmatter"]["kind"] == "fact"  # untouched

    def test_list_filters_by_tag(self, manager: MemoryManager) -> None:
        """list_memories filters by tag."""
        manager.write_memory(_memory_id("chess", hex3="111"), "A", kind="fact", tags=["chess"])
        manager.write_memory(_memory_id("cook", hex3="222"), "B", kind="fact", tags=["cooking"])
        results = manager.list_memories(tag="chess")
        assert [m["id"] for m in results] == [_memory_id("chess", hex3="111")]

    def test_list_filters_by_kind(self, manager: MemoryManager) -> None:
        """list_memories filters by kind."""
        manager.write_memory(_memory_id("p", hex3="111"), "A", kind="preference", tags=["t"])
        manager.write_memory(_memory_id("e", hex3="222"), "B", kind="event", tags=["t"])
        results = manager.list_memories(kind="event")
        assert [m["id"] for m in results] == [_memory_id("e", hex3="222")]

    def test_list_hides_superseded_by_default(self, manager: MemoryManager) -> None:
        """Superseded memories drop out of the default listing but not the full one."""
        old = _memory_id("old", hex3="111")
        new = _memory_id("new", hex3="222")
        manager.write_memory(old, "old", kind="fact", tags=["t"])
        manager.write_memory(new, "new", kind="fact", tags=["t"], supersedes=old)
        manager.update_memory(old, frontmatter_updates={"superseded_by": new})

        visible = {m["id"] for m in manager.list_memories()}
        full = {m["id"] for m in manager.list_memories(include_superseded=True)}
        assert old not in visible and new in visible
        assert {old, new}.issubset(full)

    def test_summary_extracted_from_body(self, manager: MemoryManager) -> None:
        """First non-title body line becomes the summary."""
        mid = _memory_id()
        body = "# Chess\n\nUser prefers Queen's Gambit."
        manager.write_memory(mid, body, kind="preference", tags=["chess"])
        [entry] = manager.list_memories()
        assert entry["summary"] == "User prefers Queen's Gambit."


class TestFindRelatedMemories:
    """Verify the substring-ranking search used by the dreamer."""

    def _seed(self, manager: MemoryManager) -> None:
        manager.write_memory(
            _memory_id("chess-openings", hex3="aaa"),
            "User prefers Queen's Gambit.",
            kind="preference",
            tags=["chess", "openings"],
        )
        manager.write_memory(
            _memory_id("ping-pong", hex3="bbb"),
            "User is learning ping pong and wants to read spin better.",
            kind="skill",
            tags=["ping-pong", "sports"],
        )
        manager.write_memory(
            _memory_id("french", hex3="ccc"),
            "User is French.",
            kind="fact",
            tags=["nationality", "france"],
        )

    def test_query_matches_body(self, manager: MemoryManager) -> None:
        """A keyword found only in the body ranks that memory first."""
        self._seed(manager)
        results = manager.find_related_memories(query="queen")
        assert results
        assert results[0]["id"].startswith("2026-04-17_chess-openings_")

    def test_score_prefers_more_hits(self, manager: MemoryManager) -> None:
        """More substring matches → higher rank."""
        self._seed(manager)
        results = manager.find_related_memories(query="chess queen openings")
        assert results[0]["id"].startswith("2026-04-17_chess-openings_")
        assert results[0]["score"] >= 3

    def test_tags_param_contributes(self, manager: MemoryManager) -> None:
        """Passing ``tags=`` boosts memories matching those tags."""
        self._seed(manager)
        results = manager.find_related_memories(tags=["france"])
        assert [r["id"] for r in results] == [_memory_id("french", hex3="ccc")]

    def test_empty_inputs_return_empty(self, manager: MemoryManager) -> None:
        """No query and no tags → no match."""
        self._seed(manager)
        assert manager.find_related_memories() == []

    def test_hides_superseded(self, manager: MemoryManager) -> None:
        """Superseded memories drop out of the results."""
        old = _memory_id("old", hex3="111")
        new = _memory_id("new", hex3="222")
        manager.write_memory(old, "User prefers apples.", kind="preference", tags=["fruit"])
        manager.write_memory(new, "User prefers oranges.", kind="preference", tags=["fruit"], supersedes=old)
        manager.update_memory(old, frontmatter_updates={"superseded_by": new})
        results = manager.find_related_memories(query="prefers")
        assert [r["id"] for r in results] == [new]

    def test_body_preview_carried_when_requested(self, manager: MemoryManager) -> None:
        """body_preview_chars > 0 carries a preview so callers can skip a follow-up read."""
        mid = _memory_id("chess", hex3="aaa")
        long_body = "TLDR first line.\n" + "x" * 500
        manager.write_memory(mid, long_body, kind="preference", tags=["chess"])
        results = manager.find_related_memories(query="chess", body_preview_chars=50)
        assert results
        preview = results[0]["body_preview"]
        assert preview.startswith("TLDR first line.")
        assert preview.endswith("…")
        assert len(preview) <= 51  # 50 chars + ellipsis

    def test_body_preview_absent_by_default(self, manager: MemoryManager) -> None:
        """With body_preview_chars=0 the preview field is omitted."""
        mid = _memory_id("chess", hex3="aaa")
        manager.write_memory(mid, "TLDR.", kind="preference", tags=["chess"])
        [result] = manager.find_related_memories(query="chess", body_preview_chars=0)
        assert "body_preview" not in result


class TestMarkLogProcessed:
    """Verify move pending -> processed."""

    def test_moves_log(self, manager: MemoryManager) -> None:
        """Successful move from pending/ to processed/."""
        old = manager.pending_logs_dir / "2025-06-01_12-00.log"
        old.write_text("contents", encoding="utf-8")
        manager.mark_log_processed("2025-06-01_12-00.log")
        assert not old.exists()
        assert (manager.pending_logs_dir.parent / "processed" / "2025-06-01_12-00.log").is_file()

    def test_refuses_active_session(self, manager: MemoryManager) -> None:
        """Refuse to move the active session log."""
        active = manager.session_log_path.name  # type: ignore[union-attr]
        with pytest.raises(RuntimeError):
            manager.mark_log_processed(active)


class TestIndexRenderer:
    """Verify tiered index rendering."""

    def test_empty_index(self) -> None:
        """Empty input renders with (none) placeholders."""
        out = render_index([])
        assert "## Important" in out
        assert "## Other" in out
        assert "## Older topics" in out
        assert out.count("(none)") == 3

    def test_important_renders_pinned(self) -> None:
        """Pinned memories land in the Important section, regardless of age."""
        mems = [
            {
                "id": "2025-01-01_user-name_abc",
                "summary": "User's name is Rémi.",
                "tags": ["identity"],
                "kind": "fact",
                "pinned": True,
                "created": "2025-01-01T00:00:00Z",
                "superseded_by": None,
            }
        ]
        out = render_index(mems)
        important_block = out.split("## Other")[0]
        assert "[2025-01-01_user-name_abc] User's name is Rémi." in important_block

    def test_other_grouped_by_primary_tag(self) -> None:
        """Non-pinned memories get a ### subheading per first tag in 'Other'."""
        mems = [
            {
                "id": "2026-04-10_chess_aaa",
                "summary": "Chess summary.",
                "tags": ["chess", "openings"],
                "kind": "preference",
                "pinned": False,
                "created": "2026-04-10T00:00:00Z",
                "superseded_by": None,
            },
            {
                "id": "2026-04-11_board_bbb",
                "summary": "Board game night.",
                "tags": ["board-games"],
                "kind": "event",
                "pinned": False,
                "created": "2026-04-11T00:00:00Z",
                "superseded_by": None,
            },
        ]
        out = render_index(mems)
        assert "### chess" in out
        assert "### board-games" in out

    def test_overflow_beyond_limit_collapses_to_older(self) -> None:
        """Oldest memories beyond the budget drop into 'Older topics' as counts."""
        mems = [
            {
                "id": f"2026-04-1{i}_work_aa{i}",
                "summary": f"w{i}",
                "tags": ["work"],
                "kind": "event",
                "pinned": False,
                "created": f"2026-04-1{i}T00:00:00Z",
                "superseded_by": None,
            }
            for i in range(3)
        ]
        # Keep only the newest 1 in full; the older 2 overflow to tag counts.
        out = render_index(mems, limit=1)
        other = out.split("## Other")[1].split("## Older topics")[0]
        older = out.split("## Older topics")[1]
        assert "[2026-04-12_work_aa2] w2" in other  # newest kept in full
        assert "work (2)" in older  # the older two collapsed to a count

    def test_older_section_shows_counts(self) -> None:
        """Overflow memories collapse to ranked tag counts."""
        mems = [
            {
                "id": f"2025-09-29_work_{i:03x}"[:-1] + "a",
                "summary": f"w{i}",
                "tags": ["work"],
                "kind": "event",
                "pinned": False,
                "created": "2025-09-29T00:00:00Z",
                "superseded_by": None,
            }
            for i in range(3)
        ]
        mems.append(
            {
                "id": "2025-09-29_music_zzz",
                "summary": "m",
                "tags": ["music"],
                "kind": "preference",
                "pinned": False,
                "created": "2025-09-29T00:00:00Z",
                "superseded_by": None,
            }
        )
        # Force everything into the overflow so the counts render.
        out = render_index(mems, limit=0)
        older = out.split("## Older topics")[1]
        assert "work (3)" in older
        assert "music (1)" in older
        # work appears before music because of higher count
        assert older.index("work (3)") < older.index("music (1)")

    def test_rebuild_index_writes_file(self, manager: MemoryManager) -> None:
        """rebuild_index persists a non-empty file."""
        manager.write_memory(_memory_id("chess", hex3="aaa"), "Loves chess.", kind="preference", tags=["chess"])
        rendered = rebuild_index(manager)
        assert manager.active_memory_path.read_text(encoding="utf-8") == rendered
        assert "chess" in rendered.lower()


class TestMemoryBlock:
    """Verify system prompt injection."""

    def test_empty_when_no_index(self, manager: MemoryManager) -> None:
        """No index file returns empty string."""
        assert manager.get_memory_block() == ""

    def test_includes_rendered_index(self, manager: MemoryManager) -> None:
        """Built index shows up inside the MEMORY block."""
        manager.write_memory(
            _memory_id("chess", hex3="aaa"),
            "Loves chess.",
            kind="preference",
            tags=["chess"],
        )
        rebuild_index(manager)
        block = manager.get_memory_block()
        assert "## MEMORY" in block
        assert "chess" in block.lower()
        assert "recall_memory" in block
        assert "recall_memories" in block
