"""Tests for the minimal frontmatter parser."""

import pytest

from reachy_mini_conversation_app.memory.frontmatter import (
    dump_frontmatter,
    parse_frontmatter,
)


class TestParse:
    """Verify parse_frontmatter behaviour."""

    def test_simple_scalars(self) -> None:
        """Parse strings, bools, and null values."""
        text = "---\nid: 2026-04-17_demo_abc\npinned: false\nsupersedes: null\n---\nBody here.\n"
        meta, body = parse_frontmatter(text)
        assert meta["id"] == "2026-04-17_demo_abc"
        assert meta["pinned"] is False
        assert meta["supersedes"] is None
        assert body.strip() == "Body here."

    def test_inline_list(self) -> None:
        """Parse ``[a, b, c]`` lists."""
        text = "---\ntags: [chess, openings]\nrelated_to: []\n---\nbody"
        meta, _ = parse_frontmatter(text)
        assert meta["tags"] == ["chess", "openings"]
        assert meta["related_to"] == []

    def test_quoted_string(self) -> None:
        """Parse double-quoted strings stripping the quotes."""
        text = '---\nkind: "preference"\n---\nbody'
        meta, _ = parse_frontmatter(text)
        assert meta["kind"] == "preference"

    def test_missing_close_raises(self) -> None:
        """Unterminated frontmatter raises ValueError."""
        text = "---\nid: foo\nbody without close"
        with pytest.raises(ValueError):
            parse_frontmatter(text)

    def test_no_frontmatter(self) -> None:
        """No leading fence returns empty meta and full body."""
        text = "just body"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "just body"


class TestDump:
    """Verify round-trip serialization."""

    def test_round_trip(self) -> None:
        """Dump + parse yields the original mapping."""
        meta = {
            "id": "2026-04-17_demo_abc",
            "created": "2026-04-17T10:00:00Z",
            "sources": ["a.log", "b.log"],
            "kind": "preference",
            "tags": ["chess"],
            "related_to": [],
            "pinned": False,
            "supersedes": None,
            "superseded_by": None,
        }
        body = "The body.\n"
        text = dump_frontmatter(meta, body)
        parsed_meta, parsed_body = parse_frontmatter(text)
        assert parsed_meta == meta
        assert parsed_body.strip() == "The body."
