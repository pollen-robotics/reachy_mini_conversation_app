"""Render the Important / Other / Older memory index."""

from __future__ import annotations
from typing import Any
from datetime import datetime, timezone

from reachy_mini_conversation_app.memory.dates import event_date


# Non-pinned memories shown in full before the oldest overflow collapses to tag counts.
OTHER_LIMIT = 40
OLDER_TAG_LIMIT = 15
_MIN_DATE = datetime.min.replace(tzinfo=timezone.utc)


def _fmt_entry(mem: dict[str, Any]) -> str:
    summary = mem.get("summary") or "(empty memory)"
    return f"- [{mem['id']}] {summary}"


def render_index(memories: list[dict[str, Any]], limit: int = OTHER_LIMIT) -> str:
    """Render the memory index as markdown."""
    visible = [m for m in memories if not m.get("superseded_by")]

    important = [m for m in visible if m.get("pinned")]
    others = [m for m in visible if not m.get("pinned")]

    others.sort(key=lambda m: event_date(m) or _MIN_DATE, reverse=True)
    shown = others[: max(0, limit)]
    older = others[max(0, limit) :]

    lines: list[str] = ["# Memory index", ""]

    lines.append("## Important")
    if important:
        for mem in sorted(important, key=lambda m: m["id"]):
            lines.append(_fmt_entry(mem))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Other")
    if shown:
        groups: dict[str, list[dict[str, Any]]] = {}
        for mem in shown:
            tags = mem.get("tags") or []
            primary = tags[0] if tags else "untagged"
            groups.setdefault(primary, []).append(mem)
        for tag in sorted(groups):
            lines.append(f"### {tag}")
            for mem in sorted(groups[tag], key=lambda m: event_date(m) or _MIN_DATE):
                lines.append(_fmt_entry(mem))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Older topics")
    if older:
        tag_counts: dict[str, int] = {}
        for mem in older:
            for tag in mem.get("tags") or []:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        counts = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("Tags (count), ranked by frequency:")
        truncated = counts[:OLDER_TAG_LIMIT]
        remaining = max(0, len(counts) - OLDER_TAG_LIMIT)
        for tag, count in truncated:
            lines.append(f"- {tag} ({count})")
        if remaining:
            lines.append(f"- … +{remaining} more tags")
        lines.append("")
        lines.append("Use `recall_memories(tag=...)` to load (also filters by date_from/date_to).")
    else:
        lines.append("(none)")

    return "\n".join(lines).rstrip() + "\n"


def rebuild_index(manager: Any) -> str:
    """Rebuild ``active_memory.md`` from all on-disk memories."""
    memories = manager.list_memories(include_superseded=False)
    rendered = render_index(memories)
    manager._atomic_write_active(rendered)
    return rendered
