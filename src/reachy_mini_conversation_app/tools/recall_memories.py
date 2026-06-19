"""System tool ``recall_memories``: filter stored memories by tag and/or conversation date.

Returns the full text of each matching memory (id, frontmatter, body, and the dates it
was discussed), newest first. "Date" means the date of the conversation a memory came
from (parsed from its ``sources``), not when the file was written [see ``memory/dates.py``].
"""

import logging
from typing import Any, Dict
from datetime import datetime, timezone

from reachy_mini_conversation_app.memory.dates import event_date, source_dates, present_memory
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
_MIN_DATE = datetime.min.replace(tzinfo=timezone.utc)


def _parse_date_arg(value: Any) -> datetime | None:
    """Parse a ``YYYY-MM-DD`` argument into a UTC datetime, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _in_range(memory: Dict[str, Any], date_from: datetime | None, date_to: datetime | None) -> bool:
    """Return True if any conversation date of ``memory`` falls within [from, to]."""
    days = source_dates(memory)
    if not days:
        when = event_date(memory)  # falls back to created
        if when is None:
            return False
        days = [when]
    for day in days:
        if date_from is not None and day < date_from:
            continue
        if date_to is not None and day > date_to:
            continue
        return True
    return False


class RecallMemories(Tool):
    """Filter memories by tag and/or conversation-date range, newest first."""

    name = "recall_memories"
    description = (
        "Recall stored memories filtered by topic and/or date, returning each "
        "match's full text (not just titles or IDs). Provide a `tag` (a topic shown "
        "in the MEMORY index), a date range (`date_from`/`date_to` as YYYY-MM-DD), "
        "or both; at least one is required. Dates refer to when something was "
        "discussed: for 'what did we do yesterday?', work out yesterday's date from "
        "the current date in your context and pass it as both date_from and date_to. "
        "Before calling, tell the user you're checking your memory (e.g. 'Let me "
        "think back...'). Returns up to `limit` memories (full body plus the dates "
        "each was discussed), newest first."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "tag": {
                "type": "string",
                "description": "Topic tag to match (case-sensitive). Optional.",
            },
            "date_from": {
                "type": "string",
                "description": "Earliest conversation date to include, YYYY-MM-DD. Optional.",
            },
            "date_to": {
                "type": "string",
                "description": "Latest conversation date to include, YYYY-MM-DD. Optional.",
            },
            "limit": {
                "type": "integer",
                "description": f"Max memories to return (default {DEFAULT_LIMIT}).",
            },
        },
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return up to ``limit`` memories matching the given filters, newest first."""
        if deps.memory_manager is None:
            return {"status": "memory_disabled"}

        tag = (kwargs.get("tag") or "").strip() or None
        date_from = _parse_date_arg(kwargs.get("date_from"))
        date_to = _parse_date_arg(kwargs.get("date_to"))
        if tag is None and date_from is None and date_to is None:
            return {"error": "provide at least one of: tag, date_from, date_to"}

        limit = int(kwargs.get("limit") or DEFAULT_LIMIT)
        limit = max(1, min(limit, 20))

        logger.info(
            "Tool call: recall_memories tag=%r date_from=%r date_to=%r limit=%d",
            tag,
            kwargs.get("date_from"),
            kwargs.get("date_to"),
            limit,
        )

        manager = deps.memory_manager
        matches = manager.list_memories(tag=tag) if tag is not None else manager.list_memories()

        if date_from is not None or date_to is not None:
            # A memory matches if ANY of its conversation dates falls in the range
            # (a single memory can span several days).
            matches = [m for m in matches if _in_range(m, date_from, date_to)]

        matches.sort(key=lambda m: event_date(m) or _MIN_DATE, reverse=True)

        bundle = []
        for entry in matches[:limit]:
            try:
                bundle.append(present_memory(manager.read_memory(entry["id"])))
            except FileNotFoundError:
                logger.warning("recall_memories: indexed memory %s missing on disk", entry["id"])

        return {
            "tag": tag,
            "date_from": kwargs.get("date_from"),
            "date_to": kwargs.get("date_to"),
            "returned": len(bundle),
            "total_matches": len(matches),
            "memories": bundle,
        }
