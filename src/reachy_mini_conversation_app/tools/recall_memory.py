"""System tool ``recall_memory``: read one memory file plus its related_to neighbours."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory.dates import present_memory
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RecallMemory(Tool):
    """Read an atomic memory by ID, bundled with any memories it semantically depends on."""

    name = "recall_memory"
    description = (
        "Read a specific memory by its ID. The memory index (in your MEMORY section) "
        "lists IDs in square brackets, e.g. [2026-04-17_chess-openings_a3f]. "
        "This returns the full memory body plus every memory listed in its `related_to` "
        "field; use it whenever the user asks about a past topic. "
        "Before calling, tell the user you're checking your memory "
        "(e.g. 'Let me think back...' or 'That rings a bell, one moment...'). "
        "If the ID isn't found you'll get an error and a short list of existing IDs."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "The memory ID from the MEMORY index, e.g. '2026-04-17_chess-openings_a3f'. Required."
                ),
            },
        },
        "required": ["id"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the target memory + its related_to bundle."""
        if deps.memory_manager is None:
            return {"status": "memory_disabled"}

        memory_id = (kwargs.get("id") or "").strip()
        logger.info("Tool call: recall_memory id=%r", memory_id)
        if not memory_id:
            return {"error": "id is required"}

        manager = deps.memory_manager
        try:
            target = manager.read_memory(memory_id)
        except FileNotFoundError:
            known = [m["id"] for m in manager.list_memories()]
            return {
                "error": f"memory '{memory_id}' not found",
                "known_ids_sample": known[:10],
            }
        except ValueError as e:
            return {"error": str(e)}

        related_ids = target["frontmatter"].get("related_to") or []
        related_payload = []
        for rid in related_ids:
            try:
                related_payload.append(present_memory(manager.read_memory(rid)))
            except FileNotFoundError:
                logger.warning("recall_memory: related %s missing", rid)

        return {
            "memory": present_memory(target),
            "related": related_payload,
        }
