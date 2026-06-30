import logging
from typing import Any

from reachy_mini_conversation_app.memory import forget_memory_fact
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Forget(Tool):
    """Remove one long-term memory fact."""

    name = "forget"
    description = (
        "Remove a previously saved fact from long-term memory. Call this when the user asks you to forget something, "
        "or when saved information becomes obsolete. Match by a specific free-text phrase present in the fact."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A short search phrase that should be present in the fact to remove. Matching is case-insensitive."
                ),
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Forget one memory fact by query."""
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            logger.warning("forget: empty query")
            return {"error": "query must be a non-empty string"}

        result = forget_memory_fact(deps.instance_path, query=query)
        if result.removed is None:
            logger.info("Tool call: forget query=%s no_match", query[:120])
            return {"error": f'no memory matched "{query}"; nothing was removed'}

        response: dict[str, Any] = {
            "removed": result.removed.text,
            "memory_id": result.removed.id,
        }
        if len(result.candidates) > 1:
            response["other_matches"] = [fact.text for fact in result.candidates[1:]]

        logger.info("Tool call: forget query=%s removed=%s", query[:120], result.removed.text[:120])
        return response
