"""Search memory tool for full-text search across memory blocks."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory import get_memory_manager
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class SearchMemory(Tool):
    """Search memory for information about user, past conversations, or robot context."""

    name = "search_memory"
    description = "Search memory for information about the user, past conversations, or robot context. Use this when you need to recall specific information."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information in memory",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute memory search and return results."""
        query = kwargs.get("query")

        if not query:
            return {"error": "query is required"}

        logger.info("Searching memory: query=%s", query)

        try:
            manager = get_memory_manager()
            await manager.initialize()

            results = await manager.search_blocks(query)

            if not results:
                logger.info("No results found for query: %s", query)
                return {
                    "status": "no_results",
                    "query": query,
                    "message": "No matching memory found",
                }

            # Format results for LLM
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "block_type": result.block.block_type,
                    "label": result.block.label,
                    "content": result.block.content,
                    "matches": result.matches,
                })

            logger.info("Found %d memory blocks for query: %s", len(results), query)
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
            }

        except Exception as e:
            logger.exception("Memory search failed")
            return {"error": f"Search failed: {str(e)}"}
