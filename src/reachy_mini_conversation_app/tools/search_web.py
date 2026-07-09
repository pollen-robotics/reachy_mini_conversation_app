"""Web search tool backed by DuckDuckGo."""

import asyncio
import logging
from typing import Any

from ddgs import DDGS

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

MAX_RESULTS = 10
DEFAULT_RESULTS = 5


class SearchWeb(Tool):
    """Search the web with DuckDuckGo and return a short list of results."""

    name = "search_web"
    description = (
        "Search the web for current information and return a short list of results (title, snippet, url). "
        "Use this directly when the user asks to search, check the web, look something up, find today's events, "
        "or learn what is happening now. Do not just say you'll look it up."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search the web for."},
            "max_results": {
                "type": "integer",
                "description": f"Number of results to return, 1-{MAX_RESULTS} (default {DEFAULT_RESULTS}).",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Search the web off the event loop and return structured results."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}

        try:
            max_results = int(kwargs.get("max_results", DEFAULT_RESULTS))
        except (TypeError, ValueError):
            max_results = DEFAULT_RESULTS
        max_results = max(1, min(max_results, MAX_RESULTS))

        logger.info("search_web query=%s max_results=%d", query, max_results)
        results = await asyncio.to_thread(_search, query, max_results)
        return {"query": query, "results": results}


def _search(query: str, max_results: int) -> list[dict[str, str]]:
    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=max_results)
    return [
        {"title": hit.get("title", ""), "snippet": hit.get("body", ""), "url": hit.get("href", "")} for hit in hits
    ]
