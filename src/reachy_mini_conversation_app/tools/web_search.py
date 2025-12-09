"""Web search tool using DuckDuckGo."""
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WebSearch(Tool):
    """Search the web for information using DuckDuckGo."""

    name = "web_search"
    description = (
        "Search the web for current information, facts, news, or any topic. "
        "Returns search results with titles, snippets, and links. "
        "Use this when you need up-to-date information or to answer questions about current events, "
        "facts you're unsure about, or topics outside your usual knowdledge."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web (e.g., 'latest AI news', 'weather in Paris', 'what is quantum computing')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Perform a web search using DuckDuckGo."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return {"error": "No query provided"}

        logger.info("Tool call: web_search query=%s max_results=%d", query, max_results)

        try:
            # Import here to avoid issues if library not installed
            from ddgs import DDGS

            # Perform search
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return {
                    "status": "success",
                    "query": query,
                    "message": "No results found for this query.",
                    "results": [],
                }

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    {
                        "position": i,
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", ""),
                    }
                )

            return {
                "status": "success",
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results,
            }

        except ImportError:
            error_msg = (
                "Web search library not installed. "
                "Please install with: pip install duckduckgo-search"
            )
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Web search failed: {type(e).__name__}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
