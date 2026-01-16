"""Web search tool using SerpAPI."""

import logging
import os
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class WebSearch(Tool):
    """Search the web using SerpAPI and return results for conversation."""

    name = "web_search"
    description = "Search the web for current information. Use this when the user asks about recent events, facts, or anything you need to look up."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-5)",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute web search and return formatted results."""
        query = kwargs.get("query")
        num_results = min(kwargs.get("num_results", 3), 5)

        if not query:
            return {"error": "query is required"}

        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            logger.error("SERPAPI_KEY environment variable not set")
            return {"error": "SERPAPI_KEY not configured"}

        logger.info("Web search: query=%s, num_results=%d", query, num_results)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "api_key": api_key,
                        "engine": "google",
                        "num": num_results,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

            # Log available result types for debugging
            logger.debug("SerpAPI response keys: %s", list(data.keys()))

            # Extract results from multiple possible sources
            results = []

            # Check organic results (standard web search)
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                })

            # Check news results if no organic results
            if not results:
                for item in data.get("news_results", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", item.get("source", "")),
                        "link": item.get("link", ""),
                    })

            # Check top stories if still no results
            if not results:
                for item in data.get("top_stories", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("source", {}).get("name", "") if isinstance(item.get("source"), dict) else "",
                        "link": item.get("link", ""),
                    })

            if not results:
                logger.info("No results found for query: %s", query)
                return {"status": "no_results", "query": query}

            logger.info("Found %d results for query: %s", len(results), query)
            return {
                "status": "success",
                "query": query,
                "results": results,
            }

        except httpx.HTTPStatusError as e:
            logger.exception("SerpAPI HTTP error")
            return {"error": f"Search API error: {e.response.status_code}"}
        except httpx.TimeoutException:
            logger.exception("SerpAPI timeout")
            return {"error": "Search timed out"}
        except Exception as e:
            logger.exception("Web search failed")
            return {"error": f"Search failed: {str(e)}"}
