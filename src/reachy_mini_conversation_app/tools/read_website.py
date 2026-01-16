"""Website reader tool using Jina AI Reader."""

import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class ReadWebsite(Tool):
    """Read and extract content from websites using Jina Reader."""

    name = "read_website"
    description = "Read the content of one or more website URLs. Use this to read and summarize webpages. Pass an array of URLs to read multiple pages at once."
    parameters_schema = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to read",
            },
        },
        "required": ["urls"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch website content via Jina Reader for one or more URLs."""
        urls = kwargs.get("urls", [])

        if not urls:
            return {"error": "urls is required"}

        if not isinstance(urls, list):
            urls = [urls]

        # Adjust content limit based on number of URLs
        content_limit = 4000 if len(urls) == 1 else 2000

        # Fetch all URLs concurrently
        tasks = [self._fetch_url(url, content_limit) for url in urls]
        results = await asyncio.gather(*tasks)

        logger.info("Read %d website(s)", len(results))
        return {
            "status": "success",
            "results": results,
        }

    async def _fetch_url(self, url: str, content_limit: int) -> Dict[str, Any]:
        """Fetch a single URL via Jina Reader."""
        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        jina_url = f"https://r.jina.ai/{url}"
        logger.info("Reading website: %s", url)

        try:
            headers = {}
            # Optional: Use Jina API key for higher rate limits
            jina_key = os.getenv("JINA_API_KEY")
            if jina_key:
                headers["Authorization"] = f"Bearer {jina_key}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    jina_url,
                    headers=headers,
                    timeout=30.0,
                    follow_redirects=True,
                )
                response.raise_for_status()
                content = response.text

            # Truncate if too long
            if len(content) > content_limit:
                content = content[:content_limit] + "\n\n[Content truncated...]"

            logger.info("Successfully read website: %s (%d chars)", url, len(content))
            return {
                "url": url,
                "status": "success",
                "content": content,
            }

        except httpx.HTTPStatusError as e:
            logger.exception("Jina Reader HTTP error for %s", url)
            return {"url": url, "status": "error", "error": f"HTTP {e.response.status_code}"}
        except httpx.TimeoutException:
            logger.exception("Jina Reader timeout for %s", url)
            return {"url": url, "status": "error", "error": "Timeout"}
        except Exception as e:
            logger.exception("Website read failed for %s", url)
            return {"url": url, "status": "error", "error": str(e)}
