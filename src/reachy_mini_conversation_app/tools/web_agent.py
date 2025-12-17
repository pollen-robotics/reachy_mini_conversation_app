"""Web agent browser automation tool using web-agent REST API."""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class WebAgent(Tool):
    """Execute browser automation tasks via web-agent service."""

    name = "web_agent"
    description = (
        "Execute browser automation tasks autonomously. Use this when you need to interact with websites: "
        "fill forms, click buttons, navigate pages, extract information, or perform multi-step web workflows. "
        "Provide a natural language objective and optionally a starting URL."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "Natural language description of what to accomplish in the browser",
            },
            "url": {
                "type": "string",
                "description": "Starting URL (optional, defaults to about:blank)",
                "default": "about:blank",
            },
            "wait_timeout": {
                "type": "integer",
                "description": "Page load timeout in milliseconds (optional, defaults to 5000)",
                "default": 5000,
            },
        },
        "required": ["objective"],
    }

    def _get_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        endpoint = os.getenv("WEB_AGENT_ENDPOINT", "http://localhost:8080")
        path = os.getenv("WEB_AGENT_PATH")
        provider = os.getenv("WEB_AGENT_PROVIDER", "openai")
        model = os.getenv("WEB_AGENT_MODEL", "gpt-4o")

        # Get API key based on provider
        api_key = None
        provider_lower = provider.lower()
        if provider_lower == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider_lower == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider_lower == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
        elif provider_lower == "cerebras":
            api_key = os.getenv("CEREBRAS_API_KEY")

        return {
            "endpoint": endpoint.rstrip("/"),
            "path": path,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "task_timeout": int(os.getenv("WEB_AGENT_TASK_TIMEOUT", "300")),
            "startup_timeout": int(os.getenv("WEB_AGENT_STARTUP_TIMEOUT", "120")),
        }

    async def _find_docker_compose_command(self) -> Optional[List[str]]:
        """Find the appropriate docker compose command."""
        # Try docker compose (v2) first
        if shutil.which("docker"):
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "compose",
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if proc.returncode == 0:
                return ["docker", "compose"]

        # Fall back to docker-compose (v1)
        if shutil.which("docker-compose"):
            return ["docker-compose"]

        return None

    async def _start_docker_container(self, path: str, timeout: int) -> bool:
        """Start the web-agent Docker container."""
        compose_cmd = await self._find_docker_compose_command()
        if not compose_cmd:
            logger.warning(
                "docker compose not found - assuming container is managed externally"
            )
            return True

        compose_file = Path(path) / "deployments" / "local" / "docker-compose.yml"
        if not compose_file.exists():
            logger.error("docker-compose.yml not found at %s", compose_file)
            return False

        logger.info("Starting web-agent container...")
        try:
            proc = await asyncio.create_subprocess_exec(
                *compose_cmd,
                "-f",
                str(compose_file),
                "up",
                "-d",
                cwd=str(Path(path) / "deployments" / "local"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                logger.error("Failed to start container: %s", stderr.decode())
                return False

            logger.info("Docker compose started successfully")
            return True
        except asyncio.TimeoutError:
            logger.error("Timed out starting Docker container")
            return False
        except Exception as e:
            logger.error("Error starting Docker container: %s", e)
            return False

    async def _check_health(self, client: httpx.AsyncClient, endpoint: str) -> bool:
        """Check if web-agent service is healthy."""
        try:
            response = await client.get(f"{endpoint}/status", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def _wait_for_healthy(
        self, client: httpx.AsyncClient, endpoint: str, max_wait: int
    ) -> bool:
        """Wait for web-agent to become healthy with exponential backoff."""
        start_time = time.time()
        delay = 2.0

        while time.time() - start_time < max_wait:
            if await self._check_health(client, endpoint):
                logger.info(
                    "Web-agent is healthy (waited %.1fs)", time.time() - start_time
                )
                return True

            logger.debug("Web-agent not ready, waiting %.1fs...", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

        return False

    def _extract_response_text(self, data: Any) -> str:
        """Extract response text from web-agent API response format."""
        try:
            if isinstance(data, list) and len(data) > 0:
                message = data[0]
                content = message.get("content", [])
                if isinstance(content, list):
                    # Look for output_text type first
                    for item in content:
                        if item.get("type") == "output_text":
                            return item.get("text", "")
                    # Fall back to first text item
                    for item in content:
                        if "text" in item:
                            return item.get("text", "")
                elif isinstance(content, str):
                    return content
            elif isinstance(data, dict):
                if "response" in data:
                    return str(data["response"])
                if "content" in data:
                    return str(data["content"])
            return str(data)
        except Exception:
            return str(data)

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute browser automation task via web-agent."""
        objective = kwargs.get("objective")
        url = kwargs.get("url", "about:blank")
        wait_timeout = kwargs.get("wait_timeout", 5000)

        if not objective:
            return {"error": "objective is required"}

        config = self._get_config()

        if not config["api_key"]:
            return {
                "error": f"API key not configured for provider '{config['provider']}'. "
                f"Set the appropriate environment variable (e.g., OPENAI_API_KEY)."
            }

        logger.info("Web agent: objective=%s, url=%s", objective[:100], url)

        async with httpx.AsyncClient() as client:
            # Check if service is running
            is_healthy = await self._check_health(client, config["endpoint"])

            # Auto-start if not running and path is configured
            if not is_healthy and config["path"]:
                logger.info(
                    "Web-agent not running, attempting to start Docker container..."
                )
                started = await self._start_docker_container(
                    config["path"], config["startup_timeout"]
                )
                if started:
                    is_healthy = await self._wait_for_healthy(
                        client, config["endpoint"], config["startup_timeout"]
                    )

            if not is_healthy:
                return {
                    "error": "Web-agent service is not available. "
                    "Either start it manually or set WEB_AGENT_PATH to enable auto-start."
                }

            # Build request payload
            model_config = {
                "provider": config["provider"],
                "model": config["model"],
                "api_key": config["api_key"],
            }
            payload = {
                "input": objective,
                "url": url,
                "wait_timeout": wait_timeout,
                "model": {
                    "main_model": model_config,
                    "mini_model": model_config,
                    "nano_model": model_config,
                },
            }

            try:
                start_time = time.time()
                response = await client.post(
                    f"{config['endpoint']}/v1/responses",
                    json=payload,
                    timeout=float(config["task_timeout"]),
                )
                execution_time = time.time() - start_time

                response.raise_for_status()
                data = response.json()

                # Extract response text
                result_text = self._extract_response_text(data)

                logger.info(
                    "Web agent completed in %.1fs: %s",
                    execution_time,
                    result_text[:100] if result_text else "no response",
                )

                return {
                    "status": "success",
                    "response": result_text,
                    "execution_time_seconds": round(execution_time, 1),
                    "url": url,
                }

            except httpx.TimeoutException:
                return {"error": f"Task timed out after {config['task_timeout']} seconds"}
            except httpx.HTTPStatusError as e:
                error_msg = f"API error: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg} - {error_data['error']}"
                except Exception:
                    pass
                return {"error": error_msg}
            except Exception as e:
                logger.exception("Web agent failed")
                return {"error": f"Web agent failed: {str(e)}"}
