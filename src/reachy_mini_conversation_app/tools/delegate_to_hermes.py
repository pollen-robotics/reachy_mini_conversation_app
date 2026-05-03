"""Conversation tool for delegating external tasks to Hermes."""

from __future__ import annotations
import logging
from typing import Any, Dict, cast

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class DelegateToHermes(Tool):
    """Delegate current-info, web, coding, or long-running tool work to Hermes."""

    name = "delegate_to_hermes"
    description = (
        "Delegate tasks that need current information, web research, Codex/code assistance, MCP tools, "
        "or longer-running external tool work to the user's Hermes agent. Do not use this for ordinary "
        "conversation or local Reachy movement. Home Assistant work is deferred unless Hermes is configured for it."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The concrete user-facing task Hermes should complete.",
            },
            "why_needed": {
                "type": "string",
                "description": "Briefly explain why this needs Hermes instead of local conversation.",
            },
            "response_style": {
                "type": "string",
                "enum": ["brief", "detailed", "step_by_step"],
                "description": "Preferred style for the result. Use brief for spoken answers by default.",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional web domains Hermes should prefer or limit itself to.",
            },
        },
        "required": ["task"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run the Hermes delegation tool."""
        task = str(kwargs.get("task", "")).strip()
        why_needed_raw = kwargs.get("why_needed")
        response_style = str(kwargs.get("response_style") or "brief")
        allowed_domains = kwargs.get("allowed_domains")

        if not task:
            return {
                "status": "error",
                "answer": "No task was provided for Hermes delegation.",
                "error_message": "empty_task",
            }

        if deps.hermes_client is None:
            return {
                "status": "disabled",
                "answer": "Hermes delegation is not configured on this Reachy Mini.",
                "error_message": "hermes_client_missing",
            }

        if response_style not in {"brief", "detailed", "step_by_step"}:
            response_style = "brief"

        if not isinstance(allowed_domains, list):
            allowed_domains = []
        allowed_domain_strings = [str(domain) for domain in allowed_domains if str(domain).strip()]

        logger.info("Tool call: delegate_to_hermes response_style=%s", response_style)
        result = await deps.hermes_client.delegate(
            task=task,
            why_needed=str(why_needed_raw).strip() if why_needed_raw else None,
            response_style=response_style,
            allowed_domains=allowed_domain_strings,
            context={
                "profile": "default",
                "robot": "reachy_mini",
                "local_tool": self.name,
            },
            urgency="interactive",
        )

        # BackgroundToolManager treats any non-None top-level `error` as a failed tool and drops
        # the rest of the structured payload. Preserve normalized Hermes failure details for the LLM.
        if result.get("error") is not None:
            result["error_message"] = result.pop("error")
        return cast(Dict[str, Any], result)
