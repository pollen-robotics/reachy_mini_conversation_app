import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class DoNothing(Tool):
    """Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill."""

    name = "do_nothing"
    description = "Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional reason for doing nothing (e.g., 'contemplating existence', 'saving energy', 'being mysterious')",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Do nothing - stay still and silent."""
        reason = kwargs.get("reason", "just chilling")
        logger.info("Tool call: do_nothing reason=%s", reason)
        return {"status": "doing nothing", "reason": reason}
