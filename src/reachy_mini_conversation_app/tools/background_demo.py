"""Background demo tool - demonstrates background task execution."""

import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.background_tasks import BackgroundTaskManager


logger = logging.getLogger(__name__)


class BackgroundDemoTool(Tool):
    """Demo tool that runs a task in background for testing purposes."""

    name = "background_demo"
    description = (
        "Demo tool that runs a task in the background for a specified duration. "
        "Use this to test background task functionality. "
        "Set background=true to run asynchronously without blocking."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "duration": {
                "type": "integer",
                "description": "Duration in seconds to run the demo task (default: 5)",
            },
            "background": {
                "type": "boolean",
                "description": "Run in background (true) or synchronously (false). Default: true",
            },
            "with_progress": {
                "type": "boolean",
                "description": "Track progress updates during execution (default: false)",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run the demo task."""
        duration = kwargs.get("duration", 5)
        background = kwargs.get("background", True)
        with_progress = kwargs.get("with_progress", False)

        # Validate duration
        if not isinstance(duration, int) or duration < 1:
            duration = 5
        if duration > 60:
            duration = 60  # Cap at 60 seconds for safety

        logger.info(
            f"Tool call: background_demo duration={duration} background={background} with_progress={with_progress}"
        )

        if background:
            # Run in background
            manager = BackgroundTaskManager.get_instance()

            task = await manager.start_task(
                name="background_demo",
                description=f"Demo task running for {duration} seconds",
                coroutine=self._run_demo(duration, with_progress),
                with_progress=with_progress,
            )

            return {
                "status": "started",
                "task_id": task.id,
                "duration": duration,
                "with_progress": with_progress,
                "message": f"Demo task started in background for {duration} seconds. I'll notify you when it's done.",
            }
        else:
            # Run synchronously (blocking)
            result = await self._run_demo(duration, with_progress=False)
            return result

    async def _run_demo(self, duration: int, with_progress: bool = False) -> Dict[str, Any]:
        """Run the actual demo task."""
        logger.info(f"Demo task started: duration={duration}s, with_progress={with_progress}")

        if with_progress:
            manager = BackgroundTaskManager.get_instance()
            # Get current task ID from the running tasks
            running_tasks = manager.get_running_tasks()
            current_task_id = None
            for task in running_tasks:
                if task.name == "background_demo":
                    current_task_id = task.id
                    break

            # Progress updates every second
            for i in range(duration):
                await asyncio.sleep(1)
                progress = (i + 1) / duration
                if current_task_id:
                    await manager.update_progress(
                        current_task_id,
                        progress,
                        f"{i + 1}/{duration} seconds elapsed",
                    )
                logger.debug(f"Demo task progress: {progress:.0%}")
        else:
            # Simple sleep without progress updates
            await asyncio.sleep(duration)

        logger.info(f"Demo task completed after {duration} seconds")

        return {
            "status": "success",
            "duration": duration,
            "message": f"Demo task completed successfully after {duration} seconds!",
        }
