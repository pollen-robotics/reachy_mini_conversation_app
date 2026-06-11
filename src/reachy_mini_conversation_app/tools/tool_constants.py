from enum import Enum


class ToolState(Enum):
    """Status of a background tool."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SystemTool(Enum):
    """System tools are tools that are used to manage the background tool manager."""

    TASK_STATUS = "task_status"
    TASK_CANCEL = "task_cancel"


# Pure side effects with nothing to narrate, skip the spoken follow-up.
SILENT_TOOLS: frozenset[str] = frozenset(
    {
        "dance",
        "stop_dance",
        "play_emotion",
        "stop_emotion",
        "move_head",
    }
)
