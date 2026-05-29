"""Local idle tool selection."""

from __future__ import annotations
import random
from typing import Any, Final, Iterable


_IDLE_TOOL_WEIGHTS: Final[tuple[tuple[str, float], ...]] = (
    ("idle_do_nothing", 0.60),
    ("dance", 0.16),
    ("play_emotion", 0.16),
    ("move_head", 0.08),
)
_IDLE_MOVE_HEAD_DIRECTIONS: Final[tuple[str, ...]] = ("left", "right", "up", "down", "front")


def choose_idle_tool_call(available_tool_names: Iterable[str]) -> tuple[str, dict[str, Any]] | None:
    """Choose a weighted idle tool call from the tools available to the session."""
    available = set(available_tool_names)
    candidates = [(name, weight) for name, weight in _IDLE_TOOL_WEIGHTS if name in available]
    if not candidates:
        return None

    names, weights = zip(*candidates)
    tool_name = random.choices(names, weights=weights, k=1)[0]
    if tool_name == "move_head":
        return tool_name, {"direction": random.choice(_IDLE_MOVE_HEAD_DIRECTIONS)}
    if tool_name == "idle_do_nothing":
        return tool_name, {"reason": "random idle policy selected stillness"}
    return tool_name, {}
