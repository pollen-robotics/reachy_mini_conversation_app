"""Local idle tool policy.

Idle actions are selected by the app, not by the realtime model.  The model
should not see the no-op idle tool in its function-calling tool list.
"""

from __future__ import annotations
import random
from typing import Any, Iterable
from dataclasses import dataclass


IDLE_DO_NOTHING_TOOL_NAME = "idle_do_nothing"
LLM_HIDDEN_TOOL_NAMES = frozenset({IDLE_DO_NOTHING_TOOL_NAME})

_IDLE_TOOL_WEIGHTS: tuple[tuple[str, float], ...] = (
    (IDLE_DO_NOTHING_TOOL_NAME, 0.80),
    ("dance", 0.08),
    ("play_emotion", 0.08),
    ("move_head", 0.04),
)
_IDLE_MOVE_HEAD_DIRECTIONS = ("left", "right", "up", "down", "front")


@dataclass(frozen=True)
class IdleToolChoice:
    """Concrete tool invocation selected for an idle turn."""

    tool_name: str
    arguments: dict[str, Any]


def choose_idle_tool(available_tool_names: Iterable[str]) -> IdleToolChoice | None:
    """Choose a weighted random idle tool from the currently available tools."""
    available = set(available_tool_names)
    weighted_candidates = [(name, weight) for name, weight in _IDLE_TOOL_WEIGHTS if name in available]
    if not weighted_candidates:
        return None

    names, weights = zip(*weighted_candidates)
    tool_name = random.choices(names, weights=weights, k=1)[0]

    if tool_name == IDLE_DO_NOTHING_TOOL_NAME:
        return IdleToolChoice(
            tool_name=tool_name,
            arguments={"reason": "random idle policy selected stillness"},
        )
    if tool_name == "move_head":
        return IdleToolChoice(
            tool_name=tool_name,
            arguments={"direction": random.choice(_IDLE_MOVE_HEAD_DIRECTIONS)},
        )
    return IdleToolChoice(tool_name=tool_name, arguments={})
