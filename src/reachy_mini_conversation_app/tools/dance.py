import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize dance library
try:
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

    DANCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dance library not available: {e}")
    AVAILABLE_MOVES = {}
    DANCE_AVAILABLE = False


def _build_dance_description() -> str:
    """Generate dance move descriptions dynamically from AVAILABLE_MOVES.

    Returns:
        str: Formatted description of available dance moves.

    """
    if not AVAILABLE_MOVES: #if the AVAILABLE_MOVES is empty
        return "No moves currently available."

    moves_list = []
    for move_name, (func, params, metadata) in AVAILABLE_MOVES.items():
        description = metadata.get("description", "No description available.")
        moves_list.append(f"{move_name}: {description}")

    formatted_moves = "\n                                    ".join(moves_list)
    return f"""Name of the move; use 'random' or omit for random.
                                Here is a list of the available moves:
                                    {formatted_moves}
            """


class Dance(Tool):
    """Play a named or random dance move once (or repeat). Non-blocking."""

    name = "dance"
    description = "Play a named or random dance move once (or repeat). Non-blocking."
    parameters_schema = {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": _build_dance_description(),
            },
            "repeat": {
                "type": "integer",
                "description": "How many times to repeat the move (default 1).",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a named or random dance move once (or repeat). Non-blocking."""
        if not DANCE_AVAILABLE:
            return {"error": "Dance system not available"}

        if not AVAILABLE_MOVES: #if AVAILABLE_MOVES is empty
            return {"error": "No moves currently available"}

        move_name = kwargs.get("move")
        repeat = int(kwargs.get("repeat", 1))

        logger.info("Tool call: dance move=%s repeat=%d", move_name, repeat)

        if not move_name or move_name == "random":
            import random

            move_name = random.choice(list(AVAILABLE_MOVES.keys()))

        if move_name not in AVAILABLE_MOVES:
            return {"error": f"Unknown dance move '{move_name}'. Available: {list(AVAILABLE_MOVES.keys())}"}

        # Add dance moves to queue
        movement_manager = deps.movement_manager
        for _ in range(repeat):
            dance_move = DanceQueueMove(move_name)
            movement_manager.queue_move(dance_move)

        return {"status": "queued", "move": move_name, "repeat": repeat}
