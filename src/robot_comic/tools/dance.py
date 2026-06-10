import random
import logging
from typing import Any, Dict

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize dance library
try:
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    from robot_comic.dance_emotion_moves import DanceQueueMove

    DANCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dance library not available: {e}")
    AVAILABLE_MOVES = {}
    DANCE_AVAILABLE = False


def _get_dance_denylist() -> frozenset[str]:
    """Return the current chassis-safety dance denylist from config (#542).

    Imported lazily to avoid a circular import at module load time. Always
    reflects the live config value so .env overrides applied by
    refresh_runtime_config_from_env() take effect at call time.
    """
    from robot_comic.config import config  # noqa: PLC0415

    return config.DANCE_DENYLIST


def _safe_dance_names() -> list[str]:
    """Return the available dance names with the denylist filtered out."""
    denylist = _get_dance_denylist()
    return [name for name in AVAILABLE_MOVES if name not in denylist]


def get_available_dances_and_descriptions() -> str:
    """Get formatted list of available (non-denylisted) dances with descriptions."""
    if not DANCE_AVAILABLE:
        return "Moves not available."

    safe_names = _safe_dance_names()
    if not safe_names:
        return "Moves not available."

    output = ""
    for move_name in safe_names:
        _func, _params, metadata = AVAILABLE_MOVES[move_name]
        description = metadata.get("description", "No description available.")
        output += f"{move_name}: {description}\n"
    return output


class Dance(Tool):
    """Play a named or random dance move once (or repeat). Non-blocking."""

    name = "dance"
    description = "Play a named or random dance move once (or repeat). Non-blocking."
    # NOTE: parameters_schema is a class attribute for compatibility with
    # Tool.spec(); Dance overrides spec() below so the LLM always receives an
    # enum that reflects the *current* denylist at call time (mirrors
    # PlayEmotion / #542).
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "enum": [],  # populated dynamically by spec()
                "description": "",  # populated dynamically by spec()
            },
            "repeat": {
                "type": "integer",
                "description": "How many times to repeat the move (default 1).",
            },
        },
        "required": [],
    }

    def spec(self) -> Dict[str, Any]:
        """Return the function spec with the move enum filtered by the live denylist."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {
                        "type": "string",
                        "enum": _safe_dance_names() if DANCE_AVAILABLE else [],
                        "description": (
                            "Name of the moves and their descriptions; omit for random.\n"
                            "Here is a list of the available moves, you MUST only choose from these:\n"
                            f"{get_available_dances_and_descriptions()}"
                        ),
                    },
                    "repeat": {
                        "type": "integer",
                        "description": "How many times to repeat the move (default 1).",
                    },
                },
                "required": [],
            },
        }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a named or random dance move once (or repeat). Non-blocking."""
        if not DANCE_AVAILABLE:
            return {"error": "Dance system not available"}

        safe_names = _safe_dance_names()
        if not safe_names:
            return {"error": "No moves currently available"}

        move_name = kwargs.get("move")
        repeat = int(kwargs.get("repeat", 1))

        logger.info("Tool call: dance move=%s repeat=%d", move_name, repeat)

        # Chassis-safety denylist check (#542): defense-in-depth guard — the
        # spec() override already removes these from the LLM's enum; this
        # catches the LLM ignoring the enum or the denylist changing between
        # spec generation and call time.
        if move_name and move_name in _get_dance_denylist():
            logger.warning("Refusing denylisted dance %r (REACHY_MINI_DANCE_DENYLIST chassis safety)", move_name)
            return {
                "error": (
                    f"dance '{move_name}' is denylisted by REACHY_MINI_DANCE_DENYLIST"
                    " (chassis safety). Pick a different move."
                )
            }

        if not move_name:
            move_name = random.choice(safe_names)

        if move_name not in safe_names:
            return {"error": f"Unknown dance move '{move_name}'. Available: {safe_names}"}

        # Add dance moves to queue
        movement_manager = deps.movement_manager
        speed = getattr(deps.movement_manager, "speed_factor", 1.0)
        for _ in range(repeat):
            dance_move = DanceQueueMove(move_name, speed_factor=speed)
            movement_manager.queue_move(dance_move)

        return {"status": "queued", "move": move_name, "repeat": repeat}
