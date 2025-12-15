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


class Dance(Tool):
    """Play a named or random dance move once (or repeat). Non-blocking."""

    name = "dance"
    description = "Play a named or random dance move once (or repeat). Non-blocking."
    parameters_schema = {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": """Name of the move; use 'random' or omit for random.
                                    Here is a list of the available moves:
                                        simple_nod: A simple, continuous up-and-down nodding motion.
                                        head_tilt_roll: A continuous side-to-side head roll (ear to shoulder).
                                        side_to_side_sway: A smooth, side-to-side sway of the entire head.
                                        dizzy_spin: A circular 'dizzy' head motion combining roll and pitch.
                                        stumble_and_recover: A simulated stumble and recovery with multiple axis movements. Good vibes
                                        interwoven_spirals: A complex spiral motion using three axes at different frequencies.
                                        sharp_side_tilt: A sharp, quick side-to-side tilt using a triangle waveform.
                                        side_peekaboo: A multi-stage peekaboo performance, hiding and peeking to each side.
                                        yeah_nod: An emphatic two-part yeah nod using transient motions.
                                        uh_huh_tilt: A combined roll-and-pitch uh-huh gesture of agreement.
                                        neck_recoil: A quick, transient backward recoil of the neck.
                                        chin_lead: A forward motion led by the chin, combining translation and pitch.
                                        groovy_sway_and_roll: A side-to-side sway combined with a corresponding roll for a groovy effect.
                                        chicken_peck: A sharp, forward, chicken-like pecking motion.
                                        side_glance_flick: A quick glance to the side that holds, then returns.
                                        polyrhythm_combo: A 3-beat sway and a 2-beat nod create a polyrhythmic feel.
                                        grid_snap: A robotic, grid-snapping motion using square waveforms.
                                        pendulum_swing: A simple, smooth pendulum-like swing using a roll motion.
                                        jackson_square: Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint.
                """,
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
