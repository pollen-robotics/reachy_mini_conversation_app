import json
import random
import logging
from typing import Any, Dict
from pathlib import Path

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Emotions to filter out at the tool layer. These animations have keyframe
# amplitudes large enough that on at least one chassis they swing the head
# into the surround cowling. Until per-emotion velocity clamping lands
# (see #264), drop them entirely.
BLOCKED_EMOTION_PREFIXES: tuple[str, ...] = ("lonely",)


def _get_play_emotion_denylist() -> frozenset[str]:
    """Return the current chassis-safety denylist from config.

    Imported lazily to avoid a circular import at module load time.
    Always reflects the live config value so .env overrides applied by
    refresh_runtime_config_from_env() take effect at call time.
    """
    from robot_comic.config import config  # noqa: PLC0415

    return config.PLAY_EMOTION_DENYLIST


def _is_blocked_emotion(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in BLOCKED_EMOTION_PREFIXES)


# ---------------------------------------------------------------------------
# Emotion catalog (lazy-loaded — #323 lever 2)
# ---------------------------------------------------------------------------
# ``RecordedMoves("pollen-robotics/reachy-mini-emotions-library")`` was the
# single biggest cold-boot cost in the serial tool-load budget (~2.4s on a
# Pi 5) because it eagerly JSON-parses every emotion keyframe file from the
# local HuggingFace cache. The actual move data is only needed when an
# emotion is queued — but the LLM tool spec (parameters_schema) needs the
# emotion *names* and *descriptions* at session-config time.
#
# Strategy: persist the names + descriptions to a small JSON cache next to
# this module on first successful load. Subsequent boots read the cache
# (~ms) and defer ``RecordedMoves`` construction until the first
# ``play_emotion`` tool invocation. First-ever boot still pays the full
# cost so the cache can be populated. The cache auto-rebuilds if missing.
# Operators who want to force a refresh after a library update can delete
# the cache file (``_emotion_catalog_cache.json``) — the next boot
# regenerates it from the fresh keyframes.
_CATALOG_CACHE_PATH = Path(__file__).resolve().parent / "_emotion_catalog_cache.json"

_CATALOG_NAMES: list[str] = []
_CATALOG_DESCRIPTIONS: dict[str, str] = {}
RECORDED_MOVES: Any = None  # lazy: populated on first execute, or at module
# load time if no cache exists. Treated as ``None`` until needed.
EMOTION_AVAILABLE: bool = False


def _load_catalog_from_cache() -> bool:
    """Populate ``_CATALOG_NAMES``/``_CATALOG_DESCRIPTIONS`` from the JSON cache.

    Returns True if the cache was loaded successfully.
    """
    global _CATALOG_NAMES, _CATALOG_DESCRIPTIONS, EMOTION_AVAILABLE
    if not _CATALOG_CACHE_PATH.is_file():
        return False
    try:
        data = json.loads(_CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
        names = data.get("names")
        descs = data.get("descriptions")
        if not isinstance(names, list) or not isinstance(descs, dict):
            return False
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read emotion catalog cache: %s", exc)
        return False
    _CATALOG_NAMES = [n for n in names if isinstance(n, str) and not _is_blocked_emotion(n)]
    _CATALOG_DESCRIPTIONS = {k: v for k, v in descs.items() if isinstance(k, str) and k in _CATALOG_NAMES}
    EMOTION_AVAILABLE = True
    return True


def _refresh_catalog_from_moves(moves: Any) -> None:
    """Refresh in-memory catalog + write the JSON cache from a live ``RecordedMoves``."""
    global _CATALOG_NAMES, _CATALOG_DESCRIPTIONS
    raw_names = list(moves.list_moves())
    raw_descs: dict[str, str] = {}
    for name in raw_names:
        try:
            raw_descs[name] = moves.get(name).description
        except Exception as exc:  # pragma: no cover — defensive against upstream changes
            logger.debug("Could not read description for %r: %s", name, exc)
    _CATALOG_NAMES = [n for n in raw_names if not _is_blocked_emotion(n)]
    _CATALOG_DESCRIPTIONS = {k: v for k, v in raw_descs.items() if k in _CATALOG_NAMES}
    try:
        _CATALOG_CACHE_PATH.write_text(
            json.dumps({"names": raw_names, "descriptions": raw_descs}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("Failed to write emotion catalog cache: %s", exc)


def _ensure_recorded_moves() -> Any:
    """Lazily construct ``RecordedMoves`` on first emotion playback.

    Returns the live instance (or ``None`` if the upstream library is not
    importable). Updates the JSON cache when a fresh instance is loaded.
    """
    global RECORDED_MOVES, EMOTION_AVAILABLE
    if RECORDED_MOVES is not None:
        return RECORDED_MOVES
    try:
        # noqa: PLC0415 — deferred from boot; pulls heavy reachy_mini deps
        from reachy_mini.motion.recorded_move import RecordedMoves  # noqa: PLC0415

        RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        EMOTION_AVAILABLE = True
        _refresh_catalog_from_moves(RECORDED_MOVES)
        return RECORDED_MOVES
    except ImportError as exc:
        logger.warning("Emotion library not available: %s", exc)
        EMOTION_AVAILABLE = False
        return None


# Module init: prefer the cache. If it's missing (first boot or operator
# deleted it for a forced refresh), eagerly load now so the LLM tool spec
# below has the correct enum at class-definition time.
if not _load_catalog_from_cache():
    _ensure_recorded_moves()


def _safe_emotion_names() -> list[str]:
    """Return the available emotion names with blocked prefixes and denylist filtered out."""
    denylist = _get_play_emotion_denylist()
    return [n for n in _CATALOG_NAMES if n not in denylist]


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not _CATALOG_NAMES:
        return "Emotions not available" if not EMOTION_AVAILABLE else "No emotions currently available"
    lines = ["Available emotions:"]
    for name in _CATALOG_NAMES:
        desc = _CATALOG_DESCRIPTIONS.get(name, "")
        lines.append(f" - {name}: {desc}")
    return "\n".join(lines) + "\n"


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    # NOTE: parameters_schema is defined as a class attribute for compatibility
    # with Tool.spec(); PlayEmotion overrides spec() below so the LLM always
    # receives an enum that reflects the *current* denylist at call time.
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": [],  # populated dynamically by spec()
                "description": "",  # populated dynamically by spec()
            },
        },
        "required": [],
    }

    def spec(self) -> Dict[str, Any]:
        """Return the function spec with emotion enum filtered by the live denylist."""
        safe_names = _safe_emotion_names()
        desc = get_available_emotions_and_descriptions()
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": safe_names,
                        "description": (
                            "Name of the emotion to play; omit for random.\n"
                            "Here is a list of the available emotions, you MUST only choose from these:\n"
                            f"{desc}"
                        ),
                    },
                },
                "required": [],
            },
        }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        # First-call lazy hydration of the heavy keyframe library when boot
        # used the JSON cache fast-path. Subsequent calls reuse the instance.
        moves = _ensure_recorded_moves()
        if not EMOTION_AVAILABLE or moves is None:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")

        # Defence in depth: if the LLM hallucinates a blocked emotion despite
        # it being absent from the enum, refuse here too.
        if emotion_name and _is_blocked_emotion(emotion_name):
            logger.warning("Refusing blocked emotion %r (matches BLOCKED_EMOTION_PREFIXES)", emotion_name)
            return {"error": f"Emotion '{emotion_name}' is disabled on this chassis."}

        # Chassis-safety denylist check (issue #480): defense-in-depth guard
        # against emotions that drive the head chin-down toward the cowling.
        # The spec() override already removes these from the LLM's enum; this
        # check catches any case where the LLM ignores the enum or the denylist
        # changed between spec generation and call time.
        denylist = _get_play_emotion_denylist()
        if emotion_name and emotion_name in denylist:
            logger.warning(
                "Refusing denylisted emotion %r (REACHY_MINI_PLAY_EMOTION_DENYLIST chassis safety)", emotion_name
            )
            return {
                "error": (
                    f"emotion '{emotion_name}' is denylisted by REACHY_MINI_PLAY_EMOTION_DENYLIST"
                    " (chassis safety). Pick a different emotion."
                )
            }

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Check if emotion exists
        try:
            emotion_names = _safe_emotion_names()
            if not emotion_names:
                return {"error": "No emotions currently available"}

            if not emotion_name:
                emotion_name = random.choice(emotion_names)

            if emotion_name not in emotion_names:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            from robot_comic.dance_emotion_moves import EmotionQueueMove  # noqa: PLC0415

            movement_manager = deps.movement_manager
            speed = getattr(deps.movement_manager, "speed_factor", 1.0)
            emotion_move = EmotionQueueMove(emotion_name, moves, speed_factor=speed)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
