import re
import random
import logging
import unicodedata
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False


EMOTION_INTENTS: tuple[str, ...] = (
    "random",
    "happy",
    "excited",
    "loving",
    "grateful",
    "proud",
    "success",
    "curious",
    "inquiring",
    "thinking",
    "attentive",
    "confused",
    "lost",
    "uncertain",
    "sad",
    "downcast",
    "lonely",
    "angry",
    "irritated",
    "displeased",
    "disgusted",
    "scared",
    "anxious",
    "surprised",
    "amazed",
    "calming",
    "relief",
    "impatient",
    "embarrassed",
    "uncomfortable",
    "bored",
    "tired",
    "sleepy",
    "yes",
    "yes_sad",
    "yes_understanding",
    "yes_proud",
    "no",
    "no_sad",
    "no_excited",
    "no_firm",
    "no_confused",
    "oops",
    "welcoming",
    "greeting",
    "goodbye",
    "go_away",
    "helpful",
    "dance",
    "electric",
    "dying",
)

_INTENT_TO_MOVES: dict[str, tuple[str, ...]] = {
    "happy": ("cheerful1", "laughing2", "enthusiastic2", "enthusiastic1"),
    "excited": ("enthusiastic1", "enthusiastic2", "success2"),
    "loving": ("loving1", "grateful1"),
    "grateful": ("grateful1", "helpful2", "loving1"),
    "proud": ("proud1", "proud2", "proud3"),
    "success": ("success1", "success2", "proud3"),
    "curious": ("curious1", "inquiring2", "inquiring3"),
    "inquiring": ("inquiring1", "inquiring2", "inquiring3"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "confused": ("confused1", "lost1", "incomprehensible2"),
    "lost": ("lost1", "confused1"),
    "uncertain": ("uncertain1", "resigned1"),
    "sad": ("sad1", "sad2", "downcast1"),
    "downcast": ("downcast1", "sad1"),
    "lonely": ("lonely1", "sad1"),
    "angry": ("furious1", "rage1", "irritated2", "irritated1"),
    "irritated": ("irritated1", "irritated2", "displeased2"),
    "displeased": ("displeased1", "displeased2"),
    "disgusted": ("disgusted1", "contempt1"),
    "scared": ("scared1", "fear1", "anxiety1"),
    "anxious": ("anxiety1", "fear1", "scared1"),
    "surprised": ("surprised1", "surprised2", "amazed1"),
    "amazed": ("amazed1", "surprised1"),
    "calming": ("calming1", "serenity1"),
    "relief": ("relief1", "relief2"),
    "impatient": ("impatient1", "impatient2"),
    "embarrassed": ("shy1", "uncomfortable1"),
    "uncomfortable": ("uncomfortable1", "shy1"),
    "bored": ("boredom1", "boredom2"),
    "tired": ("tired1", "exhausted1", "sleep1"),
    "sleepy": ("sleep1", "tired1"),
    "yes": ("yes1", "understanding2", "understanding1"),
    "yes_sad": ("yes_sad1", "resigned1"),
    "yes_understanding": ("understanding2", "understanding1", "yes1"),
    "yes_proud": ("proud2", "yes1", "proud3"),
    "no": ("no1", "no_sad1", "no_excited1"),
    "no_sad": ("no_sad1", "downcast1"),
    "no_excited": ("no_excited1", "no1"),
    "no_firm": ("no1",),
    "no_confused": ("confused1", "incomprehensible2", "lost1"),
    "oops": ("oops1", "oops2"),
    "welcoming": ("welcoming1", "welcoming2"),
    "greeting": ("welcoming1", "welcoming2"),
    "goodbye": ("loving1", "welcoming2"),
    "go_away": ("go_away1",),
    "helpful": ("helpful1", "helpful2"),
    "dance": ("dance1", "dance2", "dance3"),
    "electric": ("electric1",),
    "dying": ("dying1",),
}

_KEYWORD_INTENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("no", "sad"), "no_sad"),
    (("no", "confused"), "no_confused"),
    (("no", "excited"), "no_excited"),
    (("no", "firm"), "no_firm"),
    (("yes", "sad"), "yes_sad"),
    (("yes", "proud"), "yes_proud"),
    (("yes", "understanding"), "yes_understanding"),
)


def _normalize_emotion_key(value: str) -> str:
    """Normalize an emotion request for exact intent and keyword matching."""
    without_accents = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", without_accents.lower()).strip("_")


def _keyword_intent(normalized_key: str) -> str | None:
    """Return the first nuanced intent whose keywords are all present."""
    tokens = set(normalized_key.split("_"))
    for keywords, intent in _KEYWORD_INTENTS:
        if all(keyword in tokens for keyword in keywords):
            return intent
    return None


def resolve_emotion_name(requested_emotion: object, available_emotions: list[str]) -> str | None:
    """Resolve a compact intent, nuanced yes/no phrase, or recorded move ID."""
    if not available_emotions:
        return None

    requested = str(requested_emotion or "").strip()
    if not requested:
        return None

    normalized = _normalize_emotion_key(requested)
    if not normalized or normalized == "random":
        return None

    available_by_key = {_normalize_emotion_key(name): name for name in available_emotions}
    exact_move = available_by_key.get(normalized)
    if exact_move is not None:
        return exact_move

    intent = normalized if normalized in _INTENT_TO_MOVES else None
    if intent is None:
        intent = _keyword_intent(normalized)

    if intent is None:
        return None

    for candidate in _INTENT_TO_MOVES.get(intent, ()):
        if candidate in available_emotions:
            return candidate
    return None


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        if not emotion_names:
            return "No emotions currently available"

        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a robot emotion matching a requested emotional intent."
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(EMOTION_INTENTS),
                "description": (
                    "Compact emotional intent to express. Choose one of the enum values. Use nuanced "
                    "labels like no_sad, no_confused, no_excited, yes_sad, or yes_understanding when "
                    "plain yes/no loses meaning. Use random if no clear intent fits."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        requested_emotion = kwargs.get("emotion")

        logger.info("Tool call: play_emotion emotion=%s", requested_emotion)

        try:
            emotion_names = RECORDED_MOVES.list_moves()
            if not emotion_names:
                return {"error": "No emotions currently available"}

            emotion_name = resolve_emotion_name(requested_emotion, emotion_names)
            if not emotion_name:
                emotion_name = random.choice(emotion_names)

            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
