"""Gesture tool — lets the LLM trigger pre-canned movement sequences.

Personas opt in by listing ``gesture`` in their ``tools.txt``.

The tool accepts either:
- ``name=<canonical>`` — play a gesture directly by its canonical name
  (shrug, nod_yes, nod_no, point_left, point_right, scan, lean_in).
- ``beat=<abstract>`` — resolve via the active persona's ``gestures.txt``
  mapping (disapproval, agreement, punchline_setup, etc.) and then play
  the resulting canonical gesture.

Providing both ``name`` and ``beat`` is valid; ``name`` takes precedence.
"""

from __future__ import annotations
import logging
from typing import Any, Dict
from pathlib import Path

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


# Lazily import the registry so the module can be loaded even in test
# environments that don't have the full reachy_mini SDK available.
def _get_registry() -> Any:
    from robot_comic.gestures import registry  # noqa: PLC0415

    return registry


def _get_active_profile_dir() -> Path | None:
    """Return the active persona's profile directory, or None if unavailable."""
    try:
        from robot_comic.config import config  # noqa: PLC0415

        profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
        return config.PROFILES_DIRECTORY / profile
    except Exception:
        return None


class Gesture(Tool):
    """Play a named gesture (pre-canned movement sequence).

    Accepts a canonical ``name`` (shrug, nod_yes, nod_no, point_left,
    point_right, scan, lean_in) **or** an abstract ``beat`` name that
    is resolved to a canonical gesture via the active persona's
    ``gestures.txt`` mapping.
    """

    name = "gesture"
    description = (
        "Play a pre-canned physical gesture. Use either a canonical gesture "
        "name (name=) or an abstract beat name (beat=) that maps to the "
        "persona's preferred gesture for that dramatic moment. "
        "Canonical names: shrug, nod_yes, nod_no, point_left, point_right, "
        "scan, lean_in. "
        "Abstract beats: disapproval, agreement, reflection, character_switch, "
        "punchline_setup, punchline_drop, dismissal, acknowledgement, surprise, "
        "defeat, swagger, vulnerability."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": [
                    "shrug",
                    "nod_yes",
                    "nod_no",
                    "point_left",
                    "point_right",
                    "scan",
                    "lean_in",
                ],
                "description": (
                    "Canonical gesture to perform. "
                    "shrug — uncertain / comic timing; "
                    "nod_yes — affirmative; "
                    "nod_no — negative / dismissive; "
                    "point_left / point_right — direct attention; "
                    "scan — slow theatrical sweep of the audience; "
                    "lean_in — conspiratorial aside or punchline build-up."
                ),
            },
            "beat": {
                "type": "string",
                "enum": [
                    "disapproval",
                    "agreement",
                    "reflection",
                    "character_switch",
                    "punchline_setup",
                    "punchline_drop",
                    "dismissal",
                    "acknowledgement",
                    "surprise",
                    "defeat",
                    "swagger",
                    "vulnerability",
                ],
                "description": (
                    "Abstract beat name resolved to a canonical gesture via "
                    "the active persona's gestures.txt. Prefer beat= over "
                    "name= when you are expressing a dramatic moment rather "
                    "than a specific physical shape: "
                    "disapproval — rejecting a premise; "
                    "agreement — validating; "
                    "reflection — thinking pause; "
                    "character_switch — transitioning to a character voice; "
                    "punchline_setup — build-up before the joke lands; "
                    "punchline_drop — physical resolution after the punchline; "
                    "dismissal — sending someone off; "
                    "acknowledgement — greeting recognition; "
                    "surprise — caught off guard; "
                    "defeat — resigned collapse; "
                    "swagger — confident wind-up; "
                    "vulnerability — tender honest moment."
                ),
            },
        },
        # NOTE: no schema-level ``anyOf`` constraint on (name OR beat).
        # Gemini Live's BidiGenerateContentRequest validator rejects the
        # patterns we tried:
        #   1. ``anyOf: [{required:[name]}, {required:[beat]}]`` →
        #      ``required: only allowed for OBJECT`` (anyOf branch must
        #      declare type).
        #   2. ``anyOf: [{type:object, required:[name]}, ...]`` →
        #      ``required[0]: property is not defined`` (Gemini doesn't
        #      inherit parent ``properties`` into anyOf branches).
        # Both failures were caught live on 2026-05-19 (1007 close +
        # crash-loop). The either-or constraint IS enforced at runtime
        # in ``__call__`` (returns an explicit error dict when both are
        # missing), and the tool ``description`` tells the LLM to use
        # one or the other. Schema-level validation would be belt-and-
        # braces but isn't load-bearing — leave it off for portability.
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play the requested gesture, resolving beats if necessary."""
        gesture_name: str | None = kwargs.get("name")
        beat: str | None = kwargs.get("beat")

        # Resolve beat → canonical name when name is not provided directly.
        if not gesture_name and beat:
            gesture_name = self._resolve_beat(beat)
            if gesture_name is None:
                # Resolution failed — error was already logged; return gracefully.
                return {
                    "error": (
                        f"Beat {beat!r} is not mapped for the active persona. "
                        "Use name= with a canonical gesture name instead."
                    )
                }

        if not isinstance(gesture_name, str) or not gesture_name:
            return {"error": "provide either name=<canonical> or beat=<abstract>"}

        logger.info("Tool call: gesture name=%s (beat=%s)", gesture_name, beat)

        try:
            reg = _get_registry()
            reg.play(gesture_name, deps.movement_manager)
            result: Dict[str, Any] = {"status": "queued", "gesture": gesture_name}
            if beat:
                result["beat"] = beat
            return result
        except KeyError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("gesture tool failed: %s", exc)
            return {"error": f"gesture failed: {type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_beat(self, beat: str) -> str | None:
        """Resolve an abstract beat name to a canonical gesture name.

        Returns the gesture name on success, or None on failure (error
        is logged; callers should return an error dict to the LLM).
        """
        try:
            from robot_comic.gestures.registry import load_persona_beats  # noqa: PLC0415

            profile_dir = _get_active_profile_dir()
            if profile_dir is None:
                logger.warning("gesture tool: cannot determine active profile dir for beat resolution")
                return None

            persona_beats = load_persona_beats(profile_dir)
            if not persona_beats:
                logger.debug("gesture tool: no beats file for profile %s", profile_dir.name)
                return None

            reg = _get_registry()
            result: str = reg.resolve_for_persona(beat, persona_beats)
            return result
        except KeyError as exc:
            logger.warning("gesture tool: beat resolution failed: %s", exc)
            return None
        except Exception as exc:
            logger.error("gesture tool: unexpected error resolving beat %r: %s", beat, exc)
            return None
