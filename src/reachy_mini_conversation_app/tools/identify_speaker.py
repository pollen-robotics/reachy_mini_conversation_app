"""Tool for identifying the current speaker."""

from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class IdentifySpeaker(Tool):
    """Tool to identify who is currently speaking based on their voice."""

    name = "identify_speaker"
    description = (
        "Identify who is currently speaking based on their voice. "
        "Returns the speaker's name if recognized, or indicates unknown."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Identify the current speaker.

        Args:
            deps: Tool dependencies containing the speaker ID worker.
            **kwargs: Not used.

        Returns:
            Speaker identification result.

        """
        if deps.speaker_id_worker is None:
            return {
                "error": "Speaker identification is not enabled. "
                "Start the app with --speaker-identification flag."
            }

        speaker, confidence = deps.speaker_id_worker.get_current_speaker()
        if speaker:
            return {
                "status": "identified",
                "speaker": speaker,
                "confidence": round(confidence, 2),
                "message": f"I recognize {speaker} with {confidence:.0%} confidence.",
            }
        else:
            return {
                "status": "unknown",
                "speaker": None,
                "confidence": 0.0,
                "message": "I don't recognize this voice. Would you like to register?",
            }


class ListSpeakers(Tool):
    """Tool to list all registered speakers."""

    name = "list_speakers"
    description = "List all registered speakers that can be identified by voice."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """List registered speakers.

        Args:
            deps: Tool dependencies containing the speaker ID worker.
            **kwargs: Not used.

        Returns:
            List of registered speaker names.

        """
        if deps.speaker_id_worker is None:
            return {
                "error": "Speaker identification is not enabled. "
                "Start the app with --speaker-identification flag."
            }

        speakers = deps.speaker_id_worker.list_speakers()
        if speakers:
            return {
                "status": "success",
                "speakers": speakers,
                "count": len(speakers),
                "message": f"I know {len(speakers)} speaker(s): {', '.join(speakers)}.",
            }
        else:
            return {
                "status": "empty",
                "speakers": [],
                "count": 0,
                "message": "No speakers registered yet. Would you like to register?",
            }


class RemoveSpeaker(Tool):
    """Tool to remove a registered speaker."""

    name = "remove_speaker"
    description = "Remove a registered speaker from the voice recognition system."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the speaker to remove",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Remove a registered speaker.

        Args:
            deps: Tool dependencies containing the speaker ID worker.
            **kwargs: Must include 'name' parameter.

        Returns:
            Status message for the LLM to vocalize.

        """
        name = kwargs.get("name")
        if not name:
            return {"error": "Missing required parameter: name"}

        if deps.speaker_id_worker is None:
            return {
                "error": "Speaker identification is not enabled. "
                "Start the app with --speaker-identification flag."
            }

        success = deps.speaker_id_worker.remove_speaker(name)
        if success:
            return {
                "status": "success",
                "message": f"Removed {name} from the registered speakers.",
            }
        else:
            return {
                "status": "not_found",
                "error": f"Speaker '{name}' not found in the registered speakers.",
            }
