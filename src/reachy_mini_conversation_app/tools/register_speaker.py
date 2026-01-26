"""Tool for registering a new speaker."""

from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class RegisterSpeaker(Tool):
    """Tool to register a new speaker by their name.

    When called, starts audio collection mode. The speaker should talk
    for a few seconds, then call finish_speaker_registration to complete.

    """

    name = "register_speaker"
    description = (
        "Start registering a new speaker by their name. After calling this, "
        "the speaker should talk for a few seconds to record their voice."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the speaker to register",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Start speaker registration.

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

        deps.speaker_id_worker.start_registration(name)
        return {
            "status": "registration_started",
            "message": f"I'm ready to register {name}. Please speak for a few seconds, "
            "then tell me when you're done.",
        }


class FinishSpeakerRegistration(Tool):
    """Tool to finish speaker registration after collecting audio."""

    name = "finish_speaker_registration"
    description = (
        "Finish registering the speaker after they have spoken for a few seconds. "
        "Call this when the speaker indicates they are done talking."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Finish speaker registration.

        Args:
            deps: Tool dependencies containing the speaker ID worker.
            **kwargs: Not used.

        Returns:
            Status message for the LLM to vocalize.

        """
        if deps.speaker_id_worker is None:
            return {
                "error": "Speaker identification is not enabled. "
                "Start the app with --speaker-identification flag."
            }

        if not deps.speaker_id_worker.is_registering():
            return {"error": "No registration in progress. Call register_speaker first."}

        success = deps.speaker_id_worker.finish_registration()
        if success:
            return {
                "status": "success",
                "message": "Voice registered successfully! I'll recognize you next time.",
            }
        else:
            return {
                "status": "failed",
                "error": "Registration failed. Not enough audio collected. Please try again.",
            }


class CancelSpeakerRegistration(Tool):
    """Tool to cancel an ongoing speaker registration."""

    name = "cancel_speaker_registration"
    description = "Cancel the current speaker registration if one is in progress."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel speaker registration.

        Args:
            deps: Tool dependencies containing the speaker ID worker.
            **kwargs: Not used.

        Returns:
            Status message for the LLM to vocalize.

        """
        if deps.speaker_id_worker is None:
            return {
                "error": "Speaker identification is not enabled. "
                "Start the app with --speaker-identification flag."
            }

        if not deps.speaker_id_worker.is_registering():
            return {"status": "no_registration", "message": "No registration was in progress."}

        deps.speaker_id_worker.cancel_registration()
        return {"status": "cancelled", "message": "Registration cancelled."}
