import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from robot_comic.config import OPENAI_BACKEND, config, get_default_voice_for_provider
from robot_comic.prompts import get_session_voice, get_session_instructions
from robot_comic.base_realtime import BaseRealtimeHandler, to_realtime_tools_config
from robot_comic.tools.core_tools import get_active_tool_specs


logger = logging.getLogger(__name__)

__all__ = ["OpenaiRealtimeHandler"]


class OpenaiRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for the direct OpenAI Realtime API."""

    PROVIDER_ID = OPENAI_BACKEND
    SAMPLE_RATE = 24000
    REFRESH_CLIENT_ON_RECONNECT = False
    AUDIO_INPUT_COST_PER_1M = 32.0
    AUDIO_OUTPUT_COST_PER_1M = 64.0
    TEXT_INPUT_COST_PER_1M = 4.0
    TEXT_OUTPUT_COST_PER_1M = 16.0
    IMAGE_INPUT_COST_PER_1M = 5.0

    async def _prepare_startup_credentials(self) -> None:
        """Wait for the admin UI to populate the OpenAI key in env when needed.

        In sim mode the user provides credentials through the static admin
        UI at /, which writes them to the instance ``.env``. Poll the env
        until the key appears (mirrors LocalStream's headless polling).
        """
        if not self.sim_mode or config.OPENAI_API_KEY:
            return

        from robot_comic.config import refresh_runtime_config_from_env

        logger.warning("OPENAI_API_KEY not set; waiting for the admin UI at / to provide it…")
        while not config.OPENAI_API_KEY:
            await asyncio.sleep(0.2)
            try:
                refresh_runtime_config_from_env()
            except Exception:
                pass

    def _get_session_instructions(self) -> str:
        """Return OpenAI session instructions."""
        return get_session_instructions()

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured OpenAI session voice."""
        return get_session_voice(default, backend="openai")

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Return the OpenAI Realtime session config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=self._get_session_instructions(),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    transcription=AudioTranscriptionParam(model="gpt-4o-transcribe", language="en"),
                    turn_detection=ServerVad(type="server_vad", interrupt_response=True),
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured OpenAI realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        fallback = await super().get_available_voices()
        try:
            model = await self.client.models.retrieve(config.MODEL_NAME)
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None

            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            key_lower = str(key).lower()
                            if "voice" in key_lower and isinstance(value, (list, tuple)):
                                for item in value:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and isinstance(item.get("name"), str):
                                        candidates.add(item["name"])
                            else:
                                _collect(value)
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            _collect(item)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)

            voices = sorted(candidates) if candidates else fallback
            default_voice = get_default_voice_for_provider(self.PROVIDER_ID)
            if default_voice not in voices:
                voices = [default_voice, *[voice for voice in voices if voice != default_voice]]
            return voices
        except Exception:
            return fallback

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the OpenAI realtime SDK client."""
        self._realtime_connect_query = {}
        resolved_api_key = (config.OPENAI_API_KEY or "").strip()
        if not resolved_api_key:
            # In headless console mode, LocalStream blocks startup until the key is provided.
            # Unit tests may invoke this handler directly with a stubbed client.
            logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
            resolved_api_key = "DUMMY"
        return AsyncOpenAI(api_key=resolved_api_key)
