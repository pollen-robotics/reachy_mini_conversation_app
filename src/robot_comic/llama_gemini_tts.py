"""llama-server LLM + Gemini 3.1 Flash TTS response handler.

Combines:
- LLM: llama-server /v1/chat/completions (local Qwen3)
- TTS: gemini-3.1-flash-tts-preview (cloud, via Gemini client)

Select via the admin UI's composable pipeline picker (Phase 4f) or the env
vars REACHY_MINI_PIPELINE_MODE=composable +
REACHY_MINI_AUDIO_OUTPUT_BACKEND=gemini_tts + REACHY_MINI_LLM_BACKEND=llama.
"""

from __future__ import annotations
import time
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from google import genai

from robot_comic import telemetry
from robot_comic.config import (
    GEMINI_TTS_DEFAULT_VOICE,
    GEMINI_TTS_AVAILABLE_VOICES,
    config,
)
from robot_comic.prompts import get_session_voice
from robot_comic.gemini_tts import (
    SHORT_PAUSE_MS,
    SHORT_PAUSE_TAG,
    GEMINI_TTS_MODEL,
    _TTS_EXCLUSIVE_VOICES,
    _silence_pcm,
    _build_tts_config,
    _build_tts_contents,
    extract_delivery_tags,
    build_tts_system_instruction,
    load_profile_tts_instruction,
)
from robot_comic.llama_base import _OUTPUT_SAMPLE_RATE, BaseLlamaResponseHandler, split_sentences
from robot_comic.gemini_retry import (
    compute_backoff,
    is_rate_limit_error,
    describe_quota_failure,
    extract_retry_after_seconds,
)
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.chatterbox_tag_translator import strip_gemini_tags


logger = logging.getLogger(__name__)

_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY = 0.5


class LlamaGeminiTTSResponseHandler(BaseLlamaResponseHandler):
    """llama-server LLM + Gemini 3.1 Flash TTS voice output with tool dispatch."""

    _BACKEND_LABEL = "llama_gemini_tts"
    _TTS_SYSTEM = "gemini_tts"

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(deps, sim_mode, instance_path, startup_voice)
        self._client: genai.Client | None = None
        # Only restore voices valid for Gemini TTS (not Gemini Live voices)
        self._voice_override = startup_voice if startup_voice in _TTS_EXCLUSIVE_VOICES else None
        # Tracks the most recent TTS-call outcome so the synthesizer can
        # surface a rate-limit-specific message to the chat UI.
        self._last_tts_rate_limited: bool = False
        self._last_tts_quota: str | None = None

    def copy(self) -> "LlamaGeminiTTSResponseHandler":
        return LlamaGeminiTTSResponseHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        from google import genai  # deferred: google.genai.types costs ~5.5 s at boot

        await super()._prepare_startup_credentials()
        api_key = config.GEMINI_API_KEY or "DUMMY"
        self._client = genai.Client(api_key=api_key)

        llm_model = await self._fetch_llm_model_name()
        stt_model = getattr(config, "LOCAL_STT_MODEL", "unknown")
        logger.info(
            "Pipeline: Moonshine (%s) → llama-server (%s @ %s) → Gemini TTS (%s, voice=%s)",
            stt_model,
            llm_model,
            self._llama_cpp_url,
            GEMINI_TTS_MODEL,
            self.get_current_voice(),
        )

    async def _fetch_llm_model_name(self) -> str:
        assert self._http is not None
        try:
            r = await self._http.get(f"{self._llama_cpp_url}/v1/models", timeout=3.0)
            r.raise_for_status()
            data = r.json()
            return str(data["data"][0]["id"])
        except Exception:
            return self._llama_cpp_url

    # ------------------------------------------------------------------ #
    # Voice management (Gemini TTS voices)                                 #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        return list(GEMINI_TTS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        voice = self._voice_override or get_session_voice(
            backend="gemini",
            default=GEMINI_TTS_DEFAULT_VOICE,
        )
        if voice not in GEMINI_TTS_AVAILABLE_VOICES:
            return GEMINI_TTS_DEFAULT_VOICE
        return voice

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # TTS synthesis                                                        #
    # ------------------------------------------------------------------ #

    async def _synthesize_and_enqueue(
        self,
        response_text: str,
        tts_start: float | None = None,
        target_queue: "asyncio.Queue[Any] | None" = None,
    ) -> None:
        if not response_text:
            return
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        base_instruction = load_profile_tts_instruction()
        sentences = split_sentences(response_text) or [response_text]
        any_audio = False
        first_chunk = True
        for sentence in sentences:
            spoken = strip_gemini_tags(sentence)
            if not spoken:
                continue
            tags = extract_delivery_tags(sentence)
            if SHORT_PAUSE_TAG in tags:
                for frame in self._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS, _OUTPUT_SAMPLE_RATE)):
                    await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
            instruction = build_tts_system_instruction(base_instruction, tags)
            pcm_bytes = await self._call_gemini_tts(spoken, system_instruction=instruction)
            if pcm_bytes is None:
                logger.warning("Gemini TTS returned None for sentence: %r", spoken[:60])
                continue
            if first_chunk and tts_start is not None:
                telemetry.record_tts_first_audio(time.perf_counter() - tts_start, {"gen_ai.system": "gemini_tts"})
                first_chunk = False
            for frame in self._pcm_to_frames(pcm_bytes):
                from robot_comic.startup_timer import log_once

                log_once("first TTS audio frame", logger)
                telemetry.emit_first_greeting_audio_once()
                await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
            any_audio = True

        if not any_audio:
            if self._last_tts_rate_limited:
                msg = f"[Gemini TTS rate-limited (quota={self._last_tts_quota or 'unknown'}); try again later]"
            else:
                msg = "[TTS error — Gemini TTS failed]"
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": msg}))

    async def _call_gemini_tts(self, text: str, system_instruction: str | None = None) -> bytes | None:
        assert self._client is not None, "Gemini client not initialised"
        instruction = system_instruction if system_instruction is not None else load_profile_tts_instruction()
        tts_config = _build_tts_config(instruction, self.get_current_voice(), GEMINI_TTS_MODEL)
        contents = _build_tts_contents(text, instruction, GEMINI_TTS_MODEL)
        self._last_tts_rate_limited = False
        self._last_tts_quota = None
        for attempt in range(_TTS_MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=GEMINI_TTS_MODEL,
                    contents=contents,
                    config=tts_config,
                )
                data = response.candidates[0].content.parts[0].inline_data.data  # type: ignore[index,union-attr]
                return base64.b64decode(data) if isinstance(data, str) else bytes(data)  # type: ignore[arg-type]
            except Exception as exc:
                rate_limited = is_rate_limit_error(exc)
                if rate_limited:
                    quota = describe_quota_failure(exc)
                    retry_after = extract_retry_after_seconds(exc)
                    self._last_tts_rate_limited = True
                    self._last_tts_quota = quota
                    logger.warning(
                        "Gemini TTS 429 (quota=%s, attempt %d/%d); retry-after=%s",
                        quota,
                        attempt + 1,
                        _TTS_MAX_RETRIES,
                        f"{retry_after:.1f}s" if retry_after is not None else "n/a",
                    )
                else:
                    retry_after = None
                    logger.warning("Gemini TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    delay = compute_backoff(attempt, _TTS_RETRY_BASE_DELAY, retry_after)
                    await asyncio.sleep(delay)
                elif rate_limited:
                    logger.error(
                        "Gemini TTS exhausted %d retries on 429 (quota=%s); skipping audio for this turn",
                        _TTS_MAX_RETRIES,
                        describe_quota_failure(exc),
                    )
                else:
                    logger.error(
                        "Gemini TTS exhausted %d retries (last error: %s); skipping audio for this turn",
                        _TTS_MAX_RETRIES,
                        exc,
                    )
        return None
