"""Gemini TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls the
Gemini Flash text model for reasoning and tool execution, then synthesises
audio with gemini-3.1-flash-tts-preview (voice: Algenib by default).

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
"""

from __future__ import annotations
import re
import json
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

import httpx
import numpy as np
from fastrtc import AsyncStreamHandler


if TYPE_CHECKING:
    from google import genai
    from google.genai import types

from robot_comic.config import (
    GEMINI_AVAILABLE_VOICES,
    GEMINI_TTS_DEFAULT_VOICE,
    GEMINI_TTS_AVAILABLE_VOICES,
    config,
    set_custom_profile,
)
from robot_comic.openers import get_canned_opener
from robot_comic.llama_base import split_sentences
from robot_comic.joke_history import JokeHistory, default_history_path, extract_punchline_via_llm
from robot_comic.chatterbox_tag_translator import strip_gemini_tags


# Voices shared with Gemini Live may have been persisted for that backend.
# Only restore startup_voice if it is TTS-exclusive (not in the Live voice list).
_TTS_EXCLUSIVE_VOICES: frozenset[str] = frozenset(GEMINI_TTS_AVAILABLE_VOICES) - frozenset(GEMINI_AVAILABLE_VOICES)
from robot_comic.prompts import get_session_instructions
from robot_comic.gemini_live import _openai_tool_specs_to_gemini
from robot_comic.gemini_retry import (
    compute_backoff,
    is_rate_limit_error,
    describe_quota_failure,
    extract_retry_after_seconds,
)
from robot_comic.history_trim import trim_history_in_place, is_synthetic_status_marker
from robot_comic.tools.core_tools import ToolDependencies, dispatch_tool_call, get_active_tool_specs
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.tools.name_validation import record_user_transcript


logger = logging.getLogger(__name__)

GEMINI_TTS_LLM_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_TTS_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz
_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY = 0.5
_LLM_MAX_RETRIES = 4
_LLM_RETRY_BASE_DELAY = 1.0
_LLM_MAX_TOOL_ROUNDS = 5

# Fallback used when a profile does not provide its own gemini_tts.txt.
DEFAULT_TTS_SYSTEM_INSTRUCTION = (
    "Deliver this text at a fast, clipped Brooklyn pace — "
    "rapid-fire on the insults, short crisp pauses only where marked. "
    "Never drawl or over-enunciate. Keep the energy sharp."
)

# Delivery tags emitted by profile prompts (see e.g. profiles/house_comedian/instructions.txt).
# Each tag maps to a human-readable phrase used in the system_instruction suffix —
# the TTS model has more to work with than the raw tag name.
_DELIVERY_TAG_PHRASES: dict[str, str] = {
    "fast": "rapid-fire, clipped delivery",
    "slow": "slower, more deliberate pacing",
    "annoyance": "exasperated, contemptuous edge",
    "aggression": "sharp, biting attack",
    "amusement": "self-satisfied, savouring the line",
    "enthusiasm": "warm, energised lift",
}
# [short pause] is special: handled as a real audio gap before the sentence,
# not as a TTS cue (otherwise the model would add its own pause on top of the silence).
SHORT_PAUSE_TAG = "short pause"
SHORT_PAUSE_MS = 400

_ALL_TAGS = (*_DELIVERY_TAG_PHRASES.keys(), SHORT_PAUSE_TAG)
_DELIVERY_TAG_RE = re.compile(
    r"\[(" + "|".join(re.escape(t) for t in _ALL_TAGS) + r")\]",
    re.IGNORECASE,
)


def extract_delivery_tags(text: str) -> list[str]:
    """Return ordered, de-duplicated delivery-tag names found in *text*."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _DELIVERY_TAG_RE.finditer(text):
        tag = match.group(1).lower()
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def build_tts_system_instruction(base: str, tags: list[str]) -> str:
    """Append a short delivery-cue suffix to *base* for one TTS call.

    [short pause] is excluded — it becomes a real silence gap in the audio,
    not a hint to the model.
    """
    cue_tags = [t for t in tags if t != SHORT_PAUSE_TAG]
    if not cue_tags:
        return base
    cues = "; ".join(_DELIVERY_TAG_PHRASES.get(t, t) for t in cue_tags)
    return f"{base}\n\nDelivery cues for this line: {cues}."


# Models that do NOT support system_instruction (400 INVALID_ARGUMENT).
# For these models, delivery cues are prepended to the user text instead.
_TTS_NO_SYSTEM_INSTRUCTION_MODELS: frozenset[str] = frozenset(
    {
        "gemini-3.1-flash-tts-preview",
    }
)


def _build_tts_contents(text: str, instruction: str, model: str) -> str:
    """Return the TTS request content string.

    For models that reject ``system_instruction``, the instruction is prepended
    to the spoken text as a parenthetical cue so the model still gets the styling
    hint without using the unsupported field.
    """
    if model in _TTS_NO_SYSTEM_INSTRUCTION_MODELS:
        return f"({instruction}) {text}"
    return text


def _build_tts_config(
    instruction: str,
    voice_name: str,
    model: str,
) -> "types.GenerateContentConfig":
    """Return a ``GenerateContentConfig`` appropriate for *model*.

    ``system_instruction`` is omitted for models in
    ``_TTS_NO_SYSTEM_INSTRUCTION_MODELS`` to avoid 400 INVALID_ARGUMENT errors.
    """
    from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name))
    )
    if model in _TTS_NO_SYSTEM_INSTRUCTION_MODELS:
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )
    return types.GenerateContentConfig(
        system_instruction=instruction,
        response_modalities=["AUDIO"],
        speech_config=speech_config,
    )


def _silence_pcm(duration_ms: int, sample_rate: int = GEMINI_TTS_OUTPUT_SAMPLE_RATE) -> bytes:
    """Return raw int16 PCM bytes of silence at *sample_rate* for *duration_ms*."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return bytes(np.zeros(n_samples, dtype=np.int16).tobytes())


def load_profile_tts_instruction() -> str:
    """Read profiles/<name>/gemini_tts.txt, falling back to DEFAULT_TTS_SYSTEM_INSTRUCTION."""
    profile: str | None = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    if not profile:
        return DEFAULT_TTS_SYSTEM_INSTRUCTION
    try:
        path = config.PROFILES_DIRECTORY / profile / "gemini_tts.txt"
        if path.exists():
            text: str = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception as exc:
        logger.warning("Could not read gemini_tts.txt for profile %r: %s", profile, exc)
    return DEFAULT_TTS_SYSTEM_INSTRUCTION


class GeminiTTSResponseHandler(AsyncStreamHandler, ConversationHandler):
    """Request/response handler: Gemini Flash text model + Gemini TTS voice output."""

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
        )
        self.deps = deps
        self.sim_mode = sim_mode
        self.instance_path = instance_path
        # Only restore voices that are TTS-exclusive. Shared Live/TTS voices
        # (Kore, Aoede, etc.) may have been persisted for Gemini Live — ignore them
        # and default to GEMINI_TTS_DEFAULT_VOICE (Algenib) instead.
        self._voice_override: str | None = startup_voice if startup_voice in _TTS_EXCLUSIVE_VOICES else None
        self._client: genai.Client | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._conversation_history: list[dict[str, Any]] = []
        # Tracks the most recent TTS-call outcome so the dispatch loop can
        # surface a rate-limit-specific message to the chat UI.
        self._last_tts_rate_limited: bool = False
        self._last_tts_quota: str | None = None
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()

        # Attributes referenced by LocalSTTInputMixin (declared to satisfy type checker and MRO)
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    def _mark_activity(self, label: str) -> None:
        """No-op activity marker (no cost tracking in this handler)."""
        logger.debug("Activity: %s", label)

    def copy(self) -> "GeminiTTSResponseHandler":
        return GeminiTTSResponseHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        """Initialise the Gemini client.

        Phase 5e.6 idempotency: the migrated triple's factory composes
        a plain leaf handler (no :class:`LocalSTTInputMixin` shell),
        and the LLM and TTS adapters each call this method during their
        ``prepare`` lifecycle. Without a leaf-level guard duplicate
        calls would leak a fresh ``genai.Client`` every time. Guard
        the whole body so duplicate calls are cheap no-ops. The flag
        is only set on success so a failed attempt still re-tries the
        full chain.
        """
        if getattr(self, "_startup_credentials_ready", False):
            return
        from google import genai  # deferred: google.genai.types costs ~5.5 s at boot

        api_key = config.GEMINI_API_KEY or "DUMMY"
        self._client = genai.Client(api_key=api_key)
        logger.info(
            "GeminiTTS handler initialised: llm=%s tts=%s voice=%s",
            GEMINI_TTS_LLM_MODEL,
            GEMINI_TTS_MODEL,
            self.get_current_voice(),
        )
        self._startup_credentials_ready = True

    async def start_up(self) -> None:
        """Initialise credentials and block until shutdown() is called."""
        await self._prepare_startup_credentials()
        self._stop_event.clear()
        asyncio.create_task(self._send_startup_trigger(), name="gemini-tts-startup-trigger")
        await self._stop_event.wait()

    async def _send_startup_trigger(self) -> None:
        """Kick off the opening sequence.

        Default mode ("canned", #290): bypass the LLM and synthesize a
        per-persona line read from ``profiles/<persona>/openers.txt``.
        Legacy mode ("llm"): keep the synthetic ``[conversation started]``
        round-trip available for A/B comparison.
        """
        mode = getattr(config, "STARTUP_TRIGGER_MODE", "canned")
        if mode == "canned":
            await self._speak_canned_opener(get_canned_opener())
            return
        await self._dispatch_completed_transcript("[conversation started]")

    async def _speak_canned_opener(self, text: str) -> None:
        """Enqueue *text* directly to Gemini TTS and record it as a model turn."""
        if not text or not text.strip():
            logger.warning("Canned opener was empty; skipping startup speak.")
            return
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        logger.info("Canned startup opener: %r", text)
        self._conversation_history.append({"role": "model", "parts": [{"text": text}]})
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))

        try:
            base_instruction = load_profile_tts_instruction()
            sentences = split_sentences(text) or [text]
            for sentence in sentences:
                spoken = strip_gemini_tags(sentence)
                if not spoken:
                    continue
                tags = extract_delivery_tags(sentence)
                if SHORT_PAUSE_TAG in tags:
                    for frame in self._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS)):
                        await self.output_queue.put((GEMINI_TTS_OUTPUT_SAMPLE_RATE, frame))
                instruction = build_tts_system_instruction(base_instruction, tags)
                pcm_bytes = await self._call_tts_with_retry(spoken, system_instruction=instruction)
                if pcm_bytes is None:
                    continue
                for frame in self._pcm_to_frames(pcm_bytes):
                    await self.output_queue.put((GEMINI_TTS_OUTPUT_SAMPLE_RATE, frame))
        except Exception as exc:
            logger.warning("Canned opener TTS failed: %s", exc)

    async def shutdown(self) -> None:
        """Signal start_up to return and drain the output queue."""
        self._stop_event.set()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive(self, frame: Any) -> None:
        """No-op: audio input is handled by LocalSTTInputMixin.receive()."""

    async def emit(self) -> Any:
        """Yield the next audio frame or status message from the output queue."""
        from fastrtc import wait_for_item  # deferred: fastrtc pulls gradio at boot

        return await wait_for_item(self.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality profile and reset conversation history."""
        try:
            set_custom_profile(profile)
            self._conversation_history.clear()
            return f"Applied personality {profile!r}. Conversation history reset."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        return list(GEMINI_TTS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        voice = self._voice_override or GEMINI_TTS_DEFAULT_VOICE
        if voice not in GEMINI_TTS_AVAILABLE_VOICES:
            logger.warning(
                "Voice %r is not a valid Gemini TTS voice; falling back to %s", voice, GEMINI_TTS_DEFAULT_VOICE
            )
            return GEMINI_TTS_DEFAULT_VOICE
        return voice

    async def change_voice(self, voice: str) -> str:
        self._voice_override = voice
        return f"Voice changed to {voice}."

    # ------------------------------------------------------------------ #
    # Response cycle                                                       #
    # ------------------------------------------------------------------ #

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """Gemini-native response cycle: LLM → tools → TTS → audio frames."""
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        self._conversation_history.append({"role": "user", "parts": [{"text": transcript}]})
        # Record for tool-side name-validation guard (#287).
        record_user_transcript(self.deps.recent_user_transcripts, transcript)
        # Trim BEFORE building the next request so long sessions don't blow the
        # model's context window or rack up token cost.
        trim_history_in_place(self._conversation_history, role_key="role")

        try:
            response_text = await self._run_llm_with_tools()
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return

        # Guard against synthetic status markers (e.g. "[Skipped TTS: ...]")
        # leaking into LLM history. The monitor still sees them via the output
        # queue; the next Gemini call must not. See issue #306.
        if not is_synthetic_status_marker(response_text):
            self._conversation_history.append({"role": "model", "parts": [{"text": response_text}]})
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": response_text}))

        # Capture punchline for avoid-repeat history (best-effort).
        if getattr(config, "JOKE_HISTORY_ENABLED", True):
            try:
                _persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", "") or ""
                async with httpx.AsyncClient() as _jh_http:
                    _extracted = await extract_punchline_via_llm(response_text, _jh_http)
                _punchline = _extracted.get("punchline") if _extracted is not None else None
                if _punchline and _extracted is not None:
                    JokeHistory(default_history_path()).add(
                        _punchline,
                        topic=_extracted.get("topic", "") or "",
                        persona=_persona,
                    )
            except Exception as _jh_exc:
                logger.debug("joke_history capture failed: %s", _jh_exc)

        base_instruction = load_profile_tts_instruction()
        sentences = split_sentences(response_text) or [response_text]
        any_audio = False
        for sentence in sentences:
            spoken = strip_gemini_tags(sentence)
            if not spoken:
                continue
            tags = extract_delivery_tags(sentence)
            if SHORT_PAUSE_TAG in tags:
                for frame in self._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS)):
                    await self.output_queue.put((GEMINI_TTS_OUTPUT_SAMPLE_RATE, frame))
            instruction = build_tts_system_instruction(base_instruction, tags)
            pcm_bytes = await self._call_tts_with_retry(spoken, system_instruction=instruction)
            if pcm_bytes is None:
                continue
            for frame in self._pcm_to_frames(pcm_bytes):
                from robot_comic import telemetry as _telemetry
                from robot_comic.startup_timer import log_once

                log_once("first TTS audio frame", logger)
                _telemetry.emit_first_greeting_audio_once()
                await self.output_queue.put((GEMINI_TTS_OUTPUT_SAMPLE_RATE, frame))
            any_audio = True

        if not any_audio:
            if self._last_tts_rate_limited:
                msg = f"[Gemini TTS rate-limited (quota={self._last_tts_quota or 'unknown'}); try again later]"
            else:
                msg = "[TTS error — could not generate audio]"
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": msg}))

    async def _llm_generate_with_backoff(self, contents: Any, config: Any) -> Any:
        """Call generate_content with backoff on transient errors and 429s."""
        assert self._client is not None, "Client not initialised"
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                return await self._client.aio.models.generate_content(
                    model=GEMINI_TTS_LLM_MODEL,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                msg = str(exc)
                rate_limited = is_rate_limit_error(exc)
                is_retryable = rate_limited or "503" in msg or "UNAVAILABLE" in msg
                if not is_retryable or attempt == _LLM_MAX_RETRIES - 1:
                    if rate_limited and attempt == _LLM_MAX_RETRIES - 1:
                        logger.error(
                            "Gemini LLM rate-limited (quota=%s) after %d attempts; giving up",
                            describe_quota_failure(exc),
                            _LLM_MAX_RETRIES,
                        )
                    raise
                retry_after = extract_retry_after_seconds(exc) if rate_limited else None
                delay = compute_backoff(attempt, _LLM_RETRY_BASE_DELAY, retry_after)
                if rate_limited:
                    logger.warning(
                        "Gemini LLM 429 (quota=%s, attempt %d/%d); sleeping %.1fs before retry",
                        describe_quota_failure(exc),
                        attempt + 1,
                        _LLM_MAX_RETRIES,
                        delay,
                    )
                else:
                    logger.warning(
                        "Gemini LLM attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1,
                        _LLM_MAX_RETRIES,
                        msg.split("\n")[0],
                        delay,
                    )
                await asyncio.sleep(delay)

    async def _run_llm_with_tools(self) -> str:
        """Call Gemini Flash with conversation history, handling tool round-trips."""
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot
        from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

        assert self._client is not None, "Client not initialised"

        tool_specs = get_active_tool_specs(self.deps)
        function_declarations = _openai_tool_specs_to_gemini(tool_specs)
        tools_config = [types.Tool(function_declarations=function_declarations)] if function_declarations else []  # type: ignore[arg-type]
        gen_config = types.GenerateContentConfig(
            system_instruction=get_session_instructions(),
            tools=tools_config,  # type: ignore[arg-type]
        )

        history: list[Any] = list(self._conversation_history)

        for _ in range(_LLM_MAX_TOOL_ROUNDS):
            response = await self._llm_generate_with_backoff(history, gen_config)

            candidate = response.candidates[0]
            function_calls = [p.function_call for p in candidate.content.parts if p.function_call is not None]

            if not function_calls:
                return "".join(p.text for p in candidate.content.parts if p.text).strip()

            # Append model's function-call turn to history
            history.append(candidate.content)

            # Execute tools and collect responses
            response_parts: list[types.Part] = []
            for fc in function_calls:
                logger.info("GeminiTTS tool call: %s args=%s", fc.name, dict(fc.args))
                try:
                    result = await dispatch_tool_call(fc.name, json.dumps(dict(fc.args)), self.deps)
                except Exception as exc:
                    result = {"error": str(exc)}

                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(name=fc.name, response=result))
                )
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": f"🛠️ Used tool {fc.name}"})
                )

            history.append(types.Content(role="user", parts=response_parts))

        return "[Response generation reached tool call limit]"

    async def _call_tts_with_retry(self, text: str, system_instruction: str | None = None) -> bytes | None:
        """Call Gemini TTS, retrying up to 3 times on transient errors."""
        assert self._client is not None, "Client not initialised"

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
                    logger.warning("TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
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

    @staticmethod
    def _pcm_to_frames(pcm_bytes: bytes) -> "list[np.ndarray[Any, Any]]":
        """Split raw 16-bit PCM bytes into ~100 ms numpy frames."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return [
            audio[i : i + _CHUNK_SAMPLES]
            for i in range(0, len(audio), _CHUNK_SAMPLES)
            if len(audio[i : i + _CHUNK_SAMPLES]) > 0
        ]
