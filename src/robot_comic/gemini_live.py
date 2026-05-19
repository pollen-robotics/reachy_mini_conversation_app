"""Gemini Live API handler for real-time audio conversation.

Drop-in alternative to OpenaiRealtimeHandler. Uses the google-genai SDK's
Live API for bidirectional audio streaming with function calling support.

Audio formats (per Gemini Live API spec):
  Input:  16-bit PCM, 16 kHz, mono
  Output: 16-bit PCM, 24 kHz, mono
"""

from __future__ import annotations
import re
import json
import time
import uuid
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Final, Tuple, Optional
from datetime import datetime

import numpy as np
import gradio as gr
from fastrtc import AsyncStreamHandler
from numpy.typing import NDArray
from scipy.signal import resample
from opentelemetry import trace
from opentelemetry import context as otel_context


if TYPE_CHECKING:
    from fastrtc import AdditionalOutputs
    from google.genai import types

from robot_comic import telemetry
from robot_comic.config import (
    GEMINI_BACKEND,
    GEMINI_AVAILABLE_VOICES,
    DEFAULT_VOICE_BY_BACKEND,
    config,
)
from robot_comic.prompts import get_session_voice, get_session_instructions
from robot_comic.gemini_retry import (
    compute_backoff,
    is_rate_limit_error,
    describe_quota_failure,
    extract_retry_after_seconds,
)
from robot_comic.presence_monitor import PresenceMonitor
from robot_comic.tools.core_tools import (
    ToolDependencies,
    get_active_tool_specs,
)
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.camera_frame_encoding import encode_bgr_frame_as_jpeg
from robot_comic.tools.name_validation import record_user_transcript
from robot_comic.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000
GEMINI_OUTPUT_SAMPLE_RATE: Final[int] = 24000
_B64_IMAGE_RESULT_KEYS: Final[tuple[str, ...]] = ("b64_im", "b64_scene")

# Per-million-token prices for `gemini-3.1-flash-live-preview` in USD.
# Source: https://ai.google.dev/gemini-api/docs/pricing
# Last verified: 2026-05-13
# Used only for the locally-aggregated cost counter — Google bills off
# their own metering, not ours.
GEMINI_LIVE_AUDIO_INPUT_COST_PER_1M: Final[float] = 3.0
GEMINI_LIVE_AUDIO_OUTPUT_COST_PER_1M: Final[float] = 12.0
GEMINI_LIVE_TEXT_INPUT_COST_PER_1M: Final[float] = 0.75
GEMINI_LIVE_TEXT_OUTPUT_COST_PER_1M: Final[float] = 4.50


def _build_realtime_input_config() -> "types.RealtimeInputConfig":
    """Build a RealtimeInputConfig with our tuned activity-detection knobs.

    The SDK defaults are eager: short pauses, breaths, and even the assistant's
    own audio leaking into the mic can fire a fresh turn. We bias towards
    LOW sensitivity and longer silence so brief user pauses don't preempt the
    model mid-response. All four knobs are env-overridable for tuning.
    """
    from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

    _start_sensitivity_map: Dict[str, Any] = {
        "HIGH": types.StartSensitivity.START_SENSITIVITY_HIGH,
        "LOW": types.StartSensitivity.START_SENSITIVITY_LOW,
        "UNSPECIFIED": types.StartSensitivity.START_SENSITIVITY_UNSPECIFIED,
    }
    _end_sensitivity_map: Dict[str, Any] = {
        "HIGH": types.EndSensitivity.END_SENSITIVITY_HIGH,
        "LOW": types.EndSensitivity.END_SENSITIVITY_LOW,
        "UNSPECIFIED": types.EndSensitivity.END_SENSITIVITY_UNSPECIFIED,
    }
    start_sens = _start_sensitivity_map.get(
        config.GEMINI_LIVE_VAD_START_SENSITIVITY,
        types.StartSensitivity.START_SENSITIVITY_LOW,
    )
    end_sens = _end_sensitivity_map.get(
        config.GEMINI_LIVE_VAD_END_SENSITIVITY,
        types.EndSensitivity.END_SENSITIVITY_LOW,
    )
    return types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            start_of_speech_sensitivity=start_sens,
            end_of_speech_sensitivity=end_sens,
            prefix_padding_ms=config.GEMINI_LIVE_VAD_PREFIX_MS,
            silence_duration_ms=config.GEMINI_LIVE_VAD_SILENCE_MS,
        ),
    )


def _openai_tool_specs_to_gemini(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool specs to Gemini function_declarations format.

    OpenAI format:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}

    Gemini format:
        {"name": "...", "description": "...", "parameters": {...}}

    The parameters schema is mostly compatible (JSON Schema), but Gemini uses
    uppercase type names (STRING, NUMBER, OBJECT, ARRAY, BOOLEAN, INTEGER).
    """
    declarations = []
    for spec in specs:
        decl: Dict[str, Any] = {
            "name": spec["name"],
        }
        if "description" in spec:
            decl["description"] = spec["description"]
        if "parameters" in spec and spec["parameters"]:
            decl["parameters"] = _convert_schema_types(spec["parameters"])
        declarations.append(decl)
    return declarations


def _convert_schema_types(schema: Any) -> Any:
    """Recursively convert JSON Schema type strings to Gemini uppercase format."""
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    # Convert type field to uppercase
    if "type" in result:
        type_map = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }
        t = result["type"]
        if isinstance(t, str):
            result["type"] = type_map.get(t.lower(), t.upper())

    # Recurse into properties
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {k: _convert_schema_types(v) for k, v in result["properties"].items()}

    # Recurse into items (for arrays)
    if "items" in result:
        result["items"] = _convert_schema_types(result["items"])

    # Remove fields not supported by Gemini
    for unsupported_key in ("additionalProperties",):
        result.pop(unsupported_key, None)

    return result


_TTS_DELIVERY_TAG_NAMES = (
    "fast",
    "slow",
    "short pause",
    "long pause",
    "amusement",
    "annoyance",
    "aggression",
    "enthusiasm",
)
_TTS_SECTION_RE = re.compile(
    r"## GEMINI TTS DELIVERY TAGS\b.*?(?=\r?\n##|\Z)",
    re.DOTALL,
)
_TTS_TAG_RE = re.compile(r"\[(?:" + "|".join(re.escape(t) for t in _TTS_DELIVERY_TAG_NAMES) + r")\]")


def _strip_tts_delivery_tags(instructions: str) -> str:
    """Remove Gemini-TTS-specific delivery tags from system instructions.

    Strips the entire GEMINI TTS DELIVERY TAGS section and any residual
    inline [tag] patterns so Gemini Live does not read them aloud.
    """
    result = _TTS_SECTION_RE.sub("", instructions)
    result = _TTS_TAG_RE.sub("", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _load_profile_live_styling() -> str | None:
    """Read profiles/<name>/gemini_live.txt if present.

    Unlike Gemini TTS — which renders prepared text through a separate TTS
    call and supports inline [delivery_tag] markers — Gemini Live produces
    audio inside the live session. Per-profile styling here is appended to
    the base system instructions as a delivery guideline the model can
    apply while it speaks. Returns None if the profile or file is absent.
    """
    profile: str | None = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    if not profile:
        return None
    try:
        path = config.PROFILES_DIRECTORY / profile / "gemini_live.txt"
        if path.exists():
            text: str = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception as exc:
        logger.warning("Could not read gemini_live.txt for profile %r: %s", profile, exc)
    return None


def _resolve_gemini_voice(profile_voice: str) -> str:
    """Map a profile voice name to the closest Gemini voice.

    If the voice is already a valid Gemini voice (case-insensitive), use it.
    Otherwise fall back to the default.
    """
    voice_map = {v.lower(): v for v in GEMINI_AVAILABLE_VOICES}
    return voice_map.get(profile_voice.lower(), DEFAULT_VOICE_BY_BACKEND[GEMINI_BACKEND])


async def _send_b64_tool_image_to_gemini(
    session: Any,
    tool_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Send base64 JPEG tool payloads as video input and return compact JSON."""
    from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

    compact_result = dict(tool_result)
    image_key = next(
        (key for key in _B64_IMAGE_RESULT_KEYS if isinstance(compact_result.get(key), str | bytes | bytearray)),
        None,
    )
    if image_key is None:
        return compact_result

    b64_image = compact_result.pop(image_key)
    if compact_result:
        compact_result.setdefault("image", "sent_as_realtime_video_input")
        compact_result.setdefault("image_source", image_key)
    else:
        compact_result = {"status": "image_captured"}

    try:
        if isinstance(b64_image, str):
            image_bytes = base64.b64decode(b64_image)
        else:
            image_bytes = bytes(b64_image)
        await session.send_realtime_input(video=types.Blob(data=image_bytes, mime_type="image/jpeg"))
    except Exception as exc:
        compact_result["image_error"] = f"failed_to_send_realtime_video_input: {exc}"
    return compact_result


def _resolve_gemini_startup_voice(voice: str | None) -> str | None:
    """Return a valid persisted Gemini startup voice or None."""
    if voice is None:
        return None

    voice_map = {candidate.lower(): candidate for candidate in GEMINI_AVAILABLE_VOICES}
    resolved = voice_map.get(voice.lower())
    if resolved is None:
        logger.warning(
            "Ignoring persisted Gemini startup voice %r; expected one of %s",
            voice,
            GEMINI_AVAILABLE_VOICES,
        )
    return resolved


_QUESTION_RE = re.compile(
    r"""
    \?                                   # ends with ?
    |
    \b(who|what|where|when|why|how      # interrogative words
    |are\s+you|do\s+you|can\s+you
    |would\s+you|will\s+you|have\s+you
    |did\s+you|is\s+that|isn't\s+that
    |don't\s+you|doesn't\s+that
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _is_question_turn(transcript: str) -> bool:
    """Return True when the assistant's finalised transcript looks like a question.

    Heuristic: ends with '?' or contains a common interrogative.  Intentionally
    lenient so borderline prompts ("Tell me about yourself…") don't fall through.
    """
    return bool(_QUESTION_RE.search(transcript.strip()))


class GeminiLiveHandler(AsyncStreamHandler, ConversationHandler):
    """Gemini Live API handler for fastrtc Stream."""

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.sim_mode = sim_mode
        self.instance_path = instance_path
        self._voice_override: str | None = _resolve_gemini_startup_voice(startup_voice)

        self.session: Any = None  # google.genai live session
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Stop event for the receive loop
        self._stop_event: asyncio.Event = asyncio.Event()
        self._pending_user_transcript_chunks: list[str] = []
        self._pending_assistant_transcript_chunks: list[str] = []
        self._listening_state = False

        # OTel turn-span tracking
        self._session_id: str = str(uuid.uuid4())
        self._turn_id: str | None = None
        self._turn_span: Any = None
        self._turn_ctx_token: Any = None
        self._turn_start_at: float | None = None
        self._user_done_at: float | None = None
        self._llm_span: Any = None
        self._llm_start_at: float | None = None
        self._tts_span: Any = None
        self._tts_start_at: float | None = None
        self._first_audio_at: float | None = None

        # Aggregated cost in USD, computed from usage_metadata frames. Mirrors
        # the BaseRealtimeHandler.cumulative_cost field so dashboards that
        # already read that name keep working.
        self.cumulative_cost: float = 0.0

        # Presence monitor — arms after question turns, fires re-prompt nudges
        # on exponential backoff, then enters silent-wait after max_attempts.
        # Only instantiated when GEMINI_LIVE_PRESENCE_ENABLED is set.
        self._presence_monitor: PresenceMonitor | None = (
            PresenceMonitor(
                probe_callback=self._send_presence_probe,
                first_s=config.GEMINI_LIVE_PRESENCE_FIRST_S,
                backoff_factor=config.GEMINI_LIVE_PRESENCE_BACKOFF_FACTOR,
                max_attempts=config.GEMINI_LIVE_PRESENCE_MAX_ATTEMPTS,
            )
            if config.GEMINI_LIVE_PRESENCE_ENABLED
            else None
        )

    def copy(self) -> "GeminiLiveHandler":
        """Create a copy of the handler."""
        return GeminiLiveHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    def _close_turn_span(self, outcome: str) -> None:
        """End the active root turn span (idempotent) and record the turn-duration metric."""
        if self._tts_span is not None:
            self._tts_span.end()
            self._tts_span = None
        if self._llm_span is not None:
            self._llm_span.end()
            self._llm_span = None
        if self._turn_span is not None:
            self._turn_span.set_attribute("turn.outcome", outcome)
            self._turn_span.end()
            self._turn_span = None
            if self._turn_start_at is not None:
                telemetry.record_turn(
                    time.perf_counter() - self._turn_start_at,
                    {"robot.mode": "gemini", "turn.outcome": outcome},
                )
        if self._turn_ctx_token is not None:
            otel_context.detach(self._turn_ctx_token)
            self._turn_ctx_token = None
        self._turn_start_at = None
        self._user_done_at = None
        self._llm_start_at = None
        self._tts_start_at = None
        self._first_audio_at = None

    def _record_usage_metadata(self, usage: Any) -> None:
        """Convert Gemini Live usage_metadata to cost + OTel attributes.

        Gemini reports modality-broken-down token counts in
        ``prompt_tokens_details`` and ``response_tokens_details`` (each a list
        of ``ModalityTokenCount`` with ``modality`` and ``token_count``). We
        price AUDIO and TEXT separately; other modalities (IMAGE/VIDEO/etc.)
        fall back to text pricing since we have no per-modality rate for them.
        """
        if usage is None:
            return

        def _by_modality(details: Any) -> Dict[str, int]:
            buckets: Dict[str, int] = {}
            if not details:
                return buckets
            for entry in details:
                modality = getattr(entry, "modality", None)
                if modality is None:
                    continue
                key = getattr(modality, "value", str(modality)).upper()
                buckets[key] = buckets.get(key, 0) + (getattr(entry, "token_count", 0) or 0)
            return buckets

        prompt_by_mod = _by_modality(getattr(usage, "prompt_tokens_details", None))
        response_by_mod = _by_modality(getattr(usage, "response_tokens_details", None))

        prompt_total = getattr(usage, "prompt_token_count", 0) or 0
        response_total = getattr(usage, "response_token_count", 0) or 0

        audio_in = prompt_by_mod.get("AUDIO", 0)
        text_in = max(prompt_total - audio_in, 0) if not prompt_by_mod else prompt_by_mod.get("TEXT", 0)
        audio_out = response_by_mod.get("AUDIO", 0)
        text_out = max(response_total - audio_out, 0) if not response_by_mod else response_by_mod.get("TEXT", 0)

        cost = (
            audio_in * GEMINI_LIVE_AUDIO_INPUT_COST_PER_1M / 1e6
            + text_in * GEMINI_LIVE_TEXT_INPUT_COST_PER_1M / 1e6
            + audio_out * GEMINI_LIVE_AUDIO_OUTPUT_COST_PER_1M / 1e6
            + text_out * GEMINI_LIVE_TEXT_OUTPUT_COST_PER_1M / 1e6
        )
        self.cumulative_cost += cost
        logger.debug(
            "Cost: $%.4f | Cumulative: $%.4f (in=%d audio=%d text=%d, out=%d audio=%d text=%d)",
            cost,
            self.cumulative_cost,
            prompt_total,
            audio_in,
            text_in,
            response_total,
            audio_out,
            text_out,
        )

        if self._llm_span is not None:
            try:
                self._llm_span.set_attribute("gen_ai.usage.input_tokens", prompt_total)
                self._llm_span.set_attribute("gen_ai.usage.output_tokens", response_total)
                if cost > 0:
                    self._llm_span.set_attribute("gen_ai.usage.cost_usd", round(cost, 6))
            except Exception:
                pass

    def _set_listening_state(self, listening: bool) -> None:
        """Avoid queueing redundant listening-state updates."""
        if self._listening_state == listening:
            return
        self._listening_state = listening
        self.deps.movement_manager.set_listening(listening)

    async def _send_presence_probe(self, attempt: int, nudge: str) -> None:
        """Probe callback invoked by PresenceMonitor when the user has been silent.

        Sends the nudge as synthetic user-content text so the model sees it as
        an in-conversation cue rather than a system instruction.
        """
        if not self.session:
            logger.debug("PresenceMonitor probe %d: no active session, skipping nudge", attempt)
            return
        try:
            await self.session.send_realtime_input(text=nudge)
            logger.debug("PresenceMonitor probe %d sent: %r", attempt, nudge)
        except Exception as exc:
            logger.warning("PresenceMonitor probe %d: failed to send nudge: %s", attempt, exc)

    async def _flush_transcript_chunks(self, role: str, chunks: list[str]) -> None:
        """Emit one finalized transcript message for the current turn.

        For user transcripts this also (a) sets ``turn.excerpt`` on the active
        turn span so the monitor's "What" column gets populated, and (b)
        routes the transcript through the pause controller so stop/resume
        phrases work on Gemini Live the same way they do on the realtime
        handlers that inherit from BaseRealtimeHandler.
        """
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        if not chunks:
            return

        transcript = "".join(chunks).strip()
        chunks.clear()
        if not transcript:
            return

        if role == "user":
            # Record for tool-side name-validation guard (#287).
            record_user_transcript(self.deps.recent_user_transcripts, transcript)

            if self._turn_span is not None:
                try:
                    self._turn_span.set_attribute("turn.excerpt", transcript[:200])
                except Exception:
                    pass

            pause_controller = getattr(self.deps, "pause_controller", None)
            if pause_controller is not None:
                try:
                    pause_controller.handle_transcript(transcript)
                except Exception as e:
                    logger.error("pause_controller.handle_transcript raised: %s", e)

        await self.output_queue.put(AdditionalOutputs({"role": role, "content": transcript}))

    async def _mark_model_response_started(self) -> None:
        """Switch out of user-listening mode when the model begins responding."""
        await self._flush_transcript_chunks("user", self._pending_user_transcript_chunks)
        self._set_listening_state(False)
        # New assistant turn starting — cancel any pending presence probe so we
        # don't double-fire if the nudge itself triggered a model response.
        if self._presence_monitor is not None:
            self._presence_monitor.cancel()

    async def _handle_interruption(self) -> None:
        """Stop current playback and preserve any transcript already spoken."""
        logger.debug("Gemini: user interrupted")
        await self._flush_transcript_chunks("assistant", self._pending_assistant_transcript_chunks)
        self._close_turn_span("interrupted")
        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
            self._clear_queue()
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
        self._set_listening_state(True)

    async def _handle_turn_complete(self) -> None:
        """Finalize the current turn and restore post-speech motion state."""
        logger.debug("Gemini turn complete")
        await self._flush_transcript_chunks("user", self._pending_user_transcript_chunks)

        # Capture the assistant transcript *before* flushing so we can
        # classify it as a question turn for the presence monitor.
        assistant_transcript = "".join(self._pending_assistant_transcript_chunks).strip()
        await self._flush_transcript_chunks("assistant", self._pending_assistant_transcript_chunks)

        self._close_turn_span("success")
        self._set_listening_state(False)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.request_reset_after_current_audio()

        # Arm presence monitor when the assistant ends on a question.
        if self._presence_monitor is not None and assistant_transcript and _is_question_turn(assistant_transcript):
            logger.debug("Presence monitor armed after question turn")
            self._presence_monitor.arm()

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime.

        For Gemini Live, we must restart the session since there's no
        session.update equivalent.
        """
        try:
            from robot_comic.config import set_custom_profile

            set_custom_profile(profile)
            logger.info("Set custom profile to %r", profile)

            try:
                _ = get_session_instructions()
                _ = get_session_voice(backend="gemini")
            except BaseException as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Force a restart to apply new config
            if self.session is not None:
                try:
                    await self._restart_session()
                    return "Applied personality and restarted Gemini session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def change_voice(self, voice: str) -> str:
        """Change only the voice and restart the session."""
        self._voice_override = voice
        if getattr(self, "client", None) is not None:
            try:
                await self._restart_session()
                return f"Voice changed to {voice}."
            except Exception as e:
                logger.warning("Failed to restart session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the resolved Gemini voice currently selected for this handler."""
        return _resolve_gemini_voice(self._voice_override or get_session_voice(backend="gemini"))

    async def start_up(self) -> None:
        """Start the handler with retries on unexpected closure."""
        gemini_api_key = config.GEMINI_API_KEY
        if self.sim_mode and not gemini_api_key:
            # Sim mode: poll the env for the key. The admin UI writes it to
            # the instance .env when the user enters credentials.
            from robot_comic.config import refresh_runtime_config_from_env

            logger.warning("GEMINI_API_KEY not set; waiting for the admin UI at / to provide it…")
            while not gemini_api_key:
                await asyncio.sleep(0.2)
                try:
                    refresh_runtime_config_from_env()
                except Exception:
                    pass
                gemini_api_key = config.GEMINI_API_KEY
        else:
            if not gemini_api_key or not gemini_api_key.strip():
                logger.warning("GEMINI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                gemini_api_key = "DUMMY"

        from google import genai  # deferred: google.genai.types costs ~5.5 s at boot

        self.client = genai.Client(api_key=gemini_api_key)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_live_session()
                return
            except Exception as e:
                rate_limited = is_rate_limit_error(e)
                if rate_limited:
                    logger.warning(
                        "Gemini Live rate-limited (quota=%s, attempt %d/%d): %s",
                        describe_quota_failure(e),
                        attempt,
                        max_attempts,
                        e,
                    )
                else:
                    logger.warning(
                        "Gemini Live session closed unexpectedly (attempt %d/%d): %s",
                        attempt,
                        max_attempts,
                        e,
                    )
                if attempt < max_attempts:
                    retry_after = extract_retry_after_seconds(e) if rate_limited else None
                    delay = compute_backoff(attempt - 1, base_delay=1.0, retry_after_s=retry_after)
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                if rate_limited:
                    logger.error(
                        "Gemini Live exhausted %d retries on rate-limit (quota=%s); giving up",
                        max_attempts,
                        describe_quota_failure(e),
                    )
                raise
            finally:
                self.session = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one."""
        try:
            # Drop any pending presence probe; do not carry timers across reconnects.
            if self._presence_monitor is not None:
                self._presence_monitor.cancel()

            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    pass
                finally:
                    self.session = None

            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: Gemini client not initialized yet.")
                return

            try:
                self._connected_event.clear()
            except Exception:
                pass
            self._stop_event.set()  # Signal the old receive loop to stop
            await asyncio.sleep(0.1)
            self._stop_event.clear()
            asyncio.create_task(self.start_up(), name="gemini-live-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Gemini Live session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Gemini Live session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    def _build_live_config(self) -> "types.LiveConnectConfig":
        """Build the LiveConnectConfig for a Gemini Live session."""
        from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

        instructions = _strip_tts_delivery_tags(get_session_instructions())

        # Append any profile-specific delivery guidance (Brooklyn rapid-fire
        # for Rickles, drawl for Pryor, etc.) so Gemini Live speaks in the
        # persona's voice without needing those cues baked into every line.
        live_styling = _load_profile_live_styling()
        if live_styling:
            instructions = f"{instructions}\n\n## DELIVERY\n{live_styling}"

        voice = _resolve_gemini_voice(self._voice_override or get_session_voice(backend="gemini"))

        # Convert OpenAI-style tool specs to Gemini function declarations
        tool_specs = get_active_tool_specs(self.deps)
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
        function_declarations = _openai_tool_specs_to_gemini(tool_specs)

        tools_config: List[Dict[str, Any]] = []
        if function_declarations:
            tools_config.append({"function_declarations": function_declarations})

        live_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(parts=[types.Part(text=instructions)]),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    ),
                ),
            ),
            tools=tools_config,  # type: ignore[arg-type]
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=_build_realtime_input_config(),
        )

        logger.info(
            "Gemini Live config: model=%r voice=%r tools=%d vad_silence=%dms vad_prefix=%dms vad_start=%s vad_end=%s",
            config.MODEL_NAME,
            voice,
            len(function_declarations),
            config.GEMINI_LIVE_VAD_SILENCE_MS,
            config.GEMINI_LIVE_VAD_PREFIX_MS,
            config.GEMINI_LIVE_VAD_START_SENSITIVITY,
            config.GEMINI_LIVE_VAD_END_SENSITIVITY,
        )
        return live_config

    async def _handle_tool_call(self, response: Any) -> None:
        """Process a tool_call from Gemini and send the response back."""
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        if not response.tool_call or not response.tool_call.function_calls:
            return

        for fc in response.tool_call.function_calls:
            tool_name = fc.name
            call_id = fc.id or str(uuid.uuid4())
            args_dict = dict(fc.args) if fc.args else {}
            args_json_str = json.dumps(args_dict)

            logger.info(
                "Gemini tool call: tool_name=%r, call_id=%s, is_idle=%s, args=%s",
                tool_name,
                call_id,
                self.is_idle_tool_call,
                args_json_str,
            )

            bg_tool = await self.tool_manager.start_tool(
                call_id=call_id,
                tool_call_routine=ToolCallRoutine(
                    tool_name=tool_name,
                    args_json_str=args_json_str,
                    deps=self.deps,
                ),
                is_idle_tool_call=self.is_idle_tool_call,
            )

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. Tool ID: {bg_tool.tool_id}",
                    },
                ),
            )

            if self.is_idle_tool_call:
                self.is_idle_tool_call = False

            logger.info("Started background tool: %s (id=%s, call_id=%s)", tool_name, bg_tool.tool_id, call_id)

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Process the result of a completed tool and send it back to Gemini."""
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot
        from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

        if bg_tool.error is not None:
            logger.error("Tool '%s' (id=%s) failed: %s", bg_tool.tool_name, bg_tool.id, bg_tool.error)
            tool_result = {"error": bg_tool.error}
        elif bg_tool.result is not None:
            tool_result = bg_tool.result
            logger.info("Tool '%s' (id=%s) succeeded.", bg_tool.tool_name, bg_tool.id)
        else:
            logger.warning("Tool '%s' (id=%s) returned no result and no error", bg_tool.tool_name, bg_tool.id)
            tool_result = {"error": "No result returned from tool execution"}

        if not self.session:
            logger.warning("Connection closed during tool '%s' execution", bg_tool.tool_name)
            return

        try:
            if isinstance(tool_result, dict):
                original_keys = set(tool_result)
                tool_result = await _send_b64_tool_image_to_gemini(self.session, tool_result)
                if original_keys != set(tool_result):
                    if "image_error" in tool_result:
                        logger.warning(
                            "Failed to push tool image from %s to Gemini: %s",
                            bg_tool.tool_name,
                            tool_result["image_error"],
                        )
                    else:
                        logger.info("Pushed tool image from %s to Gemini via realtime video input", bg_tool.tool_name)

            console_content = json.dumps(tool_result)

            function_response = types.FunctionResponse(
                id=bg_tool.id if isinstance(bg_tool.id, str) else str(bg_tool.id),
                name=bg_tool.tool_name,
                response=tool_result,
            )
            await self.session.send_tool_response(function_responses=[function_response])

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": console_content,
                        "metadata": {
                            "title": f"🛠️ Used tool {bg_tool.tool_name}",
                            "status": "done",
                        },
                    },
                ),
            )

            if bg_tool.tool_name == "camera" and self.deps.camera_worker is not None:
                np_img = self.deps.camera_worker.get_latest_frame()
                if np_img is not None:
                    rgb_frame = np.ascontiguousarray(np_img[..., ::-1])
                else:
                    rgb_frame = None
                img = gr.Image(value=rgb_frame)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": img}),
                )

        except Exception as e:
            logger.warning("Error sending tool result to Gemini: %s", e)

    async def _video_sender_loop(self) -> None:
        """Send camera frames to Gemini Live at ~1 FPS for continuous visual context.

        Only runs when a camera_worker is available. Frames are JPEG-encoded
        and sent via send_realtime_input(video=...).
        """
        from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

        logger.info("Video sender loop started (1 FPS)")
        while not self._stop_event.is_set():
            try:
                if self.session and self.deps.camera_worker is not None:
                    frame = self.deps.camera_worker.get_latest_frame()
                    if frame is not None:
                        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
                        await self.session.send_realtime_input(
                            video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                        )
            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.debug("Video sender error (will retry): %s", e)

            await asyncio.sleep(1.0)  # 1 FPS

        logger.info("Video sender loop stopped")

    async def _run_live_session(self) -> None:
        """Establish and manage a single Gemini Live session."""
        live_config = self._build_live_config()

        async with self.client.aio.live.connect(
            model=config.MODEL_NAME,
            config=live_config,
        ) as session:
            self.session = session
            try:
                self._connected_event.set()
            except Exception:
                pass

            logger.info("Gemini Live session connected successfully")

            video_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start video sender only when explicitly enabled — continuous
                # streaming triggers 1007 errors on some Gemini Live models.
                if self.deps.camera_worker is not None and config.GEMINI_LIVE_VIDEO_STREAMING:
                    video_task = asyncio.create_task(self._video_sender_loop(), name="gemini-video-sender")

                # session.receive() yields responses for the current turn then completes.
                # We loop so the session stays alive across multiple conversation turns.
                while not self._stop_event.is_set():
                    try:
                        async for response in session.receive():
                            if self._stop_event.is_set():
                                logger.info("Stop event set, breaking receive loop")
                                break

                            # Handle server content (audio, transcription, interruption)
                            if response.server_content:
                                content = response.server_content

                                # Handle interruption / barge-in
                                if content.interrupted is True:
                                    await self._handle_interruption()

                                # Handle audio output from model
                                if content.model_turn and content.model_turn.parts:
                                    has_audio_part = any(
                                        part.inline_data and part.inline_data.data for part in content.model_turn.parts
                                    )
                                    if has_audio_part:
                                        await self._mark_model_response_started()

                                    for part in content.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            audio_bytes = part.inline_data.data
                                            if isinstance(audio_bytes, str):
                                                audio_bytes = base64.b64decode(audio_bytes)

                                            if len(audio_bytes) == 0:
                                                continue

                                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                                            if len(audio_array) == 0:
                                                continue

                                            if self.sim_mode and self.deps.head_wobbler is not None:
                                                self.deps.head_wobbler.feed(
                                                    base64.b64encode(audio_bytes).decode("utf-8")
                                                )

                                            self.last_activity_time = asyncio.get_event_loop().time()

                                            if self._first_audio_at is None and self._user_done_at is not None:
                                                self._first_audio_at = time.perf_counter()
                                                ttft_s = self._first_audio_at - self._user_done_at
                                                if self._llm_span is not None:
                                                    self._llm_span.set_attribute(
                                                        "gen_ai.server.time_to_first_token", ttft_s
                                                    )
                                                telemetry.record_ttft(
                                                    ttft_s,
                                                    {
                                                        "gen_ai.system": "gemini",
                                                        "gen_ai.request.model": config.MODEL_NAME,
                                                    },
                                                )
                                                if self._tts_span is None:
                                                    self._tts_start_at = self._first_audio_at
                                                    # Explicitly parent the TTS span under the active turn span
                                                    # so it shares the same trace even when _turn_ctx_token
                                                    # has not yet been attached (fast/interrupted turns).
                                                    _tts_ctx = (
                                                        trace.set_span_in_context(self._turn_span)
                                                        if self._turn_span is not None
                                                        else None
                                                    )
                                                    self._tts_span = telemetry.get_tracer().start_span(
                                                        "tts.synthesize", context=_tts_ctx
                                                    )

                                            from robot_comic.startup_timer import log_once

                                            log_once("first TTS audio frame", logger)
                                            telemetry.emit_first_greeting_audio_once()
                                            await self.output_queue.put(
                                                (GEMINI_OUTPUT_SAMPLE_RATE, audio_array),
                                            )

                                # Handle input transcription (user speech)
                                if content.input_transcription and content.input_transcription.text:
                                    transcript = content.input_transcription.text
                                    logger.debug("User transcript chunk: %s", transcript)
                                    if not self._pending_user_transcript_chunks:
                                        # First chunk — open turn span and record user-done timestamp
                                        self._close_turn_span("interrupted")
                                        self._turn_id = str(uuid.uuid4())
                                        now_pc = time.perf_counter()
                                        self._turn_start_at = now_pc
                                        self._user_done_at = now_pc
                                        self._first_audio_at = None
                                        self._turn_span = telemetry.get_tracer().start_span(
                                            "turn",
                                            attributes={
                                                "turn.id": self._turn_id,
                                                "session.id": self._session_id,
                                                "robot.mode": "gemini",
                                                "robot.persona": telemetry.current_persona(),
                                            },
                                        )
                                        self._turn_ctx_token = otel_context.attach(
                                            trace.set_span_in_context(self._turn_span)
                                        )
                                        self._llm_start_at = now_pc
                                        self._llm_span = telemetry.get_tracer().start_span(
                                            "llm.request",
                                            attributes={
                                                "gen_ai.system": "gemini",
                                                "gen_ai.operation.name": "chat",
                                                "gen_ai.request.model": config.MODEL_NAME,
                                            },
                                        )
                                    self._pending_user_transcript_chunks.append(transcript)
                                    self._set_listening_state(True)
                                    # User is speaking — cancel any pending presence probe.
                                    if self._presence_monitor is not None:
                                        self._presence_monitor.on_user_activity()

                                # Handle output transcription (model speech)
                                if content.output_transcription and content.output_transcription.text:
                                    transcript = content.output_transcription.text
                                    logger.debug("Assistant transcript chunk: %s", transcript)
                                    await self._mark_model_response_started()
                                    self._pending_assistant_transcript_chunks.append(transcript)

                                # Turn complete
                                if content.turn_complete:
                                    await self._handle_turn_complete()

                            # Handle tool calls
                            if response.tool_call:
                                await self._handle_tool_call(response)

                            # Cost tracking — emitted by the server at varying
                            # cadence (typically per response or per turn end).
                            usage_metadata = getattr(response, "usage_metadata", None)
                            if usage_metadata is not None:
                                self._record_usage_metadata(usage_metadata)

                    except Exception as e:
                        if self._stop_event.is_set():
                            break
                        logger.warning("Receive loop error, restarting Gemini session: %s", e)
                        raise

            finally:
                if video_task is not None:
                    video_task.cancel()
                    try:
                        await video_task
                    except asyncio.CancelledError:
                        pass
                await self.tool_manager.shutdown()

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone and send to Gemini."""
        from fastrtc import audio_to_int16  # deferred: fastrtc pulls gradio at boot
        from google.genai import types  # deferred: google.genai.types costs ~5.5 s at boot

        if not self.session:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample to 16kHz if needed
        if GEMINI_INPUT_SAMPLE_RATE != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sample_rate),
            )

        # Cast to int16
        audio_frame = audio_to_int16(audio_frame)

        # Send raw PCM bytes to Gemini
        try:
            pcm_bytes = audio_frame.tobytes()
            await self.session.send_realtime_input(audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000"))
        except Exception as e:
            logger.debug("Dropping audio frame: session not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        from fastrtc import wait_for_item  # deferred: fastrtc pulls gradio at boot

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._stop_event.set()

        if self._presence_monitor is not None:
            self._presence_monitor.shutdown()

        await self.tool_manager.shutdown()

        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.debug("session.close() error: %s", e)
            finally:
                self.session = None

        # Clear remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to Gemini."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            "You've been idle for a while. Feel free to get creative - dance, show an emotion, "
            "look around, call idle_do_nothing to stay still and silent, or just be yourself!"
        )
        if not self.session:
            logger.debug("No session, cannot send idle signal")
            return

        await self.session.send_realtime_input(text=timestamp_msg)

    async def get_available_voices(self) -> list[str]:
        """Return the list of available Gemini voices."""
        return list(GEMINI_AVAILABLE_VOICES)
