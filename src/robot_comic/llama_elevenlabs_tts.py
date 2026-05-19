"""llama-server LLM + ElevenLabs TTS response handler.

Combines:
- LLM: llama-server /v1/chat/completions (local Qwen3)
- TTS: ElevenLabs API (cloud)

Select via the admin UI's composable pipeline picker (Phase 4f) or the env
vars REACHY_MINI_PIPELINE_MODE=composable +
REACHY_MINI_AUDIO_OUTPUT_BACKEND=elevenlabs + REACHY_MINI_LLM_BACKEND=llama.
"""

import os
import time
import asyncio
import logging
from typing import Any, Optional

import httpx
import numpy as np

from robot_comic import telemetry
from robot_comic.config import (
    ELEVENLABS_DEFAULT_VOICE,
    ELEVENLABS_AVAILABLE_VOICES,
    config,
)
from robot_comic.prompts import get_session_voice
from robot_comic.gemini_tts import (
    SHORT_PAUSE_MS,
    SHORT_PAUSE_TAG,
    _silence_pcm,
    extract_delivery_tags,
)
from robot_comic.llama_base import _CHUNK_SAMPLES, _OUTPUT_SAMPLE_RATE, BaseLlamaResponseHandler, split_sentences
from robot_comic.elevenlabs_tts import load_profile_elevenlabs_config as _shared_load_profile_elevenlabs_config
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.elevenlabs_voices import resolve_voice_id_by_name
from robot_comic.chatterbox_tag_translator import strip_gemini_tags


logger = logging.getLogger(__name__)

_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY = 0.5


def load_profile_elevenlabs_config() -> dict[str, str]:
    """Delegate to the shared loader so this path picks up `.local.txt` overrides
    (e.g. the IVC voice_id). Kept as a thin wrapper for backward compatibility
    with any external imports of this name.
    """
    return _shared_load_profile_elevenlabs_config()


def apply_voice_settings_deltas(
    base_stability: float,
    base_similarity_boost: float,
    tags: list[str],
) -> dict[str, float]:
    """Map delivery tags to voice_settings adjustments.

    Returns adjusted {stability, similarity_boost} dict, clamped to [0.0, 1.0].
    """
    stability = base_stability
    similarity_boost = base_similarity_boost

    for tag in tags:
        if tag == "fast":
            similarity_boost += 0.2
            stability -= 0.1
        elif tag == "annoyance":
            similarity_boost += 0.3
            stability -= 0.15
        elif tag == "aggression":
            similarity_boost += 0.4
            stability -= 0.2
        elif tag == "slow":
            stability += 0.1
            similarity_boost -= 0.1
        elif tag == "amusement":
            similarity_boost += 0.15
        elif tag == "enthusiasm":
            similarity_boost += 0.2
            stability -= 0.05

    return {
        "stability": max(0.0, min(1.0, stability)),
        "similarity_boost": max(0.0, min(1.0, similarity_boost)),
    }


class LlamaElevenLabsTTSResponseHandler(BaseLlamaResponseHandler):
    """llama-server LLM + ElevenLabs TTS voice output with tool dispatch."""

    _BACKEND_LABEL = "llama_elevenlabs_tts"
    _TTS_SYSTEM = "elevenlabs"
    # Dispatch each sentence's TTS in parallel; a playback chain pumps the
    # per-sentence local queues into output_queue in order. Eliminates the
    # ~300-1000ms first-byte stall per sentence that's audible on long replies.
    _PARALLEL_SENTENCE_TTS = True
    # ElevenLabs Turbo v2.5 pricing: $0.50 per 1M characters (Creator tier)
    # verify against current ElevenLabs pricing
    ELEVENLABS_COST_PER_1M_CHARS: float = 0.50

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        super().__init__(deps, sim_mode, instance_path, startup_voice)
        self._http: httpx.AsyncClient | None = None
        self._last_tts_rate_limited: bool = False
        self.cumulative_cost: float = 0.0

    def copy(self) -> "LlamaElevenLabsTTSResponseHandler":
        return LlamaElevenLabsTTSResponseHandler(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        """Set up the httpx client + log the resolved pipeline.

        Phase 5e.2 idempotency: the migrated triple's factory composes
        a plain handler (no :class:`LocalSTTInputMixin` shell), and the
        LLM and TTS adapters each call this method during their
        ``prepare`` lifecycle. The mixin used to gate the call chain
        behind ``_startup_credentials_ready``; without that wrapper the
        second invocation would leak a fresh ``httpx.AsyncClient`` and
        re-fetch the model name. Guard the body so duplicate calls are
        cheap no-ops. The flag is only set on success so a failed
        attempt still re-tries.
        """
        if getattr(self, "_startup_credentials_ready", False):
            return
        await super()._prepare_startup_credentials()
        self._http = httpx.AsyncClient(timeout=30.0)

        llm_model = await self._fetch_llm_model_name()
        stt_model = getattr(config, "LOCAL_STT_MODEL", "unknown")
        logger.info(
            "Pipeline: Moonshine (%s) → llama-server (%s @ %s) → ElevenLabs TTS (voice=%s)",
            stt_model,
            llm_model,
            self._llama_cpp_url,
            self.get_current_voice(),
        )
        self._startup_credentials_ready = True

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
    # Voice management (ElevenLabs voices)                                #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        return list(ELEVENLABS_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        if self._voice_override:
            return self._voice_override
        # Admin-UI override via ELEVENLABS_VOICE env wins over the profile config so
        # users can pick a voice from the settings page without editing elevenlabs.txt.
        env_voice = (os.environ.get("ELEVENLABS_VOICE") or "").strip()
        if env_voice:
            return env_voice
        config_params = load_profile_elevenlabs_config()
        voice = config_params.get("voice") or get_session_voice(
            backend="elevenlabs",
            default=ELEVENLABS_DEFAULT_VOICE,
        )
        # We no longer gate on `voice in ELEVENLABS_AVAILABLE_VOICES` here — the
        # API returns decorated names (e.g. "Brian - Deep, Resonant and
        # Comforting") so an exact-name gate forces every profile to use the
        # verbose form. Smart resolution happens in `_resolve_voice_id` via
        # `resolve_voice_id_by_name`.
        return voice

    def _resolve_voice_id(self) -> str | None:
        """Resolve the ElevenLabs voice ID.

        Profile config `voice_id=<id>` takes precedence (e.g. PVC clones).
        Otherwise map the named voice via the dynamic voice catalog using
        word-boundary prefix matching so configured short names like
        ``voice=Brian`` resolve to API-decorated names like
        ``"Brian - Deep, Resonant and Comforting"``.
        """
        config_params = load_profile_elevenlabs_config()
        custom_id = config_params.get("voice_id")
        if custom_id:
            return custom_id
        voice_name = self.get_current_voice()
        return resolve_voice_id_by_name(voice_name, fallback_name=ELEVENLABS_DEFAULT_VOICE)

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

        out_queue = target_queue if target_queue is not None else self.output_queue
        sentences = split_sentences(response_text) or [response_text]
        any_audio = False
        # List used as a one-shot first-audio marker shared with _stream_tts_to_queue.
        # When non-empty, the first PCM chunk fires record_tts_first_audio and clears it.
        first_audio_marker: list[float] = [tts_start] if tts_start is not None else []
        for sentence in sentences:
            # Extract delivery tags before stripping so they can guide voice_settings.
            tags = extract_delivery_tags(sentence)
            # Strip Gemini-style delivery tags ([fast], [annoyance], etc.) so they
            # aren't spoken literally. [short pause] becomes a real silence gap.
            spoken = strip_gemini_tags(sentence)
            if not spoken:
                continue
            if SHORT_PAUSE_TAG in tags:
                for frame in self._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS, _OUTPUT_SAMPLE_RATE)):
                    await self._enqueue_audio_frame(frame, target_queue=out_queue)
            sentence_had_audio = await self._stream_tts_to_queue(
                spoken, first_audio_marker, tags, target_queue=out_queue
            )
            if sentence_had_audio:
                any_audio = True

        if not any_audio:
            if self._last_tts_rate_limited:
                msg = "[ElevenLabs TTS rate-limited; try again later]"
            else:
                msg = "[TTS error — ElevenLabs TTS failed]"
            # Error markers always go to the main queue so the chat UI sees them
            # regardless of whether we're streaming into a per-sentence buffer.
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": msg}))

    async def _stream_tts_to_queue(
        self,
        text: str,
        first_audio_marker: list[float] | None = None,
        tags: list[str] | None = None,
        target_queue: "asyncio.Queue[Any] | None" = None,
    ) -> bool:
        """Stream ElevenLabs TTS PCM chunks directly into ``output_queue``.

        Uses the ``/stream`` endpoint with ``output_format=pcm_24000`` so first
        audio arrives in ~100-200ms instead of ~500-1000ms with full-body POST.

        ``first_audio_marker``: when non-empty, the first PCM chunk fires
        ``telemetry.record_tts_first_audio`` (perf_counter - marker[0]) and the
        marker is cleared so subsequent sentences in the same turn don't refire.

        ``tags``: delivery tags to adjust voice_settings (e.g., [fast], [annoyance]).

        Returns True if any audio was streamed for this call.
        """
        assert self._http is not None, "HTTP client not initialised"

        api_key = config.ELEVENLABS_API_KEY
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not configured")
            return False

        voice_id = self._resolve_voice_id()
        if not voice_id:
            logger.error("Could not resolve voice ID for %s", self.get_current_voice())
            return False

        # Accumulate cost: ElevenLabs charges per character (text is already tag-stripped).
        char_count = len(text)
        cost = (char_count / 1_000_000) * self.ELEVENLABS_COST_PER_1M_CHARS
        self.cumulative_cost += cost
        if cost > 0:
            logger.debug(
                "ElevenLabs TTS cost: $%.4f (%d chars) | Cumulative: $%.4f", cost, char_count, self.cumulative_cost
            )

        config_params = load_profile_elevenlabs_config()
        base_stability = float(config_params.get("stability", "0.5"))
        base_similarity_boost = float(config_params.get("similarity_boost", "0.75"))

        # Apply per-sentence tag adjustments to voice_settings.
        voice_settings = apply_voice_settings_deltas(
            base_stability,
            base_similarity_boost,
            tags or [],
        )

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_{_OUTPUT_SAMPLE_RATE}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": voice_settings,
        }

        out_queue = target_queue if target_queue is not None else self.output_queue
        self._last_tts_rate_limited = False
        frame_bytes = _CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample
        for attempt in range(_TTS_MAX_RETRIES):
            try:
                got_audio = False
                leftover = b""
                async with self._http.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        if not got_audio:
                            got_audio = True
                            if first_audio_marker:
                                telemetry.record_tts_first_audio(
                                    time.perf_counter() - first_audio_marker[0],
                                    {"gen_ai.system": "elevenlabs"},
                                )
                                first_audio_marker.clear()
                        leftover += chunk
                        while len(leftover) >= frame_bytes:
                            frame = np.frombuffer(leftover[:frame_bytes], dtype=np.int16)
                            await self._enqueue_audio_frame(frame, target_queue=out_queue)
                            leftover = leftover[frame_bytes:]
                if leftover:
                    tail = np.frombuffer(leftover[: (len(leftover) // 2) * 2], dtype=np.int16)
                    if len(tail) > 0:
                        await self._enqueue_audio_frame(tail, target_queue=out_queue)
                return got_audio
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    self._last_tts_rate_limited = True
                    logger.warning(
                        "ElevenLabs TTS 429 (attempt %d/%d); sleeping %.1fs before retry",
                        attempt + 1,
                        _TTS_MAX_RETRIES,
                        _TTS_RETRY_BASE_DELAY * (2**attempt),
                    )
                    if attempt < _TTS_MAX_RETRIES - 1:
                        await asyncio.sleep(_TTS_RETRY_BASE_DELAY * (2**attempt))
                elif exc.response.status_code == 401:
                    logger.error("ElevenLabs API key invalid or expired")
                    return False
                else:
                    logger.warning("ElevenLabs TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                    if attempt < _TTS_MAX_RETRIES - 1:
                        await asyncio.sleep(_TTS_RETRY_BASE_DELAY)
            except Exception as exc:
                logger.warning("ElevenLabs TTS attempt %d/%d failed: %s", attempt + 1, _TTS_MAX_RETRIES, exc)
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_BASE_DELAY)

        if self._last_tts_rate_limited:
            logger.error("ElevenLabs TTS exhausted %d retries on 429; skipping audio for this turn", _TTS_MAX_RETRIES)
        else:
            logger.error("ElevenLabs TTS exhausted %d retries; skipping audio for this turn", _TTS_MAX_RETRIES)
        return False
