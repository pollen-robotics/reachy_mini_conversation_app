"""Chatterbox TTS response handler for the local-STT audio path.

Receives transcripts from Moonshine STT (via LocalSTTInputMixin), calls a
llama-server LLM via /v1/chat/completions for text generation and tool dispatch,
then synthesises audio with the Chatterbox TTS server's /tts endpoint (voice cloning mode).

Audio output: 24 kHz, mono, 16-bit PCM — matches the existing pipeline.
"""

import time
import asyncio
import logging
from typing import Any

import numpy as np

from robot_comic.config import (
    LLAMA_CPP_DEFAULT_URL,
    CHATTERBOX_DEFAULT_URL,
    AUDIO_OUTPUT_CHATTERBOX,
    CHATTERBOX_DEFAULT_GAIN,
    CHATTERBOX_DEFAULT_VOICE,
    CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_TARGET_DBFS,
    CHATTERBOX_DEFAULT_TEMPERATURE,
    CHATTERBOX_DEFAULT_EXAGGERATION,
    config,
)
from robot_comic.prompts import get_session_instructions
from robot_comic.audio_gain import normalize_gain
from robot_comic.llama_base import _OUTPUT_SAMPLE_RATE, BaseLlamaResponseHandler, split_sentences
from robot_comic.wake_on_lan import send_magic_packet
from robot_comic.tools.core_tools import get_active_tool_specs
from robot_comic.chatterbox_voice_clone import load_voice_clone_ref
from robot_comic.chatterbox_tag_translator import translate


logger = logging.getLogger(__name__)

_CHATTERBOX_SAMPLE_RATE = 24000
_TTS_MAX_RETRIES = 3
_TTS_RETRY_DELAY = 0.5

# Keep the private alias so the test-suite import (from robot_comic.chatterbox_tts import _split_sentences) still works.
_split_sentences = split_sentences


class ChatterboxTTSResponseHandler(BaseLlamaResponseHandler):
    """llama-server LLM + Chatterbox TTS voice output with tool dispatch."""

    _BACKEND_LABEL = "chatterbox"
    _TTS_SYSTEM = "chatterbox"

    def copy(self) -> "ChatterboxTTSResponseHandler":
        """Create a new instance of this handler with the same configuration."""
        return type(self)(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    def _load_profile_params(self) -> dict[str, str]:
        """Read profiles/<name>/chatterbox.txt as key=value pairs, if present."""
        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if not profile:
            return {}
        try:
            path = config.PROFILES_DIRECTORY / profile / "chatterbox.txt"
            if not path.exists():
                return {}
            params: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    params[k.strip()] = v.strip()
            return params
        except Exception as exc:
            logger.warning("Could not read chatterbox.txt for profile %r: %s", profile, exc)
            return {}

    @property
    def _chatterbox_url(self) -> str:
        return getattr(config, "CHATTERBOX_URL", CHATTERBOX_DEFAULT_URL)

    @property
    def _chatterbox_voice(self) -> str:
        if self._voice_override:
            return self._voice_override
        params = self._load_profile_params()
        return str(params.get("voice") or getattr(config, "CHATTERBOX_VOICE", CHATTERBOX_DEFAULT_VOICE))

    @property
    def _exaggeration(self) -> float:
        params = self._load_profile_params()
        return float(
            params.get("exaggeration", getattr(config, "CHATTERBOX_EXAGGERATION", CHATTERBOX_DEFAULT_EXAGGERATION))
        )

    @property
    def _cfg_weight(self) -> float:
        params = self._load_profile_params()
        return float(params.get("cfg_weight", getattr(config, "CHATTERBOX_CFG_WEIGHT", CHATTERBOX_DEFAULT_CFG_WEIGHT)))

    @property
    def _temperature(self) -> float:
        params = self._load_profile_params()
        return float(
            params.get("temperature", getattr(config, "CHATTERBOX_TEMPERATURE", CHATTERBOX_DEFAULT_TEMPERATURE))
        )

    @property
    def _gain(self) -> float:
        params = self._load_profile_params()
        return float(params.get("gain", getattr(config, "CHATTERBOX_GAIN", CHATTERBOX_DEFAULT_GAIN)))

    @property
    def _auto_gain_enabled(self) -> bool:
        return bool(getattr(config, "CHATTERBOX_AUTO_GAIN_ENABLED", True))

    @property
    def _target_dbfs(self) -> float:
        return float(getattr(config, "CHATTERBOX_TARGET_DBFS", CHATTERBOX_DEFAULT_TARGET_DBFS))

    async def _prepare_startup_credentials(self) -> None:
        """Set up Chatterbox TTS startup state.

        Phase 5e.3 idempotency: the migrated triple's factory composes
        a plain handler (no :class:`LocalSTTInputMixin` shell), and the
        LLM and TTS adapters each call this method during their
        ``prepare`` lifecycle. The mixin used to gate the call chain
        behind ``_startup_credentials_ready``; without that wrapper the
        second invocation would leak an extra ``httpx.AsyncClient`` and
        re-fire the llama-server health probe + KV-cache warmup
        (streaming POST) + Chatterbox TTS warmup (real TTS round-trip).
        Guard the body so duplicate calls are cheap no-ops. The flag is
        only set on success so a failed attempt still re-tries.

        Non-chatterbox output backend (e.g. ``AUDIO_OUTPUT_BACKEND=xtts``):
        when this handler is used only as an LLM host — not as the active
        TTS backend — the base-class httpx setup and tool-manager startup
        are still required, but the Chatterbox-specific network probes
        (llama-server health check, LLM KV-cache warmup, TTS warmup) must
        NOT fire.  They would time-out against an absent Chatterbox service
        and pollute the journal with spurious warnings on every boot.
        """
        if getattr(self, "_startup_credentials_ready", False):
            return
        await super()._prepare_startup_credentials()
        # Resolve per-persona voice-clone reference audio once at startup.
        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if profile:
            self._voice_clone_ref_path = load_voice_clone_ref(config.PROFILES_DIRECTORY / profile)
        else:
            self._voice_clone_ref_path = None

        # Gate the Chatterbox-specific probes on whether chatterbox is the
        # active TTS backend.  When another backend (e.g. xtts) is selected,
        # this handler acts only as an LLM host; its TTS machinery is unused
        # and attempting to reach the Chatterbox service would produce
        # misleading ConnectTimeout warnings in the journal on every boot.
        _active_output = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_CHATTERBOX)
        if _active_output != AUDIO_OUTPUT_CHATTERBOX:
            logger.debug(
                "ChatterboxTTS: skipping TTS probes — active output backend is %r, not %r",
                _active_output,
                AUDIO_OUTPUT_CHATTERBOX,
            )
            self._startup_credentials_ready = True
            return

        logger.info(
            "ChatterboxTTS handler initialised: llm=%s/v1/chat/completions tts=%s voice=%s exag=%.2f cfg=%.2f temp=%.2f",
            self._llama_cpp_url,
            self._chatterbox_url,
            self._chatterbox_voice,
            self._exaggeration,
            self._cfg_weight,
            self._temperature,
        )
        await self._probe_llama_health()
        await self._warmup_llm_kv_cache()
        await self._warmup_tts()
        self._startup_credentials_ready = True

    async def _probe_llama_health(self) -> None:
        """Ping llama-server /health at startup and emit a clear warning on failure.

        On success (HTTP 200): logs INFO.
        On non-200, timeout, or connection refused: logs WARNING so operators
        see it in journalctl before the first LLM call fails.

        If ``REACHY_MINI_WOL_MAC`` is set and the initial probe fails, a
        Wake-on-LAN magic packet is sent to wake the host, then the probe is
        retried once after ``REACHY_MINI_WOL_RETRY_AFTER_S`` seconds (default 3).

        Controlled by ``REACHY_MINI_LLAMA_HEALTH_CHECK`` env var (default enabled).
        Set to ``0`` to skip the probe in environments where llama-server starts later.
        """
        if not getattr(config, "LLAMA_HEALTH_CHECK_ENABLED", True):
            logger.debug("llama-server health check disabled via REACHY_MINI_LLAMA_HEALTH_CHECK=0")
            return

        url = getattr(config, "LLAMA_CPP_URL", LLAMA_CPP_DEFAULT_URL)
        health_url = f"{url}/health"
        assert self._http is not None

        probe_ok = await self._single_health_probe(url, health_url)
        if probe_ok:
            return

        # Primary probe failed — attempt WoL fallback if configured.
        wol_mac: str | None = getattr(config, "WOL_MAC", None)
        if not wol_mac:
            return

        wol_broadcast: str = getattr(config, "WOL_BROADCAST", "255.255.255.255")
        wol_delay: float = float(getattr(config, "WOL_RETRY_AFTER_S", 3.0))
        logger.info(
            "llama-server unreachable — sending Wake-on-LAN packet to %s (broadcast %s), "
            "retrying health probe in %.0f s",
            wol_mac,
            wol_broadcast,
            wol_delay,
        )
        try:
            send_magic_packet(wol_mac, broadcast=wol_broadcast)
        except Exception as wol_exc:
            logger.warning("Wake-on-LAN packet could not be sent: %s", wol_exc)

        await asyncio.sleep(wol_delay)

        retry_ok = await self._single_health_probe(url, health_url, log_success_as_wakeup=True)
        if not retry_ok:
            logger.warning(
                "llama-server still unreachable after Wake-on-LAN — first LLM call may fail",
            )

    async def _single_health_probe(
        self,
        url: str,
        health_url: str,
        *,
        log_success_as_wakeup: bool = False,
    ) -> bool:
        """Issue one GET to *health_url* and return ``True`` on HTTP 200.

        Logs INFO on success and WARNING on failure; never raises.
        """
        assert self._http is not None
        try:
            resp = await self._http.get(health_url, timeout=3.0)
            if resp.status_code == 200:
                if log_success_as_wakeup:
                    logger.info("llama-server woke up after WoL — health OK at %s", url)
                else:
                    logger.info("llama-server health OK at %s", url)
                return True
            logger.warning(
                "llama-server health probe failed at %s: HTTP %s — first LLM call may fail",
                url,
                resp.status_code,
            )
            return False
        except Exception as exc:
            logger.warning(
                "llama-server health probe failed at %s: %s — first LLM call may fail",
                url,
                exc,
            )
            return False

    async def _warmup_llm_kv_cache(self) -> None:
        """Pre-warm llama-server's KV cache with the active system prompt.

        Fires a fire-and-forget ``/v1/chat/completions`` request using the
        current system prompt and a trivial user message so that the KV cache
        is populated before the first real user turn.  This eliminates the
        cold-prefill latency (~3-5 s for Qwen3 14B) that would otherwise land
        on the user during the opening exchange.

        Gated by ``REACHY_MINI_LLM_WARMUP_ENABLED`` (default: on).
        Automatically skipped when ``REACHY_MINI_LLAMA_HEALTH_CHECK=0`` because
        there is no point warming a server whose availability is unknown.
        Failure is best-effort: logs WARNING and continues.
        """
        if not getattr(config, "LLM_WARMUP_ENABLED", True):
            logger.debug("LLM KV-cache warmup disabled via REACHY_MINI_LLM_WARMUP_ENABLED=0")
            return
        if not getattr(config, "LLAMA_HEALTH_CHECK_ENABLED", True):
            logger.debug("LLM KV-cache warmup skipped because health check is disabled")
            return

        logger.info("llm warmup: begin (priming KV cache with system prompt)")
        _t0 = time.perf_counter()
        try:
            system_prompt = get_session_instructions()
            tool_specs = get_active_tool_specs(self.deps)
            chat_tools = [
                {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} for t in tool_specs
            ]
            payload: dict[str, object] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "ping"},
                ],
                "tools": chat_tools,
                "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
            }
            assert self._http is not None
            url = f"{self._llama_cpp_url}/v1/chat/completions"
            async with self._http.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                # Drain the stream so the server considers the request complete.
                async for _ in resp.aiter_bytes():
                    pass
        except Exception as exc:
            logger.warning("llm warmup: failed (continuing): %s", exc)
            return
        _dt = time.perf_counter() - _t0
        logger.info("llm warmup: complete in %.1f s", _dt)

    async def _warmup_tts(self) -> None:
        """Send a silent warmup request to force the voice model into memory.

        Chatterbox loads the voice model lazily on the first real TTS call,
        which causes a 20-40 s pause on CPU. Doing it here at startup makes
        the cold-start visible in logs and keeps the first user turn snappy.
        """
        logger.info("chatterbox: warming up voice model …")
        _t0 = time.perf_counter()
        try:
            await self._call_chatterbox_tts(".")
        except Exception as exc:
            logger.warning("chatterbox: warmup failed (continuing): %s", exc)
            return
        _dt = time.perf_counter() - _t0
        logger.info("chatterbox: warmup complete in %.1f s", _dt)

    # ------------------------------------------------------------------ #
    # Voice management                                                     #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        """Return predefined voice filenames from the Chatterbox server."""
        assert self._http is not None
        try:
            r = await self._http.get(f"{self._chatterbox_url}/get_predefined_voices")
            r.raise_for_status()
            return [v["filename"] for v in r.json()]
        except Exception as exc:
            logger.warning("Could not fetch Chatterbox voices: %s", exc)
            return [self._chatterbox_voice]

    def get_current_voice(self) -> str:
        """Return the Chatterbox voice reference file currently in use."""
        return self._chatterbox_voice

    async def change_voice(self, voice: str) -> str:
        """Switch to a different Chatterbox voice reference file."""
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
        """Translate response_text to TTS segments and enqueue PCM frames."""
        if not response_text:
            return
        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        from robot_comic import telemetry

        persona = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default"
        segments = translate(response_text, persona=persona, use_turbo=False)
        any_audio = False
        _first_chunk = True
        for seg in segments:
            if seg.silence_ms:
                for frame in self._pcm_to_frames(self._silence_pcm(seg.silence_ms)):
                    await self._enqueue_audio_frame(frame)
                any_audio = True
            else:
                text = f"{seg.turbo_insert} {seg.text}" if seg.turbo_insert else seg.text
                for sentence in _split_sentences(text):
                    pcm = await self._call_chatterbox_tts(
                        sentence, exaggeration=seg.exaggeration, cfg_weight=seg.cfg_weight
                    )
                    if pcm:
                        if _first_chunk and tts_start is not None:
                            telemetry.record_tts_first_audio(
                                time.perf_counter() - tts_start, {"gen_ai.system": "chatterbox"}
                            )
                            _first_chunk = False
                        for frame in self._pcm_to_frames(pcm):
                            from robot_comic.startup_timer import log_once

                            log_once("first TTS audio frame", logger)
                            telemetry.emit_first_greeting_audio_once()
                            await self._enqueue_audio_frame(frame)
                        any_audio = True
        if not any_audio:
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": "[TTS error]"}))

    async def _call_chatterbox_tts(
        self,
        text: str,
        *,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> bytes | None:
        """POST to Chatterbox /tts in clone mode; return raw WAV bytes."""
        assert self._http is not None
        voice = self._chatterbox_voice
        ref_file = voice if voice.endswith(".wav") else f"{voice}.wav"
        # Per-persona voice-clone ref overrides the generic predefined voice when present.
        clone_ref = getattr(self, "_voice_clone_ref_path", None)
        payload: dict[str, object] = {
            "text": text,
            "voice_mode": "clone",
            "output_format": "wav",
            "split_text": False,
            "exaggeration": exaggeration if exaggeration is not None else self._exaggeration,
            "cfg_weight": cfg_weight if cfg_weight is not None else self._cfg_weight,
            "temperature": self._temperature,
        }
        if clone_ref is not None:
            payload["audio_prompt_path"] = str(clone_ref)
        else:
            payload["reference_audio_filename"] = ref_file

        _call_start = time.perf_counter()
        for attempt in range(_TTS_MAX_RETRIES):
            try:
                r = await self._http.post(f"{self._chatterbox_url}/tts", json=payload)
                r.raise_for_status()
                result = self._wav_to_pcm(
                    r.content,
                    gain=self._gain,
                    auto_gain=self._auto_gain_enabled,
                    target_dbfs=self._target_dbfs,
                )
                _elapsed = time.perf_counter() - _call_start
                _slow_thresh = float(getattr(config, "REACHY_MINI_TTS_SLOW_WARN_S", 10.0))
                if _elapsed > _slow_thresh:
                    logger.warning(
                        "chatterbox: TTS call took %.1f s (threshold %.0f s) — voice model may have been evicted",
                        _elapsed,
                        _slow_thresh,
                    )
                return result
            except Exception as exc:
                logger.warning(
                    "TTS attempt %d/%d failed: %s: %s", attempt + 1, _TTS_MAX_RETRIES, type(exc).__name__, exc
                )
                if attempt < _TTS_MAX_RETRIES - 1:
                    await asyncio.sleep(_TTS_RETRY_DELAY)
        return None

    @staticmethod
    def _wav_to_pcm(
        wav_bytes: bytes,
        gain: float = 1.0,
        auto_gain: bool = False,
        target_dbfs: float = -16.0,
    ) -> bytes:
        """Strip WAV header, resample to 24 kHz mono int16 PCM, and apply gain.

        When *auto_gain* is True the manual *gain* multiplier is applied first,
        then ``normalize_gain`` scales the result to *target_dbfs* dBFS.
        """
        import io
        import wave

        from scipy.signal import resample

        with wave.open(io.BytesIO(wav_bytes)) as wf:
            src_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        if src_rate != _OUTPUT_SAMPLE_RATE:
            target_len = int(len(audio) * _OUTPUT_SAMPLE_RATE / src_rate)
            audio = resample(audio, target_len)
        if gain != 1.0:
            audio = audio * gain
        pcm = np.clip(audio, -32768, 32767).astype(np.int16)
        if auto_gain:
            pcm = normalize_gain(pcm, target_dbfs=target_dbfs)
        return pcm.tobytes()
