"""Local speech-to-text frontend for realtime response-audio backends."""

import os
import sys
import time
import asyncio
import logging
from typing import Any, Tuple
from pathlib import Path
from concurrent.futures import Future as ConcurrentFuture

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
from opentelemetry import trace
from opentelemetry import context as otel_context
from openai.types.realtime import (
    RealtimeAudioConfigParam,
    RealtimeAudioConfigOutputParam,
    RealtimeResponseCreateParamsParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM

from robot_comic import telemetry
from robot_comic.pause import TranscriptDisposition
from robot_comic.config import (
    HF_BACKEND,
    OPENAI_BACKEND,
    config,
    get_default_voice_for_provider,
)
from robot_comic.prompts import get_session_voice, get_session_instructions
from robot_comic.welcome_gate import GateState, WelcomeGate
from robot_comic.base_realtime import to_realtime_tools_config
from robot_comic.openai_realtime import OpenaiRealtimeHandler
from robot_comic.tools.core_tools import ToolDependencies, get_active_tool_specs
from robot_comic.huggingface_realtime import HuggingFaceRealtimeHandler
from robot_comic.tools.name_validation import record_user_transcript


logger = logging.getLogger(__name__)

LOCAL_STT_SAMPLE_RATE = 16000

# Operator-flipped env var. When set to "1" the module emits a verbose
# stream of diagnostic logs (listener wiring, add_audio shape/dtype, periodic
# state dumps, callback fires) intended for chassis-side disambiguation of
# the four hypotheses in issue #314. Zero impact when unset.
_DIAG_ENV_VAR = "MOONSHINE_DIAG"

# Cap how many add_audio frames we log in full detail to avoid drowning the
# journal once the operator has confirmed dtype/shape are stable.
_DIAG_ADD_AUDIO_LOG_LIMIT = 20

# Cadence of the periodic state-dump log during a stall.
_DIAG_PERIODIC_LOG_INTERVAL_S = 5.0


def _diag_enabled() -> bool:
    """Return True iff the operator has enabled MOONSHINE_DIAG=1.

    Read from the environment on every call so unit tests can toggle the flag
    between subtests via monkeypatch without restarting the process.
    """
    return os.environ.get(_DIAG_ENV_VAR) == "1"


class LocalSTTDependencyError(RuntimeError):
    """Raised when the optional local STT dependency is not installed."""


def resolve_ort_model_path(model_path: Path) -> tuple[Path, str]:
    """Return (resolved_path, format_str) preferring .ort over .onnx.

    If *model_path* is a directory, we look for any ``.ort`` file inside it
    first; if none is found we look for any ``.onnx`` file and return that
    (falling back to the original path so the caller can pass it straight to
    ``Transcriber``).  If *model_path* is a file, we just swap the suffix.

    The returned *format_str* is ``"ort"`` or ``"onnx"`` and is used only for
    the startup log line.
    """
    if model_path.is_dir():
        ort_files = sorted(model_path.glob("*.ort"))
        if ort_files:
            return ort_files[0], "ort"
        onnx_files = sorted(model_path.glob("*.onnx"))
        if onnx_files:
            return model_path, "onnx"
        return model_path, "onnx"

    ort_candidate = model_path.with_suffix(".ort")
    if ort_candidate.exists():
        return ort_candidate, "ort"
    return model_path, "onnx"


def prewarm_model_file(model_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Pull the model file into the OS page cache by reading it sequentially.

    This is a no-op on Windows where the page-cache concept does not apply in
    the same way and the overhead is not worthwhile.
    """
    if sys.platform == "win32":
        return
    resolved = model_path if model_path.is_file() else model_path
    # If model_path is a directory, prewarm the first .ort or .onnx file found.
    if resolved.is_dir():
        candidates = sorted(resolved.glob("*.ort")) or sorted(resolved.glob("*.onnx"))
        if not candidates:
            return
        resolved = candidates[0]
    if not resolved.is_file():
        return
    try:
        fd = os.open(str(resolved), os.O_RDONLY)
        try:
            while True:
                chunk = os.read(fd, chunk_size)
                if not chunk:
                    break
        finally:
            os.close(fd)
        logger.debug("Moonshine page-cache prewarm complete: %s", resolved)
    except OSError as exc:
        logger.debug("Moonshine prewarm skipped (%s): %s", resolved, exc)


def _resolve_moonshine_arch(moonshine_voice: Any, model_name: str) -> Any:
    """Return a Moonshine ModelArch value, falling back to the package default."""
    model_arch = getattr(moonshine_voice, "ModelArch", None)
    if model_arch is None:
        return None
    arch_name = model_name.upper()
    if not arch_name.endswith("_STREAMING"):
        arch_name = f"{arch_name}_STREAMING"
    return getattr(model_arch, arch_name, None)


class _MoonshineListener:  # base class is attached dynamically in _build_local_stt_stream()
    """Bridge Moonshine transcript callbacks into the asyncio handler loop."""

    def __init__(self, handler: "LocalSTTInputMixin") -> None:
        self.handler = handler

    def _diag_log_callback(self, kind: str, event: Any, text: str) -> None:
        """Emit a structured callback-fired log entry when MOONSHINE_DIAG=1.

        Issue #314 hypothesis 1: these are the events that AREN'T firing on
        the stall. If the operator sees any of these in the journal during
        the failing window, hypothesis 1 (listener wiring) is invalidated
        and the bug is upstream of the callback (VAD / shape / starvation).
        """
        if not _diag_enabled():
            return
        try:
            logger.info(
                "[MOONSHINE_DIAG] callback=%s listener_id=%s stream_id=%s event_type=%s text=%r",
                kind,
                id(self),
                id(getattr(self.handler, "_local_stt_stream", None)),
                type(event).__name__,
                text[:80],
            )
        except Exception as e:  # pragma: no cover - defensive only
            logger.debug("[MOONSHINE_DIAG] callback log failed: %s", e)

    def on_line_started(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._diag_log_callback("on_line_started", event, text)
        self.handler._heartbeat.update(
            {"state": "speech_started", "last_event": "started", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("started", text)

    def on_line_updated(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._diag_log_callback("on_line_updated", event, text)
        self.handler._heartbeat.update(
            {"state": "partial", "last_event": "partial", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("partial", text)

    def on_line_text_changed(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._diag_log_callback("on_line_text_changed", event, text)
        self.handler._heartbeat.update(
            {"state": "partial", "last_event": "partial", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("partial", text)

    def on_line_completed(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._diag_log_callback("on_line_completed", event, text)
        self.handler._heartbeat.update(
            {"state": "completed", "last_event": "completed", "last_text": text, "last_event_at": time.monotonic()}
        )
        self.handler._schedule_local_stt_event("completed", text)
        # Moonshine's streaming Stream keeps the completed line as its current
        # state forever — there's no `reset()` / `start_new_line()` / `clear()`
        # in the public API (only `start`/`stop`/`add_audio`/`update_transcription`
        # in `moonshine_voice/transcriber.py`). Without rearming, the next
        # utterance is fed into a stream that never emits another
        # `started`/`partial`/`completed` event, silently dropping the turn (#279).
        # Set a flag so the input pump rebuilds the stream before the next
        # frame is pushed (see LocalSTTInputMixin.receive).
        self.handler._pending_stream_rearm = True

    def on_error(self, event: Any) -> None:
        if _diag_enabled():
            logger.info(
                "[MOONSHINE_DIAG] callback=on_error listener_id=%s stream_id=%s event=%r",
                id(self),
                id(getattr(self.handler, "_local_stt_stream", None)),
                getattr(event, "error", event),
            )
        logger.warning("Local STT error: %s", getattr(event, "error", event))
        # Surface the error to the metrics surface so the monitor can show
        # a numeric signal alongside the warning log. Instrumentation audit
        # (PR #385) Rec 3.
        telemetry.inc_errors({"subsystem": "stt", "error_type": "stream_error"})
        # If the stream errored out we also need to rearm — the underlying
        # C handle may be in a state where no further events will fire.
        self.handler._pending_stream_rearm = True

    @staticmethod
    def _text_from_event(event: Any) -> str:
        line = getattr(event, "line", event)
        text = getattr(line, "text", "")
        return text if isinstance(text, str) else str(text)


class LocalSTTInputMixin:
    """Mixin that replaces upstream mic-audio transport with local Moonshine STT.

    Post-Phase-5e (2026-05-16): this mixin is **hybrid-only**. The five
    composable triples (Moonshine + llama/gemini-text/gemini-bundled ×
    ElevenLabs/Chatterbox/Gemini-TTS) migrated off the mixin via
    sub-phases 5e.2-5e.6 — they now construct a plain
    ``*ResponseHandler`` and a standalone
    :class:`~robot_comic.adapters.moonshine_stt_adapter.MoonshineSTTAdapter`,
    with the orchestrator-level concerns (turn-span, output-queue
    publishing, listening-state, pause-controller, welcome gate,
    name-validation transcript recording) living on
    :class:`~robot_comic.composable_pipeline.ComposablePipeline`.

    The two remaining production consumers are
    :class:`LocalSTTOpenAIRealtimeHandler` and
    :class:`LocalSTTHuggingFaceRealtimeHandler` — the Phase 4c-tris
    Option B "legacy forever" hybrids whose bundled LLM+TTS half lives
    inside the realtime websocket and doesn't decompose into the
    Protocol triple. New composable triples should use the standalone
    adapter + pipeline shape from
    ``handler_factory._build_composable_*`` instead of mixing this
    class in.
    """

    # Declared here so mypy sees them on the mixin; the concrete base class
    # provides the actual values via super().__init__().
    deps: ToolDependencies
    output_queue: asyncio.Queue[Any]

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize local STT state and the response backend."""
        super().__init__(deps, sim_mode, instance_path, startup_voice)  # type: ignore[call-arg]
        # Declare timing / span attrs so mypy sees consistent types on the mixin.
        # Concrete base classes (BaseLlamaResponseHandler, BaseRealtimeHandler)
        # initialise these in their own __init__ via super().__init__() above.
        self._turn_user_done_at: float | None
        self._turn_response_created_at: float | None
        self._turn_first_audio_at: float | None
        self._turn_id: str | None
        self._turn_start_at: float | None
        self.local_stt_sample_rate = LOCAL_STT_SAMPLE_RATE
        self._local_stt_stream: Any = None
        self._local_stt_transcriber: Any = None
        self._local_stt_listener: Any = None
        # Captured during _build_local_stt_stream() so _open_local_stt_stream()
        # can rebuild the stream after a completion without reloading the model.
        self._local_stt_update_interval: float = 0.0
        self._local_stt_listener_base_cls: Any = None
        # Set by _MoonshineListener.on_line_completed / on_error and consumed by
        # receive() to tear down and recreate the streaming stream. See #279.
        self._pending_stream_rearm: bool = False
        self._local_loop: asyncio.AbstractEventLoop | None = None
        self._last_completed_transcript: str = ""
        self._last_completed_at: float = 0.0
        self._heartbeat: dict[str, Any] = {
            "state": "idle",
            "last_event": None,
            "last_text": "",
            "last_event_at": time.monotonic(),
            "audio_frames": 0,
            # Dedup bookkeeping — see `_log_heartbeat`.
            "last_logged_state": None,
            "last_logged_text": None,
            "last_logged_frames": 0,
            "last_logged_at": 0.0,
            # First-audio-frame timestamp, used to suppress the "thread-lock or
            # model stall" warning during the post-startup warm-up window.
            "first_audio_at": None,
        }
        self._heartbeat_future: "asyncio.Future[Any] | ConcurrentFuture[None] | None" = None
        # MOONSHINE_DIAG=1 bookkeeping (issue #314). Tracks how many add_audio
        # frames we have logged in detail, the wallclock of the last add_audio
        # call (so the periodic dump can show elapsed-since-last-frame), and
        # the future for the periodic state-dump task. All unused when diag
        # is disabled.
        self._diag_add_audio_logged: int = 0
        self._diag_last_add_audio_at: float | None = None
        self._diag_periodic_future: "asyncio.Future[Any] | ConcurrentFuture[None] | None" = None
        self._welcome_gate: WelcomeGate | None = self._build_welcome_gate()

    def _build_welcome_gate(self) -> WelcomeGate | None:
        """Create a WelcomeGate for the current profile when the feature is enabled.

        Returns None when REACHY_MINI_WELCOME_GATE_ENABLED is false or no
        profile directory can be resolved.
        """
        if not getattr(config, "WELCOME_GATE_ENABLED", False):
            return None

        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        profiles_dir = getattr(config, "PROFILES_DIRECTORY", None)
        if not profile or profiles_dir is None:
            logger.info("welcome gate: enabled but no profile selected — gate inactive")
            return None

        from robot_comic.welcome_gate import make_gate_for_profile

        profile_dir = Path(profiles_dir) / profile
        gate = make_gate_for_profile(profile_dir)
        logger.info("welcome gate: active for profile %r", profile)
        return gate

    async def _prepare_startup_credentials(self) -> None:
        """Let the response backend prepare itself, then initialize local STT.

        Idempotency guard (#337 Phase 5 / boot memo PR #383 V7): the three
        composable adapters (LLM, TTS, STT) each call this method on the same
        shared host during their ``prepare``/``start`` lifecycle. The
        underlying work (httpx client, Gemini client, Moonshine model load —
        ~20 s on cold boot) is idempotent today, but the redundant work
        costs real wall-clock during the start_up critical path. The first
        successful call flips ``_startup_credentials_ready``; subsequent
        calls are cheap no-ops. If the first call raises, the flag stays
        unset so retries still re-attempt — preserves the "don't lock out
        retries on failure" semantics.
        """
        if getattr(self, "_startup_credentials_ready", False):
            return
        await super()._prepare_startup_credentials()  # type: ignore[misc]
        self._local_loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._build_local_stt_stream)
        self._startup_credentials_ready = True

    def _build_local_stt_stream(self) -> None:
        """Build and start the Moonshine streaming transcriber."""
        try:
            import moonshine_voice
            from moonshine_voice import Transcriber, TranscriptEventListener, get_model_for_language
        except Exception as e:  # pragma: no cover - exercised via unit tests without dependency
            raise LocalSTTDependencyError(
                "Local STT requires the optional dependency group: install with `uv sync --extra local_stt` "
                "or `pip install -e .[local_stt]`."
            ) from e

        language = (getattr(config, "LOCAL_STT_LANGUAGE", None) or "en").strip().lower()
        model_name = getattr(config, "LOCAL_STT_MODEL", "tiny_streaming")
        update_interval = float(getattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35))
        requested_arch = _resolve_moonshine_arch(moonshine_voice, model_name)
        cache_root = Path(getattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")).expanduser()

        try:
            model_path, model_arch = get_model_for_language(
                language,
                wanted_model_arch=requested_arch,
                cache_root=cache_root,
            )
        except TypeError:
            model_path, model_arch = get_model_for_language(language, cache_root=cache_root)
            model_arch = requested_arch or model_arch

        # Prefer .ort (ONNX Runtime FlatBuffer) when available — loads ~3-5x
        # faster than plain .onnx because it skips the graph parse step.
        model_path, model_format = resolve_ort_model_path(Path(model_path))

        # Pull the model file into the OS page cache before handing it to
        # ONNX Runtime.  No-op on Windows.
        prewarm_model_file(model_path)

        logger.info(
            "Starting local Moonshine STT: language=%s model=%s format=%s update_interval=%.2fs cache=%s path=%s",
            language,
            model_name,
            model_format,
            update_interval,
            cache_root,
            model_path,
        )

        # moonshine_voice.Transcriber expects the *directory* containing the
        # model file and its tokenizer.bin sibling, not the model file itself.
        transcriber_path = model_path.parent if model_path.is_file() else model_path
        transcriber = Transcriber(
            model_path=str(transcriber_path), model_arch=model_arch, update_interval=update_interval
        )
        self._local_stt_transcriber = transcriber
        self._local_stt_update_interval = update_interval
        self._local_stt_listener_base_cls = TranscriptEventListener
        self._open_local_stt_stream()

        if config.MOONSHINE_HEARTBEAT and self._local_loop is not None:
            self._heartbeat_future = asyncio.run_coroutine_threadsafe(
                self._moonshine_heartbeat_loop(), self._local_loop
            )

        # MOONSHINE_DIAG periodic state-dump task. Independent of the regular
        # heartbeat: fires every 5s regardless of the dedup logic, so the
        # operator can diff listener_id / stream_id / audio_frames across the
        # full stall window even when the regular heartbeat is suppressed.
        if _diag_enabled() and self._local_loop is not None:
            self._diag_periodic_future = asyncio.run_coroutine_threadsafe(
                self._moonshine_diag_periodic_loop(), self._local_loop
            )

    def _open_local_stt_stream(self) -> None:
        """Create + start a fresh Moonshine stream on the current transcriber.

        Called once from `_build_local_stt_stream` at startup and again from
        `_rearm_local_stt_stream` after each completed utterance. Assumes
        `_local_stt_transcriber`, `_local_stt_update_interval` and
        `_local_stt_listener_base_cls` are already populated.
        """
        transcriber = self._local_stt_transcriber
        base_cls = self._local_stt_listener_base_cls
        stream = transcriber.create_stream(update_interval=self._local_stt_update_interval)
        listener_cls = type(
            "RobotComicMoonshineListener",
            (_MoonshineListener, base_cls),
            {},
        )
        listener = listener_cls(self)
        stream.add_listener(listener)
        stream.start()
        self._local_stt_stream = stream
        self._local_stt_listener = listener
        self._pending_stream_rearm = False
        # Reset add_audio diag counter so a rearm gets its own fresh window of
        # detailed frame logs — useful for #314 hypothesis 1 (listener wiring
        # is fine on first boot but broken after rearm, or vice versa).
        self._diag_add_audio_logged = 0
        self._diag_last_add_audio_at = None
        if _diag_enabled():
            # Snapshot the stream's public callable surface — useful for
            # hypothesis 3 (sample-rate / shape mismatch) and to confirm the
            # transcriber actually returns a stream that supports add_audio /
            # add_listener after a rearm.
            try:
                stream_methods = sorted(
                    name for name in dir(stream) if not name.startswith("_") and callable(getattr(stream, name, None))
                )
            except Exception:  # pragma: no cover - defensive only
                stream_methods = []
            logger.info(
                "[MOONSHINE_DIAG] stream_opened transcriber_id=%s stream_id=%s "
                "stream_type=%s listener_id=%s listener_type=%s update_interval=%.3fs "
                "stream_methods=%s",
                id(transcriber),
                id(stream),
                type(stream).__name__,
                id(listener),
                type(listener).__name__,
                self._local_stt_update_interval,
                stream_methods,
            )

    def _rearm_local_stt_stream(self) -> None:
        """Tear down the current Moonshine stream and start a fresh one.

        Moonshine's streaming Stream has no public reset / clear / start-new-line
        method (see `moonshine_voice/transcriber.py`: only start, stop, add_audio,
        update_transcription, add/remove_listener, close). Once a line completes,
        subsequent `add_audio` calls never emit another `started`/`partial`/
        `completed` event — the conversation silently dies (#279). The only
        recovery is to free the old stream handle and create a new one on the
        same (expensive) transcriber.

        Runs synchronously on the caller's thread. The transcriber is preserved
        so the multi-hundred-millisecond ONNX model load is NOT repeated.
        """
        old_stream = self._local_stt_stream
        old_listener = self._local_stt_listener
        if _diag_enabled():
            logger.info(
                "[MOONSHINE_DIAG] rearm_begin old_stream_id=%s old_listener_id=%s pending_rearm=%s",
                id(old_stream),
                id(old_listener),
                self._pending_stream_rearm,
            )
        self._local_stt_stream = None
        self._local_stt_listener = None
        if old_stream is not None:
            for method_name in ("stop", "close"):
                method = getattr(old_stream, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as e:
                        logger.debug("Moonshine stream %s during rearm: %s", method_name, e)
        if self._local_stt_transcriber is None:
            # Shutdown raced with us; nothing to do.
            self._pending_stream_rearm = False
            return
        self._open_local_stt_stream()
        logger.debug("Moonshine stream rearmed after line completion")

    def _schedule_local_stt_event(self, kind: str, text: str) -> None:
        """Schedule a local STT event onto the handler event loop."""
        loop = self._local_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(self._handle_local_stt_event(kind, text)))

    async def _handle_local_stt_event(self, kind: str, text: str) -> None:
        """Handle local STT lifecycle events inside the asyncio loop."""
        import uuid as _uuid

        from fastrtc import AdditionalOutputs  # deferred: fastrtc pulls gradio at boot

        transcript = (text or "").strip()
        if kind == "started":
            self._mark_activity("local_stt_speech_started")  # type: ignore[attr-defined]
            self._turn_user_done_at = None
            self._turn_response_created_at = None
            self._turn_first_audio_at = None
            # Open root turn span and STT infer child span
            if hasattr(self, "_close_turn_span"):
                self._close_turn_span("interrupted")
            _now = time.perf_counter()
            _tracer = telemetry.get_tracer()
            _turn_id = str(_uuid.uuid4())
            if hasattr(self, "_turn_id"):
                self._turn_id = _turn_id
            if hasattr(self, "_turn_start_at"):
                self._turn_start_at = _now
            if hasattr(self, "_session_id"):
                _session_id = self._session_id
            else:
                _session_id = ""
            if hasattr(self, "_current_response_has_tool_call"):
                self._current_response_has_tool_call = False
            _turn_span = _tracer.start_span(
                "turn",
                attributes={
                    "turn.id": _turn_id,
                    "session.id": _session_id,
                    "robot.mode": "local_stt",
                    "robot.persona": telemetry.current_persona(),
                },
            )
            if hasattr(self, "_turn_span"):
                self._turn_span = _turn_span
            _ctx_token = otel_context.attach(trace.set_span_in_context(_turn_span))
            if hasattr(self, "_turn_ctx_token"):
                self._turn_ctx_token = _ctx_token
            _prior_stt = getattr(self, "_stt_infer_span", None)
            if _prior_stt is not None:
                _prior_stt.end()
            self._stt_infer_span: Any = _tracer.start_span("stt.infer")
            self._stt_infer_start: float = _now
            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            return

        if kind == "partial":
            if transcript:
                self._mark_activity("local_stt_partial")  # type: ignore[attr-defined]
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
            return

        if kind != "completed" or not transcript:
            return

        now = time.perf_counter()
        if transcript == self._last_completed_transcript and now - self._last_completed_at < 0.75:
            logger.debug("Ignoring duplicate local STT completion: %s", transcript)
            return
        self._last_completed_transcript = transcript
        self._last_completed_at = now

        self._mark_activity("local_stt_completed")  # type: ignore[attr-defined]
        self.deps.movement_manager.set_listening(False)
        words = transcript.split()
        excerpt = " ".join(words[:5]) + ("…" if len(words) > 5 else "")
        stt_span = getattr(self, "_stt_infer_span", None)
        if stt_span is not None:
            stt_s = now - getattr(self, "_stt_infer_start", now)
            # Tag the stt.infer span with the excerpt *before* closing it so the
            # monitor can read it from the completed child span while the outer
            # turn span is still open (pending row).
            stt_span.set_attribute("turn.excerpt", excerpt)
            stt_span.end()
            self._stt_infer_span = None
            telemetry.record_stt(stt_s, {"gen_ai.system": "local_stt", "stt.type": "moonshine"})
        # Tag the outer turn span with the same excerpt for the completed-row path.
        _outer_span = getattr(self, "_turn_span", None)
        if _outer_span is not None:
            _outer_span.set_attribute("turn.excerpt", excerpt)
        self._turn_user_done_at = now
        self._turn_response_created_at = None
        self._turn_first_audio_at = None

        # Record for tool-side name-validation guard (#287).
        record_user_transcript(self.deps.recent_user_transcripts, transcript)

        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        pause_controller = getattr(self.deps, "pause_controller", None)
        if pause_controller is not None:
            try:
                disposition = pause_controller.handle_transcript(transcript)
            except Exception as e:
                logger.error("pause_controller.handle_transcript raised: %s", e)
                disposition = TranscriptDisposition.DISPATCH
            if disposition is TranscriptDisposition.HANDLED:
                return

        # Echo guard: suppress transcripts that arrive while TTS is still playing.
        # Pause commands are intentionally exempted (handled above).
        speaking_until = getattr(self, "_speaking_until", 0.0)
        if time.perf_counter() < speaking_until:
            logger.info("Echo guard: discarding transcript during TTS playback: %r", transcript[:60])
            return

        # Welcome-gate: if enabled and still WAITING, only check for the wake-name.
        # Do NOT dispatch to the LLM until the gate opens.
        gate = self._welcome_gate
        if gate is not None and gate.state is GateState.WAITING:
            if not gate.consider(transcript):
                logger.debug("welcome gate: WAITING — transcript not dispatched: %r", transcript[:60])
                return
            # Gate just opened — fall through and dispatch this transcript normally.

        await self._dispatch_completed_transcript(transcript)

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """Send a completed transcript to the realtime response backend.

        Override in subclasses to redirect to a different response backend.
        """
        if not self.connection:  # type: ignore[attr-defined]
            logger.debug("Local STT transcript ready but realtime connection is not connected")
            return

        await self.connection.conversation.item.create(  # type: ignore[attr-defined]
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": transcript}],
            },
        )
        await self._safe_response_create(  # type: ignore[attr-defined]
            response=RealtimeResponseCreateParamsParam(
                instructions="Answer the user's transcribed speech naturally and concisely in audio.",
            ),
        )

    # Re-log the same (state, text) at INFO at most this often, to keep a
    # periodic liveness ping in the journal without drowning it.
    _HEARTBEAT_REPEAT_INFO_INTERVAL_S: float = 30.0
    # Suppress the "thread-lock or model stall" warning during the first
    # window after the audio source starts emitting frames (covers
    # resume-from-suspend warm-up).
    _STARTUP_STALL_GRACE_S: float = 10.0

    def _log_heartbeat(self) -> None:
        """Emit one heartbeat log line with current Moonshine state.

        Dedup: a heartbeat at the same (state, text) as the previous emit
        is demoted to DEBUG (a "pure" heartbeat — only `age`/`frames` are
        moving). To preserve liveness, an INFO line is still re-emitted at
        most once every ``_HEARTBEAT_REPEAT_INFO_INTERVAL_S``.

        Idle-stall warning: suppressed during the first
        ``_STARTUP_STALL_GRACE_S`` seconds after the first audio frame, to
        avoid false positives during post-suspend warm-up.
        """
        h = self._heartbeat
        now = time.monotonic()
        age = now - h["last_event_at"]
        text_snippet = (h["last_text"] or "")[:40]

        state = h["state"]
        same_signal = state == h["last_logged_state"] and text_snippet == h["last_logged_text"]
        since_logged = now - h["last_logged_at"]
        repeat_window = self._HEARTBEAT_REPEAT_INFO_INTERVAL_S

        if not same_signal or since_logged >= repeat_window or h["last_logged_state"] is None:
            level = logging.INFO
            h["last_logged_at"] = now
        else:
            level = logging.DEBUG

        logger.log(
            level,
            "[Moonshine] state=%s  last_event=%s  age=%.1fs  frames=%d  text=%r",
            state,
            h["last_event"],
            age,
            h["audio_frames"],
            text_snippet,
        )
        h["last_logged_state"] = state
        h["last_logged_text"] = text_snippet
        h["last_logged_frames"] = h["audio_frames"]

        if state == "idle" and age > 10.0 and h["audio_frames"] > 0:
            first_audio_at = h.get("first_audio_at")
            in_startup_grace = first_audio_at is not None and (now - first_audio_at) < self._STARTUP_STALL_GRACE_S
            if not in_startup_grace:
                logger.warning(
                    "[Moonshine] idle for %.1fs with %d audio frames received — possible thread-lock or model stall",
                    age,
                    h["audio_frames"],
                )
                # Surface the idle-stall to the metrics surface; this is the
                # very symptom MOONSHINE_DIAG exists to debug (#314) and the
                # monitor should not have to scrape logs to detect it.
                # Instrumentation audit (PR #385) Rec 3.
                telemetry.inc_errors({"subsystem": "stt", "error_type": "idle_stall"})

    async def _moonshine_heartbeat_loop(self) -> None:
        """Log Moonshine state every second while the stream is active."""
        while self._local_stt_stream is not None and config.MOONSHINE_HEARTBEAT:
            self._log_heartbeat()
            await asyncio.sleep(1.0)

    def _diag_log_periodic_state(self) -> None:
        """Emit one MOONSHINE_DIAG periodic state-dump line.

        Run from `_moonshine_diag_periodic_loop` every 5s. Captures the
        information needed to diff state across a stall window:

        - listener_id / stream_id — confirm wiring is stable (#314 hyp 1)
        - audio_frames — confirm receive() is still ingesting (#314 hyp 4)
        - elapsed_since_last_add_audio_s — confirm the input pump isn't stuck
        - pending_rearm — surface a stuck rearm flag
        - last_event / last_event_age — last callback time vs now
        """
        if not _diag_enabled():
            return
        h = self._heartbeat
        now = time.monotonic()
        last_audio = self._diag_last_add_audio_at
        elapsed_audio = (now - last_audio) if last_audio is not None else None
        logger.info(
            "[MOONSHINE_DIAG] periodic listener_id=%s stream_id=%s audio_frames=%d "
            "last_add_audio_elapsed_s=%s state=%s last_event=%s last_event_age_s=%.1f "
            "pending_rearm=%s",
            id(self._local_stt_listener),
            id(self._local_stt_stream),
            h["audio_frames"],
            f"{elapsed_audio:.2f}" if elapsed_audio is not None else "never",
            h["state"],
            h["last_event"],
            now - h["last_event_at"],
            self._pending_stream_rearm,
        )

    async def _moonshine_diag_periodic_loop(self) -> None:
        """Periodically dump diag state every _DIAG_PERIODIC_LOG_INTERVAL_S.

        Runs until the stream is torn down OR MOONSHINE_DIAG is cleared at
        runtime. Operator-only — gated by _diag_enabled().
        """
        while self._local_stt_stream is not None and _diag_enabled():
            try:
                self._diag_log_periodic_state()
            except Exception as e:  # pragma: no cover - defensive only
                logger.debug("[MOONSHINE_DIAG] periodic dump failed: %s", e)
            await asyncio.sleep(_DIAG_PERIODIC_LOG_INTERVAL_S)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Feed microphone audio into the local STT stream."""
        from fastrtc import audio_to_float32  # deferred: fastrtc pulls gradio at boot

        # Moonshine has no public reset; after each `LineCompleted` the stream
        # silently stops emitting events, so the listener flags a rearm here
        # (see #279). Rebuilding before the next frame keeps audio loss to one
        # FastRTC chunk and avoids freeing a handle from inside a callback.
        if self._pending_stream_rearm:
            try:
                await asyncio.to_thread(self._rearm_local_stt_stream)
            except Exception as e:
                logger.warning("Moonshine stream rearm failed: %s", e)
                # Clear the flag so we don't tight-loop on the same failure;
                # the next completion will retry.
                self._pending_stream_rearm = False

        if self._local_stt_stream is None:
            return

        # Skip pumping mic audio while the robot is speaking. Without this,
        # Moonshine's streaming VAD treats the robot's own TTS as one
        # continuous utterance and never emits `completed`, stalling the
        # next user turn. The post-transcript echo guard alone is too late.
        speaking_until = getattr(self, "_speaking_until", 0.0)
        if time.perf_counter() < speaking_until:
            return

        input_sample_rate, audio_frame = frame
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        audio_float = audio_to_float32(audio_frame)

        if self.local_stt_sample_rate != input_sample_rate:
            audio_float = resample(
                audio_float,
                int(len(audio_frame) * self.local_stt_sample_rate / input_sample_rate),
            ).astype(np.float32, copy=False)
        audio_payload = audio_float.tolist()
        # MOONSHINE_DIAG: log the first N add_audio calls in detail. This
        # confirms (or rules out) #314 hypothesis 3 (sample-rate / shape /
        # dtype mismatch silently dropped). We log BEFORE the call so an
        # exception inside add_audio still leaves a trace.
        if _diag_enabled() and self._diag_add_audio_logged < _DIAG_ADD_AUDIO_LOG_LIMIT:
            try:
                first_val: Any = audio_payload[0] if audio_payload else None
            except Exception:  # pragma: no cover - defensive only
                first_val = None
            logger.info(
                "[MOONSHINE_DIAG] add_audio call#%d stream_id=%s listener_id=%s "
                "input_sample_rate=%s input_ndim=%d input_dtype=%s input_len=%d "
                "resampled_len=%d resampled_dtype=%s payload_type=%s payload_len=%d "
                "first_sample=%r sample_rate_arg=%d",
                self._diag_add_audio_logged + 1,
                id(self._local_stt_stream),
                id(self._local_stt_listener),
                input_sample_rate,
                audio_frame.ndim,
                str(audio_frame.dtype),
                int(audio_frame.shape[0]) if audio_frame.ndim >= 1 else 0,
                len(audio_float),
                str(audio_float.dtype),
                type(audio_payload).__name__,
                len(audio_payload),
                first_val,
                self.local_stt_sample_rate,
            )
            self._diag_add_audio_logged += 1
        try:
            self._local_stt_stream.add_audio(audio_payload, self.local_stt_sample_rate)
            self._heartbeat["audio_frames"] += 1
            self._diag_last_add_audio_at = time.monotonic()
            if self._heartbeat["first_audio_at"] is None:
                self._heartbeat["first_audio_at"] = time.monotonic()
        except Exception as e:
            if _diag_enabled():
                logger.info(
                    "[MOONSHINE_DIAG] add_audio raised stream_id=%s err_type=%s err=%s",
                    id(self._local_stt_stream),
                    type(e).__name__,
                    e,
                )
            logger.debug("Dropping local STT audio frame: %s", e)

    async def shutdown(self) -> None:
        """Shutdown realtime and local STT resources."""
        if self._heartbeat_future is not None:
            self._heartbeat_future.cancel()
            self._heartbeat_future = None
        if self._diag_periodic_future is not None:
            self._diag_periodic_future.cancel()
            self._diag_periodic_future = None
        await super().shutdown()  # type: ignore[misc]
        stream = self._local_stt_stream
        transcriber = self._local_stt_transcriber
        self._local_stt_stream = None
        self._local_stt_transcriber = None
        self._local_stt_listener = None

        def _close_local() -> None:
            for obj, methods in (
                (stream, ("stop", "close")),
                (transcriber, ("close",)),
            ):
                if obj is None:
                    continue
                for method_name in methods:
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as e:
                            logger.debug("Local STT %s failed during shutdown: %s", method_name, e)

        await asyncio.to_thread(_close_local)


class LocalSTTOpenAIRealtimeHandler(LocalSTTInputMixin, OpenaiRealtimeHandler):
    """Use local Moonshine STT for input and OpenAI Realtime for responses."""

    PROVIDER_ID = OPENAI_BACKEND
    SAMPLE_RATE = 24000
    REFRESH_CLIENT_ON_RECONNECT = False
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = OpenaiRealtimeHandler.AUDIO_OUTPUT_COST_PER_1M
    TEXT_INPUT_COST_PER_1M = OpenaiRealtimeHandler.TEXT_INPUT_COST_PER_1M
    TEXT_OUTPUT_COST_PER_1M = OpenaiRealtimeHandler.TEXT_OUTPUT_COST_PER_1M
    IMAGE_INPUT_COST_PER_1M = OpenaiRealtimeHandler.IMAGE_INPUT_COST_PER_1M

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured OpenAI voice for the local-STT response backend."""
        return get_session_voice(default, backend="openai")

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Return a text-in/audio-out realtime config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=get_session_instructions(),
            audio=RealtimeAudioConfigParam(
                output=RealtimeAudioConfigOutputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    def get_current_voice(self) -> str:
        """Return the OpenAI voice selected for local-STT responses."""
        default_voice = get_default_voice_for_provider(OPENAI_BACKEND)
        return self._voice_override or self._get_session_voice(default=default_voice)


class LocalSTTHuggingFaceRealtimeHandler(LocalSTTInputMixin, HuggingFaceRealtimeHandler):
    """Use local Moonshine STT for input and Hugging Face realtime for responses."""

    PROVIDER_ID = HF_BACKEND
    SAMPLE_RATE = 16000
    REFRESH_CLIENT_ON_RECONNECT = True
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = 0.0
    TEXT_INPUT_COST_PER_1M = 0.0
    TEXT_OUTPUT_COST_PER_1M = 0.0
    IMAGE_INPUT_COST_PER_1M = 0.0


# Backward-compatible name for tests/imports; OpenAI remains the default response backend.
LocalSTTRealtimeHandler = LocalSTTOpenAIRealtimeHandler
