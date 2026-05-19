"""OpenTelemetry instrumentation for robot_comic.

Gated behind ROBOT_INSTRUMENTATION env var:
  unset / empty  — no-op (zero overhead)
  trace          — Console span exporter only (local dev / debugging)
  remote         — OTLP + Console span exporters (SigNoz or compatible collector)

Metric console output (ROBOT_METRICS_CONSOLE):
  unset / empty  — metrics are NOT written to console (prevents journald spam)
  1 / true / yes — enable ConsoleMetricExporter for developer debug sessions

All public names are safe to call regardless of whether instrumentation is
enabled — callers never need to check ENABLED themselves.
"""

import os
import json
import logging
import threading
from typing import Any, Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
_RAW = os.getenv("ROBOT_INSTRUMENTATION", "").strip().lower()
ENABLED: bool = _RAW in {"trace", "remote"}
_REMOTE: bool = _RAW == "remote"

# ---------------------------------------------------------------------------
# OTel SDK initialisation
# ---------------------------------------------------------------------------
_SERVICE = "robot_comic"
_TURN_DURATION_BUCKETS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 10.0]
_DEFAULT_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


_SPAN_ATTRS_TO_KEEP = frozenset(
    {
        "turn.id",
        "session.id",
        "turn.outcome",
        "turn.excerpt",
        "robot.mode",
        "robot.persona",
        "gen_ai.system",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        # GenAI semconv attrs set on spans but previously dropped at export.
        # ``gen_ai.usage.api_call_count`` is set per turn in
        # ``elevenlabs_tts.py:901`` (counts LLM retries inside a single turn).
        # ``gen_ai.server.time_to_first_token`` is set on ``llm.request`` in
        # ``base_realtime.py:1028`` and ``gemini_live.py:1010``; the histogram
        # side already exports it, but operators reading raw spans want the
        # attribute too. Instrumentation audit (PR #385) §3 / §4.
        "gen_ai.usage.api_call_count",
        "gen_ai.server.time_to_first_token",
        "tool.name",
        "tool.id",
        # TTS attribution set on ``tts.synthesize`` in ``elevenlabs_tts.py:973-975``
        # but previously dropped at export. Instrumentation audit §4 "orphan
        # emissions".
        "tts.voice_id",
        "tts.char_count",
        "vad.duration_ms",
        "stt.type",
        # Supporting-event spans surfaced to the monitor's parallel lane.
        # ``event.kind=supporting`` routes the row; ``event.dur_ms`` carries
        # an externally-measured duration that may differ from the span's
        # own wall-clock (boot-timeline synthetic point events, #301/#321).
        # ``from_persona`` / ``to_persona`` / ``outcome`` carry the
        # persona.switch swap triple (#303/#330). ``aplay.exit_code`` /
        # ``aplay.command`` carry welcome-WAV playback exit detail (#324).
        "event.kind",
        "event.dur_ms",
        "from_persona",
        "to_persona",
        "outcome",
        "aplay.exit_code",
        "aplay.command",
    }
)


class CompactLineExporter(SpanExporter):
    """Writes one RCSPAN JSON line per span to stdout for live monitoring."""

    def export(self, spans: Any) -> SpanExportResult:
        """Serialize each span to a compact JSON line prefixed with RCSPAN."""
        for span in spans:
            dur_ms = (span.end_time - span.start_time) / 1_000_000
            attrs = {k: v for k, v in (span.attributes or {}).items() if k in _SPAN_ATTRS_TO_KEEP}
            line = {
                "name": span.name,
                "trace": format(span.context.trace_id, "032x"),
                "span": format(span.context.span_id, "016x"),
                "parent": format(span.parent.span_id, "016x") if span.parent else None,
                "dur_ms": round(dur_ms, 1),
                "status": span.status.status_code.name,
                "ts": span.end_time // 1_000_000,
                "attrs": attrs,
            }
            print(f"RCSPAN {json.dumps(line, separators=(',', ':'))}", flush=True)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op shutdown."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush — output is written synchronously."""
        return True


def _init_otel() -> None:
    """Set up TracerProvider and MeterProvider.  Called once at app startup."""
    resource = Resource.create({SERVICE_NAME: _SERVICE})

    # --- Tracing ---
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(CompactLineExporter()))

    if _REMOTE:
        try:
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_trace = OTLPSpanExporter()  # reads OTEL_EXPORTER_OTLP_ENDPOINT from env
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace))
            logger.info("OTel: OTLP trace exporter enabled")
        except ImportError:
            logger.warning("OTel: opentelemetry-exporter-otlp-proto-grpc not installed; skipping OTLP traces")

    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ---
    # ConsoleMetricExporter floods journald with multi-hundred-line JSON every 60 s.
    # Gate it behind ROBOT_METRICS_CONSOLE so it is opt-in for developer debug only.
    _metrics_console = os.getenv("ROBOT_METRICS_CONSOLE", "").strip().lower() in {"1", "true", "yes"}
    readers: list[Any] = []
    if _metrics_console:
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60_000))

    if _REMOTE:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=10_000))
            logger.info("OTel: OTLP metric exporter enabled")
        except ImportError:
            logger.warning("OTel: opentelemetry-exporter-otlp-proto-grpc not installed; skipping OTLP metrics")

    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(meter_provider)

    logger.info("OTel: instrumentation initialised (mode=%s)", _RAW)


def init() -> None:
    """Initialise OTel if ROBOT_INSTRUMENTATION is set.  Safe to call multiple times.

    Also drains any supporting events that were queued before init() was called
    (issue #337): the early-welcome path and its completion daemon thread
    ``emit_supporting_event`` *before* ``run()`` reaches this function, and
    without the drain those events would resolve to OTel's no-op default tracer
    and be silently dropped.
    """
    # Re-read at call time so .env loaded after module import takes effect.
    global _RAW, ENABLED, _REMOTE, _initialized
    _RAW = os.getenv("ROBOT_INSTRUMENTATION", "").strip().lower()
    ENABLED = _RAW in {"trace", "remote"}
    _REMOTE = _RAW == "remote"
    if ENABLED:
        _init_otel()
        _init_instruments()
    _initialized = True
    # Flush pre-init supporting events in original order.
    with _pending_lock:
        pending = list(_pending_supporting)
        _pending_supporting.clear()
    for name, dur_ms, extra_attrs in pending:
        _emit_supporting_event_now(name, dur_ms, extra_attrs or None)


# ---------------------------------------------------------------------------
# Tracer / meter accessors
# ---------------------------------------------------------------------------


def get_tracer() -> trace.Tracer:
    """Return the robot_comic tracer (no-op when not initialised)."""
    return trace.get_tracer(_SERVICE)


def get_meter() -> metrics.Meter:
    """Return the robot_comic meter (no-op when not initialised)."""
    return metrics.get_meter(_SERVICE)


# ---------------------------------------------------------------------------
# Metric instruments  (module-level singletons, created after init_otel())
# ---------------------------------------------------------------------------
_meter: Optional[metrics.Meter] = None

# Histograms
turn_duration: Optional[metrics.Histogram] = None
llm_operation_duration: Optional[metrics.Histogram] = None
ttft: Optional[metrics.Histogram] = None
stt_duration: Optional[metrics.Histogram] = None
tts_duration: Optional[metrics.Histogram] = None
tts_first_audio: Optional[metrics.Histogram] = None

# Counters
frame_drops: Optional[metrics.Counter] = None
playback_underruns: Optional[metrics.Counter] = None
errors: Optional[metrics.Counter] = None
empty_stop_fallback: Optional[metrics.Counter] = None


def _init_instruments() -> None:
    global _meter, turn_duration, llm_operation_duration, ttft
    global stt_duration, tts_duration, tts_first_audio
    global frame_drops, playback_underruns, errors, empty_stop_fallback

    _meter = get_meter()

    turn_duration = _meter.create_histogram(
        "robot.turn.duration",
        unit="s",
        description="End-to-end conversational turn duration",
    )
    llm_operation_duration = _meter.create_histogram(
        "gen_ai.client.operation.duration",
        unit="s",
        description="LLM request duration (OTel GenAI semconv)",
    )
    ttft = _meter.create_histogram(
        "gen_ai.server.time_to_first_token",
        unit="s",
        description="Time from LLM request sent to first token received",
    )
    stt_duration = _meter.create_histogram(
        "robot.stt.duration",
        unit="s",
        description="Speech-to-text transcription duration",
    )
    tts_duration = _meter.create_histogram(
        "robot.tts.duration",
        unit="s",
        description="Text-to-speech synthesis duration",
    )
    tts_first_audio = _meter.create_histogram(
        "robot.tts.time_to_first_audio",
        unit="s",
        description="Time from TTS request to first audio byte",
    )
    frame_drops = _meter.create_counter(
        "robot.audio.capture.frame_drops",
        unit="frames",
        description="Audio capture frames dropped due to consumer lag",
    )
    playback_underruns = _meter.create_counter(
        "robot.audio.playback.underruns",
        unit="events",
        description="Audio playback buffer underruns",
    )
    errors = _meter.create_counter(
        "robot.errors",
        unit="events",
        description="Application errors",
    )
    empty_stop_fallback = _meter.create_counter(
        "composable.empty_stop_fallback",
        unit="events",
        description=(
            "Composable pipeline canned-line fallbacks spoken in place of an empty assistant turn (issue #267)."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers for recording metrics safely (no-op when instrument is None)
# ---------------------------------------------------------------------------


def current_persona() -> str:
    """Return the active persona name for span/metric attribution (#303).

    Resolves ``config.REACHY_MINI_CUSTOM_PROFILE`` and falls back to
    ``"default"`` so the attribute is always a non-empty string, matching
    the convention used for ``robot.mode``. Safe to call from any thread —
    no IO is performed.
    """
    try:
        from robot_comic.config import config as _config

        return str(getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None) or "default")
    except Exception:
        return "default"


def record_turn(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.turn.duration histogram."""
    if turn_duration is not None:
        turn_duration.record(duration_s, attributes=attrs)


def record_llm_duration(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record gen_ai.client.operation.duration histogram."""
    if llm_operation_duration is not None:
        llm_operation_duration.record(duration_s, attributes=attrs)


def record_ttft(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record gen_ai.server.time_to_first_token histogram."""
    if ttft is not None:
        ttft.record(duration_s, attributes=attrs)


def record_stt(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.stt.duration histogram."""
    if stt_duration is not None:
        stt_duration.record(duration_s, attributes=attrs)


def record_tts(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.tts.duration histogram."""
    if tts_duration is not None:
        tts_duration.record(duration_s, attributes=attrs)


def record_tts_first_audio(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.tts.time_to_first_audio histogram."""
    if tts_first_audio is not None:
        tts_first_audio.record(duration_s, attributes=attrs)


def inc_frame_drops(count: int, attrs: dict[str, Any]) -> None:
    """Increment robot.audio.capture.frame_drops counter."""
    if frame_drops is not None and count > 0:
        frame_drops.add(count, attributes=attrs)


def inc_playback_underruns(attrs: dict[str, Any]) -> None:
    """Increment robot.audio.playback.underruns counter."""
    if playback_underruns is not None:
        playback_underruns.add(1, attributes=attrs)


def inc_errors(attrs: dict[str, Any]) -> None:
    """Increment robot.errors counter."""
    if errors is not None:
        errors.add(1, attributes=attrs)


def inc_empty_stop_fallback(attrs: dict[str, Any]) -> None:
    """Increment composable.empty_stop_fallback counter (issue #267)."""
    if empty_stop_fallback is not None:
        empty_stop_fallback.add(1, attributes=attrs)


_FIRST_GREETING_EMITTED: bool = False


def emit_first_greeting_audio_once() -> None:
    """Emit ``first_greeting.tts_first_audio`` exactly once per process (#301).

    Closes the boot-timeline by recording when the first PCM frame of the
    synthetic startup turn is enqueued. ``event.dur_ms`` carries seconds
    since ``startup_timer.STARTUP_T0`` so the operator can read the full
    boot-to-first-audible-word window directly off the supporting-event row.
    """
    global _FIRST_GREETING_EMITTED
    if _FIRST_GREETING_EMITTED:
        return
    _FIRST_GREETING_EMITTED = True
    try:
        from robot_comic.startup_timer import since_startup

        dur_ms = since_startup() * 1000
    except Exception:
        dur_ms = None
    try:
        emit_supporting_event("first_greeting.tts_first_audio", dur_ms=dur_ms)
    except Exception:
        pass


# Supporting-event queue for pre-init emissions (issue #337). The early-welcome
# path dispatches the WAV before any non-stdlib import, and the WAV-completion
# daemon thread can fire its emit before ``run()`` reaches ``telemetry.init()``.
# Without buffering, those calls resolve against OTel's no-op default tracer
# and the spans are silently dropped — so the monitor's boot-timeline lane is
# empty for everything but ``app.startup`` (which emits *after* init).
_initialized: bool = False
_pending_supporting: list[tuple[str, Optional[float], dict[str, Any]]] = []
_pending_lock = threading.Lock()


def emit_supporting_event(
    name: str,
    dur_ms: Optional[float] = None,
    *,
    extra_attrs: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a synthetic boot-timeline span (``event.kind=supporting``).

    Opens and immediately closes a span carrying ``event.kind=supporting`` so
    the monitor TUI can place it on its boot-timeline lane (issue #301).

    When *dur_ms* is supplied, it is attached as ``event.dur_ms`` so the
    operator-visible duration reflects an externally measured window rather
    than the span's own (effectively zero) wall-clock. The exporter still
    emits the span's own ``dur_ms`` at the top level; the monitor prefers
    ``attrs["event.dur_ms"]`` when present.

    ``extra_attrs`` is merged in for callers that need to attach event-specific
    metadata (e.g. ``aplay.exit_code`` on ``welcome.wav.completed``, #324). Keys
    that aren't in ``_SPAN_ATTRS_TO_KEEP`` are dropped silently by the
    exporter — keep that allowlist in sync when adding new attribute names.

    Safe to call whether or not instrumentation is enabled. Calls made before
    :func:`init` runs are buffered in ``_pending_supporting`` and flushed when
    init completes (issue #337).
    """
    if not _initialized:
        with _pending_lock:
            _pending_supporting.append((name, dur_ms, dict(extra_attrs or {})))
        return
    _emit_supporting_event_now(name, dur_ms, extra_attrs)


def _emit_supporting_event_now(
    name: str,
    dur_ms: Optional[float],
    extra_attrs: Optional[dict[str, Any]],
) -> None:
    """Emit a supporting span immediately. Internal helper, no queueing."""
    tracer = get_tracer()
    attrs: dict[str, Any] = {"event.kind": "supporting"}
    if dur_ms is not None:
        attrs["event.dur_ms"] = float(dur_ms)
    if extra_attrs:
        attrs.update(extra_attrs)
    with tracer.start_as_current_span(name, attributes=attrs):
        pass
