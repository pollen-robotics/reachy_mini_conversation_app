"""Composable conversation pipeline — Phase 2 of the pipeline refactor.

Orchestrates a 3-phase pipeline (STT → LLM → TTS) by composing backend
objects that satisfy the Protocols defined in :mod:`robot_comic.backends`
(Phase 1). This sits alongside the existing ``ConversationHandler`` ABC in
:mod:`robot_comic.conversation_handler`; it does NOT replace it yet.

The existing ABC's concrete subclasses (``BaseLlamaResponseHandler``,
``GeminiTextResponseHandler``, ``ElevenLabsTTSResponseHandler``, etc.)
express their 3-phase pipeline via inheritance — every ``(STT, LLM, TTS)``
triple is a distinct class with the LLM dial baked into the class
hierarchy. After Phase 3, swapping the LLM means swapping one object, not
introducing a new subclass.

## Phase 2 scope (this PR)

Net-new code only. Operators don't yet use ``ComposablePipeline`` — the
factory still wires the legacy class hierarchy. Phase 2.5 / Phase 3 will:

1. Write adapter classes that expose existing handler internals
   (``_run_llm_with_tools`` / ``_stream_tts_to_queue`` / ``LocalSTTInputMixin``)
   as the Protocols defined in :mod:`robot_comic.backends`.
2. Update ``handler_factory.py``'s ``composable`` branch to instantiate
   adapter-wrapped backends and inject them into a ``ComposablePipeline``.
3. Retire the legacy class hierarchy where this orchestrator covers it.

## Loop shape

For each completed user transcript:

1. Append ``{"role": "user", "content": transcript}`` to the conversation
   history.
2. Call ``llm.chat(history, tools)`` to get an :class:`LLMResponse`.
3. If the response has ``tool_calls``: dispatch each via the operator-
   supplied ``tool_dispatcher`` callback, append each result as a
   ``{"role": "tool", ...}`` history entry, and loop back to step 2.
   A per-turn ``max_tool_rounds`` cap prevents storms.
4. If the response has ``text``: append it as ``{"role": "assistant"}``
   and stream-synthesize it via ``tts.synthesize(text, tags)``, pushing
   each :class:`AudioFrame` into the operator-supplied output queue.

Tools are kept outside the LLM Protocol on purpose — the orchestrator
owns the dispatcher because tool implementations need access to
``ToolDependencies`` (robot, movement manager, camera worker, etc.),
which the LLM backend shouldn't see.

## Bundled-live backends

OpenAI Realtime / Gemini Live / HF Realtime fuse all three phases into
one session and do NOT compose into this orchestrator. The factory's
``PIPELINE_MODE`` dial (Phase 0) keeps them on their own path; this
class is only for composable-mode pipelines.
"""

from __future__ import annotations
import os
import time
import uuid
import asyncio
import difflib
import logging
import collections
from typing import TYPE_CHECKING, Any, Callable, Awaitable
from difflib import SequenceMatcher

import httpx
from opentelemetry import trace
from opentelemetry import context as otel_context

from robot_comic import telemetry
from robot_comic.pause import TranscriptDisposition
from robot_comic.config import config, set_custom_profile
from robot_comic.prompts import get_session_instructions
from robot_comic.backends import (
    ToolCall,
    AudioFrame,
    LLMBackend,
    STTBackend,
    TTSBackend,
    LLMResponse,
)
from robot_comic.echo_filter import normalize_for_echo_check
from robot_comic.history_trim import trim_history_in_place
from robot_comic.joke_history import record_joke_history
from robot_comic.welcome_gate import GateState, WelcomeGate
from robot_comic.empty_stop_fallbacks import (
    load_pool as _load_empty_stop_pool,
)
from robot_comic.empty_stop_fallbacks import (
    pick_fallback as _pick_empty_stop_fallback,
)
from robot_comic.empty_stop_fallbacks import (
    resolve_profile_dir as _resolve_empty_stop_profile_dir,
)
from robot_comic.tools.name_validation import record_user_transcript


if TYPE_CHECKING:
    from robot_comic.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


# Tool dispatcher callback. Receives a ToolCall (id + name + args) and returns
# the tool's result as a string that the orchestrator feeds back to the LLM
# as a ``{"role": "tool", "tool_call_id": ..., "content": ...}`` history entry.
ToolDispatcher = Callable[[ToolCall], Awaitable[str]]


# Default per-turn cap on consecutive tool round-trips. Matches the existing
# ``MAX_API_CALLS_PER_TURN`` in elevenlabs_tts.py (8) — Gemini 503 retries
# can otherwise burn the budget. See #286 / PR #322.
DEFAULT_MAX_TOOL_ROUNDS = 8

# Per-turn cap on calls to the *same* tool name (issue #430). Once a tool
# has been dispatched more than this many times in a turn, the orchestrator
# splices a non-persistent system nudge into the next ``llm.chat`` call
# telling the model to stop calling tools and respond in text. The nudge
# fires exactly once per turn; if the LLM still loops, ``max_tool_rounds``
# remains the hard backstop. PR #418 fixed the ``greet`` retry-trap on the
# tool side; this is the orchestrator-side safety net that covers every
# other tool without per-tool ``retry_hint`` plumbing.
MAX_SAME_TOOL_REPEATS = 2


# Phase 5f.2 — minimum-utterance filter (self-echo cascade defence).
#
# Drops transcripts that look like speaker-echo / whisper-family
# hallucinations BEFORE they reach duplicate-suppression or the LLM.
# faster-whisper's webrtcvad-based segment dispatch is aggressive enough
# that room-acoustic reverb tail occasionally slips past the input-side
# echo guard; this orchestrator-side filter is the second line of defence
# and also helps Moonshine (which can occasionally hallucinate short
# partials too — see #19).
#
# Thresholds are heuristic; on-device tuning is a one-line change here.
# Rationale + the hardware finding that motivated the constants are in
# docs/superpowers/specs/2026-05-16-phase-5f-2-echo-cascade-fix.md.
MIN_TRANSCRIPT_WORDS = 2
MIN_TRANSCRIPT_CHARS = 8


# Phase 5f.3 — content-similarity echo filter.
#
# After 5f.2 closed the short-fragment arm of the cascade, hardware
# testing on 2026-05-16 showed longer multi-sentence echoes still
# cascading (faster-whisper transcribes the assistant's own speech back
# off the chassis speaker with only minor word errors; the transcript
# clears the 5f.2 length floor and dispatches as a fresh user turn).
#
# This filter compares each completed transcript against the last N
# assistant utterances; if any similarity ratio exceeds the threshold
# the transcript is dropped. Rationale + the hardware finding behind
# the constants live in
# ``docs/superpowers/specs/2026-05-16-phase-5f-3-echo-content-similarity-filter.md``.
ECHO_HISTORY_MAXLEN = 5
ECHO_SIMILARITY_THRESHOLD = 0.65


# TTS-down cascade suppression (this PR).
#
# When the TTS service is unreachable, every ambient-noise VAD trigger fires
# a full STT → LLM → TTS round-trip that ultimately times out — burning
# CPU and keeping the cascade alive. Hardware finding 2026-05-18: xtts at
# astralplane.lan:8020 was firewall-blocked from ricci; 43 interrupted turns
# in 32 minutes, all hallucinating on ambient mic input, none producing audio.
#
# Fix: after a TTS call raises a "service down" error class
# (``httpx.ConnectTimeout``, ``httpx.ConnectError``, ``httpx.ReadTimeout``,
# ``OSError``), suppress new STT-triggered turns for
# ``REACHY_MINI_TTS_DOWN_COOLDOWN_S`` seconds. The first successful TTS call
# clears the cooldown, so recovery is automatic with no reconnect probe.
#
# The gate sits at ``_on_transcript_completed`` entry — before LLM, before
# STT telemetry — so suppressed turns cost nothing except one log line.
# The warning fires ONCE when cooldown is entered; subsequent drops are
# silent (DEBUG) so the journal doesn't flood during an outage.
_TTS_DOWN_COOLDOWN_ENV = "REACHY_MINI_TTS_DOWN_COOLDOWN_S"
_TTS_DOWN_COOLDOWN_DEFAULT = 45.0
_TTS_DOWN_COOLDOWN_MIN = 5.0
_TTS_DOWN_COOLDOWN_MAX = 600.0

# Exception types that indicate the TTS *service* is unreachable (vs a
# transient inference error that may self-heal within the same call).
# ``OSError`` covers socket-level failures (e.g. no route to host, ECONNREFUSED)
# that httpx can raise before building an HTTP response.
_TTS_DOWN_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.ReadTimeout,
    OSError,
)


def _tts_down_cooldown_s() -> float:
    """Read and clamp ``REACHY_MINI_TTS_DOWN_COOLDOWN_S``.

    Reads the env var once at call time (not at import time) so tests can
    override it via ``monkeypatch.setenv`` without module-reload gymnastics.
    Out-of-range values clamp with a warning so a misconfigured deployment
    doesn't silently break.
    """
    raw = os.environ.get(_TTS_DOWN_COOLDOWN_ENV, "")
    if not raw:
        return _TTS_DOWN_COOLDOWN_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a valid float; using default %.1f s",
            _TTS_DOWN_COOLDOWN_ENV,
            raw,
            _TTS_DOWN_COOLDOWN_DEFAULT,
        )
        return _TTS_DOWN_COOLDOWN_DEFAULT
    if value < _TTS_DOWN_COOLDOWN_MIN:
        logger.warning(
            "%s=%.1f is below minimum %.1f s; clamping",
            _TTS_DOWN_COOLDOWN_ENV,
            value,
            _TTS_DOWN_COOLDOWN_MIN,
        )
        return _TTS_DOWN_COOLDOWN_MIN
    if value > _TTS_DOWN_COOLDOWN_MAX:
        logger.warning(
            "%s=%.1f exceeds maximum %.1f s; clamping",
            _TTS_DOWN_COOLDOWN_ENV,
            value,
            _TTS_DOWN_COOLDOWN_MAX,
        )
        return _TTS_DOWN_COOLDOWN_MAX
    return value


# The normalize step moved to ``echo_filter`` (#527) so the Gemini Live
# input-transcription path can run the same comparison; the alias keeps this
# module's call sites and tests unchanged.
_normalize_for_echo_check = normalize_for_echo_check


# Duplicate-suppression window + similarity threshold (post-5f.2).
#
# The legacy mixin (``LocalSTTInputMixin._handle_local_stt_event``,
# ``local_stt_realtime.py:596``) used a 0.75s window with strict
# text-equality dedup. That misses Moonshine's "hypothesis-burst" pattern
# where one user utterance fires multiple ``on_line_completed`` events
# with slightly-different final hypotheses ("Hey Rickles" / "Hey Rickles."
# / "Hey Rickles right?"). Each variant is its own asyncio task (see
# ``adapters/moonshine_listener.py:321`` — every event gets its own
# ``create_task``); each passes the text-equality check; each runs a
# full LLM round-trip and TTS span. Caught once during 2026-05-16
# hardware validation at 16:58 — three ``tts.synthesize`` spans landed
# within ~1s with response char_counts 96/96/137 (the two 96s being the
# LLM's deterministic response to two near-identical hypotheses).
#
# Fix (Phase 5 follow-up): widen the window to 2.0s and use
# stdlib :class:`difflib.SequenceMatcher` similarity (>= 0.85) instead
# of text equality. Window widening covers the case where the LLM
# round-trip is slow enough that variants land >0.75s apart.
# Similarity catches the variants without false-positive-ing legitimate
# distinct user lines (which differ by far more than 15%).
#
# Trade-off: a legitimate "Hey Rickles!" repeated emphatically within 2s
# is suppressed. Acceptable — the comedian persona doesn't need
# sub-2s-cadence-repeat fidelity, and an open variant-burst is much
# worse (motors jitter from spurious tool calls, audio overlaps).
#
# Operators can revert behaviour with one constant: set
# ``DEDUP_SIMILARITY_THRESHOLD = 1.0`` for legacy text-equality,
# ``DEDUP_WINDOW_S = 0.75`` for the legacy window.
#
# Rationale + the 16:58 hardware finding are documented in
# ``docs/superpowers/specs/2026-05-16-phase-5-duplicate-tts-storm-investigation.md``.
DEDUP_WINDOW_S = 2.0
DEDUP_SIMILARITY_THRESHOLD = 0.85


def _is_near_duplicate(a: str, b: str) -> bool:
    """Return True iff *a* and *b* look like edit-variants of the same utterance.

    Used by :meth:`ComposablePipeline._on_transcript_completed` to widen
    the dedup beyond strict text equality so Moonshine's
    hypothesis-burst pattern is captured.

    Implementation notes:

    - Exact-match short-circuit covers the common case (and the legacy
      path) without paying for the SequenceMatcher init.
    - Length pre-filter: ``SequenceMatcher.ratio()`` is upper-bounded by
      ``2 * matches / (la + lb)`` and ``matches <= min(la, lb)``, so the
      ratio is bounded by ``2 * min(la, lb) / (la + lb)``. If that
      upper bound is below the threshold, no string content can push
      the actual ratio over the bar — skip the O(n*m) ratio call.
    - Empty-string guard returns False (an empty new transcript already
      short-circuits at the caller before reaching this function; this
      is defensive).
    """
    if a == b:
        return True
    if not a or not b:
        return False
    la, lb = len(a), len(b)
    max_possible_ratio = 2 * min(la, lb) / (la + lb)
    if max_possible_ratio < DEDUP_SIMILARITY_THRESHOLD:
        return False
    return SequenceMatcher(a=a, b=b).ratio() >= DEDUP_SIMILARITY_THRESHOLD


class ComposablePipeline:
    """Orchestrates one STT + LLM + TTS pipeline via injected backends.

    Takes three backends (each satisfying its respective Protocol from
    :mod:`robot_comic.backends`) plus optional tool wiring and an output
    queue. The bound STT callback drives the main loop — every completed
    transcript runs one full user-turn cycle.

    Not thread-safe: assumes a single asyncio loop owns this instance.
    """

    def __init__(
        self,
        stt: STTBackend,
        llm: LLMBackend,
        tts: TTSBackend,
        *,
        output_queue: "asyncio.Queue[Any] | None" = None,
        tool_dispatcher: ToolDispatcher | None = None,
        tools_spec: list[dict[str, Any]] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        system_prompt: str | None = None,
        deps: "ToolDependencies | None" = None,
        welcome_gate: WelcomeGate | None = None,
    ) -> None:
        """Build the pipeline. See class docstring for arg semantics.

        Phase 5e.2 kwargs:

        - ``deps`` — when provided, the pipeline drives
          movement-manager listening state, head-wobbler reset,
          pause-controller integration, and
          name-validation transcript recording on each turn. Pre-5e.2
          callers that route those concerns through
          :class:`LocalSTTInputMixin` on the host pass ``None`` and the
          pipeline-side callbacks become safe no-ops.
        - ``welcome_gate`` — when provided, completed transcripts are
          checked against the gate before dispatch; in ``WAITING``
          state non-matching transcripts are dropped.
        """
        self.stt = stt
        self.llm = llm
        self.tts = tts
        # The queue is heterogeneous post-5e.2: TTS pushes ``AudioFrame``;
        # the orchestrator's transcript callbacks push
        # ``fastrtc.AdditionalOutputs`` envelopes for the admin UI.
        # Drainers (``ComposableConversationHandler.emit`` via
        # ``wait_for_item``) tolerate both.
        self.output_queue: "asyncio.Queue[Any]" = output_queue or asyncio.Queue()
        self.tool_dispatcher = tool_dispatcher
        self.tools_spec = tools_spec or []
        self.max_tool_rounds = max_tool_rounds
        self.deps = deps
        self.welcome_gate = welcome_gate

        # Barge-in flush callback (Phase 5e.2). The wrapper's
        # ``_clear_queue`` setter mirrors onto this attribute so the
        # ``_on_speech_started`` hook can flush the player when a new
        # user turn begins. ``None`` until ``LocalStream`` installs it.
        self._clear_queue: Callable[[], None] | None = None

        # Per-turn STT spans + duplicate-suppression state (Phase 5e.2,
        # mirrors ``LocalSTTInputMixin._handle_local_stt_event``).
        self._turn_span: Any = None
        self._turn_ctx_token: Any = None
        self._stt_infer_span: Any = None
        self._stt_infer_start: float = 0.0
        self._turn_id: str | None = None
        self._turn_start_at: float | None = None
        self._last_completed_transcript: str = ""
        self._last_completed_at: float = 0.0

        # Phase 5f.3 — ring buffer of recent assistant utterances
        # (normalized) for content-similarity echo filtering.
        self._recent_assistant_texts: collections.deque[str] = collections.deque(maxlen=ECHO_HISTORY_MAXLEN)

        # Issue #267 — last canned line spoken by the empty-STOP fallback
        # path this session, so :meth:`_speak_assistant_text` can avoid
        # picking it twice in a row when Gemini emits two consecutive
        # empty-parts STOP responses. ``None`` until the first fallback
        # fires; persona-reset (``apply_personality``) clears it.
        self._last_empty_stop_fallback: str | None = None

        # TTS-down cascade suppression. ``_tts_down_until`` is 0.0 at start
        # (not in cooldown). Set to ``time.monotonic() + cooldown_s`` when a
        # TTS call raises a service-down exception; cleared to 0.0 on the
        # first subsequent successful TTS call.
        # ``_tts_down_warned`` gates the single entry-warning: True once the
        # WARNING has fired for the current cooldown window; reset when the
        # window expires or TTS recovers.
        self._tts_down_until: float = 0.0
        self._tts_down_warned: bool = False

        self._conversation_history: list[dict[str, Any]] = []
        if system_prompt:
            self._conversation_history.append({"role": "system", "content": system_prompt})
        self._stop_event = asyncio.Event()
        self._started = False

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def start_up(self) -> None:
        """Prepare backends, bind the STT callback, and block until shutdown.

        Mirrors the existing ``ConversationHandler`` ABC's contract: this
        coroutine runs for the entire lifetime of the pipeline. Returns
        only when :meth:`shutdown` is called.
        """
        # If shutdown() already fired (stop_event is set), don't prepare any
        # backends — they'd never be torn down because shutdown's
        # ``if not self._started`` early-return has already run.
        if self._stop_event.is_set():
            return
        await self.llm.prepare()
        await self.tts.prepare()
        await self.stt.start(
            on_completed=self._on_transcript_completed,
            on_partial=self._on_partial_transcript,
            on_speech_started=self._on_speech_started,
        )
        self._started = True
        logger.info(
            "ComposablePipeline ready (stt=%s llm=%s tts=%s)",
            type(self.stt).__name__,
            type(self.llm).__name__,
            type(self.tts).__name__,
        )
        # Issue #504 — Composable mode is input-driven only; without a kick
        # the persona stays silent until the user speaks. Fire a synthetic
        # ``[conversation started]`` transcript through the LLM at start_up
        # so the persona produces its opener, matching the legacy handlers'
        # behaviour. Gated by ``REACHY_MINI_STARTUP_GREETING`` (default 1;
        # set 0 for chassis-safe / silent boots).
        #
        # Fire-and-forget: scheduled as a separate task so start_up reaches
        # the ``_stop_event.wait()`` idle state immediately instead of
        # blocking on the LLM round-trip + TTS synthesis.
        if config.STARTUP_GREETING_ENABLED:
            logger.info("ComposablePipeline startup greeting enabled")
            asyncio.create_task(
                self._on_transcript_completed("[conversation started]"),
                name="composable-startup-greeting",
            )
        else:
            logger.info("ComposablePipeline startup greeting disabled")
        await self._stop_event.wait()

    async def shutdown(self) -> None:
        """Stop the STT pipeline, close LLM/TTS clients, release the loop."""
        if not self._started:
            self._stop_event.set()
            return
        try:
            await self.stt.stop()
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            logger.warning("STT.stop() raised: %s", exc)
        try:
            await self.llm.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM.shutdown() raised: %s", exc)
        try:
            await self.tts.shutdown()
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS.shutdown() raised: %s", exc)
        self._stop_event.set()

    # ---------------------------------------------------------------------
    # Audio input — operator pushes captured frames here.
    # ---------------------------------------------------------------------

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Forward a captured audio frame to the STT backend."""
        await self.stt.feed_audio(frame)

    # ---------------------------------------------------------------------
    # Conversation state
    # ---------------------------------------------------------------------

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Return the live conversation history (mutable)."""
        return self._conversation_history

    def reset_history(self, *, keep_system: bool = True) -> None:
        """Clear the conversation history. By default the system prompt persists."""
        if keep_system and self._conversation_history and self._conversation_history[0].get("role") == "system":
            self._conversation_history = [self._conversation_history[0]]
        else:
            self._conversation_history = []

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality, reset history + per-session TTS state, re-seed system prompt.

        Phase 5c.2 (#TBD) moved this body off
        :class:`ComposableConversationHandler` and onto the pipeline; the
        wrapper is now a thin pass-through. Persona switch is pipeline-shaped
        state surgery (history reset + system-prompt re-seed) plus a hard
        cut on TTS listening state via :meth:`TTSBackend.reset_per_session_state`,
        so the work belongs here rather than at the FastRTC-adapter
        boundary.

        Flow:

        1. :func:`set_custom_profile` — flip the global profile dial. On
           failure, return a "Failed to apply personality" string without
           touching history or per-session state (pins the partial-failure
           contract: a typo'd profile name does not half-reset the session).
        2. :meth:`reset_history` with ``keep_system=False`` — wipe
           everything, including the prior persona's system prompt.
        3. :meth:`TTSBackend.reset_per_session_state` — clear the wrapped
           handler's echo-guard accumulators (``_speaking_until``,
           ``_response_start_ts``, ``_response_audio_bytes``) so a stale
           playback deadline from the previous persona does not bleed
           into the new persona's listening window. Wired in Phase 5a.1
           on the wrapper, moved onto :class:`TTSBackend` in Phase 5c.2.
        4. Re-seed the system prompt with :func:`get_session_instructions`
           so the next LLM round-trip uses the new persona's instructions.

        Joke history is a cross-persona file
        (``~/.robot-comic/joke-history.json``) read in
        ``prompts.py:_append_joke_history`` — it intentionally persists
        across personality switches so each persona avoids the others'
        punchlines (legacy parity, see ``joke_history.format_for_prompt``
        comment "Pulls entries from ALL personas (cross-persona dedup)").
        """
        try:
            set_custom_profile(profile)
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"
        self.reset_history(keep_system=False)
        await self.tts.reset_per_session_state()
        # Phase 5f.3: drop the prior persona's echo-history so the new
        # persona's similarity filter starts clean.
        self._recent_assistant_texts.clear()
        # Issue #267: reset the empty-STOP fallback no-repeat memory so the
        # new persona's first fallback (if any) is sampled uniformly.
        self._last_empty_stop_fallback = None
        self._conversation_history.append({"role": "system", "content": get_session_instructions()})
        return f"Applied personality {profile!r}. Conversation history reset."

    # ---------------------------------------------------------------------
    # Internal — transcript-triggered turn loop
    # ---------------------------------------------------------------------

    async def _on_speech_started(self) -> None:
        """STT speech-started callback (Phase 5e.2).

        Opens the root ``turn`` span + child ``stt.infer`` span, fires
        the barge-in flush callback if registered, resets the head
        wobbler, and signals the movement manager to enter "listening"
        state. All deps-dependent steps are guarded — pipelines
        constructed without ``deps`` (pre-5e.2 host-coupled triples)
        get the span open but skip the movement/wobbler hooks.

        Mirrors :class:`LocalSTTInputMixin._handle_local_stt_event`
        (``local_stt_realtime.py:518-562``) — the started-event arm.
        """
        # Close any leftover turn span from a previous (interrupted) turn.
        if self._turn_span is not None:
            try:
                self._turn_span.set_attribute("turn.outcome", "interrupted")
                self._turn_span.end()
            except Exception:  # pragma: no cover — best-effort span cleanup
                logger.debug("Failed to close prior turn span", exc_info=True)
            self._turn_span = None
        if self._stt_infer_span is not None:
            try:
                self._stt_infer_span.end()
            except Exception:  # pragma: no cover
                logger.debug("Failed to close prior stt.infer span", exc_info=True)
            self._stt_infer_span = None

        now = time.perf_counter()
        tracer = telemetry.get_tracer()
        self._turn_id = str(uuid.uuid4())
        self._turn_start_at = now
        self._turn_span = tracer.start_span(
            "turn",
            attributes={
                "turn.id": self._turn_id,
                "robot.mode": "local_stt",
                "robot.persona": telemetry.current_persona(),
            },
        )
        self._turn_ctx_token = otel_context.attach(trace.set_span_in_context(self._turn_span))
        self._stt_infer_span = tracer.start_span("stt.infer")
        self._stt_infer_start = now

        # Barge-in flush: dump any queued TTS frames from the prior turn so
        # the user's new utterance isn't preceded by stale audio.
        if self._clear_queue is not None:
            try:
                self._clear_queue()
            except Exception:  # pragma: no cover — flush is best-effort
                logger.debug("clear_queue raised in _on_speech_started", exc_info=True)

        if self.deps is not None:
            head_wobbler = getattr(self.deps, "head_wobbler", None)
            if head_wobbler is not None:
                try:
                    head_wobbler.reset()
                except Exception:  # pragma: no cover
                    logger.debug("head_wobbler.reset raised", exc_info=True)
            movement_manager = getattr(self.deps, "movement_manager", None)
            if movement_manager is not None:
                try:
                    movement_manager.set_listening(True)
                except Exception:  # pragma: no cover
                    logger.debug("movement_manager.set_listening(True) raised", exc_info=True)

    async def _on_partial_transcript(self, transcript: str) -> None:
        """STT partial-transcript callback (Phase 5e.2).

        Publishes the in-progress transcript to the output queue as a
        ``user_partial`` row so the admin UI live-transcript widget can
        render incremental updates. Mirrors
        :class:`LocalSTTInputMixin._handle_local_stt_event`
        (``local_stt_realtime.py:564-568``).
        """
        if not transcript:
            return
        from fastrtc import AdditionalOutputs  # deferred — fastrtc pulls gradio at boot

        await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))

    async def _on_transcript_completed(self, transcript: str) -> None:
        """Handle one completed user line: append to history, run LLM, speak.

        Phase 5e.2 extension: when ``deps`` is provided the pipeline
        also drives the per-turn closing rituals previously owned by
        :class:`LocalSTTInputMixin` — duplicate suppression, ``user``
        publication, ``set_listening(False)``,
        :func:`record_user_transcript` for tool-side name validation,
        pause-controller routing, welcome-gate gating, and ``stt.infer``
        span closure. Pre-5e.2 pipelines (``deps=None``) skip all of
        that and behave as before.
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return

        # TTS-down cascade suppression gate.
        #
        # If the TTS service went down recently, suppress this turn entirely
        # so we don't burn CPU on STT inference + LLM round-trips that end
        # in another timeout. The gate fires one WARNING on entry and is
        # then silent (DEBUG) until the cooldown expires or TTS recovers.
        #
        # Placed before 5f.2 / 5f.3 / dedup so suppressed turns cost nothing
        # except a monotonic clock read and a conditional branch.
        _now = time.monotonic()
        if _now < self._tts_down_until:
            remaining = self._tts_down_until - _now
            if not self._tts_down_warned:
                # This is the first drop inside the current window. The warning
                # was already emitted when the window was opened (in
                # _speak_assistant_text), so this path is reached only if
                # _on_transcript_completed fires before _speak_assistant_text
                # has had a chance to open the window. In practice the window
                # opens inside _speak_assistant_text and this branch fires on
                # the *second* and subsequent ambient triggers.
                self._tts_down_warned = True
            logger.debug(
                "TTS still down — suppressing turn, %.0f s remaining in cooldown",
                remaining,
            )
            return

        # Phase 5f.2 — minimum-utterance filter. Drops short transcripts
        # that almost always come from speaker-echo / whisper hallucination
        # (e.g. ``"You"``, ``"Thank you"``) before they reach the LLM and
        # restart the echo-cascade. Runs BEFORE duplicate-suppression so a
        # dropped fragment never poisons the dedupe cache. See spec
        # ``docs/superpowers/specs/2026-05-16-phase-5f-2-echo-cascade-fix.md``.
        if len(transcript.split()) < MIN_TRANSCRIPT_WORDS or len(transcript) < MIN_TRANSCRIPT_CHARS:
            logger.debug("dropping short transcript as likely echo: %r", transcript)
            return

        # Phase 5f.3 — content-similarity echo filter. Drops transcripts
        # that are substantially similar to one of the last N assistant
        # utterances (i.e. the chassis mic picked up our own speaker
        # output and faster-whisper transcribed it back, only mildly
        # corrupted). Runs BEFORE duplicate-suppression so the dedupe
        # cache is never poisoned with an echoed value. See spec
        # ``docs/superpowers/specs/2026-05-16-phase-5f-3-echo-content-similarity-filter.md``.
        if self._recent_assistant_texts:
            needle = _normalize_for_echo_check(transcript)
            if needle:
                best_ratio = max(
                    difflib.SequenceMatcher(None, needle, candidate).ratio()
                    for candidate in self._recent_assistant_texts
                )
                if best_ratio >= ECHO_SIMILARITY_THRESHOLD:
                    logger.debug(
                        "dropping likely echo of own speech: ratio=%.2f, transcript=%r",
                        best_ratio,
                        transcript[:60],
                    )
                    return

        # Duplicate-suppression window. Originally mirrored the legacy
        # mixin's 0.75s strict-equality dedup (``local_stt_realtime.py:596``).
        # Phase-5 follow-up widened the window to ``DEDUP_WINDOW_S`` and
        # swapped equality for similarity (``DEDUP_SIMILARITY_THRESHOLD``)
        # to catch Moonshine's hypothesis-burst pattern; see the constants
        # at module top and the 2026-05-16-phase-5-duplicate-tts-storm
        # spec for the hardware finding that motivated the change.
        #
        # The check + set is synchronous (no ``await``) and therefore
        # atomic within a single asyncio task — two concurrent
        # ``_on_transcript_completed`` tasks for the same utterance are
        # serialised by the event loop's single-tick scheduling, so the
        # second task sees the first task's cache update and dedups
        # correctly even though both were dispatched concurrently from
        # the STT listener's ``call_soon_threadsafe`` hop.
        now = time.perf_counter()
        if now - self._last_completed_at < DEDUP_WINDOW_S and _is_near_duplicate(
            transcript, self._last_completed_transcript
        ):
            if transcript == self._last_completed_transcript:
                # Quiet path — exact-match dedup is the high-volume
                # legacy case and the journal would drown if we shouted.
                logger.debug("Ignoring duplicate transcript: %s", transcript)
            else:
                # Near-duplicate dedup is rarer and load-bearing for the
                # hardware story; log at WARNING so journalctl makes the
                # fix easy to verify on the next on-robot session.
                logger.warning(
                    "Suppressing near-duplicate transcript within %.1fs window: prev=%r new=%r",
                    DEDUP_WINDOW_S,
                    self._last_completed_transcript,
                    transcript,
                )
            return
        self._last_completed_transcript = transcript
        self._last_completed_at = now

        if self.deps is not None:
            movement_manager = getattr(self.deps, "movement_manager", None)
            if movement_manager is not None:
                try:
                    movement_manager.set_listening(False)
                except Exception:  # pragma: no cover
                    logger.debug("movement_manager.set_listening(False) raised", exc_info=True)

            # Close stt.infer span with turn excerpt before LLM round-trip
            # so monitor consumers see the excerpt while the outer turn
            # span is still open.
            words = transcript.split()
            excerpt = " ".join(words[:5]) + ("…" if len(words) > 5 else "")
            stt_span = self._stt_infer_span
            if stt_span is not None:
                try:
                    stt_span.set_attribute("turn.excerpt", excerpt)
                    stt_span.end()
                    stt_s = now - self._stt_infer_start
                    telemetry.record_stt(
                        stt_s,
                        {"gen_ai.system": "local_stt", "stt.type": "moonshine"},
                    )
                except Exception:  # pragma: no cover
                    logger.debug("Failed to close stt.infer span", exc_info=True)
                self._stt_infer_span = None
            if self._turn_span is not None:
                try:
                    self._turn_span.set_attribute("turn.excerpt", excerpt)
                except Exception:  # pragma: no cover
                    logger.debug("Failed to tag turn span", exc_info=True)

            # Tool-side name-validation guard (#287).
            recent = getattr(self.deps, "recent_user_transcripts", None)
            if recent is not None:
                record_user_transcript(recent, transcript)

            # Publish completed transcript to the admin UI.
            from fastrtc import AdditionalOutputs  # deferred — fastrtc pulls gradio

            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            # Pause-controller routing. HANDLED short-circuits dispatch.
            pause_controller = getattr(self.deps, "pause_controller", None)
            if pause_controller is not None:
                try:
                    disposition = pause_controller.handle_transcript(transcript)
                except Exception as exc:
                    logger.error("pause_controller.handle_transcript raised: %s", exc)
                    disposition = TranscriptDisposition.DISPATCH
                if disposition is TranscriptDisposition.HANDLED:
                    return

        # Welcome-gate: drop WAITING transcripts that don't match the wake name.
        gate = self.welcome_gate
        if gate is not None and gate.state is GateState.WAITING:
            if not gate.consider(transcript):
                logger.debug("welcome gate: WAITING — transcript not dispatched: %r", transcript[:60])
                return
            # Gate just opened — dispatch this transcript normally.

        self._conversation_history.append({"role": "user", "content": transcript})
        try:
            await self._run_llm_loop_and_speak()
        except Exception:
            logger.exception("ComposablePipeline turn failed")

    async def _run_llm_loop_and_speak(self) -> None:
        """Run the LLM round-trips with tool dispatch, then synthesize speech.

        Two per-turn safety nets bound the loop:

        - ``max_tool_rounds`` — hard cap on total round-trips.
        - ``MAX_SAME_TOOL_REPEATS`` — per-name repeat ceiling (issue #430).
          When a single tool is called more than the ceiling, the next
          ``llm.chat`` invocation receives a one-shot system nudge
          ("stop calling tools and respond in text"). The nudge is NOT
          appended to the persistent conversation history; it's spliced
          into a local message list for that single call so future turns
          aren't contaminated. Fires at most once per turn; if the LLM
          still loops, ``max_tool_rounds`` is the backstop.
        """
        # Lifecycle Hook #5 (#337): cap the conversation history at
        # ``REACHY_MINI_MAX_HISTORY_TURNS`` user turns so long sessions don't
        # blow the model's context window or run the token bill into the
        # ground. Once-per-user-turn cadence matches the legacy sites in
        # ``_dispatch_completed_transcript``. Legacy parity:
        # llama_base.py:506, gemini_tts.py:365, elevenlabs_tts.py:565.
        trim_history_in_place(self._conversation_history)
        tool_call_counts: collections.Counter[str] = collections.Counter()
        nudge_sent = False
        for _round in range(self.max_tool_rounds):
            looper: str | None = None
            if not nudge_sent:
                looper = next(
                    (name for name, count in tool_call_counts.items() if count > MAX_SAME_TOOL_REPEATS),
                    None,
                )
            if looper is not None:
                logger.warning(
                    "ComposablePipeline: tool %r called %d times this turn "
                    "(> MAX_SAME_TOOL_REPEATS=%d); injecting one-shot "
                    "stop-calling-tools nudge",
                    looper,
                    tool_call_counts[looper],
                    MAX_SAME_TOOL_REPEATS,
                )
                messages: list[dict[str, Any]] = list(self._conversation_history) + [
                    {
                        "role": "system",
                        "content": (
                            f"Tool '{looper}' has been called {tool_call_counts[looper]} "
                            "times this turn with no resolution. Stop calling tools and "
                            "respond to the user in text."
                        ),
                    }
                ]
                nudge_sent = True
            else:
                messages = self._conversation_history
            response = await self.llm.chat(
                messages,
                tools=self.tools_spec or None,
            )
            if response.tool_calls:
                if self.tool_dispatcher is None:
                    logger.warning(
                        "LLM requested tools but no dispatcher is configured; "
                        "ignoring %d call(s) and breaking the loop",
                        len(response.tool_calls),
                    )
                    break
                await self._dispatch_tools_and_record(response.tool_calls)
                for call in response.tool_calls:
                    tool_call_counts[call.name] += 1
                continue
            # No more tool calls — assistant text is final.
            await self._speak_assistant_text(response)
            return
        logger.warning(
            "ComposablePipeline hit max_tool_rounds=%d without an assistant text response",
            self.max_tool_rounds,
        )

    async def _dispatch_tools_and_record(self, tool_calls: tuple[ToolCall, ...]) -> None:
        """Dispatch each tool call sequentially and append results to history."""
        assert self.tool_dispatcher is not None  # caller guards
        for call in tool_calls:
            try:
                result = await self.tool_dispatcher(call)
            except Exception as exc:
                logger.exception("Tool %s raised", call.name)
                result = f"[tool error: {type(exc).__name__}: {exc}]"
            self._conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result,
                }
            )

    async def _speak_assistant_text(self, response: LLMResponse) -> None:
        """Append the assistant text to history and stream TTS frames out.

        Issue #267: Gemini 2.5 Flash sometimes returns ``finish_reason=STOP``
        with zero parts (no text, no function_call) under the combination
        of (tools-enabled config + short user input + certain persona
        prompts). The legacy ``elevenlabs_tts.py`` path already substitutes
        a canned filler for this case; the composable path used to log
        and drop the turn, which the operator perceived as a silent hang.

        Trigger: the orchestrator only reaches this method when
        ``response.tool_calls == ()`` (the loop in
        :meth:`_run_llm_loop_and_speak` ``continue``-s on any non-empty
        tool_calls), so an empty text here is exactly the empty-STOP
        case. We pick a persona-appropriate canned line, log a single
        WARNING, increment ``composable.empty_stop_fallback``, and fall
        through into the normal TTS path so the line is spoken and
        recorded in history as an ``assistant`` turn — the LLM keeps
        continuity on the next request.

        TTS-down detection: if ``tts.synthesize`` raises a service-unreachable
        exception (``httpx.ConnectTimeout``, ``httpx.ConnectError``,
        ``httpx.ReadTimeout``, ``OSError``), this method sets
        ``_tts_down_until`` to suppress future STT-triggered turns for
        ``REACHY_MINI_TTS_DOWN_COOLDOWN_S`` seconds, then re-raises so the
        outer ``_run_llm_loop_and_speak`` exception handler logs it. When TTS
        successfully streams at least one frame, ``_tts_down_until`` is cleared
        so the cooldown lifts immediately on recovery.
        """
        text = response.text
        is_empty_stop_fallback = False
        if not text.strip():
            pool = _load_empty_stop_pool(_resolve_empty_stop_profile_dir())
            fallback = _pick_empty_stop_fallback(pool, last_spoken=self._last_empty_stop_fallback)
            self._last_empty_stop_fallback = fallback
            logger.warning(
                "Assistant returned empty text (Gemini empty-STOP, issue #267); substituting canned fallback %r",
                fallback,
            )
            telemetry.inc_empty_stop_fallback({"robot.persona": telemetry.current_persona()})
            # Fall through into the normal TTS path with the canned line
            # as if Gemini had returned it directly. Continuing means the
            # line lands in conversation_history, the echo ring buffer,
            # and the TTS adapter — all the regular per-turn machinery.
            text = fallback
            is_empty_stop_fallback = True
        # Lifecycle Hook #4 (#337): capture the punchline + topic into joke
        # history so the next session's system-prompt builder can include
        # them in the "RECENT JOKES (DO NOT REPEAT)" section. Best-effort —
        # the helper swallows its own exceptions; the outer ``try`` is
        # belt-and-braces so a future code change in the helper can't kill
        # the turn before TTS runs. Legacy parity sites:
        # llama_base.py:578-594 and gemini_tts.py:380-394.
        # Skip joke-history capture for empty-STOP canned fallbacks (#267):
        # the canned filler is not a comedy punchline and would pollute the
        # cross-persona "RECENT JOKES (DO NOT REPEAT)" prompt section.
        if not is_empty_stop_fallback:
            try:
                await record_joke_history(text)
            except Exception as exc:  # pragma: no cover — helper is itself defensive
                logger.debug("record_joke_history raised through to orchestrator: %s", exc)
        self._conversation_history.append({"role": "assistant", "content": text})
        # Phase 5f.3 — record the normalized assistant text in the ring
        # buffer so the next user transcript's similarity check has the
        # latest utterance to compare against. Skip empty normalized
        # output (would dominate the SequenceMatcher ratio with garbage).
        normalized = _normalize_for_echo_check(text)
        if normalized:
            self._recent_assistant_texts.append(normalized)
        # Phase 5a.2: thread the structured ``delivery_tags`` channel from
        # ``LLMResponse`` to ``TTSBackend.synthesize(tags=...)``. Today's
        # LLM adapters leave the tuple empty, so TTS adapters fall back to
        # the today text-parsing behaviour; future PRs that surface
        # structured cues from the LLM populate the field and the consume
        # path activates without further orchestrator changes.
        # ``first_audio_marker`` (Phase 5a.2): allocate a fresh list per
        # turn so the TTS adapter can stamp first-audio-out wallclock.
        # The marker is consumed below to emit ``robot.tts.time_to_first_audio``
        # (#271 ask 1). See
        # ``docs/superpowers/specs/2026-05-16-phase-5a2-delivery-tag-plumbing.md``.
        #
        # ``response_committed_at`` (Phase 5 / #271 ask 1): record when the
        # orchestrator hands the final assistant text to the TTS backend so
        # the ``robot.tts.time_to_first_audio`` metric reflects the operator-
        # perceived wait from "response finalized" to "first audio emitted".
        # Uses ``time.monotonic()`` to match the TTS adapters' stamp clock.
        first_audio_marker: list[float] = []
        response_committed_at: float = time.monotonic()
        _first_audio_recorded: bool = False
        _frames_emitted = 0
        try:
            async for frame in self.tts.synthesize(
                text,
                tags=response.delivery_tags,
                first_audio_marker=first_audio_marker,
            ):
                # Issue #271 ask 1 — emit ``robot.tts.time_to_first_audio`` on
                # the first frame of the first TTS call of this turn. Multi-
                # sentence responses produce multiple ``synthesize`` calls (one
                # per sentence) but only the first sentence's first-audio is
                # the "user-perceived latency to first hear something"; subsequent
                # sentences' first-audio stamps are NOT re-recorded.
                #
                # The marker list is populated by the TTS adapter on its first
                # yielded frame (Phase 5a.2). If the adapter did not populate
                # it (e.g. a test stub with no marker support or an adapter that
                # yields no frames for an empty segment), the list stays empty
                # and we skip the metric rather than recording a garbage value.
                if not _first_audio_recorded and first_audio_marker:
                    _first_audio_recorded = True
                    first_audio_latency_s = first_audio_marker[0] - response_committed_at
                    telemetry.record_tts_first_audio(
                        first_audio_latency_s,
                        {"gen_ai.system": "tts", "robot.persona": telemetry.current_persona()},
                    )
                    logger.debug(
                        "tts.first_audio_latency_s=%.3f (committed_at=%.3f first_audio=%.3f)",
                        first_audio_latency_s,
                        response_committed_at,
                        first_audio_marker[0],
                    )
                # ``console.py``'s playback loop branches on
                # ``isinstance(handler_output, tuple)`` to dispatch to ALSA
                # (see ``console.py:1466``). The TTS adapters yield
                # :class:`AudioFrame` dataclasses for the Protocol-level
                # surface; the legacy ``elevenlabs_tts.py`` put-site (line
                # ~487) bypassed the wrapping and pushed
                # ``(sample_rate, ndarray)`` tuples straight onto the queue.
                # Unwrap here so downstream consumers stay shape-compatible.
                # Without this, TTS audio is silently dropped on hardware
                # (the dataclass falls through ``isinstance(..., tuple)`` and
                # ``isinstance(..., AdditionalOutputs)`` checks). Caught
                # during 2026-05-16 hardware validation on ricci.
                await self.output_queue.put((frame.sample_rate, frame.samples))
                _frames_emitted += 1
        except _TTS_DOWN_EXCEPTIONS as exc:
            # Service-unreachable failure — arm the cascade-suppression cooldown.
            cooldown_s = _tts_down_cooldown_s()
            self._tts_down_until = time.monotonic() + cooldown_s
            self._tts_down_warned = False  # allow the gate to emit one WARNING
            logger.warning(
                "TTS unreachable (%s: %s) — suppressing STT-triggered turns for %.0f s "
                "to break cascade. Set %s to adjust.",
                type(exc).__name__,
                exc,
                cooldown_s,
                _TTS_DOWN_COOLDOWN_ENV,
            )
            raise  # let _run_llm_loop_and_speak → _on_transcript_completed log it
        else:
            # Successful synthesis. If we were in a cooldown window, lift it
            # and emit a recovery log so the operator can correlate in journalctl.
            if self._tts_down_until > 0.0:
                logger.info("TTS recovered after outage — resuming normal turn processing")
                self._tts_down_until = 0.0
                self._tts_down_warned = False
