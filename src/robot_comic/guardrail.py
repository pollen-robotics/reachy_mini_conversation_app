"""Disengagement guardrail for high-intensity personas.

``EngagementMonitor`` inspects each user turn for discomfort signals using a
lightweight keyword/regex heuristic — no LLM call, no network, O(n) on the
phrase list.

Activation is persona-aware: only profiles listed in ``GUARDRAIL_PROFILES``
activate the monitor.  By default the set is empty; external persona libraries
can register profiles that opt in.  The feature can be force-disabled
for any profile via the ``REACHY_MINI_GUARDRAIL_ENABLED`` environment variable.

Optional LLM-scored mode
------------------------
When ``REACHY_MINI_GUARDRAIL_LLM_SCORING=1`` is set **and** an ``http_client``
is passed to ``analyze()``, the monitor sends a lightweight one-shot prompt to
the local llama-server to score discomfort more accurately.  On any parse or
network error it falls back transparently to the heuristic score.

Soften notes
------------
Each guardrail-active persona has its own soften note, injected at runtime
(not persisted) into the system prompt when ``should_soften`` is True.
Use :func:`get_soften_note` to retrieve the note for a given persona name.

Calibration logging
-------------------
Every ``analyze()`` invocation emits a DEBUG line::

    guardrail.calibration persona=<name> heuristic_score=<float>
    llm_score=<float|null> consecutive_discomfort=<int> should_soften=<bool>
"""

from __future__ import annotations
import os
import re
import json
import logging
from typing import Any


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-persona soften notes
# ---------------------------------------------------------------------------

# The note prepended (at runtime only) to the system prompt when the monitor
# decides the persona should soften.  Not persisted anywhere.
# Per-persona notes are registered by external persona libraries via
# SOFTEN_NOTES.  The generic fallback is used for any persona not listed here.
SOFTEN_NOTES: dict[str, str] = {}

_SOFTEN_NOTE_GENERIC = (
    "NOTE: the user seems uncomfortable with the current intensity — "
    "ease back and pivot to lighter material for the next few turns."
)

# Backwards-compat alias — returns the generic note.
SOFTEN_NOTE: str = _SOFTEN_NOTE_GENERIC


def get_soften_note(persona: str | None) -> str:
    """Return the persona-specific soften note, or the generic fallback."""
    if persona and persona in SOFTEN_NOTES:
        return SOFTEN_NOTES[persona]
    return _SOFTEN_NOTE_GENERIC


# ---------------------------------------------------------------------------
# Guardrail-active profiles
# ---------------------------------------------------------------------------

# Profiles that opt in to the guardrail by default.
# External persona libraries can register additional profiles here at import
# time by adding to this frozenset (reassign with a union).  The built-in
# house_comedian persona does not require the guardrail.
GUARDRAIL_PROFILES: frozenset[str] = frozenset()

# Number of consecutive discomfort turns required before ``should_soften``
# fires.  Override via ``REACHY_MINI_GUARDRAIL_THRESHOLD``.
_DEFAULT_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Discomfort signal patterns
# ---------------------------------------------------------------------------
# Phrases that indicate the user wants to disengage or tone things down.
# Each entry is compiled into a case-insensitive regex with word-boundary
# anchors where helpful.  The list is intentionally conservative to avoid
# false positives on content-level disagreement (which intense personas
# *should* engage).
_DISCOMFORT_PHRASES: list[str] = [
    r"\bstop\b",
    r"\benough\b",
    r"\btone.?it.?down\b",
    r"\bless.?aggressive\b",
    r"\bchange.?the.?topic\b",
    r"\bchange.?subject\b",
    r"\blet.?s.?change\b",
    r"\bokay.?enough\b",
    r"\bthat.?s.?enough\b",
    r"\btoo.?much\b",
    r"\bback.?off\b",
    r"\bease.?up\b",
    r"\bcalm.?down\b",
    r"\bnot.?funny\b",
    r"\bnot.?comfortable\b",
    r"\buncomfortable\b",
    r"\bplease.?stop\b",
    r"\bi.?don.?t.?like.?this\b",
    r"\bi.?don.?t.?want.?to\b",
    r"\bcan.?we.?talk.?about.?something.?else\b",
    r"\bcan.?we.?change\b",
    r"\bnever.?mind\b",
    r"\bforget.?it\b",
    r"\bquit.?it\b",
    r"\bjust.?stop\b",
    r"\bwhatever\b",
    r"\bi.?give.?up\b",
    r"\bleave.?me.?alone\b",
    r"\bgo.?away\b",
]

# Compile once at module load.
_DISCOMFORT_RE: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _DISCOMFORT_PHRASES]

# Minimum word count below which a response is treated as "very short"
# (a mild secondary signal — not enough alone, raises score by less).
_SHORT_RESPONSE_WORDS = 3
_SHORT_RESPONSE_SCORE_CONTRIBUTION = 0.3


def _score_discomfort(text: str) -> float:
    """Return a discomfort score in ``[0.0, 1.0]`` for a single user turn.

    Scoring rules:
    - Empty / blank input → 0.6 (silence is a strong disengagement signal).
    - Any discomfort phrase match → 1.0 (hard signal; return immediately).
    - Very short response (≤ ``_SHORT_RESPONSE_WORDS`` words) without a phrase
      match → ``_SHORT_RESPONSE_SCORE_CONTRIBUTION`` (weak signal).
    - Otherwise → 0.0.
    """
    stripped = text.strip()
    if not stripped:
        return 0.6

    for pattern in _DISCOMFORT_RE:
        if pattern.search(stripped):
            return 1.0

    words = stripped.split()
    if len(words) <= _SHORT_RESPONSE_WORDS:
        return _SHORT_RESPONSE_SCORE_CONTRIBUTION

    return 0.0


def _guardrail_enabled_for_profile(profile: str | None) -> bool:
    """Return whether the guardrail is active for the given profile name.

    Resolution order:
    1. ``REACHY_MINI_GUARDRAIL_ENABLED`` env var (explicit override, any profile).
    2. Profile membership in ``GUARDRAIL_PROFILES``.
    """
    raw = os.getenv("REACHY_MINI_GUARDRAIL_ENABLED", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # No explicit override — use profile membership.
    return (profile or "") in GUARDRAIL_PROFILES


def _llm_scoring_enabled() -> bool:
    """Return True when REACHY_MINI_GUARDRAIL_LLM_SCORING is set to a truthy value."""
    raw = os.getenv("REACHY_MINI_GUARDRAIL_LLM_SCORING", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_threshold() -> int:
    """Return the consecutive-discomfort threshold from env or default."""
    raw = os.getenv("REACHY_MINI_GUARDRAIL_THRESHOLD", "").strip()
    if raw:
        try:
            val = int(raw)
            if val >= 1:
                return val
            logger.warning(
                "REACHY_MINI_GUARDRAIL_THRESHOLD=%r must be >= 1; using default %d", raw, _DEFAULT_THRESHOLD
            )
        except ValueError:
            logger.warning(
                "REACHY_MINI_GUARDRAIL_THRESHOLD=%r is not an integer; using default %d", raw, _DEFAULT_THRESHOLD
            )
    return _DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# LLM scoring prompt
# ---------------------------------------------------------------------------

_LLM_SCORE_PROMPT = (
    "Score the user's discomfort with the conversation from 0.0 (engaged/enjoying) to "
    '1.0 (uncomfortable/wants to stop). Return JSON: {"score": <float>, "reason": <short string>}. '
    "User text: "
)


class EngagementMonitor:
    r"""Stateful per-session monitor for user disengagement signals.

    Typical lifecycle::

        monitor = EngagementMonitor(profile="house_comedian")
        score, should_soften = monitor.analyze(user_text)
        if should_soften:
            note = get_soften_note("house_comedian")
            instructions = note + "\n\n" + base_instructions

    The monitor is a no-op (``score=0.0, should_soften=False``) when the
    active profile is not in ``GUARDRAIL_PROFILES`` and no env-override is set.

    Optional LLM scoring
    --------------------
    Pass an ``http_client`` with a ``post(url, json, timeout)`` interface to
    ``analyze()`` when ``REACHY_MINI_GUARDRAIL_LLM_SCORING=1`` is set.  The
    monitor will call :meth:`score_via_llm` asynchronously and use the result
    instead of the heuristic.  Falls back to heuristic on any error.
    """

    def __init__(self, profile: str | None = None) -> None:
        """Initialise the monitor for the given profile (or no profile)."""
        self._profile = profile
        self._enabled = _guardrail_enabled_for_profile(profile)
        self._threshold = _get_threshold()
        self._consecutive_discomfort: int = 0
        self._last_score: float = 0.0
        logger.debug(
            "EngagementMonitor init: profile=%r enabled=%s threshold=%d",
            profile,
            self._enabled,
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return whether the guardrail is active for this monitor's profile."""
        return self._enabled

    @property
    def consecutive_discomfort(self) -> int:
        """Number of consecutive turns with a discomfort score above threshold."""
        return self._consecutive_discomfort

    @property
    def last_score(self) -> float:
        """Discomfort score from the most recent ``analyze()`` call."""
        return self._last_score

    async def score_via_llm(self, user_text: str, http_client: Any) -> float:  # noqa: ANN401
        """Score *user_text* discomfort via a one-shot LLM call.

        Sends a prompt to the local llama-server (same endpoint used by the
        handler's ``_http`` client) asking for a JSON score in ``[0.0, 1.0]``.

        Args:
            user_text: The raw transcript of the latest user turn.
            http_client: An object with a ``post(url, json, timeout)`` async
                coroutine compatible with ``httpx.AsyncClient``.

        Returns:
            The LLM-scored discomfort value, or the heuristic fallback on any
            parse/network error.

        """
        from robot_comic import config as _cfg  # local import to avoid circular

        prompt = _LLM_SCORE_PROMPT + user_text
        try:
            resp = await http_client.post(
                _cfg.config.LLAMA_CPP_URL + "/completion",
                json={
                    "prompt": prompt,
                    "max_tokens": 80,
                    "temperature": 0.1,
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # llama.cpp /completion returns {"content": "..."}
            content = data.get("content", "")
            parsed = json.loads(content)
            score = float(parsed["score"])
            score = max(0.0, min(1.0, score))
            logger.debug(
                "EngagementMonitor LLM score=%.2f reason=%r",
                score,
                parsed.get("reason", ""),
            )
            return score
        except Exception as exc:
            logger.debug(
                "EngagementMonitor: LLM scoring failed (%s), falling back to heuristic",
                exc,
            )
            return _score_discomfort(user_text)

    def analyze(
        self,
        user_text: str,
        llm_score: float | None = None,
    ) -> tuple[float, bool]:
        """Inspect *user_text* and update engagement state.

        Args:
            user_text: The raw transcript of the latest user turn.
            llm_score: Pre-computed LLM discomfort score (0.0–1.0).  When
                provided **and** ``REACHY_MINI_GUARDRAIL_LLM_SCORING`` is
                enabled, this value is used instead of the heuristic.  Callers
                that want LLM scoring should ``await score_via_llm(...)`` first
                and pass the result here.

        Returns:
            ``(discomfort_score, should_soften)`` where:
            - ``discomfort_score`` is in ``[0.0, 1.0]``.
            - ``should_soften`` is ``True`` when the monitor recommends that
              the persona ease back for the next few turns.

        """
        if not self._enabled:
            logger.debug(
                "guardrail.calibration persona=%r heuristic_score=0.0 llm_score=null"
                " consecutive_discomfort=0 should_soften=False",
                self._profile,
            )
            return 0.0, False

        heuristic_score = _score_discomfort(user_text)

        use_llm = _llm_scoring_enabled() and llm_score is not None
        effective_score: float = llm_score if (use_llm and llm_score is not None) else heuristic_score
        log_llm = f"{llm_score:.2f}" if llm_score is not None else "null"

        self._last_score = effective_score

        # A score ≥ 0.5 counts as a discomfort signal.
        if effective_score >= 0.5:
            self._consecutive_discomfort += 1
            logger.debug(
                "EngagementMonitor: discomfort score=%.2f consecutive=%d threshold=%d",
                effective_score,
                self._consecutive_discomfort,
                self._threshold,
            )
        else:
            if self._consecutive_discomfort > 0:
                logger.debug(
                    "EngagementMonitor: engagement signal (score=%.2f) — resetting consecutive counter",
                    effective_score,
                )
            self._consecutive_discomfort = 0

        should_soften = self._consecutive_discomfort >= self._threshold
        if should_soften:
            logger.info(
                "EngagementMonitor: soften triggered (consecutive=%d >= threshold=%d)",
                self._consecutive_discomfort,
                self._threshold,
            )

        # Calibration log — one structured line per analyze() call.
        logger.debug(
            "guardrail.calibration persona=%r heuristic_score=%.2f llm_score=%s"
            " consecutive_discomfort=%d should_soften=%s",
            self._profile,
            heuristic_score,
            log_llm,
            self._consecutive_discomfort,
            should_soften,
        )

        return effective_score, should_soften

    def reset(self) -> None:
        """Reset engagement state (e.g., on new session or profile switch)."""
        self._consecutive_discomfort = 0
        self._last_score = 0.0
        logger.debug("EngagementMonitor: state reset for profile=%r", self._profile)

    def update_profile(self, profile: str | None) -> None:
        """Update the active profile and re-evaluate whether the guardrail is enabled."""
        self._profile = profile
        self._enabled = _guardrail_enabled_for_profile(profile)
        self.reset()
        logger.debug(
            "EngagementMonitor: profile updated to %r, enabled=%s",
            profile,
            self._enabled,
        )
