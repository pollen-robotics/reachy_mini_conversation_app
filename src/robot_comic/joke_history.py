"""Persistent joke history for avoid-repeat prompt injection.

Stores the last N punchlines across sessions in ``~/.robot-comic/joke-history.json``
and appends a "don't repeat" section to the system prompt at each session start.

Enable/disable via the ``REACHY_MINI_JOKE_HISTORY_ENABLED`` env var (default: True).
LLM extraction can be toggled via ``REACHY_MINI_JOKE_HISTORY_LLM_EXTRACT_ENABLED``
(default: True).  When disabled (or when the llama-server is unreachable), the
module falls back to the ``last_sentence_of`` heuristic.
"""

from __future__ import annotations
import os
import re
import json
import math
import logging
import tempfile
from typing import TYPE_CHECKING, Any
from pathlib import Path
from datetime import datetime, timezone


if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = Path.home() / ".robot-comic"
_DEFAULT_HISTORY_FILENAME = "joke-history.json"
_DEFAULT_MAX_ENTRIES = 50
_DEFAULT_RECENT_N = 10

# Time-decay half-life: 7 days expressed as the τ in exp(-age_days / τ).
_DECAY_TAU_DAYS: float = 7.0
# Entries whose decayed weight drops below this threshold are excluded from
# the avoid-list (at τ=7 this corresponds to ~17 days = weight ≈ 0.085).
_MIN_WEIGHT_THRESHOLD: float = 0.1

# Split on sentence-ending punctuation, keeping the terminator attached.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Prompt sent to llama-server for punchline extraction.
_EXTRACT_PROMPT = (
    "Given this assistant turn, return JSON with exactly two keys: "
    '"punchline" (one-sentence punchline, or null if this is setup/banter/not a joke) '
    'and "topic" (1-3 word topic tag, e.g. "appearance", "hockey puck", "marriage"). '
    "Be terse. Return valid JSON only — no prose.\n\nAssistant turn: {text}"
)
# Timeout for the punchline-extraction LLM call (seconds).
_EXTRACT_TIMEOUT_S: float = 0.5
# Max tokens for the extraction response.
_EXTRACT_MAX_TOKENS: int = 100


def last_sentence_of(text: str) -> str:
    """Extract the trailing sentence from *text*.

    Splits on ``.``, ``!``, or ``?`` boundaries and returns the last
    non-empty segment.  Returns the whole string (stripped) when no
    sentence terminator is found, or an empty string when *text* is empty.
    """
    text = text.strip()
    if not text:
        return ""
    parts = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    return parts[-1] if parts else text


def default_history_path() -> Path:
    """Return the default joke-history file path, creating the parent dir if needed."""
    try:
        _DEFAULT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create joke-history directory %s: %s", _DEFAULT_HISTORY_DIR, exc)
    return _DEFAULT_HISTORY_DIR / _DEFAULT_HISTORY_FILENAME


async def extract_punchline_via_llm(
    assistant_text: str,
    http_client: "httpx.AsyncClient",
    *,
    llama_url: str = "http://astralplane.lan:8080",
) -> dict[str, Any] | None:
    """Ask llama-server to extract the punchline and topic from *assistant_text*.

    Returns a dict ``{"punchline": str | None, "topic": str}`` on success.
    Returns ``None`` when the feature is disabled or when a hard error occurs
    (caller should skip the ``add()`` call rather than falling back).

    Falls back to the ``last_sentence_of`` heuristic (returning a dict) when
    the server is unreachable or returns non-JSON output.

    Args:
        assistant_text: The full assistant response text to analyse.
        http_client: Shared ``httpx.AsyncClient`` from the calling handler.
        llama_url: Base URL of the llama-server instance.

    """
    import httpx as _httpx

    from robot_comic.config import config  # avoid circular at module load

    if not getattr(config, "JOKE_HISTORY_LLM_EXTRACT_ENABLED", True):
        # Feature disabled — use heuristic directly.
        return {"punchline": last_sentence_of(assistant_text), "topic": ""}

    prompt = _EXTRACT_PROMPT.format(text=assistant_text)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _EXTRACT_MAX_TOKENS,
        "temperature": 0.1,
        "stream": False,
    }
    # Use the caller's shared client, but enforce a tight per-request timeout.
    _timeout = _httpx.Timeout(_EXTRACT_TIMEOUT_S)

    try:
        resp = await http_client.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=_timeout,
        )
        resp.raise_for_status()
        raw_content: str = resp.json()["choices"][0]["message"]["content"]
        # Strip markdown code fences if present.
        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```[a-z]*\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content.strip())
        parsed = json.loads(raw_content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned non-dict JSON")
        punchline = parsed.get("punchline")
        topic = str(parsed.get("topic") or "").strip()
        if punchline is not None:
            punchline = str(punchline).strip() or None
        logger.debug("joke_history LLM extract: punchline=%r topic=%r", punchline, topic)
        return {"punchline": punchline, "topic": topic}
    except (_httpx.TimeoutException, _httpx.NetworkError, _httpx.ConnectError) as exc:
        logger.debug("joke_history LLM extract network error (%s) — using heuristic fallback", exc)
        return {"punchline": last_sentence_of(assistant_text), "topic": ""}
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.debug("joke_history LLM extract parse error (%s) — using heuristic fallback", exc)
        return {"punchline": last_sentence_of(assistant_text), "topic": ""}
    except Exception as exc:
        logger.debug("joke_history LLM extract unexpected error (%s) — using heuristic fallback", exc)
        return {"punchline": last_sentence_of(assistant_text), "topic": ""}


async def record_joke_history(
    response_text: str,
    *,
    persona: str | None = None,
) -> None:
    """Capture a punchline + topic from *response_text* into joke history.

    Lifecycle Hook #4 (#337): the composable pipeline (``ComposablePipeline``)
    calls this after a final, non-empty assistant text round to keep the
    avoid-repeat joke history fed without going through the legacy
    ``_run_turn`` / ``GeminiTTSResponseHandler.response`` paths. Mirrors the
    legacy capture sites (``llama_base.py:578-594`` and
    ``gemini_tts.py:380-394``) but owns its own ``httpx`` client lifecycle so
    the orchestrator doesn't have to.

    Best-effort:

    - Returns silently if the feature is disabled
      (``config.JOKE_HISTORY_ENABLED=False``) or if *response_text* is
      empty / whitespace.
    - Swallows any exception from the extraction call or the file write
      (logged at DEBUG). Joke history must never crash a turn.

    Args:
        response_text: The final assistant text emitted for the turn.
        persona: Persona name to record alongside the punchline. When
            ``None`` (default), falls back to
            ``config.REACHY_MINI_CUSTOM_PROFILE`` or an empty string.

    """
    if not response_text or not response_text.strip():
        return

    from robot_comic.config import config  # avoid circular at module load

    if not getattr(config, "JOKE_HISTORY_ENABLED", True):
        return

    resolved_persona = persona if persona is not None else (getattr(config, "REACHY_MINI_CUSTOM_PROFILE", "") or "")

    try:
        import httpx as _httpx

        async with _httpx.AsyncClient() as http:
            extracted = await extract_punchline_via_llm(response_text, http)
        punchline = extracted.get("punchline") if extracted is not None else None
        if punchline and extracted is not None:
            JokeHistory(default_history_path()).add(
                punchline,
                topic=extracted.get("topic", "") or "",
                persona=resolved_persona,
            )
    except Exception as exc:
        logger.debug("joke_history capture failed: %s", exc)


def _entry_weight(entry: dict[str, Any], now: datetime) -> float:
    """Compute the time-decay weight for a history entry.

    Uses ``weight = exp(-age_days / τ)`` where τ = ``_DECAY_TAU_DAYS``.
    Entries with no parseable timestamp are treated as age=0 (weight=1.0).
    """
    ts_raw = entry.get("ts", "")
    if not ts_raw:
        return 1.0
    try:
        ts = datetime.fromisoformat(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (now - ts).total_seconds() / 86400.0
        return math.exp(-max(age_days, 0.0) / _DECAY_TAU_DAYS)
    except (ValueError, OSError):
        return 1.0


class JokeHistory:
    """FIFO store for recent punchlines with prompt-injection support.

    Args:
        path: Path to the JSON file used for persistence.
        max_entries: Maximum number of entries to keep (oldest are dropped).

    """

    def __init__(self, path: Path, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        """Initialise the store, loading any existing entries from *path*."""
        self._path = path
        self._max_entries = max_entries
        self._entries: list[dict[str, Any]] = self.load()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def load(self) -> list[dict[str, Any]]:
        """Load entries from disk.  Returns ``[]`` if the file is missing or unreadable."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning("joke-history file %s has unexpected format; starting fresh", self._path)
        except Exception as exc:
            logger.warning("Could not read joke-history from %s: %s", self._path, exc)
        return []

    def save(self, entries: list[dict[str, Any]]) -> None:
        """Atomically write *entries* to disk (tmp file + rename)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".joke-history-",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(entries, fh, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Could not save joke-history to %s: %s", self._path, exc)

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #

    def add(self, punchline: str, topic: str = "", persona: str = "") -> None:
        """Append a new punchline entry and auto-save.

        Truncates the in-memory list to the last ``max_entries`` before saving.

        Args:
            punchline: The extracted or heuristic punchline text.
            topic: Short topic tag (1-3 words).
            persona: The active persona/profile name at capture time.

        """
        punchline = punchline.strip()
        if not punchline:
            logger.debug("joke_history.add: empty punchline, skipping")
            return
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "punchline": punchline,
            "topic": topic.strip(),
            "persona": persona.strip(),
        }
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        self.save(self._entries)

    # ------------------------------------------------------------------ #
    # Query / formatting                                                   #
    # ------------------------------------------------------------------ #

    def recent(self, n: int = _DEFAULT_RECENT_N) -> list[dict[str, Any]]:
        """Return the last *n* entries in chronological order (oldest first)."""
        return list(self._entries[-n:])

    def format_for_prompt(
        self,
        n: int = _DEFAULT_RECENT_N,
        *,
        min_weight_threshold: float = _MIN_WEIGHT_THRESHOLD,
    ) -> str:
        """Return a formatted block ready to drop into the system prompt.

        Pulls entries from ALL personas (cross-persona dedup), applies
        exponential time-decay weighting, drops entries below
        *min_weight_threshold*, then returns the top *n* by weight.

        Returns an empty string when there are no qualifying entries so callers
        can skip the section entirely.

        Args:
            n: Maximum number of entries to include.
            min_weight_threshold: Entries with ``weight < min_weight_threshold``
                are excluded entirely (default 0.1, ~17 days at τ=7 d).

        """
        if not self._entries:
            return ""

        now = datetime.now(timezone.utc)

        # Score every entry and filter out stale ones.
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._entries:
            w = _entry_weight(entry, now)
            if w >= min_weight_threshold:
                scored.append((w, entry))

        if not scored:
            return ""

        # Sort descending by weight (most recent / highest weight first),
        # then take at most *n*.
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:n]

        # Re-sort by timestamp (oldest→newest) so the prompt reads chronologically.
        top.sort(key=lambda t: t[1].get("ts", ""))

        lines = ["## RECENT JOKES (DO NOT REPEAT)", ""]
        lines.append("Recent jokes told across all personas (avoid repeating these themes/punchlines):")
        for _w, entry in top:
            punchline = entry.get("punchline", "").strip()
            if not punchline:
                continue
            persona = entry.get("persona", "").strip()
            topic = entry.get("topic", "").strip()
            persona_prefix = f"[{persona}] " if persona else ""
            topic_suffix = f" (topic: {topic})" if topic else ""
            lines.append(f"- {persona_prefix}{punchline}{topic_suffix}")
        lines.append("")
        return "\n".join(lines)
