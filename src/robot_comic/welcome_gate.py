"""Welcome-gate state machine.

Gates the conversational handler's audio output until the user speaks a
persona wake-name.  The gate has two states:

- WAITING  -- welcome narration plays; Moonshine feeds transcripts here but
              they are NOT forwarded to the LLM.
- GATED    -- wake-name detected; normal handler processing resumes.

Usage::

    gate = WelcomeGate(["rickles", "don rickles"])
    if gate.consider("Hey, Rickles!"):
        # state is now GATED -- open the handler
        pass
"""

from __future__ import annotations
import logging
from enum import Enum, auto
from pathlib import Path


logger = logging.getLogger(__name__)

# Name of the optional per-profile file that lists one wake-name per line.
WAKE_NAMES_FILENAME = "wake_names.txt"


class GateState(Enum):
    """States of the welcome gate."""

    WAITING = auto()
    GATED = auto()


def _levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    # Keep only two rows to save memory.
    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr[0] = i
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,  # insertion
                prev[j] + 1,  # deletion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[len_b]


class WelcomeGate:
    """State machine that gates audio until the persona wake-name is heard.

    Parameters
    ----------
    persona_names:
        List of acceptable wake-names (case-insensitive).  Each name is
        checked as a substring of the transcript AND via Levenshtein distance
        against each word / word-pair in the transcript.
    threshold:
        Maximum edit distance to accept as a fuzzy match.  Default is 2,
        which handles one-character mishearings ("rickless" -> "rickles").

    """

    def __init__(self, persona_names: list[str], threshold: int = 2) -> None:
        """Initialise the gate with a list of acceptable wake-names."""
        self._names: list[str] = [n.strip().lower() for n in persona_names if n.strip()]
        if not self._names:
            raise ValueError("persona_names must not be empty")
        self._threshold = threshold
        self.state: GateState = GateState.WAITING

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consider(self, transcript: str) -> bool:
        """Feed a completed transcript; return True if the gate opens.

        If the gate is already GATED this always returns True without doing
        additional work.

        Parameters
        ----------
        transcript:
            The raw text from the STT engine.

        """
        if self.state is GateState.GATED:
            return True

        lowered = transcript.strip().lower()
        matched_name = self._match(lowered)
        if matched_name is not None:
            self.state = GateState.GATED
            logger.info("welcome gate: opened by %r", matched_name)
            return True

        return False

    def reset(self) -> None:
        """Return the gate to WAITING state (e.g. on app restart)."""
        self.state = GateState.WAITING

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_punctuation(word: str) -> str:
        """Strip leading and trailing non-alphanumeric characters from a word."""
        return word.strip("!?,.:;\"'()-[]{}…")

    def _match(self, lowered_transcript: str) -> str | None:
        """Return the matched name string or None.

        Matching strategy (in order):
        1. Exact substring of the whole transcript (fast path).
        2. Fuzzy Levenshtein match against each word (or word n-gram for
           multi-word names) in the transcript, after stripping punctuation.
        """
        for name in self._names:
            # Fast path: exact substring (handles names embedded in words like
            # "rickles" inside "hey rickles, how are you?").
            if name in lowered_transcript:
                return name

            # Fuzzy path: compare the name against every n-gram of words in
            # the transcript where n == number of words in the name.
            # Strip punctuation from each word before comparing so that
            # trailing "!" or "," do not inflate the edit distance.
            name_words = name.split()
            n = len(name_words)
            raw_words = lowered_transcript.split()
            transcript_words = [self._strip_punctuation(w) for w in raw_words]

            if n == 1:
                # Single-word name: compare against each transcript word.
                for word in transcript_words:
                    if word and _levenshtein(name, word) <= self._threshold:
                        return name
            else:
                # Multi-word name: compare against each sliding n-gram.
                for i in range(len(transcript_words) - n + 1):
                    ngram = " ".join(transcript_words[i : i + n])
                    if _levenshtein(name, ngram) <= self._threshold:
                        return name

        return None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _names_from_profile_dir(profile_dir: Path) -> list[str]:
    """Derive wake-names from a profile directory.

    Priority order:
    1. ``wake_names.txt`` — explicit list, one name per non-empty line.
    2. Directory stem split on underscores — e.g. "house_comedian" → ["comedian",
       "house comedian"].  The last word alone is added first (most likely to be
       said in isolation), followed by the full space-joined name.
    """
    # 1. Explicit wake_names.txt overrides everything.
    wake_file = profile_dir / WAKE_NAMES_FILENAME
    if wake_file.is_file():
        try:
            lines = wake_file.read_text(encoding="utf-8").splitlines()
            explicit = [ln.strip() for ln in lines if ln.strip()]
            if explicit:
                return explicit
        except Exception:
            pass

    # 2. Derive from directory name.
    stem = profile_dir.name
    parts = [p for p in stem.split("_") if p]
    if not parts:
        return [stem]

    derived: list[str] = []
    # Last word (most likely spoken alone: "rickles", "carlin", …)
    if len(parts) > 1:
        derived.append(parts[-1])
    # Full name joined with spaces: "don rickles"
    derived.append(" ".join(parts))
    return derived


def make_gate_for_profile(profile_dir: Path, threshold: int = 2) -> WelcomeGate:
    """Create a :class:`WelcomeGate` pre-loaded with names for *profile_dir*.

    Parameters
    ----------
    profile_dir:
        Path to the profile directory (e.g. ``profiles/house_comedian``).
    threshold:
        Levenshtein threshold passed to :class:`WelcomeGate`.

    """
    names = _names_from_profile_dir(profile_dir)
    logger.debug("welcome gate: wake-names for %s = %r", profile_dir.name, names)
    return WelcomeGate(names, threshold=threshold)
