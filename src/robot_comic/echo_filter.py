"""Shared self-echo detection helpers (#527).

The normalize step is extracted from ``composable_pipeline`` (Phase 5f.3, the
content-similarity echo filter) so realtime handlers can run the same check on
their input-transcription paths. The fragment matcher is new for the realtime
case: Gemini Live transcribes a *slice* of the robot's own riff back as user
input ("you know I feel like I should just call the…"), and a plain
``SequenceMatcher.ratio()`` of a short fragment against the full riff text is
low even for an exact substring — so candidates are scanned window-wise.
"""

from __future__ import annotations
import re
from typing import Iterable
from difflib import SequenceMatcher


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Ultra-short utterances ("ok", "ja") appear inside almost any riff; treating
# them as echoes would swallow real user back-channel speech. Mirrors the
# Phase 5f.2 minimum-utterance constants.
MIN_ECHO_WORDS = 3
MIN_ECHO_CHARS = 12


def normalize_for_echo_check(text: str) -> str:
    """Return a comparison-only form of ``text`` for echo similarity scoring.

    Lowercases, replaces every non-word non-whitespace character with a
    single space (so punctuation, apostrophes, ellipses all collapse),
    and squashes whitespace runs. The output is used only for similarity
    ratios — the original transcript string is what gets dispatched.
    """
    if not text:
        return ""
    lowered = text.lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", no_punct)
    return collapsed.strip()


def fragment_echo_ratio(needle: str, candidate: str) -> float:
    """Best similarity of ``needle`` against same-length windows of ``candidate``.

    Both arguments must already be normalized. Exact containment
    short-circuits to 1.0; otherwise a window of ``len(needle)`` (plus slack)
    slides over ``candidate`` and the best ``SequenceMatcher`` ratio wins.
    Texts are short (transcript-sized), so the scan cost is negligible at
    chunk cadence.
    """
    if not needle or not candidate:
        return 0.0
    if needle in candidate:
        return 1.0
    n = len(needle)
    if n >= len(candidate):
        return SequenceMatcher(None, needle, candidate).ratio()
    step = max(1, n // 4)
    best = 0.0
    for start in range(0, len(candidate) - n + 1, step):
        window = candidate[start : start + n + step]
        matcher = SequenceMatcher(None, needle, window)
        if matcher.real_quick_ratio() <= best:
            continue
        best = max(best, matcher.ratio())
    return best


def is_echo_of_recent(text: str, recent_normalized: Iterable[str], *, threshold: float) -> bool:
    """Return True when ``text`` looks like an echo of a recent assistant utterance.

    ``recent_normalized`` holds already-normalized assistant texts (newest or
    oldest first — order is irrelevant). Inputs shorter than the minimum
    utterance bounds never flag, so genuine back-channel speech ("ok", "ja")
    always passes through to reset pacing.
    """
    needle = normalize_for_echo_check(text)
    if not needle or len(needle) < MIN_ECHO_CHARS or len(needle.split()) < MIN_ECHO_WORDS:
        return False
    return any(fragment_echo_ratio(needle, candidate) >= threshold for candidate in recent_normalized if candidate)
