"""Pre-processing layer that translates Gemini Live delivery tags to Chatterbox TTS parameters.

Gemini Live uses inline prosody tags ([fast], [slow], [annoyance], etc.) to shape delivery.
Chatterbox has no equivalent prosody tags — instead it exposes two generation parameters:
  - exaggeration (float 0-2): controls expressiveness; 0=flat, 1=normal, 2=dramatic
  - cfg_weight   (float 0-1): controls adherence to reference; high=literal, low=dynamic

Chatterbox Turbo also supports paralinguistic sound-event tags ([chuckle], [laugh], etc.)
which are not prosody but actual audio events.

Usage (Gemini pipeline — tags pass through unchanged):
    Don't use this module. Tags are passed directly to Gemini Live.

Usage (local Chatterbox pipeline):
    segments = translate(llm_output, persona="house_comedian", use_turbo=True)
    for seg in segments:
        if seg.silence_ms:
            audio_chunks.append(make_silence(seg.silence_ms))
        else:
            text = (seg.turbo_insert + " " + seg.text) if seg.turbo_insert else seg.text
            audio_chunks.append(
                chatterbox.generate(text, exaggeration=seg.exaggeration, cfg_weight=seg.cfg_weight)
            )
    final_audio = concatenate(audio_chunks)

    # Or just strip tags for a simpler single-call approach:
    clean_text = strip_gemini_tags(llm_output)
    audio = chatterbox.generate(clean_text, **PERSONA_BASELINES["house_comedian"])
"""

from __future__ import annotations
import re
from typing import Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tag → Chatterbox parameter mapping
# ---------------------------------------------------------------------------

# (exaggeration, cfg_weight, turbo_paralinguistic_insert_or_None)
_TAG_PARAMS: dict[str, tuple[float, float, Optional[str]]] = {
    "fast": (1.2, 0.30, None),
    "slow": (0.8, 0.80, None),
    "annoyance": (1.5, 0.40, None),
    "aggression": (1.6, 0.35, None),
    "amusement": (1.3, 0.40, "[chuckle]"),  # Turbo paralinguistic on amusement
    "enthusiasm": (1.4, 0.30, None),
}

SILENCE_PADDING_MS = 400  # ms of silence to insert for [short pause]


# ---------------------------------------------------------------------------
# Per-persona baseline parameters
# ---------------------------------------------------------------------------
# These establish the resting exaggeration level for each comic's voice.
# Tags push exaggeration above/below this absolute value, not relative to it.
# Tune these against actual Chatterbox output with the cloned reference audio.

PERSONA_BASELINES: dict[str, dict[str, float]] = {
    "house_comedian": {"exaggeration": 0.95, "cfg_weight": 0.50},  # measured, present baseline
}

_DEFAULT_BASELINE: dict[str, float] = {"exaggeration": 1.00, "cfg_weight": 0.50}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ChatterboxSegment:
    """One generation unit for the Chatterbox TTS pipeline.

    If silence_ms > 0: insert audio silence of that duration; text will be empty.
    Otherwise: generate speech from text using exaggeration and cfg_weight.
    If turbo_insert is set (Turbo model only): prepend to text before generation.
    """

    text: str
    exaggeration: float
    cfg_weight: float
    turbo_insert: Optional[str] = None
    silence_ms: int = 0


# ---------------------------------------------------------------------------
# Tag patterns
# ---------------------------------------------------------------------------

# Matches [short pause] and any single-word bracketed tag like [fast], [amusement], etc.
_TAG_RE = re.compile(r"(\[short pause\]|\[\w+\])")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate(
    text: str,
    persona: str = "default",
    use_turbo: bool = True,
) -> list[ChatterboxSegment]:
    """Split LLM output at Gemini delivery tag boundaries and map each segment to
    Chatterbox generation parameters.

    Tags come BEFORE the text they apply to (e.g. "[fast] He said take a number.").
    [short pause] produces a silence-only segment with no text.
    Unknown tags fall back to the persona baseline.
    Tags are stripped from the text; Chatterbox never sees them as words.

    Args:
        text:      Raw LLM output containing Gemini delivery tags.
        persona:   Profile directory name (e.g. "house_comedian"). Used to look
                   up per-persona baseline exaggeration/cfg_weight.
        use_turbo: If True, insert Chatterbox Turbo paralinguistic events where
                   appropriate (e.g. [chuckle] on [amusement] segments).

    Returns:
        Ordered list of ChatterboxSegment — one per generation call, plus any
        silence segments for [short pause].

    """
    baseline = PERSONA_BASELINES.get(persona, _DEFAULT_BASELINE)
    base_exag = baseline["exaggeration"]
    base_cfg = baseline["cfg_weight"]

    # Split at tag boundaries, keeping the tags as tokens.
    # Result: [pre_text, tag, text, tag, text, ...]
    parts = _TAG_RE.split(text)

    segments: list[ChatterboxSegment] = []
    prev_exag = base_exag
    prev_cfg = base_cfg

    # parts[0] is any text before the first tag (no associated tag)
    if parts[0].strip():
        segments.append(
            ChatterboxSegment(
                text=parts[0].strip(),
                exaggeration=base_exag,
                cfg_weight=base_cfg,
            )
        )

    # Remaining parts alternate: tag at odd index, text at even index
    i = 1
    while i < len(parts):
        tag = parts[i]
        chunk = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        if tag == "[short pause]":
            # Silence segment — use prev params so the pause "sounds like" the last segment
            segments.append(
                ChatterboxSegment(
                    text="",
                    exaggeration=prev_exag,
                    cfg_weight=prev_cfg,
                    silence_ms=SILENCE_PADDING_MS,
                )
            )
            # Any text following the pause before the next tag uses prev params
            if chunk:
                segments.append(
                    ChatterboxSegment(
                        text=chunk,
                        exaggeration=prev_exag,
                        cfg_weight=prev_cfg,
                    )
                )
        else:
            tag_name = tag[1:-1]  # strip square brackets
            params = _TAG_PARAMS.get(tag_name)
            if params:
                exag, cfg, turbo_insert = params
            else:
                exag, cfg, turbo_insert = base_exag, base_cfg, None

            if chunk:
                insert = turbo_insert if use_turbo else None
                segments.append(
                    ChatterboxSegment(
                        text=chunk,
                        exaggeration=exag,
                        cfg_weight=cfg,
                        turbo_insert=insert,
                    )
                )
                prev_exag, prev_cfg = exag, cfg

    return segments


def strip_gemini_tags(text: str) -> str:
    """Remove all Gemini delivery tags from text.

    Use this for a simple single-call Chatterbox approach where per-segment
    parameter variation is not needed — just clean the text and call generate()
    once with the persona baseline parameters.
    """
    return _TAG_RE.sub("", text).strip()
