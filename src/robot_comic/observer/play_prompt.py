"""Tier-1.5 closing-the-loop actuator: play a trigger prompt at the robot.

The autonomous *trigger* half of the loop. It plays audio from the
coding-laptop speaker so the robot's microphone hears it — exactly the
``docs/references/audio-playback-recipe.md`` path — letting an agent exercise
hear-and-respond behaviours without a human talking. Pair it with the Tier-0
event log and the Tier-1 audio witness to assert the robot actually reacted.

Two input modes (provide exactly one):

* ``wav`` — play an existing WAV file.
* ``text`` — synthesize speech with piper, then play it (so the robot's STT can
  recognise the words, e.g. ``"Hey Richie, tell me a joke"``).

Because this makes the robot move and speak, it is guarded (see
``docs/closing-the-loop.md`` → *Tradeoffs & guardrails*):

* **Opt-in per session** — refuses unless ``ROBOT_PLAY_PROMPT_ENABLED`` is
  truthy, so it can never fire from a stale config.
* **Duration cap** — refuses clips longer than ``ROBOT_PLAY_PROMPT_MAX_DURATION_S``.
* **Rate limited** — a ``ROBOT_PLAY_PROMPT_COOLDOWN_S`` minimum gap between plays
  and a ``ROBOT_PLAY_PROMPT_MAX_PLAYS`` hard cap per process, so a runaway agent
  loop stops itself rather than blasting the room.

The policy logic is pure and unit-tested; ``sounddevice``/``numpy`` (playback)
and ``piper`` (TTS) are imported lazily so this module — and the MCP server that
exposes it — stay importable without an audio stack. Install the extras where
the speaker lives (native Windows recommended)::

    uv pip install sounddevice numpy piper-tts
"""

from __future__ import annotations
import os
import time
import wave
from typing import Any, Mapping, Callable, Optional


# Opt-in flag: the actuator refuses to make sound unless this is truthy.
ENABLED_ENV = "ROBOT_PLAY_PROMPT_ENABLED"
# Reject clips longer than this (seconds) — a stray long file shouldn't play.
MAX_DURATION_S = float(os.getenv("ROBOT_PLAY_PROMPT_MAX_DURATION_S", "15"))
# Minimum gap between plays (seconds) and hard per-process cap on play count.
COOLDOWN_S = float(os.getenv("ROBOT_PLAY_PROMPT_COOLDOWN_S", "3"))
MAX_PLAYS = int(os.getenv("ROBOT_PLAY_PROMPT_MAX_PLAYS", "20"))
# piper voice model (.onnx); required for text mode. No default — must be set.
PIPER_MODEL_ENV = "ROBOT_PLAY_PROMPT_PIPER_MODEL"

_TRUTHY = {"1", "true", "yes", "on"}


class PlayPromptError(RuntimeError):
    """A play was refused (disabled, rate-limited, too long, bad input)."""


def is_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the opt-in env flag is set to a truthy value."""
    env = os.environ if env is None else env
    return env.get(ENABLED_ENV, "").strip().lower() in _TRUTHY


class RateLimiter:
    """Cooldown + per-session play cap. Pure, with an injectable monotonic clock.

    State lives in the process, so the cap resets only on server restart — which
    is the intended "stop condition" for an unattended loop.
    """

    def __init__(self, *, cooldown_s: float = COOLDOWN_S, max_plays: int = MAX_PLAYS) -> None:
        """Configure the minimum gap (``cooldown_s``) and hard cap (``max_plays``)."""
        self.cooldown_s = cooldown_s
        self.max_plays = max_plays
        self._count = 0
        self._last_ts: Optional[float] = None

    def check(self, now: float) -> None:
        """Raise ``PlayPromptError`` if a play at ``now`` would break a limit."""
        if self._count >= self.max_plays:
            raise PlayPromptError(
                f"per-session cap reached ({self.max_plays} plays) — restart the "
                "server to reset; this is the stop condition for unattended loops"
            )
        if self._last_ts is not None and (now - self._last_ts) < self.cooldown_s:
            wait = self.cooldown_s - (now - self._last_ts)
            raise PlayPromptError(f"cooldown active — wait {wait:.1f}s (min {self.cooldown_s}s between plays)")

    def record(self, now: float) -> None:
        """Mark a successful play at ``now`` against the cooldown and the cap."""
        self._last_ts = now
        self._count += 1

    @property
    def plays_remaining(self) -> int:
        """Plays left before the per-session cap refuses further actuation."""
        return max(0, self.max_plays - self._count)


# Process-wide limiter shared by every call that doesn't inject its own (the MCP
# server is one long-lived process, so the cap spans the whole agent session).
_LIMITER = RateLimiter()


def wav_duration_s(path: str) -> float:
    """Duration of a WAV file in seconds, via the stdlib ``wave`` module only.

    Dependency-free so the duration guard is unit-testable without an audio
    stack. Returns ``0.0`` for a malformed/zero-rate file (the caller treats a
    non-positive duration as a refusal).
    """
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
    except (wave.Error, OSError):
        return 0.0
    return frames / float(rate) if rate else 0.0


def synthesize_piper(text: str, *, model: Optional[str] = None) -> str:
    """Synthesize ``text`` to a temp WAV with piper; return its path.

    Lazy-imports ``piper`` (``uv pip install piper-tts``). The caller is
    responsible for deleting the returned temp file once played.
    """
    import tempfile

    from piper import PiperVoice

    model = (model if model is not None else os.getenv(PIPER_MODEL_ENV, "")).strip()
    if not model:
        raise PlayPromptError(f"text mode needs a piper voice model — set {PIPER_MODEL_ENV} to a .onnx voice path")
    voice = PiperVoice.load(model)
    fd, out_path = tempfile.mkstemp(prefix="robot_play_prompt_", suffix=".wav")
    os.close(fd)
    with wave.open(out_path, "wb") as wf:
        # synthesize_wav writes the WAV header + samples; plain synthesize() only
        # yields AudioChunks and leaves the file headerless (piper-tts >= 1.3).
        voice.synthesize_wav(text, wf)
    return out_path


def play_wav(path: str, *, blocking: bool = True) -> None:
    """Play a WAV file through the default output device. Lazy audio imports.

    Decodes with the stdlib ``wave`` module + ``numpy`` and plays via
    ``sounddevice`` (``uv pip install sounddevice numpy``). Not unit-tested —
    needs real hardware; tests inject a fake player instead.
    """
    import numpy as np
    import sounddevice as sd

    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise PlayPromptError(f"unsupported WAV sample width: {width} bytes")
    data = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        data = data.reshape(-1, channels)
    sd.play(data, rate)
    if blocking:
        sd.wait()


def play_prompt(
    *,
    wav: Optional[str] = None,
    text: Optional[str] = None,
    max_duration_s: float = MAX_DURATION_S,
    env: Optional[Mapping[str, str]] = None,
    limiter: Optional[RateLimiter] = None,
    clock: Optional[Callable[[], float]] = None,
    synthesizer: Optional[Callable[[str], str]] = None,
    player: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Play a trigger prompt at the robot, enforcing the safety guards.

    Provide exactly one of ``wav`` (a file path) or ``text`` (synthesized via
    ``synthesizer``). Returns a summary dict on success. Raises
    ``PlayPromptError`` on any refusal (disabled, bad input, rate-limited, too
    long). The ``env``/``limiter``/``clock``/``synthesizer``/``player`` seams are
    injectable so the policy is testable without audio hardware.
    """
    env = os.environ if env is None else env
    limiter = _LIMITER if limiter is None else limiter
    clock = time.monotonic if clock is None else clock
    synthesizer = synthesize_piper if synthesizer is None else synthesizer
    player = play_wav if player is None else player

    if not is_enabled(env):
        raise PlayPromptError(f"actuator disabled — set {ENABLED_ENV}=1 to opt in for this session")
    if (wav is None) == (text is None):
        raise PlayPromptError("provide exactly one of `wav` or `text`")

    now = clock()
    limiter.check(now)  # refuse before doing any work (synthesis/playback)

    synthesized = False
    if wav is not None:
        source = "wav"
        wav_path = wav
        if not os.path.isfile(wav_path):
            raise PlayPromptError(f"wav not found: {wav_path}")
    else:
        assert text is not None  # narrowed by the exactly-one check above
        source = "text"
        wav_path = synthesizer(text)
        synthesized = True

    try:
        duration_s = wav_duration_s(wav_path)
        if duration_s <= 0:
            raise PlayPromptError(f"clip has zero or unreadable duration: {wav_path}")
        if duration_s > max_duration_s:
            raise PlayPromptError(f"clip {duration_s:.1f}s exceeds max {max_duration_s:.1f}s")
        player(wav_path)
        limiter.record(now)
    finally:
        if synthesized:
            # Don't keep synthesized audio around (privacy: keep room audio local
            # and short-lived). A user-supplied `wav` is left untouched.
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    return {
        "played": True,
        "source": source,
        "duration_s": round(duration_s, 3),
        "plays_remaining": limiter.plays_remaining,
    }
