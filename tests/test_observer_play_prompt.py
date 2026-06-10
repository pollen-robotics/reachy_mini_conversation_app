"""Tests for Tier-1.5 of the closing-the-loop observer (play-prompt actuator).

Covers the pure safety policy — opt-in gate, exactly-one-input validation, the
duration cap, and the cooldown / per-session-cap rate limiter — plus the
stdlib-only ``wav_duration_s`` helper. The real ``sounddevice`` playback and
``piper`` synthesis need hardware/models, so they're injected as fakes here.
"""

from __future__ import annotations
import wave
from pathlib import Path

import pytest

from robot_comic.observer import play_prompt as pp
from robot_comic.observer.play_prompt import RateLimiter, PlayPromptError, play_prompt


ENABLED = {"ROBOT_PLAY_PROMPT_ENABLED": "1"}


def _write_wav(path: Path, *, duration_s: float, rate: int = 16_000) -> str:
    """Write a silent mono 16-bit WAV of ``duration_s`` seconds; return its path."""
    nframes = int(duration_s * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * nframes)
    return str(path)


class _FakePlayer:
    """Records the paths it was asked to play instead of touching audio hardware."""

    def __init__(self) -> None:
        self.played: list[str] = []

    def __call__(self, path: str) -> None:
        self.played.append(path)


class _Clock:
    """A controllable monotonic clock for deterministic cooldown tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# is_enabled (opt-in gate)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_is_enabled_truthy(value: str) -> None:
    assert pp.is_enabled({pp.ENABLED_ENV: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_is_enabled_falsey(value: str) -> None:
    assert pp.is_enabled({pp.ENABLED_ENV: value}) is False


def test_is_enabled_missing_key() -> None:
    assert pp.is_enabled({}) is False


# ---------------------------------------------------------------------------
# play_prompt input validation + opt-in
# ---------------------------------------------------------------------------
def test_disabled_refuses(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", duration_s=0.2)
    with pytest.raises(PlayPromptError, match="disabled"):
        play_prompt(wav=wav, env={}, player=_FakePlayer())


def test_requires_exactly_one_input_both() -> None:
    with pytest.raises(PlayPromptError, match="exactly one"):
        play_prompt(wav="a.wav", text="hi", env=ENABLED, player=_FakePlayer())


def test_requires_exactly_one_input_neither() -> None:
    with pytest.raises(PlayPromptError, match="exactly one"):
        play_prompt(env=ENABLED, player=_FakePlayer())


def test_wav_not_found_refuses() -> None:
    with pytest.raises(PlayPromptError, match="not found"):
        play_prompt(wav="/no/such/file.wav", env=ENABLED, player=_FakePlayer())


# ---------------------------------------------------------------------------
# wav_duration_s (stdlib only)
# ---------------------------------------------------------------------------
def test_wav_duration_reads_real_wav(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", duration_s=0.5)
    assert pp.wav_duration_s(wav) == pytest.approx(0.5, abs=1e-3)


def test_wav_duration_missing_or_bad(tmp_path: Path) -> None:
    assert pp.wav_duration_s("/no/such.wav") == 0.0
    bad = tmp_path / "bad.wav"
    bad.write_text("not a wav", encoding="utf-8")
    assert pp.wav_duration_s(str(bad)) == 0.0


# ---------------------------------------------------------------------------
# Duration cap
# ---------------------------------------------------------------------------
def test_duration_cap_refuses_long_clip(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "long.wav", duration_s=1.0)
    player = _FakePlayer()
    with pytest.raises(PlayPromptError, match="exceeds max"):
        play_prompt(wav=wav, env=ENABLED, max_duration_s=0.5, player=player)
    assert player.played == [], "a clip over the cap must not be played"


def test_zero_duration_refused(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "empty.wav", duration_s=0.0)
    with pytest.raises(PlayPromptError, match="zero or unreadable"):
        play_prompt(wav=wav, env=ENABLED, player=_FakePlayer())


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
def test_successful_wav_play(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "ok.wav", duration_s=0.4)
    player = _FakePlayer()
    limiter = RateLimiter(cooldown_s=0.0, max_plays=5)
    result = play_prompt(wav=wav, env=ENABLED, player=player, limiter=limiter, clock=_Clock())
    assert result["played"] is True
    assert result["source"] == "wav"
    assert result["duration_s"] == pytest.approx(0.4, abs=1e-3)
    assert result["plays_remaining"] == 4
    assert player.played == [wav]


def test_text_mode_synthesizes_and_cleans_up(tmp_path: Path) -> None:
    synth_wav = _write_wav(tmp_path / "tts.wav", duration_s=0.3)
    player = _FakePlayer()

    def fake_synth(text: str) -> str:
        assert text == "tell me a joke"
        return synth_wav

    result = play_prompt(
        text="tell me a joke",
        env=ENABLED,
        player=player,
        synthesizer=fake_synth,
        limiter=RateLimiter(cooldown_s=0.0),
        clock=_Clock(),
    )
    assert result["source"] == "text"
    assert result["played"] is True
    assert player.played == [synth_wav]
    assert not Path(synth_wav).exists(), "synthesized temp WAV should be cleaned up after playing"


# ---------------------------------------------------------------------------
# RateLimiter (pure)
# ---------------------------------------------------------------------------
def test_rate_limiter_cooldown() -> None:
    rl = RateLimiter(cooldown_s=3.0, max_plays=10)
    rl.check(100.0)
    rl.record(100.0)
    with pytest.raises(PlayPromptError, match="cooldown"):
        rl.check(101.5)  # only 1.5s later
    rl.check(103.0)  # exactly cooldown later -> allowed


def test_rate_limiter_session_cap() -> None:
    rl = RateLimiter(cooldown_s=0.0, max_plays=2)
    rl.check(0.0)
    rl.record(0.0)
    rl.check(1.0)
    rl.record(1.0)
    with pytest.raises(PlayPromptError, match="per-session cap"):
        rl.check(2.0)
    assert rl.plays_remaining == 0


# ---------------------------------------------------------------------------
# Rate limiting through play_prompt
# ---------------------------------------------------------------------------
def test_cooldown_enforced_through_play_prompt(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "c.wav", duration_s=0.2)
    player = _FakePlayer()
    limiter = RateLimiter(cooldown_s=3.0, max_plays=10)
    clock = _Clock(0.0)

    play_prompt(wav=wav, env=ENABLED, player=player, limiter=limiter, clock=clock)
    clock.t = 1.0
    with pytest.raises(PlayPromptError, match="cooldown"):
        play_prompt(wav=wav, env=ENABLED, player=player, limiter=limiter, clock=clock)
    clock.t = 4.0
    play_prompt(wav=wav, env=ENABLED, player=player, limiter=limiter, clock=clock)
    assert player.played == [wav, wav], "only the two non-cooldown plays go through"


def test_session_cap_enforced_through_play_prompt(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "s.wav", duration_s=0.2)
    player = _FakePlayer()
    limiter = RateLimiter(cooldown_s=0.0, max_plays=1)
    play_prompt(wav=wav, env=ENABLED, player=player, limiter=limiter, clock=_Clock())
    with pytest.raises(PlayPromptError, match="per-session cap"):
        play_prompt(wav=wav, env=ENABLED, player=player, limiter=limiter, clock=_Clock(1000.0))
    assert player.played == [wav]
