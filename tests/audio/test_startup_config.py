"""Tests for Reachy Mini audio startup configuration."""

from __future__ import annotations
from types import SimpleNamespace

from reachy_mini_conversation_app.audio.startup_config import (
    AUDIO_STARTUP_CONFIG,
    WRITE_SETTLE_SECONDS,
    apply_audio_startup_config,
)


class FakeAudio:
    """Fake SDK audio wrapper."""

    def __init__(self, *, result: bool = True, error: Exception | None = None) -> None:
        """Initialize the fake audio wrapper."""
        self.result = result
        self.error = error
        self.calls: list[tuple[object, bool, float]] = []

    def apply_audio_config(
        self,
        config: object,
        *,
        verify: bool = True,
        write_settle_seconds: float = WRITE_SETTLE_SECONDS,
    ) -> bool:
        """Record SDK audio config calls."""
        if self.error is not None:
            raise self.error
        self.calls.append((config, verify, write_settle_seconds))
        return self.result


def test_apply_audio_startup_config_uses_sdk_audio_config_api() -> None:
    """Startup config should delegate writes and verification to the SDK audio API."""
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot)

    assert applied is True
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, True, WRITE_SETTLE_SECONDS)]


def test_apply_audio_startup_config_forwards_sdk_options() -> None:
    """SDK verification options should stay configurable for tests and callers."""
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot, verify=False, write_settle_seconds=0)

    assert applied is True
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, False, 0)]


def test_apply_audio_startup_config_returns_false_without_audio() -> None:
    """Startup should continue when the SDK audio object is unavailable."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=None))

    applied = apply_audio_startup_config(robot)

    assert applied is False


def test_apply_audio_startup_config_returns_false_without_sdk_api() -> None:
    """Startup should continue when the installed SDK does not expose audio config helpers."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=object()))

    applied = apply_audio_startup_config(robot)

    assert applied is False


def test_apply_audio_startup_config_returns_false_when_sdk_returns_false() -> None:
    """SDK application failures should be reported without raising."""
    audio = FakeAudio(result=False)
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot)

    assert applied is False
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, True, WRITE_SETTLE_SECONDS)]


def test_apply_audio_startup_config_returns_false_when_sdk_raises() -> None:
    """Unexpected SDK audio config errors should not prevent app startup."""
    audio = FakeAudio(error=RuntimeError("audio board unavailable"))
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot)

    assert applied is False
    assert audio.calls == []
