"""Tests for Reachy Mini audio startup configuration."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock

from reachy_mini_conversation_app.audio.startup_config import (
    PARAMETERS,
    AUDIO_STARTUP_CONFIG,
    apply_audio_startup_config,
)


class FakeReSpeaker:
    """Fake XVF3800 control surface."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        """Initialize the fake control surface."""
        self.fail_on = fail_on
        self.closed = False
        self.writes: list[tuple[str, tuple[float | int, ...]]] = []
        self.values: dict[str, tuple[float | int, ...]] = {}

    def write(self, name: str, data_list: tuple[float | int, ...]) -> None:
        """Record writes and optionally fail one parameter."""
        if name == self.fail_on:
            raise RuntimeError("write failed")
        values = tuple(data_list)
        self.writes.append((name, values))
        self.values[name] = values

    def read(self, name: str) -> tuple[float | int, ...] | list[int] | None:
        """Return the last written value in the SDK's readback shape."""
        values = self.values.get(name)
        if values is None:
            return None
        if PARAMETERS[name][4] == "int32":
            int_value = int(values[0])
            return [0, *int_value.to_bytes(4, byteorder="little", signed=True)]
        return values

    def close(self) -> None:
        """Record that the temporary control handle was closed."""
        self.closed = True


def test_apply_audio_startup_config_uses_existing_robot_respeaker() -> None:
    """Existing media ReSpeaker handle should be reused instead of reopening USB."""
    respeaker = FakeReSpeaker()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_respeaker=respeaker)))
    factory = MagicMock()

    applied = apply_audio_startup_config(robot, respeaker_factory=factory)

    assert applied is True
    assert respeaker.writes == list(AUDIO_STARTUP_CONFIG)
    assert respeaker.closed is False
    factory.assert_not_called()


def test_apply_audio_startup_config_falls_back_to_usb_discovery() -> None:
    """USB discovery is used when the robot media object does not expose a ReSpeaker."""
    respeaker = FakeReSpeaker()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_respeaker=None)))

    applied = apply_audio_startup_config(robot, respeaker_factory=lambda: respeaker)

    assert applied is True
    assert respeaker.writes == list(AUDIO_STARTUP_CONFIG)
    assert respeaker.closed is True


def test_apply_audio_startup_config_returns_false_without_device() -> None:
    """Startup should continue when the audio control device is unavailable."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=None))

    applied = apply_audio_startup_config(robot, respeaker_factory=lambda: None)

    assert applied is False


def test_apply_audio_startup_config_continues_after_write_failure() -> None:
    """A single failed parameter should not prevent later startup writes."""
    respeaker = FakeReSpeaker(fail_on="PP_MIN_NS")
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_respeaker=respeaker)))

    applied = apply_audio_startup_config(robot)

    assert applied is False
    assert ("PP_MIN_NS", (0.8,)) not in respeaker.writes
    assert respeaker.writes == [item for item in AUDIO_STARTUP_CONFIG if item[0] != "PP_MIN_NS"]


def test_apply_audio_startup_config_returns_false_when_readback_does_not_match() -> None:
    """A write that does not stick should be reported as a failed application."""

    class NonPersistingReSpeaker(FakeReSpeaker):
        def read(self, name: str) -> tuple[float | int, ...] | list[int] | None:
            if name == "PP_MGSCALE":
                return (1000.0, 1.0, 1.0)
            return super().read(name)

    respeaker = NonPersistingReSpeaker()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_respeaker=respeaker)))

    applied = apply_audio_startup_config(robot, write_settle_seconds=0)

    assert applied is False
    assert ("PP_MGSCALE", (4.0, 1.0, 1.0)) in respeaker.writes
