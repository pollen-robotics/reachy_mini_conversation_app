"""Startup configuration for the Reachy Mini audio processor."""

from __future__ import annotations
import time
import struct
import logging
from typing import Protocol
from collections.abc import Callable, Sequence

from reachy_mini.media.audio_control_utils import PARAMETERS, init_respeaker_usb


AudioControlValue = float | int
AudioStartupParameter = tuple[str, tuple[AudioControlValue, ...]]
WRITE_SETTLE_SECONDS = 0.1
VERIFY_TOLERANCE = 1e-3

AUDIO_STARTUP_CONFIG: tuple[AudioStartupParameter, ...] = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


class ReSpeakerControl(Protocol):
    """Minimal interface used to configure the XVF3800 audio processor."""

    def write(self, name: str, data_list: Sequence[AudioControlValue]) -> None:
        """Write an XVF3800 parameter."""

    def read(self, name: str) -> object:
        """Read an XVF3800 parameter."""


def apply_audio_startup_config(
    robot: object,
    *,
    logger: logging.Logger | None = None,
    respeaker_factory: Callable[[], ReSpeakerControl | None] = init_respeaker_usb,
    write_settle_seconds: float = WRITE_SETTLE_SECONDS,
) -> bool:
    """Apply the tuned XVF3800 audio configuration for the conversation app."""
    log = logger or logging.getLogger(__name__)
    respeaker = _respeaker_from_robot(robot)
    should_close_respeaker = False

    if respeaker is None:
        log.debug("No existing ReSpeaker control handle on robot media; trying USB discovery.")
        try:
            respeaker = respeaker_factory()
        except Exception as exc:
            log.warning("Skipping Reachy audio startup config: ReSpeaker discovery failed: %s", exc)
            return False
        should_close_respeaker = respeaker is not None

    if respeaker is None:
        log.warning("Skipping Reachy audio startup config: ReSpeaker USB device not found.")
        return False

    failures: list[str] = []
    try:
        for name, values in AUDIO_STARTUP_CONFIG:
            try:
                respeaker.write(name, values)
                if write_settle_seconds > 0:
                    time.sleep(write_settle_seconds)
                actual_values = _read_parameter_values(respeaker, name)
                if not _values_match(actual_values, values):
                    failures.append(f"{name}: expected {_format_values(values)}, got {_format_values(actual_values)}")
                    log.warning(
                        "Audio startup parameter verification failed for %s: expected %s, got %s",
                        name,
                        _format_values(values),
                        _format_values(actual_values),
                    )
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                log.warning("Failed to apply audio startup parameter %s=%s: %s", name, _format_values(values), exc)
    finally:
        if should_close_respeaker:
            _close_respeaker(respeaker, log)

    if failures:
        log.warning("Reachy audio startup config completed with %d failed parameter(s).", len(failures))
        return False

    log.info("Applied and verified Reachy audio startup config: %s", _format_config(AUDIO_STARTUP_CONFIG))
    return True


def _respeaker_from_robot(robot: object) -> ReSpeakerControl | None:
    media = getattr(robot, "media", None)
    audio = getattr(media, "audio", None)
    respeaker = getattr(audio, "_respeaker", None)
    return respeaker


def _close_respeaker(respeaker: ReSpeakerControl, logger: logging.Logger) -> None:
    close = getattr(respeaker, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        logger.debug("Error closing temporary ReSpeaker control handle: %s", exc)


def _format_config(config: tuple[AudioStartupParameter, ...]) -> str:
    return ", ".join(f"{name}={_format_values(values)}" for name, values in config)


def _format_values(values: Sequence[AudioControlValue] | None) -> str:
    if values is None:
        return "unreadable"
    return " ".join(str(value) for value in values)


def _read_parameter_values(respeaker: ReSpeakerControl, name: str) -> tuple[AudioControlValue, ...] | None:
    raw_values = respeaker.read(name)
    parameter = PARAMETERS.get(name)
    if raw_values is None or parameter is None:
        return None

    value_count = int(parameter[2])
    value_type = str(parameter[4])

    if value_type in {"float", "radians"}:
        if not isinstance(raw_values, Sequence):
            return None
        return tuple(float(value) for value in raw_values[:value_count])

    if value_type in {"int32", "uint32"}:
        return _decode_int32_values(raw_values, value_count, signed=value_type == "int32")

    if value_type == "uint8":
        if not isinstance(raw_values, Sequence):
            return None
        offset = 1 if len(raw_values) == value_count + 1 else 0
        return tuple(int(value) for value in raw_values[offset : offset + value_count])

    return None


def _decode_int32_values(raw_values: object, value_count: int, *, signed: bool) -> tuple[int, ...] | None:
    if not isinstance(raw_values, Sequence):
        return None

    if len(raw_values) == value_count * 4 + 1:
        payload = bytes(int(value) & 0xFF for value in raw_values[1:])
        format_char = "i" if signed else "I"
        return tuple(int(value) for value in struct.unpack("<" + format_char * value_count, payload))

    if len(raw_values) >= value_count:
        return tuple(int(value) for value in raw_values[:value_count])

    return None


def _values_match(
    actual_values: Sequence[AudioControlValue] | None,
    expected_values: Sequence[AudioControlValue],
) -> bool:
    if actual_values is None or len(actual_values) != len(expected_values):
        return False

    return all(
        abs(float(actual) - float(expected)) <= VERIFY_TOLERANCE
        for actual, expected in zip(actual_values, expected_values)
    )
