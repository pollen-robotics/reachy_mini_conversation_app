"""Startup configuration for the Reachy Mini audio processor."""

from __future__ import annotations
import logging
from collections.abc import Sequence


AudioControlValue = float | int
AudioStartupParameter = tuple[str, tuple[AudioControlValue, ...]]
WRITE_SETTLE_SECONDS = 0.1

AUDIO_STARTUP_CONFIG: tuple[AudioStartupParameter, ...] = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


def apply_audio_startup_config(
    robot: object,
    *,
    logger: logging.Logger | None = None,
    verify: bool = True,
    write_settle_seconds: float = WRITE_SETTLE_SECONDS,
) -> bool:
    """Apply the tuned XVF3800 audio configuration for the conversation app."""
    log = logger or logging.getLogger(__name__)
    audio = getattr(getattr(robot, "media", None), "audio", None)

    if audio is None:
        log.warning("Skipping Reachy audio startup config: robot media audio is unavailable.")
        return False

    apply_audio_config = getattr(audio, "apply_audio_config", None)
    if not callable(apply_audio_config):
        log.warning("Skipping Reachy audio startup config: SDK audio config API is unavailable.")
        return False

    try:
        applied = bool(
            apply_audio_config(
                AUDIO_STARTUP_CONFIG,
                verify=verify,
                write_settle_seconds=write_settle_seconds,
            )
        )
    except Exception as exc:
        log.warning("Skipping Reachy audio startup config: SDK audio config failed: %s", exc)
        return False

    if applied:
        log.info("Applied Reachy audio startup config: %s", _format_config(AUDIO_STARTUP_CONFIG))
    else:
        log.warning("Reachy audio startup config was not applied.")

    return applied


def _format_config(config: Sequence[AudioStartupParameter]) -> str:
    return ", ".join(f"{name}={' '.join(str(value) for value in values)}" for name, values in config)
