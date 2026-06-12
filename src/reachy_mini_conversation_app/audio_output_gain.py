"""App-level audio output gain control.

Applies a linear gain multiplier to outgoing audio frames before they reach
the robot speaker.  The value is read per-frame in play_loop() so changes
take effect on the next speech output without a restart.
"""

import os

_DEFAULT_GAIN_DB = 0.0
_MIN_GAIN_DB = 0.0
_MAX_GAIN_DB = 24.0


def db_to_linear(db: float) -> float:
    """Convert a decibel value to a linear multiplier."""
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert a linear multiplier back to decibels."""
    import math

    if linear <= 0:
        return _MIN_GAIN_DB
    return 20.0 * math.log10(linear)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_initial_gain_db() -> float:
    raw = os.environ.get("FAMILIAR_AUDIO_GAIN_DB", "")
    if raw.strip():
        try:
            return _clamp(float(raw), _MIN_GAIN_DB, _MAX_GAIN_DB)
        except (ValueError, TypeError):
            pass
    return _DEFAULT_GAIN_DB


_gain_db: float = _load_initial_gain_db()
_gain_linear: float = db_to_linear(_gain_db)


def get_gain_db() -> float:
    """Return the current output gain in decibels."""
    return _gain_db


def get_gain_linear() -> float:
    """Return the current output gain as a linear multiplier."""
    return _gain_linear


def set_gain_db(db: float) -> None:
    """Set the output gain (clamped to 0–24 dB). Updates take effect next frame."""
    global _gain_db, _gain_linear
    _gain_db = _clamp(db, _MIN_GAIN_DB, _MAX_GAIN_DB)
    _gain_linear = db_to_linear(_gain_db)


def reload_from_env() -> None:
    """Re-read FAMILIAR_AUDIO_GAIN_DB from environment (called after .env refresh)."""
    set_gain_db(_load_initial_gain_db())
