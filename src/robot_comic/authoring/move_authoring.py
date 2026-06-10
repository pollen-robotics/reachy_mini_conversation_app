"""Author reachy_mini ``RecordedMove`` JSON from a declarative keyframe spec (#538).

A :class:`KeyframeSpec` is a small, human-authorable description of a custom
gesture: a list of :class:`Keyframe` waypoints (time + head pose as
roll/pitch/yaw degrees and x/y/z metres + antennas + body_yaw) plus a sample
rate (``fps``) and an easing curve.  :func:`build_recorded_move` resamples the
spec onto a uniform time grid, interpolating each segment with the chosen
easing, and emits the exact JSON shape consumed by
``reachy_mini.motion.recorded_move.RecordedMove``::

    {
      "description": str,
      "time": [t0, t1, ...],                       # seconds, monotonic, t0 == 0
      "set_target_data": [
        {"head": <4x4 list>, "antennas": [l, r], "body_yaw": float},
        ...
      ]
    }

Conventions match ``reachy_mini.utils.create_head_pose`` and the canonical
gestures in ``robot_comic.gestures.gestures``:

- Head orientation is built from extrinsic XYZ Euler angles (roll, pitch, yaw)
  in **degrees**.
- yaw > 0 turns left; pitch > 0 looks down; roll > 0 tilts the right ear down.
- Head translation (x, y, z) is in **metres** in the world frame.
- Antennas are a ``(left, right)`` pair in **radians**; body_yaw is in radians.

Heavy numeric imports (numpy/scipy) are deferred to call time per the codebase's
import-budget convention; the module is cheap to import.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from dataclasses import field, dataclass


if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray


# Easing curves shared with reachy_mini's interpolation utilities.  We mirror
# the names from ``reachy_mini.utils.interpolation.InterpolationTechnique`` so an
# author can use the same vocabulary the SDK uses for gotos.
_VALID_EASINGS = frozenset({"linear", "minjerk", "ease_in_out", "cartoon"})

# Default authoring sample rate.  The canonical recorded-move datasets are
# recorded at ~100 Hz; matching that keeps the screen's finite-difference
# velocity/accel/jerk estimates on the same footing as the presets.
DEFAULT_FPS = 100.0


@dataclass
class Keyframe:
    """A single authoring waypoint.

    All fields default to the neutral pose so an author only specifies the axes
    that move.  Angles are degrees; translation is metres; antennas/body_yaw are
    radians.
    """

    t: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    antennas: Tuple[float, float] = (0.0, 0.0)
    body_yaw: float = 0.0


@dataclass
class KeyframeSpec:
    """A declarative custom-move spec: ordered keyframes + sample rate + easing."""

    name: str
    description: str
    keyframes: List[Keyframe] = field(default_factory=list)
    fps: float = DEFAULT_FPS
    easing: str = "minjerk"

    def validate(self) -> None:
        """Raise :class:`ValueError` if the spec cannot be built into a move."""
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if self.easing not in _VALID_EASINGS:
            raise ValueError(f"unknown easing {self.easing!r}; valid: {sorted(_VALID_EASINGS)}")
        if len(self.keyframes) < 2:
            raise ValueError("a KeyframeSpec needs at least two keyframes")
        if self.keyframes[0].t != 0.0:
            raise ValueError(f"first keyframe must start at t=0.0, got t={self.keyframes[0].t!r}")
        for prev, cur in zip(self.keyframes, self.keyframes[1:]):
            if cur.t <= prev.t:
                raise ValueError(f"keyframe times must be strictly increasing; got {prev.t!r} followed by {cur.t!r}")

    @property
    def duration(self) -> float:
        """Total move duration in seconds (time of the last keyframe)."""
        return float(self.keyframes[-1].t)


def _ease(alpha: float, method: str) -> float:
    """Map a normalised segment parameter ``alpha in [0, 1]`` through an easing curve.

    Mirrors ``reachy_mini.utils.interpolation.time_trajectory`` but is local so
    authoring stays import-light and does not depend on SDK internals beyond the
    documented curve shapes.
    """
    a = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
    if method == "linear":
        return a
    if method == "minjerk":
        return 10 * a**3 - 15 * a**4 + 6 * a**5
    if method == "ease_in_out":
        if a < 0.5:
            return 2 * a * a
        return 1 - ((-2 * a + 2) ** 2) / 2
    if method == "cartoon":
        c1 = 1.70158
        c2 = c1 * 1.525
        if a < 0.5:
            return ((2 * a) ** 2 * ((c2 + 1) * 2 * a - c2)) / 2
        return (((2 * a - 2) ** 2 * ((c2 + 1) * (2 * a - 2) + c2)) + 2) / 2
    # validate() guards this, but be defensive.
    raise ValueError(f"unknown easing {method!r}")


def _sample_times(duration: float, fps: float) -> List[float]:
    """Uniform sample grid on ``[0, duration]`` inclusive of both endpoints."""
    n_intervals = max(1, round(duration * fps))
    return [i * duration / n_intervals for i in range(n_intervals + 1)]


def _segment_for(t: float, keyframes: List[Keyframe]) -> Tuple[Keyframe, Keyframe, float]:
    """Return ``(kf_lo, kf_hi, alpha)`` bracketing time ``t`` (alpha in [0, 1])."""
    # Clamp into range; times outside are pinned to the endpoints.
    if t <= keyframes[0].t:
        return keyframes[0], keyframes[1], 0.0
    if t >= keyframes[-1].t:
        return keyframes[-2], keyframes[-1], 1.0
    for lo, hi in zip(keyframes, keyframes[1:]):
        if lo.t <= t <= hi.t:
            span = hi.t - lo.t
            alpha = 0.0 if span <= 0.0 else (t - lo.t) / span
            return lo, hi, alpha
    # Unreachable given the guards above.
    return keyframes[-2], keyframes[-1], 1.0


def _head_matrix(roll: float, pitch: float, yaw: float, x: float, y: float, z: float) -> "NDArray[np.float64]":
    """Build a 4x4 head pose from RPY (degrees) + XYZ (metres)."""
    import numpy as np

    from reachy_mini.utils import create_head_pose

    pose = create_head_pose(x, y, z, roll, pitch, yaw, degrees=True)
    return np.asarray(pose, dtype=np.float64)


def build_recorded_move(spec: KeyframeSpec) -> Dict[str, Any]:
    """Turn a :class:`KeyframeSpec` into ``RecordedMove`` JSON (plain dict).

    The returned dict is JSON-serialisable and round-trips through
    ``reachy_mini.motion.recorded_move.RecordedMove``.

    Parameters
    ----------
    spec:
        The declarative keyframe spec.  Validated before sampling.

    Returns
    -------
    dict
        ``{"description", "time", "set_target_data"}`` ready for ``json.dump``.

    """
    spec.validate()

    times = _sample_times(spec.duration, spec.fps)
    set_target_data: List[Dict[str, Any]] = []

    for t in times:
        lo, hi, alpha = _segment_for(t, spec.keyframes)
        blend = _ease(alpha, spec.easing)

        roll = lo.roll + (hi.roll - lo.roll) * blend
        pitch = lo.pitch + (hi.pitch - lo.pitch) * blend
        yaw = lo.yaw + (hi.yaw - lo.yaw) * blend
        x = lo.x + (hi.x - lo.x) * blend
        y = lo.y + (hi.y - lo.y) * blend
        z = lo.z + (hi.z - lo.z) * blend
        ant_l = lo.antennas[0] + (hi.antennas[0] - lo.antennas[0]) * blend
        ant_r = lo.antennas[1] + (hi.antennas[1] - lo.antennas[1]) * blend
        body_yaw = lo.body_yaw + (hi.body_yaw - lo.body_yaw) * blend

        head = _head_matrix(roll, pitch, yaw, x, y, z)
        set_target_data.append(
            {
                "head": head.tolist(),
                "antennas": [float(ant_l), float(ant_r)],
                "body_yaw": float(body_yaw),
            }
        )

    return {
        "description": spec.description,
        "time": [float(t) for t in times],
        "set_target_data": set_target_data,
    }
