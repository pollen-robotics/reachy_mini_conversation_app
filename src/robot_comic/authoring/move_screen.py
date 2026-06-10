"""Offline safety pre-screen for a candidate ``RecordedMove`` (#538).

Before a hand-authored move ever reaches the robot, this screen checks it on the
workstation — no daemon, no SDK I/O beyond the (optional) analytical kinematics
engine.  A move that fails is rejected with one or more named reasons; a gentle
move passes clean.

What the screen checks
----------------------
1. **Structural validity** — the JSON has aligned ``time`` / ``set_target_data``
   arrays, ≥2 frames, and well-formed 4x4 head matrices.
2. **Kinematic envelope** — every frame's head RPY must lie inside the same safe
   envelope ``robot_comic.motion_safety`` clamps to at runtime (pitch/yaw/roll
   limits).  This mirrors the on-robot clamp layer so the screen rejects moves
   the clamp would otherwise silently mutate.
3. **Reachability** (optional, ``check_kinematics``) — when reachy_mini's
   analytical kinematics is importable, each head pose is run through IK→FK; a
   pose the solver cannot reach upright is flagged.  Cheap, dependency-light
   (Rust bindings), and degrades gracefully to a no-op if unavailable so the
   screen still runs on hosts without the kinematics package.
4. **Dynamics** — per-axis angular velocity, acceleration, and jerk are
   estimated by finite differences across the (uniform) time grid and compared
   against limits.  This is the offline analogue of the on-robot velocity cap
   (``cap_head_velocity``): a recording that snaps too far too fast is the root
   cause of the cowling-slam class of bugs (#264/#272), so it is rejected here
   before playback.

The dynamics limits default to values consistent with the on-robot velocity cap
(``HEAD_MAX_VEL_RAD_S`` = 1.5 rad/s) with headroom for the higher-order terms,
and are fully overridable via :class:`ScreenLimits`.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import field, dataclass

from robot_comic import motion_safety


if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray


# Default dynamics limits.  The velocity limit matches the on-robot per-axis
# velocity cap (``motion_safety.HEAD_MAX_VEL_RAD_S``); accel/jerk limits are
# generous multiples chosen so a smoothly-eased authored move passes while a
# snap rejects.  All are per-axis bounds on the RPY trajectory (rad/s, rad/s^2,
# rad/s^3).
_DEFAULT_MAX_VELOCITY_RAD_S = motion_safety.HEAD_MAX_VEL_RAD_S
_DEFAULT_MAX_ACCEL_RAD_S2 = 30.0
_DEFAULT_MAX_JERK_RAD_S3 = 900.0


@dataclass
class ScreenLimits:
    """Tunable limits for :func:`screen_recorded_move`.

    Envelope limits default to the live ``motion_safety`` envelope so the screen
    agrees with the on-robot clamp.  Dynamics limits bound the per-axis RPY
    trajectory derivatives.
    """

    # Kinematic envelope (radians).  Default to the live motion-safety envelope.
    pitch_min_rad: float = motion_safety.HEAD_PITCH_MIN_RAD
    pitch_max_rad: float = motion_safety.HEAD_PITCH_MAX_RAD
    yaw_min_rad: float = motion_safety.HEAD_YAW_MIN_RAD
    yaw_max_rad: float = motion_safety.HEAD_YAW_MAX_RAD
    roll_min_rad: float = motion_safety.HEAD_ROLL_MIN_RAD
    roll_max_rad: float = motion_safety.HEAD_ROLL_MAX_RAD

    # Dynamics (per-axis, on the RPY trajectory).
    max_velocity_rad_s: float = _DEFAULT_MAX_VELOCITY_RAD_S
    max_acceleration_rad_s2: float = _DEFAULT_MAX_ACCEL_RAD_S2
    max_jerk_rad_s3: float = _DEFAULT_MAX_JERK_RAD_S3

    # Small numeric tolerance so a pose authored *at* the limit is not rejected
    # by floating-point round-trip noise through the rotation matrix.
    envelope_tol_rad: float = math.radians(0.5)


@dataclass
class ScreenResult:
    """Outcome of a screen run.

    ``passed`` is True only when ``reasons`` is empty.  ``metrics`` carries the
    peak dynamics figures (and reachability stats when checked) for reporting.
    """

    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def _extract_rpy(head: Any) -> Optional["NDArray[np.float64]"]:
    """Return extrinsic-XYZ RPY (radians) for a 4x4 head matrix, or None if invalid."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    try:
        mat = np.asarray(head, dtype=np.float64)
        if mat.shape != (4, 4):
            return None
        rot = mat[:3, :3]
        # Reject non-orthonormal rotation blocks up front.
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-3):
            return None
        return np.asarray(Rotation.from_matrix(rot).as_euler("xyz", degrees=False), dtype=np.float64)
    except Exception:
        return None


def _max_abs_finite_diff(values: "NDArray[np.float64]", times: "NDArray[np.float64]") -> float:
    """Peak absolute first difference of ``values`` w.r.t. ``times`` (per column)."""
    import numpy as np

    dt = np.diff(times)
    dt = np.where(dt <= 0.0, np.inf, dt)  # guard divide-by-zero / non-monotonic
    deriv = np.diff(values, axis=0) / dt[:, None]
    if deriv.size == 0:
        return 0.0
    return float(np.max(np.abs(deriv)))


def _check_reachability(heads: List[Any], reasons: List[str], metrics: Dict[str, float]) -> None:
    """Flag any head pose the analytical kinematics cannot reach upright.

    Degrades to a no-op (recording a metric of -1) if the kinematics package is
    not importable on this host, so the screen still runs cross-platform.
    """
    import numpy as np

    try:
        from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics
    except Exception:
        metrics["reachability_checked"] = 0.0
        return

    try:
        kin = AnalyticalKinematics(automatic_body_yaw=True)
    except Exception:
        metrics["reachability_checked"] = 0.0
        return

    metrics["reachability_checked"] = 1.0
    unreachable = 0
    from reachy_mini.utils.interpolation import distance_between_poses

    # The analytical FK is iterative and warm-started from the previous
    # solution, so the first solve after a large pose jump can carry a transient
    # residual.  We settle FK with a couple of extra iterations and use a
    # generous residual threshold: the goal is to catch poses the solver cannot
    # realise upright at all (non-finite joints, or a residual far larger than
    # warm-start noise), not to assert millimetre fidelity.
    unhinged_threshold = 100.0  # magic-mm (mm + deg); warm-start noise ~30
    for idx, head in enumerate(heads):
        pose = np.asarray(head, dtype=np.float64)
        try:
            joints = np.asarray(kin.ik(pose), dtype=np.float64)
            if not np.all(np.isfinite(joints)):
                unreachable += 1
                if unreachable == 1:
                    reasons.append(
                        f"unreachable head pose at frame {idx} (inverse kinematics produced non-finite joints)"
                    )
                continue
            achieved = kin.fk(joints, no_iterations=5)
            _, _, unhinged = distance_between_poses(pose, achieved)
            if unhinged > unhinged_threshold:
                unreachable += 1
                if unreachable == 1:
                    reasons.append(f"unreachable head pose at frame {idx} (IK/FK residual {unhinged:.1f} magic-mm)")
        except Exception as exc:
            unreachable += 1
            if unreachable == 1:
                reasons.append(f"kinematics rejected head pose at frame {idx}: {exc}")
    metrics["unreachable_frames"] = float(unreachable)


def screen_recorded_move(
    move: Dict[str, Any],
    limits: Optional[ScreenLimits] = None,
    *,
    check_kinematics: bool = True,
) -> ScreenResult:
    """Screen a ``RecordedMove`` dict offline; return a pass/fail with reasons.

    Parameters
    ----------
    move:
        A ``RecordedMove`` JSON dict (``time`` + ``set_target_data``).
    limits:
        Optional :class:`ScreenLimits` override.  Defaults to the live
        motion-safety envelope plus the default dynamics caps.
    check_kinematics:
        When True (default), run the analytical IK/FK reachability check if the
        kinematics package is importable.  Disable for a pure envelope+dynamics
        screen.

    Returns
    -------
    ScreenResult
        ``passed`` is True only when no reason was recorded.

    """
    import numpy as np

    lim = limits or ScreenLimits()
    reasons: List[str] = []
    metrics: Dict[str, float] = {
        "max_velocity_rad_s": 0.0,
        "max_acceleration_rad_s2": 0.0,
        "max_jerk_rad_s3": 0.0,
    }

    # 1. Structural validity ------------------------------------------------
    times_raw = move.get("time")
    data = move.get("set_target_data")
    if not isinstance(times_raw, list) or not isinstance(data, list):
        reasons.append("malformed move: missing 'time' or 'set_target_data' list")
        return ScreenResult(False, reasons, metrics)
    if len(times_raw) != len(data):
        reasons.append(f"malformed move: time has {len(times_raw)} entries but set_target_data has {len(data)}")
        return ScreenResult(False, reasons, metrics)
    if len(times_raw) < 2:
        reasons.append("malformed move: need at least 2 frames")
        return ScreenResult(False, reasons, metrics)

    times = np.asarray(times_raw, dtype=np.float64)
    if not np.all(np.diff(times) > 0.0):
        reasons.append("malformed move: timestamps must be strictly increasing")
        return ScreenResult(False, reasons, metrics)

    # 2. Kinematic envelope (and collect RPY for dynamics) ------------------
    rpy_rows: List["NDArray[np.float64]"] = []
    heads: List[Any] = []
    axis_names = ("roll", "pitch", "yaw")
    axis_bounds = (
        (lim.roll_min_rad, lim.roll_max_rad),
        (lim.pitch_min_rad, lim.pitch_max_rad),
        (lim.yaw_min_rad, lim.yaw_max_rad),
    )
    violated_axes: set[str] = set()

    for idx, frame in enumerate(data):
        head = frame.get("head") if isinstance(frame, dict) else None
        if head is None:
            reasons.append(f"malformed move: frame {idx} has no 'head' matrix")
            return ScreenResult(False, reasons, metrics)
        rpy = _extract_rpy(head)
        if rpy is None:
            reasons.append(f"malformed move: frame {idx} head is not a valid 4x4 pose")
            return ScreenResult(False, reasons, metrics)
        heads.append(head)
        rpy_rows.append(rpy)

        for axis_idx, (name, (lo, hi)) in enumerate(zip(axis_names, axis_bounds)):
            value = float(rpy[axis_idx])
            if value < lo - lim.envelope_tol_rad or value > hi + lim.envelope_tol_rad:
                violated_axes.add(name)

    for name in sorted(violated_axes):
        reasons.append(
            f"{name} exceeds the safe envelope/limit "
            f"(outside [{math.degrees(dict(zip(axis_names, [b[0] for b in axis_bounds]))[name]):.1f}, "
            f"{math.degrees(dict(zip(axis_names, [b[1] for b in axis_bounds]))[name]):.1f}] deg)"
        )

    # 3. Dynamics: velocity / acceleration / jerk ---------------------------
    rpy_arr = np.vstack(rpy_rows)  # (n_frames, 3)
    # Unwrap each axis so a wrap across +-pi does not register as a huge step.
    rpy_unwrapped = np.asarray(np.unwrap(rpy_arr, axis=0), dtype=np.float64)

    max_vel = _max_abs_finite_diff(rpy_unwrapped, times)
    metrics["max_velocity_rad_s"] = max_vel
    if max_vel > lim.max_velocity_rad_s:
        reasons.append(f"angular velocity {max_vel:.2f} rad/s exceeds limit {lim.max_velocity_rad_s:.2f} rad/s")

    # Build a velocity series at midpoints for higher-order differences.
    dt = np.diff(times)
    dt_safe = np.where(dt <= 0.0, np.inf, dt)
    vel = np.diff(rpy_unwrapped, axis=0) / dt_safe[:, None]
    mid_times = (times[:-1] + times[1:]) / 2.0

    if vel.shape[0] >= 2:
        max_acc = _max_abs_finite_diff(vel, mid_times)
        metrics["max_acceleration_rad_s2"] = max_acc
        if max_acc > lim.max_acceleration_rad_s2:
            reasons.append(
                f"angular acceleration {max_acc:.1f} rad/s^2 exceeds limit {lim.max_acceleration_rad_s2:.1f} rad/s^2"
            )

        acc_dt = np.diff(mid_times)
        acc_dt_safe = np.where(acc_dt <= 0.0, np.inf, acc_dt)
        acc = np.diff(vel, axis=0) / acc_dt_safe[:, None]
        acc_mid_times = (mid_times[:-1] + mid_times[1:]) / 2.0
        if acc.shape[0] >= 2:
            max_jerk = _max_abs_finite_diff(acc, acc_mid_times)
            metrics["max_jerk_rad_s3"] = max_jerk
            if max_jerk > lim.max_jerk_rad_s3:
                reasons.append(f"angular jerk {max_jerk:.0f} rad/s^3 exceeds limit {lim.max_jerk_rad_s3:.0f} rad/s^3")

    # 4. Reachability (optional, best-effort) -------------------------------
    if check_kinematics:
        _check_reachability(heads, reasons, metrics)

    return ScreenResult(passed=not reasons, reasons=reasons, metrics=metrics)
