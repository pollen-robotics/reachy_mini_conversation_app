"""Dance and emotion moves for the movement queue system.

This module implements dance moves and emotions as Move objects that can be queued
and executed sequentially by the MovementManager.
"""

from __future__ import annotations
import logging
from typing import Tuple, Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_dances_library.dance_move import DanceMove


logger = logging.getLogger(__name__)


class DanceQueueMove(Move):  # type: ignore
    """Wrapper for dance moves to work with the movement queue system."""

    def __init__(self, move_name: str, speed_factor: float = 1.0):
        """Initialize a DanceQueueMove."""
        self.dance_move = DanceMove(move_name)
        self.move_name = move_name
        self.speed_factor = max(0.1, min(2.0, speed_factor))

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.dance_move.duration) / self.speed_factor

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate dance move at time t."""
        try:
            head_pose, antennas, body_yaw = self.dance_move.evaluate(t * self.speed_factor)

            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating dance move '{self.move_name}' at t={t}: {e}")
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class EmotionQueueMove(Move):  # type: ignore
    """Wrapper for emotion moves to work with the movement queue system."""

    def __init__(self, emotion_name: str, recorded_moves: RecordedMoves, speed_factor: float = 1.0):
        """Initialize an EmotionQueueMove."""
        self.emotion_move = recorded_moves.get(emotion_name)
        self.emotion_name = emotion_name
        self.speed_factor = max(0.1, min(2.0, speed_factor))

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.emotion_move.duration) / self.speed_factor

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate emotion move at time t."""
        try:
            head_pose, antennas, body_yaw = self.emotion_move.evaluate(t * self.speed_factor)

            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating emotion '{self.emotion_name}' at t={t}: {e}")
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class GotoQueueMove(Move):  # type: ignore
    """Wrapper for goto moves to work with the movement queue system."""

    def __init__(
        self,
        target_head_pose: NDArray[np.float32],
        start_head_pose: NDArray[np.float32] | Callable[[], NDArray[np.float32]] | None = None,
        target_antennas: Tuple[float, float] = (0, 0),
        start_antennas: Tuple[float, float] | Callable[[], Tuple[float, float]] | None = None,
        target_body_yaw: float = 0,
        start_body_yaw: float | Callable[[], float] | None = None,
        duration: float = 1.0,
        speed_factor: float = 1.0,
        ease: bool = False,
    ):
        """Initialize a GotoQueueMove.

        ``ease`` (default False keeps the legacy linear ramp): when True, the
        interpolation parameter ``t`` is run through a cubic smoothstep
        (3t² − 2t³) so velocity is zero at both endpoints. Used by single-shot
        head pointing (#264) to avoid the snap at the start and end of each
        move when discrete ``move_head`` calls chain together.

        ``start_head_pose`` accepts three forms:

        * ``None`` — neutral-pose fallback (legacy gesture path, unchanged).
        * ``NDArray`` — explicit array captured at call time (legacy path).
        * ``Callable[[], NDArray]`` — lazy sentinel: invoked exactly once on the
          **first** ``evaluate()`` call so the start pose reflects where the head
          *actually* is when the move begins executing, not where it was when the
          LLM queued the call.  This is the fix for the "head whip" bug where
          rapid sequential ``move_head`` calls all captured a mid-swing pose.

        ``start_antennas`` and ``start_body_yaw`` accept the same lazy-callable
        sentinel so gestures can begin from the live commanded pose on every
        axis, not just the head (#521 transition snap).
        """
        self.speed_factor = max(0.1, min(2.0, speed_factor))
        self._duration = duration / self.speed_factor
        self.target_head_pose = target_head_pose
        self.start_head_pose: NDArray[np.float32] | Callable[[], NDArray[np.float32]] | None = start_head_pose
        self._resolved_start_pose: NDArray[np.float32] | None = None if callable(start_head_pose) else start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas: Tuple[float, float] | Callable[[], Tuple[float, float]] = (
            start_antennas if callable(start_antennas) else (start_antennas or (0, 0))
        )
        self._resolved_start_antennas: Tuple[float, float] | None = (
            None if callable(start_antennas) else (start_antennas or (0, 0))
        )
        self.target_body_yaw = target_body_yaw
        self.start_body_yaw: float | Callable[[], float] = (
            start_body_yaw if callable(start_body_yaw) else (start_body_yaw or 0)
        )
        self._resolved_start_body_yaw: float | None = None if callable(start_body_yaw) else (start_body_yaw or 0)
        self.ease = ease

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Cubic smoothstep: derivative is zero at t=0 and t=1."""
        return t * t * (3.0 - 2.0 * t)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate goto move at time t.

        Linear interpolation by default; cubic smoothstep when ``ease=True``
        was passed at construction.

        On the **first** call, if ``start_head_pose`` was supplied as a
        callable, it is invoked here (dequeue time) and the result is cached in
        ``_resolved_start_pose``.  Subsequent calls reuse the cached array.
        If the callable raises, we fall back to neutral pose so the worker loop
        does not crash.
        """
        try:
            from reachy_mini.utils import create_head_pose
            from reachy_mini.utils.interpolation import linear_pose_interpolation

            # Resolve callable start pose exactly once (dequeue-time capture).
            if callable(self.start_head_pose) and self._resolved_start_pose is None:
                try:
                    self._resolved_start_pose = self.start_head_pose()
                except Exception as exc:
                    logger.warning(
                        "GotoQueueMove: start_head_pose callable raised %s; falling back to neutral pose",
                        exc,
                    )
                    self._resolved_start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

            # Same dequeue-time capture for antennas and body yaw (#521).
            if self._resolved_start_antennas is None:
                try:
                    resolved = self.start_antennas() if callable(self.start_antennas) else self.start_antennas
                    self._resolved_start_antennas = (float(resolved[0]), float(resolved[1]))
                except Exception as exc:
                    logger.warning(
                        "GotoQueueMove: start_antennas callable raised %s; falling back to (0, 0)",
                        exc,
                    )
                    self._resolved_start_antennas = (0.0, 0.0)
            if self._resolved_start_body_yaw is None:
                try:
                    self._resolved_start_body_yaw = float(
                        self.start_body_yaw() if callable(self.start_body_yaw) else self.start_body_yaw
                    )
                except Exception as exc:
                    logger.warning(
                        "GotoQueueMove: start_body_yaw callable raised %s; falling back to 0.0",
                        exc,
                    )
                    self._resolved_start_body_yaw = 0.0
            start_antennas = self._resolved_start_antennas
            start_body_yaw = self._resolved_start_body_yaw

            # Clamp t to [0, 1] for interpolation
            t_clamped = max(0, min(1, t / self.duration))
            t_eff = self._smoothstep(t_clamped) if self.ease else t_clamped

            # Use resolved/cached start pose if available, otherwise neutral.
            # For the callable path, _resolved_start_pose is set above.
            # For the array path, _resolved_start_pose mirrors start_head_pose.
            # For the None path, fall back to neutral (legacy gesture behavior).
            if self._resolved_start_pose is not None:
                start_pose = self._resolved_start_pose
            else:
                start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

            # Interpolate head pose
            head_pose = linear_pose_interpolation(start_pose, self.target_head_pose, t_eff)

            # Interpolate antennas - return as numpy array
            antennas = np.array(
                [
                    start_antennas[0] + (self.target_antennas[0] - start_antennas[0]) * t_eff,
                    start_antennas[1] + (self.target_antennas[1] - start_antennas[1]) * t_eff,
                ],
                dtype=np.float64,
            )

            # Interpolate body yaw
            body_yaw = start_body_yaw + (self.target_body_yaw - start_body_yaw) * t_eff

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating goto move at t={t}: {e}")
            # Return target pose on error - convert to float64
            target_head_pose_f64 = self.target_head_pose.astype(np.float64)
            target_antennas_array = np.array([self.target_antennas[0], self.target_antennas[1]], dtype=np.float64)
            return (target_head_pose_f64, target_antennas_array, self.target_body_yaw)
