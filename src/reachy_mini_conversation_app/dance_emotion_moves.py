"""Dance and emotion moves for the movement queue system.

This module implements dance moves and emotions as Move objects that can be queued
and executed sequentially by the MovementManager.
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_dances_library.dance_move import DanceMove


logger = logging.getLogger(__name__)


class DanceQueueMove(Move):  # type: ignore
    """Wrapper for dance moves to work with the movement queue system."""

    def __init__(self, move_name: str):
        """Initialize a DanceQueueMove."""
        self.dance_move = DanceMove(move_name)
        self.move_name = move_name

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.dance_move.duration)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate dance move at time t."""
        try:
            # Get the pose from the dance move
            head_pose, antennas, body_yaw = self.dance_move.evaluate(t)

            # Convert to numpy array if antennas is tuple and return in official Move format
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating dance move '{self.move_name}' at t={t}: {e}")
            # Return neutral pose on error
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class EmotionQueueMove(Move):  # type: ignore
    """Wrapper for emotion moves to work with the movement queue system."""

    def __init__(self, emotion_name: str, recorded_moves: RecordedMoves):
        """Initialize an EmotionQueueMove."""
        self.emotion_move = recorded_moves.get(emotion_name)
        self.emotion_name = emotion_name

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.emotion_move.duration)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate emotion move at time t."""
        try:
            # Get the pose from the emotion move
            head_pose, antennas, body_yaw = self.emotion_move.evaluate(t)

            # Convert to numpy array if antennas is tuple and return in official Move format
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating emotion '{self.emotion_name}' at t={t}: {e}")
            # Return neutral pose on error
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class GotoQueueMove(Move):  # type: ignore
    """Wrapper for goto moves to work with the movement queue system."""

    def __init__(
        self,
        target_head_pose: NDArray[np.float32],
        start_head_pose: NDArray[np.float32] | None = None,
        target_antennas: Tuple[float, float] = (0, 0),
        start_antennas: Tuple[float, float] | None = None,
        target_body_yaw: float = 0,
        start_body_yaw: float | None = None,
        duration: float = 1.0,
    ):
        """Initialize a GotoQueueMove."""
        self._duration = duration
        self.target_head_pose = target_head_pose
        self.start_head_pose = start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas = start_antennas or (0, 0)
        self.target_body_yaw = target_body_yaw
        self.start_body_yaw = start_body_yaw or 0

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate goto move at time t using linear interpolation."""
        try:
            from reachy_mini.utils import create_head_pose
            from reachy_mini.utils.interpolation import linear_pose_interpolation

            # Clamp t to [0, 1] for interpolation
            t_clamped = max(0, min(1, t / self.duration))

            # Use start pose if available, otherwise neutral
            if self.start_head_pose is not None:
                start_pose = self.start_head_pose
            else:
                start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

            # Interpolate head pose
            head_pose = linear_pose_interpolation(start_pose, self.target_head_pose, t_clamped)

            # Interpolate antennas - return as numpy array
            antennas = np.array(
                [
                    self.start_antennas[0] + (self.target_antennas[0] - self.start_antennas[0]) * t_clamped,
                    self.start_antennas[1] + (self.target_antennas[1] - self.start_antennas[1]) * t_clamped,
                ],
                dtype=np.float64,
            )

            # Interpolate body yaw
            body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * t_clamped

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating goto move at t={t}: {e}")
            # Return target pose on error - convert to float64
            target_head_pose_f64 = self.target_head_pose.astype(np.float64)
            target_antennas_array = np.array([self.target_antennas[0], self.target_antennas[1]], dtype=np.float64)
            return (target_head_pose_f64, target_antennas_array, self.target_body_yaw)
