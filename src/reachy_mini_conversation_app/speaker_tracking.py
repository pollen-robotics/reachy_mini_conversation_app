import math
import time
import logging
import threading
from collections.abc import Callable

import httpx
import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.io.protocol import DoaSnapshot


logger = logging.getLogger(__name__)


class SpeakerTracking:
    """Turn body and head toward stable speech bearings from the robot's daemon."""

    def __init__(self) -> None:
        """Initialize a stationary speaker target."""
        self._candidate: float | None = None
        self._candidate_since = 0.0
        self._last_observation = 0.0
        self._target_yaw = 0.0
        self._yaw = 0.0
        self._head_yaw = 0.0
        self._last_update: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, robot: ReachyMini, on_direction: Callable[[float, bool, float], None]) -> None:
        """Poll daemon readings off the movement thread and report world bearings."""
        self._stop.clear()

        def poll() -> None:
            endpoint = httpx.URL(scheme="http", host=robot.client.host, port=robot.client.port, path="/api/state/doa")
            last_warning = -math.inf
            with httpx.Client(timeout=0.5, trust_env=False) as client:
                while not self._stop.is_set():
                    try:
                        response = client.get(endpoint)
                        response.raise_for_status()
                        payload = response.json()
                        if payload is not None:
                            reading = DoaSnapshot.model_validate(payload)
                            if not math.isfinite(reading.angle) or not 0.0 <= reading.angle <= math.pi:
                                raise ValueError("Invalid speaker direction angle")
                            head_pose = robot.get_current_head_pose()
                            bearing = head_pose[:3, :3] @ np.array(
                                [math.sin(reading.angle), math.cos(reading.angle), 0.0]
                            )
                            on_direction(math.atan2(bearing[1], bearing[0]), reading.speech_detected, time.monotonic())
                    except (httpx.HTTPError, ValueError, ConnectionError, TimeoutError) as exc:
                        now = time.monotonic()
                        if now - last_warning >= 10.0:
                            logger.warning("Speaker direction unavailable: %s", exc)
                            last_warning = now
                    self._stop.wait(0.1)

        self._thread = threading.Thread(target=poll, daemon=True, name="speaker-direction")
        self._thread.start()

    def stop(self) -> None:
        """Stop polling without changing the robot's pose."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            if self._thread.is_alive():
                logger.warning("Speaker direction reader is still stopping")
            else:
                self._thread = None

    def observe(self, yaw: float, speech: bool, observed_at: float, now: float, enabled: bool) -> None:
        """Accept a bearing after sustained speech, rejecting stale or unstable readings."""
        if not enabled or not speech or not math.isfinite(yaw) or not 0.0 <= now - observed_at <= 0.5:
            self._candidate = None
            return
        difference = (
            math.atan2(math.sin(yaw - self._candidate), math.cos(yaw - self._candidate))
            if self._candidate is not None
            else math.inf
        )
        if abs(difference) > math.radians(8) or observed_at - self._last_observation > 0.3:
            self._candidate = yaw
            self._candidate_since = observed_at
        elif self._candidate is not None:
            self._candidate += 0.25 * difference
        self._last_observation = observed_at
        if self._candidate is not None and observed_at - self._candidate_since >= 0.25:
            target = float(np.clip(self._candidate, -math.radians(150), math.radians(150)))
            if abs(target - self._target_yaw) >= math.radians(8):
                self._target_yaw = target

    def apply(
        self, head: NDArray[np.float32], body_yaw: float, now: float, allow_turn: bool, manual: bool = False
    ) -> tuple[NDArray[np.float32], float]:
        """Apply a held speaker orientation with turns limited to 30 degrees per second."""
        elapsed = min(max(now - self._last_update, 0.0), 0.1) if self._last_update is not None else 0.0
        self._last_update = now
        pose_yaw = math.atan2(float(head[1, 0]), float(head[0, 0]))
        if manual:
            self._yaw = body_yaw
            self._head_yaw = pose_yaw
            return head, body_yaw
        if allow_turn:
            step = (self._target_yaw - self._yaw) * (1.0 - math.exp(-elapsed / 0.35))
            self._yaw += float(np.clip(step, -math.radians(30) * elapsed, math.radians(30) * elapsed))
            step = (self._target_yaw - self._head_yaw) * (1.0 - math.exp(-elapsed / 0.35))
            self._head_yaw += float(np.clip(step, -math.radians(30) * elapsed, math.radians(30) * elapsed))
        # Primary poses may already include the previous world yaw after an explicit move.
        rotation = create_head_pose(yaw=self._head_yaw - pose_yaw, degrees=False)
        return (rotation @ head).astype(np.float32), self._yaw
