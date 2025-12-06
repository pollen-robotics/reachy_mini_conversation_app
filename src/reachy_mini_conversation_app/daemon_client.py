"""Daemon API client wrapper.

Provides simplified interface to daemon REST API for robot control.
Mimics ReachyMini interface for compatibility with existing code.
"""

import logging
from typing import Tuple

import numpy as np
import requests
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class DaemonClient:
    """Client for Reachy Mini daemon REST API.

    Provides interface compatible with ReachyMini for movement control.
    """

    def __init__(self, daemon_url: str = "http://localhost:8100"):
        """Initialize daemon client.

        Args:
            daemon_url: Base URL of daemon API (default: http://localhost:8100)
        """
        self.daemon_url = daemon_url.rstrip("/")
        self._body_yaw_warning_logged = False  # Only log warning once
        self._check_connection()

    def _check_connection(self) -> None:
        """Verify daemon is accessible."""
        try:
            response = requests.get(f"{self.daemon_url}/api/daemon/status", timeout=2)
            response.raise_for_status()
            logger.info(f"Connected to daemon at {self.daemon_url}")
        except requests.RequestException as e:
            logger.error(f"Failed to connect to daemon at {self.daemon_url}: {e}")
            raise ConnectionError(f"Daemon not accessible at {self.daemon_url}") from e

    def set_target(
        self,
        head: NDArray[np.float32] | None = None,
        antennas: Tuple[float, float] | None = None,
        body_yaw: float | None = None,
    ) -> None:
        """Send target positions to robot via daemon API.

        Args:
            head: 4x4 transformation matrix for head pose (meters)
            antennas: Tuple of (left, right) antenna positions (radians)
            body_yaw: Body yaw angle (radians)
        """
        # Build request payload
        payload = {}

        if head is not None:
            # Convert 4x4 matrix to XYZRPYPose format
            x, y, z = head[0, 3], head[1, 3], head[2, 3]
            roll, pitch, yaw = R.from_matrix(head[:3, :3]).as_euler("xyz")

            payload["target_head_pose"] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            }

        if antennas is not None:
            payload["target_antennas"] = [float(antennas[0]), float(antennas[1])]

        if body_yaw is not None:
            payload["target_body_yaw"] = float(body_yaw)

        try:
            response = requests.post(
                f"{self.daemon_url}/api/move/set_target",
                json=payload,
                timeout=0.1,  # Fast timeout for 100Hz loop
            )
            response.raise_for_status()
        except requests.Timeout:
            # Timeout is acceptable for high-frequency commands
            pass
        except requests.RequestException as e:
            logger.error(f"Failed to send set_target: {e}")
            raise

    def get_current_joint_positions(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get current joint positions from daemon.

        Returns:
            Tuple of (timestamps, joint_positions)
            - timestamps: Empty array (not provided by daemon)
            - joint_positions: [left_antenna, right_antenna] in radians
        """
        try:
            response = requests.get(
                f"{self.daemon_url}/api/state/present_antenna_joint_positions",
                timeout=0.5,
            )
            response.raise_for_status()
            antennas = response.json()  # Returns tuple [left, right]

            # Return format matching ReachyMini: (timestamps, joints)
            timestamps = np.array([])
            joints = np.array(antennas, dtype=np.float64)
            return (timestamps, joints)

        except requests.RequestException as e:
            logger.error(f"Failed to get joint positions: {e}")
            raise

    def get_current_head_pose(self) -> NDArray[np.float32]:
        """Get current head pose from daemon.

        Returns:
            4x4 transformation matrix (meters)
        """
        try:
            response = requests.get(
                f"{self.daemon_url}/api/state/present_head_pose",
                params={"use_pose_matrix": False},  # Get XYZRPYPose format
                timeout=0.5,
            )
            response.raise_for_status()
            pose_data = response.json()

            # Convert XYZRPYPose to 4x4 matrix
            rotation = R.from_euler("xyz", [pose_data["roll"], pose_data["pitch"], pose_data["yaw"]])
            pose_matrix = np.eye(4, dtype=np.float32)
            pose_matrix[:3, 3] = [pose_data["x"], pose_data["y"], pose_data["z"]]
            pose_matrix[:3, :3] = rotation.as_matrix()

            return pose_matrix

        except requests.RequestException as e:
            logger.error(f"Failed to get head pose: {e}")
            raise

    def get_status(self) -> dict:
        """Get daemon status.

        Returns:
            Status dictionary from daemon
        """
        try:
            response = requests.get(f"{self.daemon_url}/api/daemon/status", timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get daemon status: {e}")
            raise

    def look_at_image(self, u: int, v: int) -> NDArray[np.float32]:
        """Calculate head pose to look at pixel position in camera image.

        Args:
            u: Horizontal pixel coordinate
            v: Vertical pixel coordinate

        Returns:
            4x4 transformation matrix for target head pose
        """
        try:
            response = requests.get(
                f"{self.daemon_url}/api/kinematics/look_at_image",
                params={"u": u, "v": v, "use_pose_matrix": False},
                timeout=0.5,
            )
            response.raise_for_status()
            pose_data = response.json()

            # Convert XYZRPYPose to 4x4 matrix
            rotation = R.from_euler("xyz", [pose_data["roll"], pose_data["pitch"], pose_data["yaw"]])
            pose_matrix = np.eye(4, dtype=np.float32)
            pose_matrix[:3, 3] = [pose_data["x"], pose_data["y"], pose_data["z"]]
            pose_matrix[:3, :3] = rotation.as_matrix()

            return pose_matrix

        except requests.RequestException as e:
            logger.error(f"Failed to call look_at_image: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from daemon (no-op for HTTP client)."""
        logger.info("Daemon client disconnected")
