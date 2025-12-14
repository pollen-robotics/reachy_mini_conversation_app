from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


try:
    from supervision import Detections
    from ultralytics import YOLO  # type: ignore
except ImportError as e:
    raise ImportError(
        "To use YOLO head tracker, please install the extra dependencies: pip install '.[yolo_vision]'",
    ) from e
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)


class HeadTracker:
    """Lightweight head tracker using YOLO for face detection."""

    def __init__(
        self,
        model_repo: str = "AdamCodd/YOLOv11n-face-detection",
        model_filename: str = "model.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLO-based head tracker.

        Args:
            model_repo: HuggingFace model repository
            model_filename: Model file name
            confidence_threshold: Minimum confidence for face detection
            device: Device to run inference on ('cpu' or 'cuda')

        """
        self.confidence_threshold = confidence_threshold

        try:
            # Download and load YOLO model
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
            self.model = YOLO(model_path).to(device)
            logger.info(f"YOLO face detection model loaded from {model_repo}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _select_best_face(self, detections: Detections) -> int | None:
        """Select the best face based on confidence and area (largest face with highest confidence).

        Args:
            detections: Supervision detections object

        Returns:
            Index of best face or None if no valid faces

        """
        if detections.xyxy.shape[0] == 0:
            return None

        # Check if confidence is available
        if detections.confidence is None:
            return None

        # Filter by confidence threshold
        valid_mask = detections.confidence >= self.confidence_threshold
        if not np.any(valid_mask):
            return None

        valid_indices = np.where(valid_mask)[0]

        # Calculate areas for valid detections
        boxes = detections.xyxy[valid_indices]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Combine confidence and area (weighted towards larger faces)
        confidences = detections.confidence[valid_indices]
        scores = confidences * 0.7 + (areas / np.max(areas)) * 0.3

        # Return index of best face
        best_idx = valid_indices[np.argmax(scores)]
        return int(best_idx)

    def _bbox_to_mp_coords(self, bbox: NDArray[np.float32], w: int, h: int) -> NDArray[np.float32]:
        """Convert bounding box center to MediaPipe-style coordinates [-1, 1].

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            w: Image width
            h: Image height

        Returns:
            Center point in [-1, 1] coordinates

        """
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        # Normalize to [0, 1] then to [-1, 1]
        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0

        return np.array([norm_x, norm_y], dtype=np.float32)

    def get_head_position(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.float32] | None, float | None]:
        """Get head position from face detection.

        Args:
            img: Input image

        Returns:
            Tuple of (eye_center [-1,1], roll_angle)

        """
        h, w = img.shape[:2]

        try:
            # Run YOLO inference
            results = self.model(img, verbose=False)
            detections = Detections.from_ultralytics(results[0])

            # Select best face
            face_idx = self._select_best_face(detections)
            if face_idx is None:
                logger.debug("No face detected above confidence threshold")
                return None, None

            bbox = detections.xyxy[face_idx]

            if detections.confidence is not None:
                confidence = detections.confidence[face_idx]
                logger.debug(f"Face detected with confidence: {confidence:.2f}")

            # Get face center in [-1, 1] coordinates
            face_center = self._bbox_to_mp_coords(bbox, w, h)

            # Roll is 0 since we don't have keypoints for precise angle estimation
            roll = 0.0

            return face_center, roll

        except Exception as e:
            logger.error(f"Error in head position detection: {e}")
            return None, None
