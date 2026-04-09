from __future__ import annotations
import logging

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTrackerResult


try:
    from supervision import Detections
    from ultralytics import YOLO  # type: ignore
except ImportError as e:
    raise ImportError(
        "To use YOLO head tracker, please install the extra dependencies: pip install '.[yolo_vision]'",
    ) from e
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)


class YoloHeadTracker:
    """Lightweight head tracker using YOLO for face detection."""

    def __init__(
        self,
        model_repo: str = "AdamCodd/YOLOv11n-face-detection",
        model_filename: str = "model.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLO-based head tracker."""
        self.confidence_threshold = confidence_threshold

        try:
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
            self.model = YOLO(model_path).to(device)
            logger.info("YOLO face detection model loaded from %s", model_repo)
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise

    def _select_best_face(self, detections: Detections) -> int | None:
        """Select the best face based on confidence and area (largest face with highest confidence)."""
        if detections.xyxy.shape[0] == 0:
            return None

        if detections.confidence is None:
            return None

        valid_mask = detections.confidence >= self.confidence_threshold
        if not np.any(valid_mask):
            return None

        valid_indices = np.where(valid_mask)[0]
        boxes = detections.xyxy[valid_indices]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        confidences = detections.confidence[valid_indices]
        scores = confidences * 0.7 + (areas / np.max(areas)) * 0.3

        best_idx = valid_indices[np.argmax(scores)]
        return int(best_idx)

    def _bbox_to_mp_coords(self, bbox: NDArray[np.float32], w: int, h: int) -> NDArray[np.float32]:
        """Convert bounding box center to MediaPipe-style coordinates [-1, 1]."""
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0

        return np.array([norm_x, norm_y], dtype=np.float32)

    def get_head_position(self, img: NDArray[np.uint8]) -> HeadTrackerResult:
        """Get head position from face detection."""
        h, w = img.shape[:2]

        try:
            results = self.model(img, verbose=False)
            detections = Detections.from_ultralytics(results[0])

            face_idx = self._select_best_face(detections)
            if face_idx is None:
                logger.debug("No face detected above confidence threshold")
                return None, None

            bbox = detections.xyxy[face_idx]

            if detections.confidence is not None:
                confidence = detections.confidence[face_idx]
                logger.debug("Face detected with confidence: %.2f", confidence)

            face_center = self._bbox_to_mp_coords(bbox, w, h)
            return face_center, 0.0

        except Exception as e:
            logger.error("Error in head position detection: %s", e)
            return None, None
