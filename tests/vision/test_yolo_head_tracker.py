"""Unit tests for the YOLO head tracker module."""

import builtins
import sys
from typing import TYPE_CHECKING, Any, Callable, Generator, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from supervision import Detections


# Create mock classes for supervision
class MockDetections:
    """Mock supervision.Detections class."""

    def __init__(
        self,
        xyxy: NDArray[np.float32],
        confidence: NDArray[np.float32] | None = None,
    ) -> None:
        """Initialize mock detections."""
        self.xyxy = xyxy
        self.confidence = confidence

    @classmethod
    def from_ultralytics(cls, results: MagicMock) -> "MockDetections":
        """Mock factory method."""
        detections: MockDetections = results._detections
        return detections


# Mock heavy dependencies before importing the module
@pytest.fixture(autouse=True)
def mock_heavy_imports() -> Generator[dict[str, Any], None, None]:
    """Mock ultralytics, supervision, and huggingface_hub before importing."""
    mock_ultralytics = MagicMock()
    mock_supervision = MagicMock()
    mock_supervision.Detections = MockDetections

    mock_huggingface = MagicMock()
    mock_huggingface.hf_hub_download.return_value = "/fake/model/path.pt"

    # Inject mocks into sys.modules
    with patch.dict(
        sys.modules,
        {
            "ultralytics": mock_ultralytics,
            "supervision": mock_supervision,
            "huggingface_hub": mock_huggingface,
        },
    ):
        yield {
            "ultralytics": mock_ultralytics,
            "supervision": mock_supervision,
            "huggingface_hub": mock_huggingface,
        }


class TestHeadTrackerInit:
    """Tests for HeadTracker initialization."""

    def test_init_default_params(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test HeadTracker initializes with default parameters."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        assert tracker.confidence_threshold == 0.3
        mock_heavy_imports["huggingface_hub"].hf_hub_download.assert_called_once()
        mock_heavy_imports["ultralytics"].YOLO.assert_called_once()

    def test_init_custom_params(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test HeadTracker initializes with custom parameters."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker(
            model_repo="custom/repo",
            model_filename="custom.pt",
            confidence_threshold=0.5,
            device="cuda",
        )

        assert tracker.confidence_threshold == 0.5
        mock_heavy_imports["huggingface_hub"].hf_hub_download.assert_called_with(
            repo_id="custom/repo", filename="custom.pt"
        )
        mock_model.to.assert_called_with("cuda")

    def test_init_model_load_failure(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test HeadTracker raises on model load failure."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_heavy_imports["huggingface_hub"].hf_hub_download.side_effect = Exception(
            "Download failed"
        )

        with pytest.raises(Exception, match="Download failed"):
            HeadTracker()


class TestSelectBestFace:
    """Tests for HeadTracker._select_best_face method."""

    def test_select_best_face_no_detections(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _select_best_face returns None for empty detections."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        # Empty detections
        detections = MockDetections(
            xyxy=np.array([], dtype=np.float32).reshape(0, 4),
            confidence=np.array([], dtype=np.float32),
        )

        result = tracker._select_best_face(cast("Detections", detections))
        assert result is None

    def test_select_best_face_no_confidence(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _select_best_face returns None when confidence is None."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        detections = MockDetections(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidence=None,
        )

        result = tracker._select_best_face(cast("Detections", detections))
        assert result is None

    def test_select_best_face_below_threshold(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _select_best_face returns None when all below threshold."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker(confidence_threshold=0.5)

        detections = MockDetections(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidence=np.array([0.3], dtype=np.float32),  # Below 0.5 threshold
        )

        result = tracker._select_best_face(cast("Detections", detections))
        assert result is None

    def test_select_best_face_single_detection(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _select_best_face with single valid detection."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker(confidence_threshold=0.3)

        detections = MockDetections(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidence=np.array([0.8], dtype=np.float32),
        )

        result = tracker._select_best_face(cast("Detections", detections))
        assert result == 0

    def test_select_best_face_multiple_detections(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _select_best_face selects best from multiple detections."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker(confidence_threshold=0.3)

        # Three faces: small high-conf, large medium-conf, small low-conf
        detections = MockDetections(
            xyxy=np.array(
                [
                    [100, 100, 150, 150],  # Small (2500 area)
                    [200, 200, 400, 400],  # Large (40000 area)
                    [500, 500, 550, 550],  # Small (2500 area)
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.7, 0.4], dtype=np.float32),
        )

        result = tracker._select_best_face(cast("Detections", detections))
        # Large face with medium confidence should win due to area weighting
        assert result == 1

    def test_select_best_face_filters_low_confidence(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _select_best_face filters out low confidence detections."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker(confidence_threshold=0.5)

        detections = MockDetections(
            xyxy=np.array(
                [
                    [100, 100, 500, 500],  # Large but low conf
                    [200, 200, 250, 250],  # Small but high conf
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.2, 0.8], dtype=np.float32),
        )

        result = tracker._select_best_face(cast("Detections", detections))
        # Only index 1 passes threshold
        assert result == 1


class TestBboxToMpCoords:
    """Tests for HeadTracker._bbox_to_mp_coords method."""

    def test_bbox_center_of_image(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test bbox at center of image returns (0, 0)."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        # Box centered at (320, 240) in 640x480 image
        bbox = np.array([270, 190, 370, 290], dtype=np.float32)
        result = tracker._bbox_to_mp_coords(bbox, w=640, h=480)

        np.testing.assert_array_almost_equal(result, [0.0, 0.0], decimal=5)

    def test_bbox_top_left(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test bbox at top-left returns approximately (-1, -1)."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        # Box at top-left corner
        bbox = np.array([0, 0, 10, 10], dtype=np.float32)
        result = tracker._bbox_to_mp_coords(bbox, w=640, h=480)

        # Center at (5, 5) -> normalized (-0.984, -0.979)
        assert result[0] < -0.9
        assert result[1] < -0.9

    def test_bbox_bottom_right(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test bbox at bottom-right returns approximately (1, 1)."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        # Box at bottom-right corner
        bbox = np.array([630, 470, 640, 480], dtype=np.float32)
        result = tracker._bbox_to_mp_coords(bbox, w=640, h=480)

        # Center at (635, 475) -> normalized (~0.984, ~0.979)
        assert result[0] > 0.9
        assert result[1] > 0.9

    def test_bbox_returns_float32(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test _bbox_to_mp_coords returns float32 array."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        tracker = HeadTracker()

        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        result = tracker._bbox_to_mp_coords(bbox, w=640, h=480)

        assert result.dtype == np.float32
        assert len(result) == 2


class TestGetHeadPosition:
    """Tests for HeadTracker.get_head_position method."""

    def test_get_head_position_no_face(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test get_head_position returns (None, None) when no face detected."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Empty detections
        mock_result = MagicMock()
        mock_result._detections = MockDetections(
            xyxy=np.array([], dtype=np.float32).reshape(0, 4),
            confidence=np.array([], dtype=np.float32),
        )
        mock_model.return_value = [mock_result]

        tracker = HeadTracker()

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        position, roll = tracker.get_head_position(img)

        assert position is None
        assert roll is None

    def test_get_head_position_with_face(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test get_head_position returns valid coords when face detected."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Face centered in image
        mock_result = MagicMock()
        mock_result._detections = MockDetections(
            xyxy=np.array([[270, 190, 370, 290]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
        )
        mock_model.return_value = [mock_result]

        tracker = HeadTracker()

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        position, roll = tracker.get_head_position(img)

        assert position is not None
        assert len(position) == 2
        # Face is centered, so position should be near (0, 0)
        assert abs(position[0]) < 0.1
        assert abs(position[1]) < 0.1
        assert roll == 0.0

    def test_get_head_position_returns_roll_zero(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test get_head_position always returns roll=0."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        mock_result = MagicMock()
        mock_result._detections = MockDetections(
            xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
            confidence=np.array([0.8], dtype=np.float32),
        )
        mock_model.return_value = [mock_result]

        tracker = HeadTracker()

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, roll = tracker.get_head_position(img)

        # Roll is always 0 since no keypoints for angle estimation
        assert roll == 0.0

    def test_get_head_position_inference_error(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test get_head_position handles inference errors gracefully."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Model raises exception during inference
        mock_model.side_effect = RuntimeError("CUDA error")

        tracker = HeadTracker()

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        position, roll = tracker.get_head_position(img)

        assert position is None
        assert roll is None

    def test_get_head_position_off_center_face(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test get_head_position with off-center face."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Face at top-left quadrant
        mock_result = MagicMock()
        mock_result._detections = MockDetections(
            xyxy=np.array([[50, 50, 150, 150]], dtype=np.float32),  # Center at (100, 100)
            confidence=np.array([0.85], dtype=np.float32),
        )
        mock_model.return_value = [mock_result]

        tracker = HeadTracker()

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        position, roll = tracker.get_head_position(img)

        assert position is not None
        # Position should be negative (left and up from center)
        assert position[0] < 0  # Left of center
        assert position[1] < 0  # Above center


class TestGetHeadPositionEdgeCases:
    """Edge case tests for get_head_position."""

    def test_get_head_position_confidence_is_none_after_detection(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test get_head_position skips confidence logging when confidence is None."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Create detections with valid face index BUT confidence is None
        # This tests line 134->139 branch where we have a valid face_idx but
        # detections.confidence is None, so we skip the logging
        mock_result = MagicMock()
        # We need _select_best_face to return a valid index, but confidence to be None
        # for the confidence logging check at line 134

        # Create a custom detection mock that has xyxy but confidence=None
        # but _select_best_face will still return a valid index
        class SpecialDetections:
            def __init__(self) -> None:
                self.xyxy = np.array([[270, 190, 370, 290]], dtype=np.float32)
                # Note: confidence is initially set for _select_best_face
                # but we'll set it to None after
                self._confidence: NDArray[np.float32] | None = np.array([0.9], dtype=np.float32)

            @property
            def confidence(self) -> NDArray[np.float32] | None:
                return self._confidence

            @confidence.setter
            def confidence(self, value: NDArray[np.float32] | None) -> None:
                self._confidence = value

            @classmethod
            def from_ultralytics(cls, results: Any) -> "SpecialDetections":
                detections: SpecialDetections = results._detections
                return detections

        # Create the detections that will pass _select_best_face but have None confidence later
        detection_obj = MockDetections(
            xyxy=np.array([[270, 190, 370, 290]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),  # Valid for _select_best_face
        )

        mock_result._detections = detection_obj
        mock_model.return_value = [mock_result]

        tracker = HeadTracker()

        # Now set confidence to None AFTER the model is set up
        # but this won't help since the detection is created fresh each call

        # Actually, let's patch the tracker's _select_best_face to return 0
        # and use detections with confidence=None
        detection_obj_no_conf = MockDetections(
            xyxy=np.array([[270, 190, 370, 290]], dtype=np.float32),
            confidence=None,  # No confidence
        )

        mock_result._detections = detection_obj_no_conf
        # Patch _select_best_face to return 0 despite None confidence
        original_select = tracker._select_best_face
        object.__setattr__(tracker, "_select_best_face", lambda d: 0)  # Always return first face

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        position, roll = tracker.get_head_position(img)

        # Should still return valid position, just without confidence logging
        assert position is not None
        assert roll == 0.0

        # Restore original method
        object.__setattr__(tracker, "_select_best_face", original_select)


class TestHeadTrackerImportFailure:
    """Tests for import failure handling."""

    def test_import_failure_raises_import_error(self) -> None:
        """Test ImportError is raised when dependencies are missing."""
        import sys
        import importlib

        # Save original modules
        modules_to_remove = [
            "reachy_mini_conversation_app.vision.yolo_head_tracker",
            "supervision",
            "ultralytics",
        ]
        saved_modules = {name: sys.modules.get(name) for name in modules_to_remove}

        try:
            # Remove modules
            for name in modules_to_remove:
                if name in sys.modules:
                    del sys.modules[name]

            # Mock the import to fail
            original_import: Callable[..., Any] = builtins.__import__

            def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name in ("supervision", "ultralytics"):
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import

            try:
                with pytest.raises(ImportError, match="To use YOLO head tracker"):
                    importlib.import_module("reachy_mini_conversation_app.vision.yolo_head_tracker")
            finally:
                builtins.__import__ = original_import
        finally:
            # Restore original modules
            for name, module in saved_modules.items():
                if module is not None:
                    sys.modules[name] = module
                elif name in sys.modules:
                    del sys.modules[name]


class TestHeadTrackerIntegration:
    """Integration tests for HeadTracker."""

    def test_full_detection_pipeline(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test complete detection pipeline."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Simulate detection result
        mock_result = MagicMock()
        mock_result._detections = MockDetections(
            xyxy=np.array([[200, 150, 400, 350]], dtype=np.float32),
            confidence=np.array([0.92], dtype=np.float32),
        )
        mock_model.return_value = [mock_result]

        tracker = HeadTracker(confidence_threshold=0.5)

        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        position, roll = tracker.get_head_position(img)

        assert position is not None
        assert isinstance(position, np.ndarray)
        assert position.dtype == np.float32
        assert -1 <= position[0] <= 1
        assert -1 <= position[1] <= 1
        assert roll == 0.0

    def test_multiple_frames(self, mock_heavy_imports: dict[str, Any]) -> None:
        """Test processing multiple frames."""
        from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_heavy_imports["ultralytics"].YOLO.return_value = mock_model

        # Different positions for each frame
        positions = [
            np.array([[100, 100, 200, 200]], dtype=np.float32),
            np.array([[300, 200, 400, 300]], dtype=np.float32),
            np.array([[400, 300, 500, 400]], dtype=np.float32),
        ]

        tracker = HeadTracker()
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        results = []
        for pos in positions:
            mock_result = MagicMock()
            mock_result._detections = MockDetections(
                xyxy=pos,
                confidence=np.array([0.8], dtype=np.float32),
            )
            mock_model.return_value = [mock_result]

            position, _ = tracker.get_head_position(img)
            results.append(position)

        # All should return valid positions
        assert all(r is not None for r in results)
        # Positions should be different
        assert results[0] is not None and results[1] is not None
        assert not np.allclose(results[0], results[1])
        assert results[2] is not None
        assert not np.allclose(results[1], results[2])
