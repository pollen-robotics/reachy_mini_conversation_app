"""Tests for the Roast tool."""

from __future__ import annotations
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_comic.tools.core_tools import ToolDependencies


_PROFILE_PATH = Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "roast.py"


def _load_roast_module():
    spec = importlib.util.spec_from_file_location("roast_test_module", _PROFILE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["roast_test_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def roast_mod():
    """Module-scoped fixture that loads the roast module from its profile path."""
    return _load_roast_module()


@pytest.fixture
def Roast(roast_mod):
    """Fixture that returns the Roast class from the loaded module."""
    return roast_mod.Roast


@pytest.fixture
def parse_extraction(roast_mod):
    """Fixture that returns the _parse_extraction helper from the loaded module."""
    return roast_mod._parse_extraction


def make_deps(scan_response: str = "PERSON: center", extraction_response: str = "") -> ToolDependencies:
    """Build a mock ToolDependencies whose vision_processor returns the given canned responses."""
    deps = MagicMock(spec=ToolDependencies)
    deps.motion_duration_s = 0.0
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    deps.camera_worker = MagicMock()
    deps.camera_worker.get_latest_frame.return_value = frame
    deps.vision_processor = MagicMock()
    deps.vision_processor.process_image.side_effect = [scan_response, extraction_response]
    deps.movement_manager = MagicMock()
    deps.reachy_mini = MagicMock()
    deps.reachy_mini.get_current_head_pose.return_value = MagicMock()
    deps.reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    return deps


EXTRACTION_TEXT = (
    "hair: thinning on top, attempting a combover\n"
    "clothing: wrinkled button-down, tucked in badly\n"
    "build: stocky, comfortable with it\n"
    "expression: trying to look relaxed, not pulling it off\n"
    "standout: the mustache. that's the whole story.\n"
    "energy: nervous, fidgety"
)


# --- _parse_extraction ---


def test_parse_extraction_returns_all_fields(parse_extraction):
    """All six labelled fields are extracted correctly from a well-formed response."""
    result = parse_extraction(EXTRACTION_TEXT)
    assert result["hair"] == "thinning on top, attempting a combover"
    assert result["clothing"] == "wrinkled button-down, tucked in badly"
    assert result["build"] == "stocky, comfortable with it"
    assert result["expression"] == "trying to look relaxed, not pulling it off"
    assert result["standout"] == "the mustache. that's the whole story."
    assert result["energy"] == "nervous, fidgety"


def test_parse_extraction_missing_field_returns_unknown(parse_extraction):
    """Fields absent from the response are substituted with 'unknown'."""
    result = parse_extraction("hair: thinning\nclothing: wrinkled")
    assert result["hair"] == "thinning"
    assert result["clothing"] == "wrinkled"
    assert result["build"] == "unknown"
    assert result["standout"] == "unknown"


# --- Roast tool ---


@pytest.mark.asyncio
async def test_roast_returns_no_subject_when_no_person_detected(Roast):
    """When the scene scan finds no person, the tool returns {no_subject: True}."""
    deps = make_deps(scan_response="NO PERSON")
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert result == {"no_subject": True}
    assert deps.camera_worker.get_latest_frame.call_count == 1


@pytest.mark.asyncio
async def test_roast_moves_head_left_when_person_on_left(Roast):
    """A person on the left side triggers a head-left movement command."""
    deps = make_deps(scan_response="PERSON: left", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    deps.movement_manager.queue_move.assert_called_once()
    left_move = deps.movement_manager.queue_move.call_args[0][0]
    assert left_move.target_head_pose is not None


@pytest.mark.asyncio
async def test_roast_moves_head_right_when_person_on_right(Roast):
    """A person on the right side triggers a head-right movement command."""
    deps = make_deps(scan_response="PERSON: right", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    deps.movement_manager.queue_move.assert_called_once()
    right_move = deps.movement_manager.queue_move.call_args[0][0]
    assert right_move.target_head_pose is not None


@pytest.mark.asyncio
async def test_roast_left_and_right_produce_different_head_poses(Roast):
    """Left and right scene positions map to numerically distinct head-pose targets."""
    deps_left = make_deps(scan_response="PERSON: left", extraction_response=EXTRACTION_TEXT)
    deps_right = make_deps(scan_response="PERSON: right", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps_left)
        await Roast()(deps_right)
    left_pose = deps_left.movement_manager.queue_move.call_args[0][0].target_head_pose
    right_pose = deps_right.movement_manager.queue_move.call_args[0][0].target_head_pose
    import numpy as np

    assert not np.allclose(left_pose, right_pose), "left and right head poses must differ"


@pytest.mark.asyncio
async def test_roast_returns_parsed_fields_on_success(Roast):
    """A successful run returns a dict with all six parsed roast-target fields."""
    deps = make_deps(scan_response="PERSON: center", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert result["hair"] == "thinning on top, attempting a combover"
    assert result["standout"] == "the mustache. that's the whole story."
    assert result["energy"] == "nervous, fidgety"


@pytest.mark.asyncio
async def test_roast_captures_two_frames(Roast):
    """The camera is sampled exactly twice: once for the scene scan and once for extraction."""
    deps = make_deps(scan_response="PERSON: center", extraction_response=EXTRACTION_TEXT)
    with patch("asyncio.sleep"):
        await Roast()(deps)
    assert deps.camera_worker.get_latest_frame.call_count == 2


@pytest.mark.asyncio
async def test_roast_returns_error_when_camera_worker_unavailable(Roast):
    """Missing camera_worker returns an error dict rather than raising."""
    deps = make_deps()
    deps.camera_worker = None
    result = await Roast()(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_roast_returns_error_when_no_frame_available(Roast):
    """A None frame from the camera worker returns an error dict."""
    deps = make_deps()
    deps.camera_worker.get_latest_frame.return_value = None
    result = await Roast()(deps)
    assert "error" in result


@pytest.mark.asyncio
async def test_roast_falls_back_to_b64_when_no_vision_processor(Roast):
    """Without a local vision processor the tool returns a base64 frame for remote interpretation."""
    deps = make_deps()
    deps.vision_processor = None
    with patch("asyncio.sleep"):
        result = await Roast()(deps)
    assert "b64_scene" in result
    assert "extraction_prompt" in result
