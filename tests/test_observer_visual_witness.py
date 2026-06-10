"""Tests for the Tier-2 visual witness (capture_frame / describe_scene).

Exercises the orchestration and frame-size logic with injected capture / save /
load / describe seams, so no camera, OpenCV, or vision model is needed. The real
``cv2`` capture and SmolVLM2 describe paths run only with hardware/a model.
"""

from __future__ import annotations
import sys

import pytest

from robot_comic.observer import visual_witness as vw
from robot_comic.observer.visual_witness import VisualWitnessError, capture_frame, describe_scene


def _fake_frame(width: int = 640, height: int = 480):
    np = pytest.importorskip("numpy")
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _frame_size
# ---------------------------------------------------------------------------
def test_frame_size_reads_width_height() -> None:
    assert vw._frame_size(_fake_frame(640, 480)) == (640, 480)


def test_frame_size_rejects_shapeless() -> None:
    with pytest.raises(VisualWitnessError, match="2-D shape"):
        vw._frame_size(object())


# ---------------------------------------------------------------------------
# capture_frame
# ---------------------------------------------------------------------------
def test_capture_frame_saves_and_reports_size() -> None:
    frame = _fake_frame(320, 240)
    saved: list = []

    def fake_saver(f) -> str:
        saved.append(f)
        return "/tmp/frame_test.jpg"

    result = capture_frame(capture=lambda: frame, saver=fake_saver)
    assert result == {"captured": True, "path": "/tmp/frame_test.jpg", "width": 320, "height": 240}
    assert saved == [frame]


# ---------------------------------------------------------------------------
# _command_capture (host capture-command fallback)
# ---------------------------------------------------------------------------
def test_command_capture_runs_command_and_loads_output(tmp_path) -> None:
    frame = _fake_frame()
    out = tmp_path / "shot.jpg"
    loaded: list = []

    def fake_loader(path: str):
        loaded.append(path)
        return frame

    cmd = f"\"{sys.executable}\" -c \"open(r'{out}', 'wb').write(b'jpg')\""
    assert vw._command_capture(cmd, str(out), loader=fake_loader) is frame
    assert loaded == [str(out)]


def test_command_capture_requires_output_path() -> None:
    with pytest.raises(VisualWitnessError, match="CAPTURE_OUTPUT"):
        vw._command_capture("true", "")


def test_command_capture_rejects_failing_command(tmp_path) -> None:
    out = tmp_path / "shot.jpg"
    with pytest.raises(VisualWitnessError, match="exit"):
        vw._command_capture(f'"{sys.executable}" -c "import sys; sys.exit(3)"', str(out))


def test_command_capture_removes_stale_output_and_rejects_missing(tmp_path) -> None:
    out = tmp_path / "shot.jpg"
    out.write_bytes(b"stale")
    cmd = f'"{sys.executable}" -c "pass"'  # succeeds but writes nothing
    with pytest.raises(VisualWitnessError, match="wrote nothing"):
        vw._command_capture(cmd, str(out))
    assert not out.exists()


def test_default_capture_prefers_capture_command(monkeypatch, tmp_path) -> None:
    frame = _fake_frame()
    out = tmp_path / "shot.jpg"
    monkeypatch.setattr(vw, "CAPTURE_COMMAND", "capture-it")
    monkeypatch.setattr(vw, "CAPTURE_OUTPUT", str(out))
    seen = {}

    def fake_command_capture(command: str, output: str, **kwargs):
        seen["command"] = command
        seen["output"] = output
        return frame

    monkeypatch.setattr(vw, "_command_capture", fake_command_capture)
    assert vw._default_capture(0) is frame
    assert seen == {"command": "capture-it", "output": str(out)}


# ---------------------------------------------------------------------------
# describe_scene
# ---------------------------------------------------------------------------
def test_describe_scene_rejects_empty_prompt() -> None:
    with pytest.raises(VisualWitnessError, match="non-empty"):
        describe_scene("   ", describer=lambda f, p: "x")


def test_describe_scene_captures_then_describes() -> None:
    frame = _fake_frame()
    seen = {}

    def fake_describer(f, prompt: str) -> str:
        seen["frame"] = f
        seen["prompt"] = prompt
        return "the antennas are raised"

    result = describe_scene(
        "did the antennas move?",
        capture=lambda: frame,
        saver=lambda f: "/tmp/frame_cap.jpg",
        describer=fake_describer,
    )
    assert result["description"] == "the antennas are raised"
    assert result["prompt"] == "did the antennas move?"
    assert result["frame_path"] == "/tmp/frame_cap.jpg"
    assert seen["frame"] is frame


def test_describe_scene_uses_image_path_via_loader() -> None:
    frame = _fake_frame()
    loaded: list = []

    def fake_loader(path: str):
        loaded.append(path)
        return frame

    result = describe_scene(
        "what is in the frame?",
        image_path="/tmp/some.jpg",
        loader=fake_loader,
        describer=lambda f, p: "a small robot",
    )
    assert loaded == ["/tmp/some.jpg"]
    assert result["frame_path"] == "/tmp/some.jpg"
    assert result["description"] == "a small robot"
