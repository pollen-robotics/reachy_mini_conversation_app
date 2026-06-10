"""Tier-2 closing-the-loop visual witness: capture a frame + describe the scene.

The on-demand visual leg of the loop (``docs/closing-the-loop.md`` → Tier 2). A
camera on the *coding laptop*, aimed at the robot, grabs a still and a vision
model answers a question about it ("did the antennas move? is the head
centred?") — so an agent can confirm a *motion* the audio/event tiers can't see.

Expensive and laggy, so this is on-demand only — never a continuous stream.

Two entry points:

* ``capture_frame()``   — grab one frame, save a JPEG, return its path + size.
* ``describe_scene(p)`` — grab (or load) a frame and run the VLM with prompt ``p``.

The describe half reuses the repo's local SmolVLM2 (``vision.local_vision``); the
heavy imports (``cv2`` for capture, torch/transformers via ``local_vision``) are
lazy, and the capture/describe/save/load seams are injectable so the logic is
unit-tested without a camera or a model. Frames are written under
``VISUAL_WITNESS_FRAME_DIR`` (outside the repo — keep room imagery local and
short-lived, per the doc's privacy guardrail).

Env:

* ``VISUAL_WITNESS_CAMERA_DEVICE``  — OpenCV device index (default ``0``).
* ``VISUAL_WITNESS_FRAME_DIR``      — where JPEGs are saved.
* ``VISUAL_WITNESS_CAPTURE_COMMAND`` — optional shell command that grabs one
  frame on the *host* and writes it to ``VISUAL_WITNESS_CAPTURE_OUTPUT``. When
  set it replaces the OpenCV capture entirely — the escape hatch for setups
  where no V4L2 device exists (e.g. WSL2, where webcam streaming over usbip
  stalls; an ``ffmpeg.exe -f dshow …`` one-liner on the Windows host works).
* ``VISUAL_WITNESS_CAPTURE_OUTPUT`` — path (as seen by *this* process) where
  the command leaves the image, e.g. ``/mnt/c/...`` for a Windows-drive target.
"""

from __future__ import annotations
import os
import time
from typing import Any, Callable, Optional


DEFAULT_DEVICE = int(os.getenv("VISUAL_WITNESS_CAMERA_DEVICE", "0"))
FRAME_DIR = os.getenv("VISUAL_WITNESS_FRAME_DIR", os.path.expanduser("~/.robot_comic/observer/frames"))
CAPTURE_COMMAND = os.getenv("VISUAL_WITNESS_CAPTURE_COMMAND", "")
CAPTURE_OUTPUT = os.getenv("VISUAL_WITNESS_CAPTURE_OUTPUT", "")
CAPTURE_COMMAND_TIMEOUT_S = 30.0


class VisualWitnessError(RuntimeError):
    """Capture or describe failed (no camera, unreadable frame, bad prompt)."""


def _command_capture(
    command: str,
    output: str,
    *,
    loader: Optional[Callable[[str], Any]] = None,
) -> Any:
    """Run a host capture command, then load the image it wrote to ``output``.

    The stale ``output`` (if any) is removed first so a silently-failing
    command can't hand back a previous frame.
    """
    import subprocess

    if not output:
        raise VisualWitnessError(
            "VISUAL_WITNESS_CAPTURE_COMMAND is set but VISUAL_WITNESS_CAPTURE_OUTPUT is not; "
            "it must point at the image file the command writes"
        )
    if os.path.exists(output):
        os.remove(output)
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=CAPTURE_COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise VisualWitnessError(
            f"capture command timed out after {CAPTURE_COMMAND_TIMEOUT_S:g}s: {command!r}"
        ) from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise VisualWitnessError(f"capture command failed (exit {proc.returncode}): {detail or command!r}")
    if not os.path.exists(output):
        raise VisualWitnessError(f"capture command succeeded but wrote nothing to {output!r}")
    return (loader or _default_loader)(output)


def _default_capture(device: int) -> Any:
    """Grab one BGR frame: via the host capture command if configured, else OpenCV."""
    if CAPTURE_COMMAND:
        return _command_capture(CAPTURE_COMMAND, CAPTURE_OUTPUT)
    import cv2

    cap = cv2.VideoCapture(device)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VisualWitnessError(f"could not read a frame from camera device {device!r}")
        return frame
    finally:
        cap.release()


def _default_loader(path: str) -> Any:
    """Load an image file as a BGR frame. Lazy ``cv2`` import."""
    import cv2

    frame = cv2.imread(path)
    if frame is None:
        raise VisualWitnessError(f"could not read image: {path}")
    return frame


def _default_saver(frame: Any, save_dir: Optional[str] = None) -> str:
    """Write a BGR frame to a timestamped JPEG and return its path. Lazy ``cv2``."""
    import cv2

    out_dir = save_dir or FRAME_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"frame_{int(time.time() * 1000)}.jpg")
    if not cv2.imwrite(path, frame):
        raise VisualWitnessError(f"could not write frame to {path}")
    return path


_PROCESSOR: Any = None


def _default_describer(frame: Any, prompt: str) -> str:
    """Describe a BGR frame with the local SmolVLM2 model (lazy, cached).

    Loads + initializes the vision processor once per process (the model load is
    expensive), then reuses it. Imports torch/transformers lazily via
    ``vision.local_vision``.
    """
    global _PROCESSOR
    if _PROCESSOR is None:
        from robot_comic.vision.local_vision import initialize_vision_processor

        _PROCESSOR = initialize_vision_processor()
    return str(_PROCESSOR.process_image(frame, prompt))


def _frame_size(frame: Any) -> tuple[int, int]:
    """Return ``(width, height)`` from a BGR ndarray-like frame."""
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2:
        raise VisualWitnessError("captured frame has no 2-D shape")
    return int(shape[1]), int(shape[0])


def capture_frame(
    *,
    device: int = DEFAULT_DEVICE,
    capture: Optional[Callable[[], Any]] = None,
    saver: Optional[Callable[[Any], str]] = None,
    save_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Grab one frame, save it as a JPEG, and return its path + dimensions.

    ``capture``/``saver`` are injectable so this is testable without a camera.
    """
    capture = capture or (lambda: _default_capture(device))
    saver = saver or (lambda frame: _default_saver(frame, save_dir))
    frame = capture()
    width, height = _frame_size(frame)
    return {"captured": True, "path": saver(frame), "width": width, "height": height}


def describe_scene(
    prompt: str,
    *,
    device: int = DEFAULT_DEVICE,
    image_path: Optional[str] = None,
    capture: Optional[Callable[[], Any]] = None,
    saver: Optional[Callable[[Any], str]] = None,
    loader: Optional[Callable[[str], Any]] = None,
    describer: Optional[Callable[[Any, str], str]] = None,
) -> dict[str, Any]:
    """Capture (or load) a frame and return a vision-model answer to ``prompt``.

    With ``image_path`` it describes that file; otherwise it grabs a fresh frame
    from ``device`` and saves it. All side-effecting seams are injectable so the
    orchestration is unit-tested without a camera or the VLM.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise VisualWitnessError("prompt must be a non-empty string")
    describer = describer or _default_describer

    if image_path is not None:
        loader = loader or _default_loader
        frame = loader(image_path)
        frame_path = image_path
    else:
        capture = capture or (lambda: _default_capture(device))
        saver = saver or (lambda f: _default_saver(f, None))
        frame = capture()
        frame_path = saver(frame)

    return {"prompt": prompt, "description": describer(frame, prompt), "frame_path": frame_path}
