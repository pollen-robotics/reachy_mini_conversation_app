# Reachy Mini conversation demo

Conversational demo for the Reachy Mini humanoid robot combining OpenAI's realtime APIs, optional vision pipelines, and choreographed motion libraries. The project currently targets internal validation; this document captures the steps needed to run it today and highlights the gaps we still need to close before a public launch.

## Overview
- Real-time audio conversation loop powered by the OpenAI realtime API and `fastrtc` for low-latency streaming.
- Motion control queue that blends scripted dances, recorded emotions, idle breathing, and speech-reactive head wobbling.
- Optional camera worker with YOLO or MediaPipe-based head tracking and LLM-accessible scene capture.
- Simulation flag and non-camera modes stubbed in; hardware robot remains the primary path for now.

## Features
- Async tool dispatch integrates robot motion, camera capture, and optional facial recognition helpers.
- Gradio web UI provides audio chat and transcript display.
- Movement manager keeps real-time control in a dedicated thread with safeguards against abrupt pose changes.
- `.env` driven configuration for OpenAI credentials and Hugging Face caches.

## Requirements
- Python 3.10 or newer (tested with CPython 3.12.1 via `uv`).
- Linux environment with build tooling and GStreamer/GTK headers for `PyGObject`:
  - `sudo apt install build-essential pkg-config python3-venv libgirepository1.0-dev gstreamer1.0-plugins-good gstreamer1.0-pulseaudio libatlas-base-dev` (adjust to your distro).
- Reachy Mini robot or the simulator (simulator wiring is incomplete; see TODOs).
- Microphone, speakers/headphones, and optionally a USB camera supported by `opencv-python`.

## Installation

### Using uv
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12.1  # Create a virtual environment with Python 3.12.1
source .venv/bin/activate
uv sync
```

To include optional vision dependencies:
```
uv sync --extra local_vision        # For local PyTorch/Transformers vision
uv sync --extra yolo_vision         # For YOLO-based vision
uv sync --extra mediapipe_vision    # For MediaPipe-based vision
uv sync --extra all_vision          # For all vision features
```

You can combine extras or include dev dependencies:
```
uv sync --extra all_vision --group dev
```

### Using pip (Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

Install optional extras depending on the feature set you need:

```bash
# Vision stacks (choose at least one if you plan to run head tracking)
pip install -e .[local_vision]
pip install -e .[yolo_vision]
pip install -e .[mediapipe_vision]
pip install -e .[all_vision]        # installs every vision extra

# Tooling for development workflows
pip install -e .[dev]
```

Some wheels (e.g. PyTorch) are large and require compatible CUDA or CPU builds. Expect the `local_vision` extra to take significantly more disk space than YOLO or MediaPipe.

## Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers. | GPU recommended; installs large packages (~2 GB).
| `yolo_vision` | YOLOv8 tracking via `ultralytics` and `supervision`. | CPU friendly; supports the `--head-tracker yolo` option.
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe. | Works on CPU; enables `--head-tracker mediapipe`.
| `all_vision` | Convenience alias installing every vision extra. | Only use if you need to experiment with all providers.
| `dev` | Developer tooling (`pytest`, `ruff`). | Add on top of either base or `all_vision` environments.

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the required values, notably the OpenAI API key.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. Grants access to the OpenAI realtime endpoint.
| `MODEL_NAME` | Override the realtime model (defaults to `gpt-4o-realtime-preview`).
| `OPENAI_VISION_MODEL` | Model used when sending captured images back to OpenAI.
| `HF_HOME` | Cache directory for local Hugging Face downloads.
| `HF_TOKEN` | Optional token for private Hugging Face models (needed for some emotion libraries).

## Running the demo

Activate your virtual environment, ensure the Reachy Mini robot (or simulator) is reachable, then launch:

```bash
reachy-mini-conversation-demo
```

The app starts a Gradio UI served locally. When running on a headless host, combine `--headless` with SSH port forwarding or use a browser on the machine itself.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--sim` | `False` | Intended to toggle the simulator (currently parsed but not wired through to `ReachyMini`). |
| `--head-tracker {yolo,mediapipe}` | `None` | Select a head-tracking backend when a camera is available. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or head tracking. |
| `--headless` | `False` | Suppress launching the Gradio UI (useful on remote machines). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples
- Run on hardware with MediaPipe head tracking:

  ```bash
  reachy-mini-conversation-demo --head-tracker mediapipe
  ```

- Disable the camera pipeline (audio-only conversation):

  ```bash
  reachy-mini-conversation-demo --no-camera
  ```

- Prepare for simulator work (flag currently no-op but reserved):

  ```bash
  reachy-mini-conversation-demo --sim --headless
  ```

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and optionally query a vision backend. | Requires camera worker; vision analysis depends on selected extras. |
| `head_tracking` | Enable or disable face-tracking offsets. | Camera worker with configured head tracker. |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Requires access to private choreography library and movement manager. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face assets. | Needs `HF_TOKEN` and the recorded emotions dataset. |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `get_person_name` | Attempt DeepFace-based recognition of the current person. | Disabled by default (`ENABLE_FACE_RECOGNITION=False`); requires `deepface` and a local face database. |
| `do_nothing` | Explicitly remain idle. | Core install only. |

## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive—offload blocking work using the helpers in `tools.py`.

## TODO before public release
- [ ] Wire the `--sim` flag through to `ReachyMini` and document simulator prerequisites.
- [ ] Replace the `git+ssh` dependencies with published wheels or read-only URLs so new users can install without deploy keys.
- [ ] Audit motion, audio, and prompt assets so that `package-data` lists every required non-Python file.
- [ ] Provide cross-platform installation notes (macOS, Windows) and verify PyGObject availability.
- [ ] Add integration tests that exercise the movement manager and tool dispatch in simulation mode.
- [ ] Record screenshots or short clips of the Gradio UI for the README once design stabilises.

## License

Reachy Mini Conversation Demo is released under the terms described in `LICENSE`.
