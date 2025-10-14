# Reachy Mini conversation demo

Conversational demo for the Reachy Mini robot combining OpenAI's realtime APIs, vision pipelines, and choreographed motion libraries.

![Reachy Mini Dance](src/reachy_mini_conversation_demo/images/reachy_mini_dance.gif)

## Architecture

The demo follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="src/reachy_mini_conversation_demo/images/conversation_demo_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Overview
- Real-time audio conversation loop powered by the OpenAI realtime API and `fastrtc` for low-latency streaming.
- Vision processing uses gpt-realtime by default (when camera tool is used), with optional local vision processing using SmolVLM2 model running on-device (CPU/GPU/MPS) via `--local-vision` flag.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and face-tracking.
- Async tool dispatch integrates robot motion, camera capture, and optional face-tracking capabilities through a Gradio web UI with live transcripts.

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

### Using pip (test on Ubuntu 24.04)

```bash
python -m venv .venv # Create a virtual environment
source .venv/bin/activate
pip install -e .
```

Install optional extras depending on the feature set you need:

```bash
# Vision stacks (choose at least one if you plan to run face tracking)
pip install -e .[local_vision]
pip install -e .[yolo_vision]
pip install -e .[mediapipe_vision]
pip install -e .[all_vision]        # installs every vision extra

# Tooling for development workflows
pip install -e .[dev]
```

Some wheels (e.g. PyTorch) are large and require compatible CUDA or CPU builds—make sure your platform matches the binaries pulled in by each extra.

## Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers. | GPU recommended; ensure compatible PyTorch builds for your platform.
| `yolo_vision` | YOLOv8 tracking via `ultralytics` and `supervision`. | CPU friendly; supports the `--head-tracker yolo` option.
| `mediapipe_vision` | Lightweight landmark tracking with MediaPipe. | Works on CPU; enables `--head-tracker mediapipe`.
| `all_vision` | Convenience alias installing every vision extra. | Install when you want the flexibility to experiment with every provider.
| `dev` | Developer tooling (`pytest`, `ruff`). | Add on top of either base or `all_vision` environments.

## Configuration

1. Copy `.env.example` to `.env`.
2. Fill in the required values, notably the OpenAI API key.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. Grants access to the OpenAI realtime endpoint.
| `MODEL_NAME` | Override the realtime model (defaults to `gpt-realtime`). Used for both conversation and vision (unless `--local-vision` flag is used).
| `HF_HOME` | Cache directory for local Hugging Face downloads (only used with `--local-vision` flag, defaults to `./cache`).
| `HF_TOKEN` | Optional token for Hugging Face models (only used with `--local-vision` flag, falls back to `huggingface-cli login`).
| `LOCAL_VISION_MODEL` | Hugging Face model path for local vision processing (only used with `--local-vision` flag, defaults to `HuggingFaceTB/SmolVLM2-2.2B-Instruct`).

## Running the demo

Activate your virtual environment, ensure the Reachy Mini robot (or simulator) is reachable, then launch:

```bash
reachy-mini-conversation-demo
```

By default, the app runs in console mode for direct audio interaction. Use the `--gradio` flag to launch a web UI served locally at http://127.0.0.1:7860/ (required when running in simulation mode). With a camera attached, vision is handled by the gpt-realtime model when the camera tool is used. For local vision processing, use the `--local-vision` flag to process frames periodically using the SmolVLM2 model. Additionally, you can enable face tracking via YOLO or MediaPipe pipelines depending on the extras you installed.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Select a face-tracking backend when a camera is available. YOLO is implemented locally, MediaPipe comes from the `reachy_mini_toolbox` package. Requires the matching optional extra. |
| `--no-camera` | `False` | Run without camera capture or face tracking. |
| `--local-vision` | `False` | Use local vision model (SmolVLM2) for periodic image processing instead of gpt-realtime vision. Requires `local_vision` extra to be installed. |
| `--gradio` | `False` | Launch the Gradio web UI. Without this flag, runs in console mode. Required when running in simulation mode. |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |


### Examples
- Run on hardware with MediaPipe face tracking:

  ```bash
  reachy-mini-conversation-demo --head-tracker mediapipe
  ```

- Run with local vision processing (requires `local_vision` extra):

  ```bash
  reachy-mini-conversation-demo --local-vision
  ```

- Disable the camera pipeline (audio-only conversation):

  ```bash
  reachy-mini-conversation-demo --no-camera
  ```

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `camera` | Capture the latest camera frame and send it to gpt-realtime for vision analysis. | Requires camera worker; uses gpt-realtime vision by default. |
| `head_tracking` | Enable or disable face-tracking offsets (not facial recognition - only detects and tracks face position). | Camera worker with configured head tracker. |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face assets. | Needs `HF_TOKEN` for the recorded emotions dataset. |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `do_nothing` | Explicitly remain idle. | Core install only. |

## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive => offload blocking work using the helpers in `tools.py`.

## License
Apache 2.0
