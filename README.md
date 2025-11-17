# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining OpenAI's realtime APIs, vision pipelines, and choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Architecture

The app follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Overview
- Real-time audio conversation loop powered by the OpenAI realtime API and `fastrtc` for low-latency streaming.
- Vision processing uses gpt-realtime by default (when camera tool is used), with optional local vision processing using SmolVLM2 model running on-device (CPU/GPU/MPS) via `--local-vision` flag.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and face-tracking.
- Async tool dispatch integrates robot motion, camera capture, and optional face-tracking capabilities through a Gradio web UI with live transcripts.

## Installation

> [!IMPORTANT]
> Windows support is currently experimental and has not been extensively tested. Use with caution.

### Using uv
You can set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12.1  # Create a virtual environment with Python 3.12.1
source .venv/bin/activate
uv sync
```

> [!NOTE]
> To reproduce the exact dependency set from this repo's `uv.lock`, run `uv sync` with `--locked` (or `--frozen`). This ensures `uv` installs directly from the lockfile without re-resolving or updating any versions.

To include optional dependencies:
```
uv sync --extra reachy_mini_wireless # For wireless Reachy Mini with GStreamer support
uv sync --extra local_vision         # For local PyTorch/Transformers vision
uv sync --extra yolo_vision          # For YOLO-based vision
uv sync --extra mediapipe_vision     # For MediaPipe-based vision
uv sync --extra all_vision           # For all vision features
```

You can combine extras or include dev dependencies:
```
uv sync --extra all_vision --group dev
```

### Using pip

```bash
python -m venv .venv # Create a virtual environment
source .venv/bin/activate
pip install -e .
```

Install optional extras depending on the feature set you need:

```bash
# Wireless Reachy Mini support
pip install -e .[reachy_mini_wireless]

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
| `reachy_mini_wireless` | Wireless Reachy Mini with GStreamer support. | Required for wireless versions of Reachy Mini, includes GStreamer dependencies.
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

## Running the app

Activate your virtual environment, ensure the Reachy Mini robot (or simulator) is reachable, then launch:

```bash
reachy-mini-conversation-app
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
  reachy-mini-conversation-app --head-tracker mediapipe
  ```

- Run with local vision processing (requires `local_vision` extra):

  ```bash
  reachy-mini-conversation-app --local-vision
  ```

- Disable the camera pipeline (audio-only conversation):

  ```bash
  reachy-mini-conversation-app --no-camera
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

## Using custom profiles
Create custom profiles with dedicated instructions and enabled tools! 

Set `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `src/reachy_mini_conversation_app/profiles/<name>/` (see `.env.example`). If unset, the `default` profile is used.

Each profile requires two files: `instructions.txt` (prompt text) and `tools.txt` (list of allowed tools), and optionally contains custom tools implementations.

### Custom instructions
Write plain-text prompts in `instructions.txt`. To reuse shared prompt pieces, add lines like:
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
Each placeholder pulls the matching file under `src/reachy_mini_conversation_app/prompts/` (nested paths allowed). See `src/reachy_mini_conversation_app/profiles/example/` for a reference layout.

### Enabling tools
List enabled tools in `tools.txt`, one per line; prefix with `#` to comment out. For example:

```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the shared library `src/reachy_mini_conversation_app/tools/` (e.g., `dance`, `head_tracking`). 

### Custom tools
On top of built-in tools found in the shared library, you can implement custom tools specific to your profile by adding Python files in the profile folder. 
Custom tools must subclass `reachy_mini_conversation_app.tools.core_tools.Tool` (see `profiles/example/sweep_look.py`).


#### Local custom tools using the rmscript scripting language
You can define custom tools using the `rmscript` scripting language and add them to tools.txt like the other tools.
See examples in the `src/reachy_mini_conversation_app/profiles/example/` profile.

## RMscript language

rmscript is a natural language-inspired programming language designed to make robot programming accessible and fun for children. It compiles to Python code that controls the Reachy Mini robot.

Example, create a `hello.rmscript` file with the following content:
```rmscript
DESCRIPTION Wave hello to someone
antenna up
wait 1s
antenna down
look left
look right
look center
```

Test the script using the provided runner (after starting the reachy-mini-daemon):

```
python src/reachy_mini_conversation_app/rmscript/run_rmscript.py path/to/hello.rmscript
```

### File Structure

Every rmscript file has a simple structure:

```rmscript
DESCRIPTION Wave hello to someone
# Your commands here
look left
wait 1s
```

- **Tool name**: Automatically derived from the filename (e.g., `wave_hello.rmscript` → tool name is `wave_hello`) Don't forget to add it to the tools.txt file to have the tool actually loaded.
- **DESCRIPTION** (optional): One-line description used for LLM tool registration

You can use # to add comments throughout the script.


### Basic Commands

```rmscript
# Comments start with #

# Movement commands
look left
turn right
antenna up
head forward 10

# Wait command
wait 2s
wait 0.5s

# Camera command
picture

# Sound playback
play mysound
play othersound pause
```

rmscript is case-insensitive for keywords.

### Look (Head Orientation)

Control the robot's head orientation (pitch and yaw):

```rmscript
look left          # Turn head left (30° default)
look right 45      # Turn head right 45°
look up           # Tilt head up (30° default)
look down 20      # Tilt head down 20°
look center       # Return to center position

# Synonyms
look straight     # Same as center
look neutral      # Same as center
```


### Turn (Body Rotation)

Rotate the robot's body (the head rotates together with the body):

```rmscript
turn left         # Rotate body and head left (30° default)
turn right 90     # Rotate body and head right 90°
turn center       # Face forward
```


### Antenna

Control the antenna positions using multiple syntaxes:

**Clock Position (Numeric 0-12):**
```rmscript
antenna both 0       # 0 o'clock = 0° (straight up)
antenna both 3       # 3 o'clock = 90° (external/right)
antenna both 6       # 6 o'clock = 180° (straight down)
antenna both 9       # 9 o'clock = -90° (internal/left)
antenna left 4.5     # Left antenna to 4.5 o'clock (135°)
```
The clock position is `as seen from the user facing the robot`, while the left/right antenna designation is from the robot's perspective.

**Clock Keywords:**
```rmscript
antenna both high    # 0° (high position)
antenna both ext     # 90° (external)
antenna both low     # 180° (low position)
antenna both int     # -90° (internal)
```

**Directional Keywords (Natural Language):**
```rmscript
antenna both up      # 0° (up)
antenna both right   # 90° (right/external)
antenna both down    # 180° (down)
antenna both left    # -90° (left/internal)

# Individual antenna control
antenna left up      # Left antenna pointing up
antenna right down   # Right antenna pointing down
```
### Head Translation

Move the head forward/back/left/right/up/down in space:

```rmscript
head forward 10    # Move head forward 10mm
head back 5        # Move head back 5mm
head left 8        # Move head left 8mm
head right 8       # Move head right 8mm
head up 5          # Move head up 5mm
head down 3        # Move head down 3mm
```


### Tilt (Head Roll)

Tilt the head side-to-side:

```rmscript
tilt left 15       # Tilt head left
tilt right 15      # Tilt head right
tilt center        # Return to level
```

### Wait

Pause between movements:

```rmscript
wait 1s           # Wait 1 second
wait 0.5s         # Wait 0.5 seconds
wait 2.5s         # Wait 2.5 seconds
```

**Important:** The `s` suffix is **required** for consistency. `wait 1` will produce a compilation error.

### Picture

Take a picture with the robot's camera and add it to the conversation:

```rmscript
picture           # Take a picture
```

The picture command captures a frame from the camera and returns it as a base64-encoded image.

### Sound Playback

Play a sound file stored in the local profile directory:

```rmscript
play mysound         # Play mysound.wav
```

Will search for `mysound.*` in the profile directory and play it, using sound extensions (.wav, .mp3, etc.).
It will be played async, so the script continues executing while the sound plays.
If you want to pause the script until the sound finishes, use the `pause` keyword:

```rmscript
play mysound pause    # Play mysound.wav and wait until it finishes
```

you can also use other keywords like "block",...

### Reference document

See `src/reachy_mini_conversation_app/rmscript/rmscript_reference_doc.md` for the full reference document.
(WIP : it might not be fully up to date with the latest changes)

## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive => offload blocking work using the helpers in `tools.py`.

## License
Apache 2.0
