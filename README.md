---
title: Robot Comic
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Robot Comic — comedic Reachy Mini voice app
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Robot Comic

---

## Project Persona

The comedian profile shipped in this repository is **The House Comedian** — a synthetic
stand-up MC persona designed to test low-latency robotic social interaction in a comedic
context. The persona is an original composite that does not reference or represent any
real individual.

Operator-specific persona content (including any real-comedian stylistic research) is
kept separately in the operator's private persona library and loaded at runtime via the
`REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` environment variable.  See
[docs/external_profiles.md](docs/external_profiles.md) for how to configure external profiles.

---

Conversational app for the Reachy Mini robot combining realtime voice backends, vision pipelines, and choreographed motion libraries.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents
- [Project Persona](#project-persona)
- [External Profiles](docs/external_profiles.md)
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Advanced features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview
- Real-time audio conversation loop with `fastrtc` for low-latency streaming. Supported backends:
  - **Hugging Face** - default, using the built-in Hugging Face server or your own local endpoint.
  - **OpenAI Realtime** (`gpt-realtime`) - requires `OPENAI_API_KEY`.
  - **Gemini Live** (`gemini-3.1-flash-live-preview`) - requires `GEMINI_API_KEY`.
- Vision processing uses the selected realtime backend by default (when the camera tool is used), with optional on-device local vision using SmolVLM2 (CPU/GPU/MPS) via `--local-vision`.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble and head-tracking.
- Async tool dispatch integrates robot motion, camera capture, and optional head-tracking capabilities through a browser admin UI with live transcripts.

## Architecture

The app follows a layered architecture connecting the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary><b>Using uv (recommended)</b></summary>

Set up the project quickly using [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

> **Note:** To reproduce the exact dependency set from this repo's `uv.lock`, run `uv sync --frozen`. This ensures `uv` installs directly from the lockfile without re-resolving or updating any versions.

**Install optional features:**
```bash
uv sync --extra local_vision         # Local PyTorch/Transformers vision
uv sync --extra yolo_vision          # YOLO face-detection backend for head tracking
uv sync --extra local_stt            # On-device Moonshine speech-to-text
uv sync --extra all_vision           # All vision features
```

Combine extras or include dev dependencies:
```bash
uv sync --extra all_vision --group dev
```

</details>

<details>
<summary><b>Using pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Install optional features:**
```bash
pip install -e .[local_vision]          # Local vision stack
pip install -e .[yolo_vision]           # YOLO face-detection backend for head tracking
pip install -e .[all_vision]            # All vision features
pip install -e .[dev]                   # Development tools
```

Some wheels (like PyTorch) are large and require compatible CUDA or CPU builds—make sure your platform matches the binaries pulled in by each extra.

</details>

### Optional dependency groups

| Extra | Purpose | Notes |
|-------|---------|-------|
| `local_vision` | Run the local VLM (SmolVLM2) through PyTorch/Transformers | GPU recommended. Ensure compatible PyTorch builds for your platform. |
| `yolo_vision` | YOLOv11n face detection via `ultralytics` and `supervision` | Used as the `yolo` head-tracking backend. Runs on CPU (default). GPU improves performance. |
| `local_stt` | On-device speech recognition via Moonshine | Enables the `local_stt` backend, which transcribes speech on the robot and uses OpenAI Realtime for response audio. |
| `all_vision` | Convenience alias installing every vision extra | Install when you want the flexibility to experiment with every provider. |
| `dev` | Developer tooling (`pytest`, `ruff`, `mypy`) | Development-only dependencies. Use `--group dev` with uv or `[dev]` with pip. |

**Note:** `dev` is a dependency group (not an optional dependency). With uv, use `--group dev`. With pip, use `[dev]`.

## Configuration

The default setup uses the Hugging Face backend and does not require an API key.

Copy `.env.example` to `.env` when you want to switch backends, provide API keys, or point Hugging Face at your own local endpoint.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI Realtime mode. |
| `GEMINI_API_KEY` | Required for Gemini mode. Also accepts `GOOGLE_API_KEY`. Get one at [aistudio.google.com](https://aistudio.google.com/apikey). **Note**: the free tier of `gemini-3.1-flash-tts-preview` (used by `gemini_tts`) is capped at ~10 requests/day; enable billing on the key for sustained use or expect 429 `RESOURCE_EXHAUSTED` with a rate-limit message in the chat UI. |
| `BACKEND_PROVIDER` | Realtime backend to use: `huggingface` (default), `openai`, `gemini`, or `local_stt`. |
| `MODEL_NAME` | Optional model override for OpenAI Realtime or Gemini Live. Defaults to `gpt-realtime` for OpenAI and `gemini-3.1-flash-live-preview` for Gemini. Hugging Face uses the server's model selection. |
| `HF_REALTIME_CONNECTION_MODE` | Hugging Face connection selector: `deployed` uses the built-in Hugging Face server; `local` uses `HF_REALTIME_WS_URL`. Defaults to `deployed`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for your own Hugging Face backend. Accepts either a base URL like `ws://127.0.0.1:8765/v1` or the full websocket URL `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `REACHY_MINI_HEAD_TRACKER` | Optional startup head-tracking backend: `mediapipe` or `yolo`. Set to `mediapipe` on Reachy Mini Wireless to opt in on app launch without CLI flags. Use `off` or leave unset to disable. |
| `HF_HOME` | Cache directory for local Hugging Face downloads (only used with `--local-vision` flag, defaults to `./cache`). |
| `HF_TOKEN` | Optional token for Hugging Face access (for gated/private assets). |
| `LOCAL_VISION_MODEL` | Hugging Face model path for local vision processing (only used with `--local-vision` flag, defaults to `HuggingFaceTB/SmolVLM2-2.2B-Instruct`). |
| `LOCAL_STT_PROVIDER` | Local STT engine. Currently `moonshine`. |
| `LOCAL_STT_RESPONSE_BACKEND` | Response voice backend for local STT: `openai` or `huggingface`. Defaults to `openai`. |
| `LOCAL_STT_CACHE_DIR` | Cache directory for Moonshine model downloads. Defaults to `./cache/moonshine_voice`. |
| `LOCAL_STT_LANGUAGE` | Language code for the local STT model. Defaults to `en`. |
| `LOCAL_STT_MODEL` | Moonshine model selector: `tiny_streaming` or `small_streaming`. Defaults to `tiny_streaming`. |
| `REACHY_MINI_MAX_HISTORY_TURNS` | Cap on retained user turns in handler-managed conversation history (`gemini_tts` and llama-server response handlers). Default `20`; set `0` to disable trimming. Realtime backends (OpenAI/HF/Gemini Live) manage history server-side and ignore this. |
| `LOCAL_STT_UPDATE_INTERVAL` | Partial transcript update interval in seconds. Defaults to `0.35`. |

### Local speech-to-text

For lower-latency turn detection and private on-device transcription, install the local STT extra and select the local backend:

```bash
uv sync --extra local_stt
```

```env
BACKEND_PROVIDER=local_stt
OPENAI_API_KEY=sk-...
LOCAL_STT_PROVIDER=moonshine
LOCAL_STT_RESPONSE_BACKEND=openai
LOCAL_STT_CACHE_DIR=./cache/moonshine_voice
LOCAL_STT_LANGUAGE=en
LOCAL_STT_MODEL=tiny_streaming
LOCAL_STT_UPDATE_INTERVAL=0.35
```

This mode runs Moonshine speech recognition locally on the robot, then sends only the finalized transcript text to the selected response backend for response generation, voice output, and tool calling. Use `LOCAL_STT_RESPONSE_BACKEND=huggingface` to pair local transcription with the Hugging Face realtime backend instead of OpenAI.

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy. This is the default for a new install; set it explicitly only when you want to switch back from a saved local endpoint:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=deployed
```

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as Robot Comic:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
BACKEND_PROVIDER=huggingface
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

When using the headless settings UI, selecting `Hugging Face` lets you choose either the built-in server or a local `host:port` target. The UI writes `HF_REALTIME_CONNECTION_MODE` for you, and the local path writes `HF_REALTIME_WS_URL` with a default of `localhost:8765`.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching the app. If you see a `TimeoutError`, it means the daemon isn't started. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup instructions.

The app runs in **headless mode** by default — the admin UI is served at http://127.0.0.1:7860/. For workstation/simulation runs, add `--sim` to also bridge audio to your browser via FastRTC (mounted at `/chat`); the same admin UI remains at `/`. Vision and head-tracking options are described in the CLI table below.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Select a head-tracking backend when a camera is available. `mediapipe` comes from the `reachy_mini_toolbox` package and is installed by default; `yolo` uses a local YOLO face detector and requires the `yolo_vision` extra. For app launches without CLI flags, set `REACHY_MINI_HEAD_TRACKER=mediapipe` in the app instance `.env`. |
| `--no-camera` | `False` | Run without camera capture or head tracking. |
| `--local-vision` | `False` | Use the local vision model (SmolVLM2) for camera-tool requests instead of the selected realtime backend. Requires `local_vision` extra to be installed. |
| `--sim` | `False` | Simulation/dev mode. Bridges audio to the browser via FastRTC at `/chat` and serves the admin UI at `/`. Auto-enabled when the Reachy Mini daemon reports simulation. `--gradio` is accepted as a deprecated alias. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples

```bash
# Run with MediaPipe head tracking
reachy-mini-conversation-app --head-tracker mediapipe

# Run with the YOLO face-detection backend for head tracking
reachy-mini-conversation-app --head-tracker yolo

# Run with local vision processing (requires local_vision extra)
reachy-mini-conversation-app --local-vision

# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Simulation / workstation dev mode (browser audio at /chat, admin UI at /)
reachy-mini-conversation-app --sim
```

> [!WARNING]
> `--local-vision` is not supported when running Robot Comic directly on Reachy Mini Wireless / the Raspberry Pi. For local vision, keep the daemon running on the robot and start Robot Comic from your laptop or workstation instead.

## LLM tools exposed to the assistant

| Tool | Action | Dependencies |
|------|--------|--------------|
| `move_head` | Queue a head pose change (left/right/up/down/front). `down` is intended only for explicit user requests, not normal conversational beats. | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend or the local vision model. | Requires camera worker. Uses local vision when `--local-vision` is enabled. |
| `head_tracking` | Enable or disable head-tracking offsets (not identity recognition - only detects and tracks head position). | Camera worker with configured head tracker (`--head-tracker`). |
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |

## Advanced features

Built-in motion content is published as open Hugging Face datasets:
- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

<details>
<summary><b>Custom profiles</b></summary>

Create custom profiles with dedicated instructions and enabled tools.

For normal usage, select a profile from the UI and save it for startup. That selection is persisted in `startup_settings.json`.

If no startup settings have been saved yet, you can still seed startup from the environment with `REACHY_MINI_CUSTOM_PROFILE=<name>` to load `profiles/<name>/`. If neither is set, the `default` profile is used.

Each profile should include `instructions.txt` (prompt text). `tools.txt` (list of allowed tools) is recommended. If missing for a non-default profile, the app falls back to `profiles/default/tools.txt`. Profiles can optionally contain custom tool implementations.

**Custom instructions:**

Write plain-text prompts in `instructions.txt`. To reuse shared prompt pieces, add lines like:
```
[passion_for_lobster_jokes]
[identities/witty_identity]
```
Each placeholder pulls the matching file under `src/robot_comic/prompts/` (nested paths allowed). See `profiles/example/` for a reference layout.

**Enabling tools:**

List enabled tools in `tools.txt`, one per line. Prefix with `#` to comment out:
```
play_emotion
# move_head

# My custom tool defined locally
sweep_look
```
Tools are resolved first from Python files in the profile folder (custom tools), then from the core library `src/robot_comic/tools/` (like `dance`, `head_tracking`).

**Custom tools:**

On top of built-in tools found in the core library, you can implement custom tools specific to your profile by adding Python files in the profile folder.
Custom tools must subclass `robot_comic.tools.core_tools.Tool` (see `profiles/example/sweep_look.py`).

**Edit personalities from the UI:**

The admin UI at http://127.0.0.1:7860/ (served in both headless and `--sim` modes) exposes a Personality section:
- Select among available profiles (folders under `profiles/`) or the built‑in default.
- Click "Apply" to update the current session instructions live.
- Create a new personality by entering a name and instructions text. It stores files under `profiles/<name>/` and copies `tools.txt` from the `default` profile.

Note: The "Personality" panel updates the conversation instructions. Tool sets are loaded at startup from `tools.txt` and are not hot‑reloaded.

</details>

<details>
<summary><b>Locked profile mode</b></summary>

To create a locked variant of the app that cannot switch profiles, edit `src/robot_comic/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "house_comedian"  # Lock to this profile
```
When `LOCKED_PROFILE` is set, the app always uses that profile, ignoring saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and the admin UI personality picker. The UI shows "(locked)" and disables all profile editing controls.
This is useful for creating dedicated clones of the app with a fixed personality. Clone scripts can simply edit this constant to lock the variant.

</details>

<details>
<summary><b>External profiles and tools</b></summary>

You can extend the app with profiles/tools stored outside the repository defaults.

- Core profiles are under `profiles/`.
- Core tools are under `src/robot_comic/tools/`.

**Recommended layout:**

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       ├── instructions.txt
│       ├── tools.txt        # optional (see fallback behavior below)
│       └── voice.txt        # optional
└── external_tools/
    └── my_custom_tool.py
```

**Environment variables:**

Set these values in your `.env` when you want env-driven external profile/tool selection:

```env
# Optional fallback/manual profile selector:
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# Optional convenience mode:
# AUTOLOAD_EXTERNAL_TOOLS=1
```

**Loading behavior:**

- **Default/strict mode**: `tools.txt` defines enabled tools explicitly. Every name in `tools.txt` must resolve to either a built-in tool (`src/robot_comic/tools/`) or an external tool module in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- **Convenience mode** (`AUTOLOAD_EXTERNAL_TOOLS=1`): all valid `*.py` tool files in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` are auto-added.
- **External profile fallback**: if the selected external profile has no `tools.txt`, the app falls back to built-in `profiles/default/tools.txt`.

This supports both:
1. Downloaded external tools used with built-in/default profile.
2. Downloaded external profiles used with built-in default tools.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Contributing

We welcome bug fixes, features, profiles, and documentation improvements. Please review our
[contribution guide](CONTRIBUTING.md) for branch conventions, quality checks, and PR workflow.

Quick start:
- Fork and clone the repo
- Follow the [installation steps](#installation) (include the `dev` dependency group)
- Run contributor checks listed in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0 — base application code (original Pollen Robotics fork).

The `house_comedian` persona content in `profiles/house_comedian/` is an original
synthetic work produced for this project. It is separately licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
You may share and adapt it for non-commercial purposes with attribution.

Operator-specific persona content that references real individuals is kept in a private
companion repository and is not distributed here.
