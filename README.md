# Reachy Mini conversation demo

Working repo, we should turn this into a ReachyMini app at some point maybe ?

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

### Using pip
Alternatively, you can install using pip in editable mode:

```bash
python -m venv .venv  # Create a virtual environment
source .venv/bin/activate
pip install -e .
```

To include optional vision dependencies:
```
pip install -e .[local_vision]
pip install -e .[yolo_vision]
pip install -e .[mediapipe_vision]
pip install -e .[all_vision]
```

To include dev dependencies:
```
pip install -e .[dev]
```

## Run

```bash
reachy-mini-conversation-demo
```

## Command line arguments

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--sim` | *(flag)* | off | Run in **simulation mode** (no physical robot required). |
| `--head-tracker` | `yolo`, `mediapipe` | `None` | Enable **head tracking** using the specified tracker:<br>• **yolo** → YOLO-based head tracker.<br>• **mediapipe** → MediaPipe-based head tracker.<br>*(Automatically disabled when `--sim` is active)* |
| `--no-camera` | *(flag)* | off | Disable **camera usage** entirely. |
| `--gradio` | *(flag)* | off | **⚠️ Under construction** - Open Gradio interface (currently not implemented). |
| `--debug` | *(flag)* | off | Enable **debug logging** (default log level is INFO). |

## Examples
- Simulated run:
```
reachy-mini-conversation-demo --sim
```
- Run with YOLO head tracking:
```
reachy-mini-conversation-demo --head-tracker yolo
```
- Run with MediaPipe head tracking and debug logging:
```
reachy-mini-conversation-demo --head-tracker mediapipe --debug
```
