---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini !
tags:
 - reachy_mini
 - reachy_mini_python_app
---

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
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).<br>
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
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude (only used with `linus` developer profile).
| `ANTHROPIC_MODEL` | Claude model to use (only used with `linus` profile, defaults to `claude-sonnet-4-20250514`).
| `GITHUB_TOKEN` | GitHub Personal Access Token with `repo`, `issues`, `pull_requests` scopes (only used with `linus` profile).
| `GITHUB_DEFAULT_OWNER` | Default GitHub owner/org for repository operations (optional, only used with `linus` profile).
| `GITHUB_OWNER_EMAIL` | Email for git commits (optional, defaults to `<owner>@users.noreply.github.com`, only used with `linus` profile).

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

- Run with Gradio web interface:

  ```bash
  reachy-mini-conversation-app --gradio
  ```

### Troubleshooting

- Timeout error:
If you get an error like this:
  ```bash
  TimeoutError: Timeout while waiting for connection with the server.
  ```
It probably means that the Reachy Mini's daemon isn't running. Install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) and start the daemon.

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

### Edit personalities from the UI
When running with `--gradio`, open the "Personality" accordion:
- Select among available profiles (folders under `src/reachy_mini_conversation_app/profiles/`) or the built‑in default.
- Click "Apply" to update the current session instructions live.
- Create a new personality by entering a name and instructions text; it stores files under `profiles/<name>/` and copies `tools.txt` from the `default` profile.

Note: The "Personality" panel updates the conversation instructions. Tool sets are loaded at startup from `tools.txt` and are not hot‑reloaded.

## Linus Developer Profile

The `linus` profile transforms Reachy Mini into a developer assistant that can generate and execute code, and interact with GitHub.

### Setup

1. Set the profile in your `.env`:
   ```bash
   REACHY_MINI_CUSTOM_PROFILE=linus
   ```

2. Configure the required API keys:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GITHUB_TOKEN=your_github_token  # Optional, for GitHub features
   ```

### Developer Tools

#### Code Generation & Execution

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `code` | Generate code using Claude API. Saves to `~/reachy_code/`. | No |
| `execute_code` | Execute generated Python/Shell scripts. | Yes |
| `code_move_to_repo` | Move generated code from `~/reachy_code/` to a repository. | No |

#### Repository Management

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_clone` | Clone repositories to `~/reachy_repos/`. | No |
| `github_pull` | Pull latest changes from remote. | No |
| `github_push` | Push local commits to remote. | Yes |
| `github_list_repos` | List all locally cloned repositories. | No |
| `github_exec` | Execute whitelisted commands (npm, pytest, ruff, etc.) in a repo. | No |

#### File Operations

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_list_files` | List files and directories in a repository. | No |
| `github_read_file` | Read file content with optional AI analysis (Claude/OpenAI). | No |
| `github_write_file` | Create or modify files in a repository. | No |
| `github_edit_file` | AI-assisted file editing with optional model file reference. | No |

#### Git Status & History

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_status` | Show repository status (staged, modified, untracked files). | No |
| `github_diff` | Show file differences (staged, unstaged, between commits). | No |
| `github_log` | Show commit history with filters (author, date, branch). | No |

#### Staging & Commits

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_add` | Stage files for commit (git add). | No |
| `github_rm` | Remove files from repository. | Yes |
| `github_restore` | Restore files (unstage or discard changes). | Yes (for worktree) |
| `github_discard` | Discard unstaged changes. | Yes |
| `github_commit` | Commit staged changes with semantic-release format. Supports auto-generated messages and pre-commit checks via `.reachy/commit_rules.yaml`. | Yes |

#### Branch Management

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_branch` | Create, switch, list, or delete branches. | No |
| `github_reset` | Reset commits (soft, mixed, hard modes). | Yes (for hard) |
| `github_rebase` | Rebase current branch onto another. | Yes |

#### GitHub Issues

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_issue` | Create issues on GitHub repositories. | No |
| `github_list_issues` | List issues with filters (state, labels). | No |
| `github_update_issue` | Update issue title, body, or labels. | No |
| `github_comment_issue` | Add a comment to an issue. | No |
| `github_close_issue` | Close an issue. | No |

#### GitHub Pull Requests

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_create_pr` | Create a pull request. | Yes |
| `github_merge_pr` | Merge a pull request (supports merge, squash, rebase). | Yes |
| `github_list_prs` | List pull requests with filters. | No |
| `github_update_pr` | Update PR title, body, or base branch. | No |
| `github_comment_pr` | Add a comment to a pull request. | No |
| `github_pr_comment` | Comment on pull requests (alias). | No |
| `github_close_pr` | Close a pull request. | No |

#### CI/CD Integration

| Tool | Description | Confirmation Required |
|------|-------------|----------------------|
| `github_pr_checks` | Get CI check status and errors for a PR. | No |
| `github_ci_logs` | Get GitHub Actions workflow logs. | No |

### Pre-commit Checks

Linus can enforce code quality checks before committing by reading a `.reachy/commit_rules.yaml` file in the repository. This works similarly to how Claude Code reads `.claude/` configuration.

#### Setup

Create a `.reachy/commit_rules.yaml` file in your repository:

```yaml
# Pre-commit checks - all must pass before committing
pre_commit:
  - name: lint
    command: ruff check .
    required: true

  - name: format_check
    command: ruff format --check .
    required: true

  - name: type_check
    command: mypy src/
    required: false  # Warning only, won't block commit

  - name: tests
    command: pytest
    required: true

  - name: coverage
    command: pytest --cov=src --cov-fail-under=80
    required: false

# Auto-fix commands (run before checks if enabled)
auto_fix:
  enabled: true
  commands:
    - ruff check --fix .
    - ruff format .
```

#### Behavior

- When `github_commit` is called, Linus checks for `.reachy/commit_rules.yaml`
- If present, all `pre_commit` checks run before the commit
- `required: true` checks must pass or the commit is blocked
- `required: false` checks warn but allow the commit
- `auto_fix` commands run first to auto-correct issues (if enabled)
- Use `skip_checks=true` to bypass checks (not recommended)

### Example Usage

Ask Linus to:
- "Write a Python script that calculates Fibonacci numbers"
- "Create an issue on my-repo about the login bug"
- "Clone the reachy_mini repository"
- "Comment on PR #42 with my review"
- "Show me the git status of reachy_mini"
- "Read and analyze the main.py file"
- "Create a new branch called feature/new-api"
- "Commit my changes with an auto-generated message"
- "Check the CI status of PR #15"
- "Rebase my branch onto main"




## Creating Custom Tools

This section explains how to create custom tools for the conversation app, including support for background execution.

### Basic Tool Structure

All tools must subclass `Tool` from `reachy_mini_conversation_app.tools.core_tools`:

```python
from typing import Any, Dict
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

class MyCustomTool(Tool):
    """Description visible to the LLM."""

    name = "my_tool"
    description = "What this tool does - the LLM uses this to decide when to call it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of the parameter",
            },
            "optional_param": {
                "type": "integer",
                "description": "An optional parameter with default",
            },
        },
        "required": ["param1"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool - must be async."""
        param1 = kwargs.get("param1", "")
        optional_param = kwargs.get("optional_param", 10)

        # Your tool logic here
        result = f"Processed {param1}"

        # Return a dict - the LLM will vocalize this
        return {
            "status": "success",
            "message": result,
        }
```

### Tool Dependencies

The `ToolDependencies` dataclass provides access to robot systems:

| Attribute | Type | Description |
|-----------|------|-------------|
| `reachy_mini` | `ReachyMini` | Robot instance for direct control |
| `movement_manager` | `MovementManager` | Queue movements, set listening state |
| `camera_worker` | `CameraWorker \| None` | Get camera frames |
| `vision_manager` | `VisionManager \| None` | Process images with vision models |
| `head_wobbler` | `HeadWobbler \| None` | Audio-reactive head motion |
| `motion_duration_s` | `float` | Default motion duration |
| `background_task_manager` | `BackgroundTaskManager \| None` | Manage background tasks |

### Background Task Execution

For long-running operations (API calls, file downloads, builds), use background execution to avoid blocking the conversation.

#### Enabling Background Support

1. Set `supports_background = True` on your tool class
2. Add a `background` parameter to your schema
3. Use `BackgroundTaskManager` to run the task asynchronously

```python
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.background_tasks import BackgroundTaskManager

class MyLongRunningTool(Tool):
    name = "my_long_task"
    description = "A tool that takes time to complete."
    supports_background = True  # Enable background support

    parameters_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"},
            "background": {
                "type": "boolean",
                "description": "Run in background (default: true for long operations)",
            },
        },
        "required": ["input"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        input_data = kwargs.get("input", "")
        background = kwargs.get("background", True)

        if background:
            manager = BackgroundTaskManager.get_instance()
            task = await manager.start_task(
                name="my_long_task",
                description=f"Processing {input_data}",
                coroutine=self._do_work(input_data),
            )
            return {
                "status": "started",
                "task_id": task.id,
                "message": f"Task started in background. I'll notify you when done.",
            }

        # Synchronous execution
        return await self._do_work(input_data)

    async def _do_work(self, input_data: str) -> Dict[str, Any]:
        """The actual work - runs in background or synchronously."""
        import asyncio
        await asyncio.sleep(10)  # Simulate long operation
        return {
            "status": "success",
            "message": f"Completed processing {input_data}",
        }
```

#### Progress Tracking

For tasks where progress can be measured, enable progress tracking:

```python
async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
    manager = BackgroundTaskManager.get_instance()

    task = await manager.start_task(
        name="download_task",
        description="Downloading large file",
        coroutine=self._download_with_progress(),
        with_progress=True,  # Enable progress tracking
    )

    return {"status": "started", "task_id": task.id}

async def _download_with_progress(self) -> Dict[str, Any]:
    manager = BackgroundTaskManager.get_instance()
    running_tasks = manager.get_running_tasks()

    # Find our task ID
    task_id = None
    for task in running_tasks:
        if task.name == "download_task":
            task_id = task.id
            break

    # Update progress as work proceeds
    for i in range(10):
        await asyncio.sleep(1)
        if task_id:
            await manager.update_progress(
                task_id,
                progress=(i + 1) / 10,
                message=f"Downloaded {(i + 1) * 10}%",
            )

    return {"status": "success", "message": "Download complete"}
```

### Background Task Tools

Three built-in tools help users interact with background tasks:

| Tool | Description |
|------|-------------|
| `task_status` | Check status of running background tasks |
| `task_cancel` | Cancel a running task (requires confirmation) |
| `background_demo` | Demo tool for testing background execution |

### Best Practices

1. **Return user-friendly messages**: The LLM vocalizes your return values
2. **Handle errors gracefully**: Return `{"error": "message"}` for failures
3. **Use async/await**: All tool methods must be async
4. **Require confirmation for destructive actions**: Add a `confirmed` parameter
5. **Background for long operations**: Anything > 5 seconds should support background mode
6. **Log important actions**: Use `logging.getLogger(__name__)`

### Tool Registration

Tools are auto-discovered when:
1. They subclass `Tool`
2. The module is imported (via `tools.txt` in your profile)

Add your tool to `profiles/<your_profile>/tools.txt`:
```
# My custom tools
my_tool
my_long_task
```

## Development workflow
- Install the dev group extras: `uv sync --group dev` or `pip install -e .[dev]`.
- Run formatting and linting: `ruff check .`.
- Execute the test suite: `pytest`.
- When iterating on robot motions, keep the control loop responsive => offload blocking work using the helpers in `tools.py`.

## License
Apache 2.0
