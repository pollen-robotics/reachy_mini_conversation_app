# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses the **central `/venvs/apps_venv`** shared with
`reachy_mini_conversation_app`. There is no local `.venv`. The package is
installed editable, so source changes in `src/` take effect immediately.

Set this once in your shell (or add to `~/.bashrc`):

```bash
export UV_PROJECT_ENVIRONMENT=/venvs/apps_venv
```

See `DEVELOPMENT.md` for the full uv workflow and venv rationale.

## Commands

```bash
# Dependencies — use uv pip, NOT uv sync (shared venv)
uv pip install -e .                  # Reinstall after editing pyproject.toml
uv pip install -e .[local_stt]       # Add local STT (Moonshine) — required on-robot
uv pip install -e .[all_vision]      # Add all vision extras
uv lock                              # Update the lock file

# Lint / format / type-check
ruff check . --fix                   # Lint and auto-fix
ruff format .                        # Format code
mypy --pretty --show-error-codes     # Type check

# Tests
/venvs/apps_venv/bin/python -m pytest tests/ -v
/venvs/apps_venv/bin/python -m pytest tests/test_openai_realtime.py -v
/venvs/apps_venv/bin/python -m pytest tests/ -k "test_name" -v

# Run the app
python -m robot_comic.main           # Console/headless mode (default; on-robot autostart path)
python -m robot_comic.main --sim     # Simulation/dev mode: FastRTC audio chat at /chat + admin UI at /
python -m robot_comic.main --debug   # Verbose logging

# On-robot logs
journalctl -u reachy-app-autostart -f   # Live app log
```

## Architecture

This is a conversational voice app for the **Reachy Mini robot**, implementing a comedian persona (Don Rickles-style). The key abstraction layers are:

### Handler (Voice Backend)
All realtime voice handlers inherit from `BaseRealtimeHandler` (`base_realtime.py`). Concrete backends:
- `huggingface_realtime.py` — Hugging Face (default)
- `openai_realtime.py` — OpenAI Realtime API
- `gemini_live.py` — Gemini Live
- `local_stt_realtime.py` — Local Moonshine STT

Backend is selected via `BACKEND_PROVIDER` env var in `config.py`.

### Movement Manager (`moves.py`)
Runs a 60 Hz control loop in a background thread. Distinguishes:
- **Primary moves** — exclusive (dances, explicit emotion poses, breathing) queued via `MoveQueue`
- **Secondary moves** — additive on top of primary (head wobble, speech-reactive sway)

Move classes inherit from `Move` and compose world-space offsets.

### Tool System (`tools/`)
LLM-callable tools (dance, camera, head tracking, emotions, etc.) are Python classes inheriting from `Tool`. Tool specs live in `tool_constants.py`. The `BackgroundToolManager` queues and dispatches tool calls from the handler without blocking audio streaming.

### Profiles / Personality System (`profiles/`)
Each profile directory contains:
- `instructions.txt` — system prompt / persona definition
- `tools.txt` — enabled tools list
- Optional `.py` files — custom tool implementations

External profiles can be loaded from `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`. The `LOCKED_PROFILE` config option pins the app to a single persona. Profile packaging into the wheel is handled by `BuildPyWithProfiles` in `setup.py`.

### Camera & Vision (`camera_worker.py`, `vision/`)
Camera runs in a dedicated thread. Head tracking is pluggable: `mediapipe` (in-process, default) or `yolo` (subprocess). Local vision uses SmolVLM2; backend vision uses the LLM provider's vision API.

### Audio (`audio/`)
`HeadWobbler` produces speech-reactive secondary head motion. `startup_settings.json` persists UI profile/voice selections across restarts.

### Admin/Settings UI
The user-facing admin UI is the same in headless (`/`) and sim (`/`) modes — only the audio bridge differs (local speakers vs FastRTC at `/chat`). When making admin/settings UI changes, edit these files:
- `src/robot_comic/static/index.html` — markup
- `src/robot_comic/static/main.js` — client-side fetch + handlers
- `src/robot_comic/static/main.css` — styles
- `src/robot_comic/console.py` — FastAPI route handlers registered inside `LocalStream.init_admin_ui` (look for `@self._settings_app.get/post(...)`)
- `src/robot_comic/headless_personality_ui.py` — the profile/personality picker routes

In sim mode (`--sim`), `main.py` constructs a settings-only `LocalStream(handler=None, robot=None, ...)` purely to call `init_admin_ui()`. The FastRTC `Stream.ui` is mounted at `/chat`. There is no longer a separate Gradio personality UI.

## Key Conventions

- **Threading model**: MovementManager, CameraWorker, and HeadWobbler each own a dedicated thread. The main thread drives the asyncio event loop for audio streaming.
- **Startup config**: Persisted to `startup_settings.json`; reloaded on app restart from the admin UI.
- **Config refresh**: `config.py` centralizes all env-var reading; backends call `refresh()` at startup.
- **Conversation history bound**: Handlers that keep client-side chat history (`gemini_tts.py`, `llama_base.py`) trim to the last `REACHY_MINI_MAX_HISTORY_TURNS` user turns (default 20, set `0` to disable) via `history_trim.trim_history_in_place`. Trimming runs *before* each request is built and cuts on user-turn boundaries so tool round-trips ride along. Realtime backends (OpenAI/HF/Gemini Live) manage history server-side and are not bounded by this knob.
- **Gemini 429 handling**: All Gemini call sites (TTS + LLM in `gemini_tts.py`, the reconnect loop in `gemini_live.py`) use `gemini_retry.compute_backoff` to honour `Retry-After` and embedded `RetryInfo`, capped at 60s. On exhaustion they log an `ERROR` naming the quota; the chat UI gets a rate-limit-specific message. Free tier is ~10 req/day so this path is hot.
- **Cross-platform**: All code must work on Linux, macOS, and Windows.
- **Git branches**: `feat/<issue>-<desc>`, `fix/<issue>-<desc>`. `main` is the release branch; PRs require CI (lint + types + tests) to pass.
- **Git remote**: The only remote is `origin` (TheKentuckian/robot_comic). There is no `upstream`. **Never** fetch from, push to, reference, or run any `gh` command targeting the upstream fork (`pollen-robotics/reachy_mini_conversation_app`) — not for issues, PRs, releases, or any other action. All GitHub interactions are scoped to `origin` only.
- **Pre-commit hook**: A secrets-and-audio guard lives in `.githooks/pre-commit`. Enable it once per clone:
  ```bash
  git config core.hooksPath .githooks
  ```
  It blocks API keys and stray archival audio from being committed to the public repo. Use `git commit --no-verify` only if you are certain a flagged item is safe.
