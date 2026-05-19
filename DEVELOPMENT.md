# Development Notes

## Environment setup

This project does **not** use a local `.venv`. Instead it shares the central
`/venvs/apps_venv` with `reachy_mini_conversation_app` — the two apps run
from the same interpreter on the robot, so their library versions are always
in sync at runtime.

The package is installed as an editable install inside `apps_venv`:

```
Location: /venvs/apps_venv/lib/python3.12/site-packages
Editable project location: /home/pollen/apps/robot_comic
```

This means **code changes in `src/` take effect immediately** — no reinstall
step for day-to-day development.

## uv workflow

Tell uv to target the central venv instead of creating a local one:

```bash
export UV_PROJECT_ENVIRONMENT=/venvs/apps_venv
```

Add that line to `~/.bashrc` once, then the usual commands work from the
project directory:

| Task | Command |
|---|---|
| Reinstall after editing `pyproject.toml` | `uv pip install -e .` |
| Install with extras | `uv pip install -e .[local_stt]` |
| Add a one-off package | `uv pip install some-package` |
| Update the lock file | `uv lock` |
| Lint / format / type-check | `uv pip install -e .[dev]` then call tools directly |
| Run tests | `/venvs/apps_venv/bin/python -m pytest tests/ -v` |

> **Do not use `uv sync` or `uv add`** — both prune the environment down to
> a single project's dependency set and will remove packages belonging to
> the other app sharing the venv.

Note: `CLAUDE.md` still references `uv sync` / `uv run` which assumed a
local `.venv`. Those commands need updating to match this workflow.

## Extras

| Extra | Installs | Required for |
|---|---|---|
| `local_stt` | `moonshine-voice` | Local Moonshine STT backend (default) |
| `faster_whisper_stt` | `faster-whisper`, `webrtcvad` | Alternate local STT (Phase 5f, A/B against Moonshine; webrtcvad picked over silero-vad in 5f.1 to keep the install under the chassis eMMC budget) |
| `local_vision` | torch, transformers, accelerate | SmolVLM2 local vision |
| `yolo_vision` | ultralytics, supervision, opencv | YOLO head tracking |

`moonshine-voice` is installed in `apps_venv` and is required when
`BACKEND_PROVIDER=local_stt` (the on-robot default). The autostart launcher
(`/usr/local/bin/reachy-app-autostart.py`) runs with `/venvs/apps_venv/bin/python`,
so any extras needed at runtime must be installed into that venv.

### Switching STT backend (Moonshine vs faster-whisper)

The composable triples support either Moonshine or faster-whisper as the
on-robot STT (Phase 5f / issue #387). Default is Moonshine. To A/B against
faster-whisper:

```bash
uv pip install -e .[faster_whisper_stt]
export REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper
export REACHY_MINI_AUDIO_OUTPUT_BACKEND=chatterbox   # or elevenlabs / gemini_tts
python -m robot_comic.main
```

Revert by unsetting `REACHY_MINI_AUDIO_INPUT_BACKEND` (defaults to
`moonshine`) or setting it back to `moonshine` explicitly. Both STT
backends live side-by-side; this is the operator A/B window before
deciding which is the long-term default.

`faster_whisper` is not yet paired with the realtime-output hybrids
(`openai_realtime_output`, `hf_output`) — those still require
`AUDIO_INPUT_BACKEND=moonshine`.

## Shared venv and version alignment

`reachy_mini_conversation_app` (the upstream this project was forked from) is
installed from a frozen HuggingFace snapshot, not a local editable. Both apps
share `apps_venv`, so runtime library versions are always identical regardless
of what each `pyproject.toml` specifies.

When Pollen updates the upstream snapshot, check that the shared dep pins
(`gradio`, `huggingface-hub`, `fastrtc`, `openai`, `reachy-mini`) remain
compatible with the versions in this project's `pyproject.toml`.

## Development workflow

When working on this repo, **push incremental changes to your branch early and often**—don't wait to finish everything locally before validating. GitHub Actions will automatically run:

- `pytest-fast` — unit tests, fast subset (`-m 'not slow and not integration and not hardware'`), Linux + Windows. Required for merge. 90 s wall-clock budget.
- `pytest-full` — same suite minus `hardware`, advisory on PRs. Gating on `push: branches: [main]` (opens a GitHub issue on regression; no auto-revert).
- `ruff check` — linting
- `ruff format` — code formatting
- `mypy` — type checking

**Workflow:**
1. Create a feature branch: `feat/<issue>-<desc>` or `fix/<issue>-<desc>`
2. Commit and push your changes
3. Check the workflow results in the branch's commit checks or PR
4. Address any failures and push fixes
5. Once CI passes, open a PR or continue development

This keeps iteration cycles tight and catches platform-specific or dependency issues early.

## Test tiering

The test suite is organised into three concentric loops. Each step
forward trades coverage for latency.

| Loop | Command | Latency target | When to use |
|---|---|---|---|
| Inner | `pytest --testmon` | < 5 s for an edit | While writing code. testmon re-runs only the tests whose dependency graph was touched by your edit. |
| Pre-push | `pytest -n auto --ignore=tests/vision/test_local_vision.py` | ~70 s on a fast laptop | Before `git push`. Catches xdist-only races + tests testmon's graph missed. |
| CI fast | `pytest-fast` workflow | ~90 s | Automatic on every PR. Required for merge. |
| CI full | `pytest-full` workflow | ~3-5 min | Automatic. Advisory on PRs; gating on push-to-main. |

### testmon setup

The `dev` group installs `pytest-testmon`. First-time setup:

```bash
.venv/bin/pytest --testmon  # one full run to build .testmondata
```

Subsequent runs re-use `.testmondata` and execute only the affected
subset. `.testmondata` is gitignored.

### Worked example: "I changed `src/robot_comic/composable_pipeline.py`"

1. `pytest --testmon` — testmon runs `test_composable_pipeline.py`,
   `test_composable_conversation_handler.py`, `test_phase_5b_*`, and any
   integration test whose imports transitively touched the changed
   module. ~5-10 s.
2. Iterate. testmon hot-loops on each edit.
3. When the change is settled and the testmon run is green:
   `pytest -n auto --ignore=tests/vision/test_local_vision.py`. Full
   parallel suite, ~70 s. This catches:
   - xdist-only races (tests that share global state).
   - Tests testmon didn't pick up because the dependency lives outside
     the import graph (string-based dispatch, env-var-driven config,
     etc.).
   - The slow-marked subset that testmon would skip-via-marker.
4. `git push`. `pytest-fast` runs in CI; if it goes green, the PR is
   merge-ready from a test-gating perspective.

### Test markers

Three pytest markers tier the suite. Apply markers conservatively —
only when the test is genuinely in that category.

| Marker | Apply when | Example |
|---|---|---|
| `slow` | Test wall clock exceeds 500 ms. | `tests/test_cold_boot_imports.py` (subprocess-per-test, ~20-25 s each) |
| `integration` | Test shells out, opens network sockets, downloads ML models, or otherwise exercises external surfaces. | The `tests/integration/` smoke suite; the cold-boot subprocess checks. |
| `hardware` | Test requires real ALSA / serial / GPIO devices. Not currently used in CI — these only run on the robot. | Direct ALSA RW capture probes; reachy_mini daemon `set_mode/enabled` round-trips. |

A test can have multiple markers (`slow + integration` is common). Find
slow candidates with `pytest --durations=25`. Find tests in a given tier
with `pytest -m slow --collect-only`.

The fast PR gate (`pytest-fast` workflow) runs
`-m 'not slow and not integration and not hardware'`. Anything that
shouldn't block merge gets one of those three markers.

## Persistent STT service (reachy-stt)

The `reachy-stt` systemd service hosts faster-whisper in memory so it
survives app restarts. The app talks to it over a Unix-domain socket
at `/run/reachy-stt/reachy-stt.sock`.

**Status:** PR-B (this) ships the unit file but not the client adapter
yet — PR-C wires the app to use it. Until then the service runs idle
and the app continues loading faster-whisper in-process.

**Install on the robot:**
(see header comment in `deploy/systemd/reachy-stt.service`)

**Local dev (Windows / macOS without UDS):**

    uvicorn robot_comic.stt_service.server:app --host 127.0.0.1 --port 8021

(--uds is Linux-only.)

**Warm-load behavior:** The unit runs unconditionally but stays idle
(~30 MB resident) unless `REACHY_MINI_AUDIO_INPUT_BACKEND` in
`/home/pollen/.robot_comic/.env` selects an on-device STT backend. The
`scripts/reachy-stt-warm-load.sh` helper makes that decision and POSTs
`/preload` only when needed.

## Running on the robot

The robot autostart flow:
1. `wait-for-reachy-daemon.sh` polls `http://127.0.0.1:8000/api/motors/get_mode`
   every 200 ms until the daemon responds (up to 10 s).  Exits 1 on timeout so
   systemd can retry rather than proceed with motors disabled.
   Previously: `ExecStartPre=/bin/sleep 30` (wasted ~25 s on every boot).
2. Enables motors via `curl … motors/set_mode/enabled`
3. Queues the wake-up animation via `curl … move/play/wake_up`
4. Execs `/venvs/apps_venv/bin/python -u -m robot_comic.main`

The systemd unit and poll script live in this repo:
- `scripts/wait-for-reachy-daemon.sh` — copy to `/usr/local/bin/` (chmod +x)
- `deploy/systemd/reachy-app-autostart.service` — copy to `/etc/systemd/system/`

See the header comment in `deploy/systemd/reachy-app-autostart.service` for
one-liner install instructions.

Daemon logs go to `/tmp/daemon.jsonl` (tmpfs — cleared on reboot).
App logs go to journald: `journalctl -u reachy-app-autostart -f`.

### Log retention

On first setup, run:

```bash
sudo ./scripts/install-pi-journald.sh
```

This caps journald at 500 MB and 2-week retention so the SD card doesn't fill
up.  It also creates `/var/log/journal/` so logs survive reboots (without that
directory, journald writes to the RAM-backed `/run/log/journal` and all history
is lost on every reboot).
