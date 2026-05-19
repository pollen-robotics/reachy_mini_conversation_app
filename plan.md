# Robot Comic install plan

## Understanding

This repo is a public fork of the Reachy Mini conversation app that should be installed as a separate Reachy Mini app named `robot_comic` / `Robot Comic`. It must not overwrite or uninstall the official conversation app already installed on this Wireless Reachy Mini.

Desired identifiers:

- Python package/module: `robot_comic`
- Reachy Mini app entry point: `robot_comic`
- App class: `RobotComic`
- Display name: `Robot Comic`
- Runtime environment: `/venvs/apps_venv/`

## Technical approach

1. [x] Use the existing clone at `/home/pollen/apps/robot_comic`.
2. [x] Prefer `uv` now that it is installed in `/venvs/apps_venv/bin/uv`; fall back to `/venvs/apps_venv/bin/python -m pip` if needed.
3. [x] Verify and, where needed, fix packaging metadata, entry points, imports, package data, CLI naming, and app class naming so this remains standalone from the official conversation app.
4. [x] Preserve the `reachy_mini_python_app` README tag and existing persona/custom behavior.
5. [x] Install editable into `/venvs/apps_venv/`.
6. [x] Validate with `reachy-mini-app-assistant check .` and entry point inspection.
7. [x] Restart `reachy-mini-daemon` only if installation and checks succeed.

## Post-install state (2026-05-19)

The app runs in production on the Wireless Reachy Mini. Autostart unit
is `reachy-app-autostart` (see `journalctl -u reachy-app-autostart -f`).
The code is editable-installed into the shared `/venvs/apps_venv` per
`CLAUDE.md`. Reinstall after `pyproject.toml` changes via `uv pip
install -e .`.

## Pointers

- Operational status: `NOW.md` (30-day rolling window).
- Architecture overview / fork divergence summary: `FORK_STATUS.md`.
- Development workflow: `DEVELOPMENT.md`.
