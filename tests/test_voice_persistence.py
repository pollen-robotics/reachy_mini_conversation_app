"""Tests for voice persistence on POST /voices/apply (#491).

Verifies that a successful voice change is mirrored to startup_settings.json
and that a failed change_voice call does NOT modify the file.
"""

from __future__ import annotations
import asyncio
import threading
from types import SimpleNamespace
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot_comic.startup_settings import STARTUP_SETTINGS_FILENAME, read_startup_settings
from robot_comic.headless_personality_ui import mount_personality_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop() -> asyncio.AbstractEventLoop:
    """Start a background event loop and return it (runs for the test lifetime)."""
    loop = asyncio.new_event_loop()

    def _run() -> None:
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return loop


def _build_app(
    tmp_path: Path,
    *,
    change_voice_result: str = "Voice changed.",
    change_voice_raises: Exception | None = None,
) -> tuple[FastAPI, asyncio.AbstractEventLoop]:
    """Build a minimal FastAPI app with /voices/apply mounted.

    Returns (app, loop).  The loop is a real background event loop so that
    ``asyncio.run_coroutine_threadsafe`` inside the route handler works.
    """
    loop = _make_loop()
    app = FastAPI()

    # Minimal handler stub that exposes change_voice
    async def _change_voice(voice: str) -> str:
        if change_voice_raises is not None:
            raise change_voice_raises
        return change_voice_result

    handler = SimpleNamespace(change_voice=_change_voice)

    mount_personality_routes(
        app,
        handler,  # type: ignore[arg-type]
        lambda: loop,
        instance_path=tmp_path,
    )
    return app, loop


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_apply_voice_persists_to_startup_settings(tmp_path: Path) -> None:
    """POST /voices/apply with a valid voice writes the voice to startup_settings.json."""
    app, _ = _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post("/voices/apply", json={"voice": "Puck"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True

    settings = read_startup_settings(tmp_path)
    assert settings.voice == "Puck"


def test_apply_voice_persists_only_voice_does_not_clobber_profile(tmp_path: Path) -> None:
    """Voice persistence must not overwrite an existing profile in startup_settings.json."""
    from robot_comic.startup_settings import write_startup_settings

    write_startup_settings(tmp_path, profile="house_comedian", voice="old_voice")

    app, _ = _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post("/voices/apply", json={"voice": "Shimmer"})
    assert resp.status_code == 200

    settings = read_startup_settings(tmp_path)
    assert settings.voice == "Shimmer"
    assert settings.profile == "house_comedian"


def test_apply_voice_overwrites_previously_persisted_voice(tmp_path: Path) -> None:
    """A second POST /voices/apply should update the persisted voice."""
    app, _ = _build_app(tmp_path)
    client = TestClient(app)

    client.post("/voices/apply", json={"voice": "Puck"})
    resp = client.post("/voices/apply", json={"voice": "Alloy"})

    assert resp.status_code == 200
    settings = read_startup_settings(tmp_path)
    assert settings.voice == "Alloy"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_apply_voice_does_not_persist_on_change_voice_failure(tmp_path: Path) -> None:
    """When change_voice raises, startup_settings.json must NOT be modified."""
    app, _ = _build_app(tmp_path, change_voice_raises=RuntimeError("TTS unavailable"))
    client = TestClient(app)

    resp = client.post("/voices/apply", json={"voice": "Puck"})

    assert resp.status_code == 500
    # The file should not exist (or, if a pre-existing file was present, unchanged)
    settings_file = tmp_path / STARTUP_SETTINGS_FILENAME
    assert not settings_file.exists()


def test_apply_voice_does_not_overwrite_existing_settings_on_failure(tmp_path: Path) -> None:
    """A failed change_voice call must not touch a pre-existing startup_settings.json."""
    from robot_comic.startup_settings import write_startup_settings

    write_startup_settings(tmp_path, profile="house_comedian", voice="original_voice")

    app, _ = _build_app(tmp_path, change_voice_raises=RuntimeError("TTS down"))
    client = TestClient(app)

    resp = client.post("/voices/apply", json={"voice": "Puck"})
    assert resp.status_code == 500

    # The file must be untouched
    settings = read_startup_settings(tmp_path)
    assert settings.voice == "original_voice"
    assert settings.profile == "house_comedian"


# ---------------------------------------------------------------------------
# No instance_path — should still return success without crashing
# ---------------------------------------------------------------------------


def test_apply_voice_without_instance_path_succeeds_without_persisting(tmp_path: Path) -> None:
    """When no instance_path is passed, voice change still works; nothing is written to disk."""
    loop = _make_loop()
    app = FastAPI()

    async def _change_voice(voice: str) -> str:
        return "Voice changed."

    handler = SimpleNamespace(change_voice=_change_voice)
    mount_personality_routes(
        app,
        handler,  # type: ignore[arg-type]
        lambda: loop,
        # no instance_path — default behaviour
    )

    client = TestClient(app)
    resp = client.post("/voices/apply", json={"voice": "Puck"})

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # No file written anywhere in tmp_path
    assert not (tmp_path / STARTUP_SETTINGS_FILENAME).exists()
