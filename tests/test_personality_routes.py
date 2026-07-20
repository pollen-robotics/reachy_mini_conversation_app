"""Route-level coverage for personality/voice endpoints not exercised elsewhere.

`test_personality_delete.py` covers the delete guards and `test_console.py` the
apply/voice happy paths; this pins down save, the error branches of apply/apply
voice, the current-voice fallback, and the non-default load path.
"""

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import reachy_mini_conversation_app.personality as personality_mod
import reachy_mini_conversation_app.personality_routes as routes_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.personality_routes import mount_personality_routes


def _client(handler: object | None = None, get_loop=lambda: None, **kwargs) -> TestClient:
    app = FastAPI()
    mount_personality_routes(app, handler=handler or MagicMock(), get_loop=get_loop, **kwargs)
    return TestClient(app)


@pytest.fixture
def running_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Provide an event loop running in a background thread, for run_coroutine_threadsafe routes."""
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait(timeout=1.0)
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_save_writes_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving a valid profile writes it under the user root and returns its selection value."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)

    resp = _client().post(
        "/personalities/save",
        json={"name": "Chatty Bot", "instructions": "Be brief.", "tools_text": "", "voice": get_default_voice()},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["value"] == "user_personalities/Chatty_Bot"
    assert (tmp_path / "user_personalities" / "Chatty_Bot").is_dir()


def test_save_rejects_invalid_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A name that sanitizes to empty is rejected with 400."""
    resp = _client().post("/personalities/save", json={"name": "***", "instructions": "x"})

    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_name"


def test_save_reports_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while writing the profile is surfaced as a 500."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(routes_mod, "_write_profile", _boom)

    resp = _client().post("/personalities/save", json={"name": "ok_name", "instructions": "x"})

    assert resp.status_code == 500
    assert "disk full" in resp.json()["error"]


def test_current_voice_uses_callback() -> None:
    """The current-voice endpoint returns the backend voice."""
    resp = _client(get_current_voice=lambda: "Serena").get("/voices/current")

    assert resp.status_code == 200
    assert resp.json() == {"voice": "Serena"}


def test_current_voice_falls_back_on_error() -> None:
    """A failing voice lookup falls back to the default voice."""

    def _boom() -> str:
        raise RuntimeError("backend down")

    resp = _client(get_current_voice=_boom).get("/voices/current")

    assert resp.status_code == 200
    assert resp.json() == {"voice": get_default_voice()}


def test_apply_rejects_locked_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a profile is locked, apply is refused with 403."""
    monkeypatch.setattr(routes_mod, "LOCKED_PROFILE", "user_personalities/locked")

    resp = _client().post("/personalities/apply", json={"name": "user_personalities/other"})

    assert resp.status_code == 403
    assert resp.json()["error"] == "profile_locked"


def test_apply_returns_503_without_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying a different profile without a running loop returns 503."""
    monkeypatch.setattr(routes_mod, "LOCKED_PROFILE", None)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    resp = _client(get_loop=lambda: None).post("/personalities/apply", json={"name": "user_personalities/other"})

    assert resp.status_code == 503
    assert resp.json()["error"] == "loop_unavailable"


def test_apply_reports_backend_failure(
    monkeypatch: pytest.MonkeyPatch, running_loop: asyncio.AbstractEventLoop
) -> None:
    """A backend apply that raises is surfaced as a 500."""
    monkeypatch.setattr(routes_mod, "LOCKED_PROFILE", None)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    async def _boom(_profile: str | None) -> str:
        raise RuntimeError("apply failed")

    resp = _client(get_loop=lambda: running_loop, apply_personality=_boom).post(
        "/personalities/apply", json={"name": "user_personalities/other"}
    )

    assert resp.status_code == 500
    assert "apply failed" in resp.json()["error"]


def test_apply_voice_requires_a_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A voice-apply request with no voice is rejected with 400."""
    resp = _client().post("/voices/apply", json={})

    assert resp.status_code == 400
    assert resp.json()["error"] == "missing_voice"


def test_apply_voice_returns_503_without_loop() -> None:
    """A voice-apply request without a running loop returns 503."""
    resp = _client(get_loop=lambda: None).post("/voices/apply?voice=Serena")

    assert resp.status_code == 503
    assert resp.json()["error"] == "loop_unavailable"


def test_apply_voice_uses_json_body(running_loop: asyncio.AbstractEventLoop) -> None:
    """A voice supplied in the JSON body (not the query) is applied."""
    change_voice = AsyncMock(return_value="Voice changed to Serena.")

    resp = _client(get_loop=lambda: running_loop, change_voice=change_voice).post(
        "/voices/apply", json={"voice": "Serena"}
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "status": "Voice changed to Serena."}
    change_voice.assert_awaited_once_with("Serena")


def test_voices_uses_running_loop(running_loop: asyncio.AbstractEventLoop) -> None:
    """The voices endpoint resolves through the running loop when one is available."""
    get_voices = AsyncMock(return_value=["Serena", "Aiden"])

    resp = _client(get_loop=lambda: running_loop, get_voices=get_voices).get("/voices")

    assert resp.status_code == 200
    assert resp.json() == ["Serena", "Aiden"]


def test_load_reads_profile_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a user profile with a saved voice reports that voice, not the default."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    personality_mod._write_profile("voiced", "Be brief.", "", "Serena")

    resp = _client().get("/personalities/load", params={"name": "user_personalities/voiced"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["voice"] == "Serena"
    assert body["uses_default_voice"] is False
