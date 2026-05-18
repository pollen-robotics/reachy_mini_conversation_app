"""Tests for the new pause-phrases + restart admin endpoints in console.py."""

from __future__ import annotations
import threading
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot_comic.pause import (
    DEFAULT_STOP_PHRASES,
    DEFAULT_RESUME_PHRASES,
    DEFAULT_SWITCH_PHRASES,
    DEFAULT_SHUTDOWN_PHRASES,
    PauseController,
)
from robot_comic.console import LocalStream
from robot_comic.pause_settings import PAUSE_SETTINGS_FILENAME


def _make_stream(
    tmp_path: Path,
    *,
    pause_controller: PauseController | None = None,
    stop_event: threading.Event | None = None,
    movement_manager: object | None = None,
) -> tuple[FastAPI, LocalStream]:
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(
        MagicMock(),
        robot,
        settings_app=app,
        instance_path=str(tmp_path),
        app_stop_event=stop_event,
        pause_controller=pause_controller,
        movement_manager=movement_manager,
    )
    stream.init_admin_ui()
    return app, stream


def test_get_pause_phrases_returns_defaults_when_unsaved(tmp_path: Path) -> None:
    """GET /pause_phrases with no saved file returns nulls for saved and defaults for effective."""
    app, _ = _make_stream(tmp_path)
    client = TestClient(app)

    resp = client.get("/pause_phrases")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["saved"] == {"stop": None, "resume": None, "shutdown": None, "switch": None}
    assert data["effective"]["stop"] == list(DEFAULT_STOP_PHRASES)
    assert data["effective"]["resume"] == list(DEFAULT_RESUME_PHRASES)
    assert data["effective"]["shutdown"] == list(DEFAULT_SHUTDOWN_PHRASES)
    assert data["effective"]["switch"] == list(DEFAULT_SWITCH_PHRASES)


def test_post_pause_phrases_persists_and_returns_canonical_form(tmp_path: Path) -> None:
    """POST /pause_phrases writes the file and returns the normalised state."""
    app, _ = _make_stream(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/pause_phrases",
        json={
            "stop": ["  System Pause  ", "system pause"],
            "resume": ["Continue"],
            "shutdown": None,
            "switch": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["saved"]["stop"] == ["system pause"]
    assert data["saved"]["resume"] == ["continue"]
    assert data["saved"]["shutdown"] is None
    assert data["saved"]["switch"] == []
    # Effective falls back to defaults when saved is None
    assert data["effective"]["shutdown"] == list(DEFAULT_SHUTDOWN_PHRASES)
    # Effective is the empty list when saved is the empty list
    assert data["effective"]["switch"] == []
    assert (tmp_path / PAUSE_SETTINGS_FILENAME).exists()


def test_post_pause_phrases_applies_live_when_controller_present(tmp_path: Path) -> None:
    """When a PauseController is wired, POST updates it in-place and reports applied_live=True."""
    controller = PauseController(
        clear_move_queue=MagicMock(),
        on_shutdown=MagicMock(),
    )
    app, _ = _make_stream(tmp_path, pause_controller=controller)
    client = TestClient(app)

    resp = client.post(
        "/pause_phrases",
        json={"stop": ["please halt"], "resume": None, "shutdown": None, "switch": None},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied_live"] is True
    assert controller.get_phrases()["stop"] == ("please halt",)


def test_post_pause_phrases_without_controller_reports_not_applied(tmp_path: Path) -> None:
    """Without a controller wired, save still succeeds but applied_live is False."""
    app, _ = _make_stream(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/pause_phrases",
        json={"stop": ["please halt"], "resume": None, "shutdown": None, "switch": None},
    )
    assert resp.status_code == 200
    assert resp.json()["applied_live"] is False


def test_post_admin_restart_sets_stop_event(tmp_path: Path) -> None:
    """POST /admin/restart sets the supplied threading.Event for graceful shutdown."""
    stop_event = threading.Event()
    app, _ = _make_stream(tmp_path, stop_event=stop_event)
    client = TestClient(app)

    resp = client.post("/admin/restart")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "graceful" in body["message"].lower() or "shutting down" in body["message"].lower()
    assert stop_event.is_set()


def test_post_admin_restart_503_when_no_stop_event(tmp_path: Path) -> None:
    """When no stop event was plumbed in, the endpoint reports 503."""
    app, _ = _make_stream(tmp_path)
    client = TestClient(app)

    resp = client.post("/admin/restart")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "no_stop_event"


def test_round_trip_phrase_save_and_get(tmp_path: Path) -> None:
    """POST then GET reflects the previously persisted state."""
    app, _ = _make_stream(tmp_path)
    client = TestClient(app)

    client.post(
        "/pause_phrases",
        json={"stop": ["robot stop"], "resume": None, "shutdown": None, "switch": None},
    )
    resp = client.get("/pause_phrases")
    assert resp.status_code == 200
    data = resp.json()
    assert data["saved"]["stop"] == ["robot stop"]
    assert data["saved"]["resume"] is None


def test_get_movement_speed_returns_current_factor(tmp_path: Path) -> None:
    """GET /movement_speed reports the live ``speed_factor`` and the slider bounds."""
    mm = SimpleNamespace(speed_factor=0.6, set_speed_factor=MagicMock())
    app, _ = _make_stream(tmp_path, movement_manager=mm)
    client = TestClient(app)

    resp = client.get("/movement_speed")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"ok": True, "value": 0.6, "min": 0.1, "max": 2.0, "step": 0.05}


def test_post_movement_speed_updates_manager(tmp_path: Path) -> None:
    """POST /movement_speed calls ``set_speed_factor`` and echoes the new value."""

    class _MM:
        def __init__(self) -> None:
            self.speed_factor = 0.6
            self.calls: list[float] = []

        def set_speed_factor(self, value: float) -> None:
            self.calls.append(value)
            self.speed_factor = max(0.1, min(2.0, value))

    mm = _MM()
    app, _ = _make_stream(tmp_path, movement_manager=mm)
    client = TestClient(app)

    resp = client.post("/movement_speed", json={"value": 1.4})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "value": 1.4}
    assert mm.calls == [1.4]


def test_post_movement_speed_persists_value_across_restarts(tmp_path: Path) -> None:
    """POST /movement_speed writes the clamped value to startup_settings.json (#309)."""
    from robot_comic.startup_settings import read_startup_settings

    class _MM:
        def __init__(self) -> None:
            self.speed_factor = 0.6

        def set_speed_factor(self, value: float) -> None:
            self.speed_factor = max(0.1, min(2.0, value))

    mm = _MM()
    app, _ = _make_stream(tmp_path, movement_manager=mm)
    client = TestClient(app)

    resp = client.post("/movement_speed", json={"value": 0.45})
    assert resp.status_code == 200

    persisted = read_startup_settings(tmp_path)
    assert persisted.movement_speed == 0.45


def test_post_movement_speed_preserves_existing_profile_and_voice(tmp_path: Path) -> None:
    """Updating the slider must not clobber a previously-saved profile/voice."""
    from robot_comic.startup_settings import read_startup_settings, write_startup_settings

    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer")

    class _MM:
        def __init__(self) -> None:
            self.speed_factor = 0.6

        def set_speed_factor(self, value: float) -> None:
            self.speed_factor = max(0.1, min(2.0, value))

    mm = _MM()
    app, _ = _make_stream(tmp_path, movement_manager=mm)
    client = TestClient(app)

    resp = client.post("/movement_speed", json={"value": 1.2})
    assert resp.status_code == 200

    settings = read_startup_settings(tmp_path)
    assert settings.profile == "house_comedian"
    assert settings.voice == "shimmer"
    assert settings.movement_speed == 1.2


def test_movement_speed_503_when_no_manager(tmp_path: Path) -> None:
    """Both endpoints return 503 when no movement_manager is wired in."""
    app, _ = _make_stream(tmp_path)
    client = TestClient(app)

    get_resp = client.get("/movement_speed")
    assert get_resp.status_code == 503
    assert get_resp.json()["error"] == "no_movement_manager"

    post_resp = client.post("/movement_speed", json={"value": 1.0})
    assert post_resp.status_code == 503
    assert post_resp.json()["error"] == "no_movement_manager"


def test_post_movement_speed_rejects_invalid_value(tmp_path: Path) -> None:
    """Non-numeric payloads return 400 without touching the manager."""
    mm = SimpleNamespace(speed_factor=0.6, set_speed_factor=MagicMock())
    app, _ = _make_stream(tmp_path, movement_manager=mm)
    client = TestClient(app)

    resp = client.post("/movement_speed", json={"value": "fast"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_value"
    mm.set_speed_factor.assert_not_called()
