"""Tests for the GET /api/xtts/voices admin endpoint (#438)."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from robot_comic.console import LocalStream


def _make_client() -> TestClient:
    """Build a TestClient backed by a minimal LocalStream with admin UI."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), None, settings_app=app)
    stream.init_admin_ui()
    return TestClient(app)


class TestXttsVoicesEndpoint:
    """Route existence, schema, and fallback checks for /api/xtts/voices."""

    def test_route_exists_and_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GET /api/xtts/voices returns 200 with a voice list on success."""
        from robot_comic.adapters import xtts_tts_adapter as adapter_mod

        async def _fake_get_available_voices(self: Any) -> list[str]:
            return ["don_rickles", "george_carlin"]

        async def _fake_shutdown(self: Any) -> None:
            pass

        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "get_available_voices",
            _fake_get_available_voices,
        )
        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "shutdown",
            _fake_shutdown,
        )

        client = _make_client()
        resp = client.get("/api/xtts/voices")

        assert resp.status_code == 200
        body = resp.json()
        assert "voices" in body
        assert body["voices"] == ["don_rickles", "george_carlin"]

    def test_route_path_matches_main_js(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route is reachable at exactly /api/xtts/voices (the path main.js calls)."""
        from robot_comic.adapters import xtts_tts_adapter as adapter_mod

        async def _fake_get_available_voices(self: Any) -> list[str]:
            return ["don_rickles"]

        async def _fake_shutdown(self: Any) -> None:
            pass

        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "get_available_voices",
            _fake_get_available_voices,
        )
        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "shutdown",
            _fake_shutdown,
        )

        client = _make_client()
        # Exact path must match the URL used in populateXttsVoices() in main.js
        resp = client.get("/api/xtts/voices")
        assert resp.status_code == 200

    def test_returns_ok_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Response body includes ok: true when the adapter succeeds."""
        from robot_comic.adapters import xtts_tts_adapter as adapter_mod

        async def _fake_get_available_voices(self: Any) -> list[str]:
            return ["voice_a", "voice_b", "voice_c"]

        async def _fake_shutdown(self: Any) -> None:
            pass

        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "get_available_voices",
            _fake_get_available_voices,
        )
        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "shutdown",
            _fake_shutdown,
        )

        client = _make_client()
        resp = client.get("/api/xtts/voices")

        body = resp.json()
        assert body.get("ok") is True
        assert isinstance(body["voices"], list)
        assert len(body["voices"]) == 3

    def test_adapter_fallback_still_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the adapter falls back to [current_voice] (server unreachable),
        the route still returns 200 — not a 500 — because XttsTTSAdapter
        get_available_voices() handles exceptions internally and returns a list."""
        from robot_comic.config import config
        from robot_comic.adapters import xtts_tts_adapter as adapter_mod

        monkeypatch.setattr(config, "XTTS_DEFAULT_SPEAKER_KEY", "don_rickles", raising=False)

        async def _fallback_voices(self: Any) -> list[str]:
            # Simulates the adapter's own fallback path (server unreachable)
            return [self._current_voice]

        async def _fake_shutdown(self: Any) -> None:
            pass

        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "get_available_voices",
            _fallback_voices,
        )
        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "shutdown",
            _fake_shutdown,
        )

        client = _make_client()
        resp = client.get("/api/xtts/voices")

        assert resp.status_code == 200
        body = resp.json()
        assert "voices" in body
        assert len(body["voices"]) >= 1

    def test_route_returns_500_when_adapter_raises_unexpectedly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If get_available_voices() raises (bypassing the adapter's own guard),
        the route catches it and returns 500 with an error field."""
        from robot_comic.adapters import xtts_tts_adapter as adapter_mod

        async def _boom(self: Any) -> list[str]:
            raise RuntimeError("unexpected adapter failure")

        async def _fake_shutdown(self: Any) -> None:
            pass

        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "get_available_voices",
            _boom,
        )
        monkeypatch.setattr(
            adapter_mod.XttsTTSAdapter,
            "shutdown",
            _fake_shutdown,
        )

        client = _make_client()
        resp = client.get("/api/xtts/voices")

        assert resp.status_code == 500
        body = resp.json()
        assert "error" in body

    def test_shutdown_called_even_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """adapter.shutdown() is called in the finally block even when an error occurs."""
        from robot_comic.adapters import xtts_tts_adapter as adapter_mod

        shutdown_calls: list[int] = []

        async def _boom(self: Any) -> list[str]:
            raise RuntimeError("simulated failure")

        async def _tracking_shutdown(self: Any) -> None:
            shutdown_calls.append(1)

        monkeypatch.setattr(adapter_mod.XttsTTSAdapter, "get_available_voices", _boom)
        monkeypatch.setattr(adapter_mod.XttsTTSAdapter, "shutdown", _tracking_shutdown)

        client = _make_client()
        client.get("/api/xtts/voices")

        assert shutdown_calls == [1], "shutdown() must be called exactly once via finally"
