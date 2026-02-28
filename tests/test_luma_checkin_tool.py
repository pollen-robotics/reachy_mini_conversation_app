import asyncio
import json
import sys
import types
from unittest.mock import MagicMock

import httpx

# Minimal stub so core_tools can be imported in environments without reachy_mini installed.
if "reachy_mini" not in sys.modules:
    _reachy_stub = types.ModuleType("reachy_mini")

    class _ReachyMini:
        pass

    _reachy_stub.ReachyMini = _ReachyMini
    sys.modules["reachy_mini"] = _reachy_stub

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.luma_checkin import LumaCheckin


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def test_luma_checkin_success_by_name(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("LUMA_DIRECT_CHECKIN_ENABLED", "true")
        monkeypatch.setenv("LUMA_API_KEY", "test-public-key")
        monkeypatch.setenv("LUMA_SESSION_COOKIE", "session=test-cookie")
        monkeypatch.setenv("LUMA_ACTIVE_EVENT_API_ID", "evt-test")

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/v1/event/get-guests"):
                return httpx.Response(
                    200,
                    json={
                        "entries": [
                            {"guest": {"api_id": "gst-kai", "name": "Kai", "email": "kai@example.com"}},
                        ]
                    },
                )
            if request.url.path.endswith("/event/admin/update-check-in"):
                body = json.loads(request.content.decode("utf-8"))
                assert body["event_api_id"] == "evt-test"
                assert body["rsvp_api_id"] == "gst-kai"
                assert request.headers.get("cookie") == "session=test-cookie"
                return httpx.Response(200, json={"ok": True})
            if request.url.path.endswith("/event/admin/get-guest"):
                assert request.url.params.get("event_api_id") == "evt-test"
                assert request.url.params.get("guest_api_id") == "gst-kai"
                return httpx.Response(
                    200,
                    json={"guest": {"api_id": "gst-kai", "last_checked_in_at": "2026-02-28T00:00:00Z"}},
                )
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        tool = LumaCheckin()
        monkeypatch.setattr(
            LumaCheckin,
            "_make_client",
            lambda self, timeout: httpx.AsyncClient(timeout=timeout, transport=httpx.MockTransport(handler)),
        )

        result = await tool(_deps(), name="Kai")
        assert result["status"] == "checked_in"
        assert result["verified"] is True
        assert result["event_api_id"] == "evt-test"
        assert result["guest_api_id"] == "gst-kai"

    asyncio.run(_run())


def test_luma_checkin_success_by_qr_url(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("LUMA_DIRECT_CHECKIN_ENABLED", "true")
        monkeypatch.setenv("LUMA_API_KEY", "test-public-key")
        monkeypatch.setenv("LUMA_SESSION_COOKIE", "session=test-cookie")
        monkeypatch.delenv("LUMA_ACTIVE_EVENT_API_ID", raising=False)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/v1/event/get-guest"):
                assert request.url.params.get("event_id") == "evt-qr"
                assert request.url.params.get("id") == "pk-123"
                return httpx.Response(200, json={"guest": {"api_id": "gst-qr", "name": "Kai"}})
            if request.url.path.endswith("/event/admin/update-check-in"):
                body = json.loads(request.content.decode("utf-8"))
                assert body["event_api_id"] == "evt-qr"
                assert body["rsvp_api_id"] == "gst-qr"
                return httpx.Response(200, json={"ok": True})
            if request.url.path.endswith("/event/admin/get-guest"):
                return httpx.Response(
                    200,
                    json={"guest": {"api_id": "gst-qr", "last_checked_in_at": "2026-02-28T01:00:00Z"}},
                )
            raise AssertionError(f"unexpected request: {request.method} {request.url}")

        tool = LumaCheckin()
        monkeypatch.setattr(
            LumaCheckin,
            "_make_client",
            lambda self, timeout: httpx.AsyncClient(timeout=timeout, transport=httpx.MockTransport(handler)),
        )

        result = await tool(_deps(), checkin_url="https://luma.com/check-in/evt-qr?pk=pk-123")
        assert result["status"] == "checked_in"
        assert result["verified"] is True
        assert result["event_api_id"] == "evt-qr"
        assert result["guest_api_id"] == "gst-qr"

    asyncio.run(_run())


def test_luma_checkin_requires_session_cookie(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("LUMA_DIRECT_CHECKIN_ENABLED", "true")
        monkeypatch.setenv("LUMA_API_KEY", "test-public-key")
        monkeypatch.delenv("LUMA_SESSION_COOKIE", raising=False)
        monkeypatch.delenv("LUMA_SESSION_COOKIE_FILE", raising=False)

        tool = LumaCheckin()
        result = await tool(_deps(), name="Kai", event_api_id="evt-x")
        assert "error" in result
        assert "session cookie" in result["error"].lower()

    asyncio.run(_run())
