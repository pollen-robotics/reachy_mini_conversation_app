"""Tests for Hermes delegation HTTP client."""

from __future__ import annotations
import json
import urllib.error
from typing import Any

import pytest

from reachy_mini_conversation_app.hermes_client import HermesDelegationClient


def _json_response(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


@pytest.mark.asyncio
async def test_delegate_success_normalizes_responses_output_shape() -> None:
    """Normalize a successful OpenAI Responses-style output payload."""
    captured: dict[str, Any] = {}

    def transport(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> tuple[int, bytes]:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(body.decode("utf-8"))
        captured["timeout_seconds"] = timeout_seconds
        return 200, _json_response(
            {
                "id": "resp_123",
                "status": "completed",
                "model": "hermes-agent",
                "details": {"foo": "bar"},
                "actions_taken": [{"tool": "firecrawl_search"}, "ignored"],
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Detroit is cold today.",
                                "annotations": [{"title": "Forecast", "url": "https://example.test/weather"}],
                            }
                        ],
                    }
                ],
            }
        )

    client = HermesDelegationClient(
        base_url="http://hermes.test:8080/",
        api_token="secret-token",
        timeout_seconds=12.0,
        transport=transport,
    )

    result = await client.delegate(task="weather in Detroit", session_id="session-1")

    assert result["status"] == "ok"
    assert result["answer"] == "Detroit is cold today."
    assert result["citations"] == [{"title": "Forecast", "url": "https://example.test/weather"}]
    assert result["task_id"] == "resp_123"
    assert result["details"]["foo"] == "bar"
    assert result["actions_taken"] == [{"tool": "firecrawl_search"}]
    assert captured["url"] == "http://hermes.test:8080/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["headers"]["X-Reachy-Session-Id"] == "session-1"
    assert captured["body"]["model"] == "hermes-agent"
    assert captured["body"]["store"] is False
    assert "weather in Detroit" in captured["body"]["input"]


@pytest.mark.asyncio
async def test_delegate_timeout_returns_normalized_timeout() -> None:
    """Convert transport timeouts to a structured timeout result."""

    def transport(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> tuple[int, bytes]:
        raise TimeoutError("slow")

    client = HermesDelegationClient(
        base_url="http://hermes.test:8080",
        api_token="secret-token",
        timeout_seconds=1.0,
        transport=transport,
    )

    result = await client.delegate(task="look this up")

    assert result["status"] == "timeout"
    assert result["error"] == "timeout"
    assert "timeout" in result["answer"].lower()


@pytest.mark.asyncio
async def test_delegate_http_error_does_not_expose_response_body() -> None:
    """Normalize HTTP auth/client errors without echoing raw response bodies."""

    def transport(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> tuple[int, bytes]:
        return 401, b'{"error":"token secret-token is invalid"}'

    client = HermesDelegationClient(
        base_url="http://hermes.test:8080",
        api_token="secret-token",
        transport=transport,
    )

    result = await client.delegate(task="web search")

    assert result["status"] == "error"
    assert result["error"] == "http_error"
    assert result["details"] == {"http_status": 401}
    assert "secret-token" not in json.dumps(result)


@pytest.mark.asyncio
async def test_delegate_unavailable_for_url_error() -> None:
    """Normalize unreachable Hermes network errors."""

    def transport(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> tuple[int, bytes]:
        raise urllib.error.URLError("connection refused")

    client = HermesDelegationClient(
        base_url="http://hermes.test:8080",
        api_token="secret-token",
        transport=transport,
    )

    result = await client.delegate(task="web search")

    assert result["status"] == "unavailable"
    assert result["error"] == "network_unavailable"


@pytest.mark.asyncio
async def test_delegate_missing_configuration_is_unavailable() -> None:
    """Avoid network calls when base URL or token is missing."""
    called = False

    def transport(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> tuple[int, bytes]:
        nonlocal called
        called = True
        return 200, b"{}"

    client = HermesDelegationClient(base_url="", api_token="", transport=transport)

    result = await client.delegate(task="web search")

    assert result["status"] == "unavailable"
    assert result["error"] == "missing_base_url_or_token"
    assert called is False
