"""Tests for the delegate_to_hermes conversation tool."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini_conversation_app.tools.delegate_to_hermes import DelegateToHermes


@pytest.mark.asyncio
async def test_delegate_tool_disabled_without_client() -> None:
    """Return a speakable disabled result when no Hermes client is injected."""
    tool = DelegateToHermes()
    deps = SimpleNamespace(hermes_client=None)

    result = await tool(deps, task="search the web")  # type: ignore[arg-type]

    assert result["status"] == "disabled"
    assert result["error_message"] == "hermes_client_missing"
    assert "not configured" in result["answer"]
    assert "error" not in result


@pytest.mark.asyncio
async def test_delegate_tool_calls_client_and_preserves_structured_failure() -> None:
    """Call the injected client and avoid a top-level `error` key for background manager compatibility."""
    calls: list[dict[str, Any]] = []

    class FakeHermesClient:
        async def delegate(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {
                "status": "timeout",
                "answer": "Hermes timed out.",
                "error": "timeout",
            }

    tool = DelegateToHermes()
    deps = SimpleNamespace(hermes_client=FakeHermesClient())

    result = await tool(
        deps,  # type: ignore[arg-type]
        task="find current news",
        why_needed="needs current information",
        response_style="detailed",
        allowed_domains=["example.com", 123],
    )

    assert result == {
        "status": "timeout",
        "answer": "Hermes timed out.",
        "error_message": "timeout",
    }
    assert calls[0]["task"] == "find current news"
    assert calls[0]["why_needed"] == "needs current information"
    assert calls[0]["response_style"] == "detailed"
    assert calls[0]["allowed_domains"] == ["example.com", "123"]
    assert calls[0]["urgency"] == "interactive"
