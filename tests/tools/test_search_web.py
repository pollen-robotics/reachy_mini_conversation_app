"""Tests for the search_web tool."""

from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.search_web as search_web
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.search_web import SearchWeb


class _FakeDDGS:
    def __enter__(self) -> "_FakeDDGS":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        return [{"title": "Example", "body": "A snippet", "href": "https://example.com"}]


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_search_web_returns_structured_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful search returns title/snippet/url results."""
    monkeypatch.setattr(search_web, "DDGS", _FakeDDGS)

    result = await SearchWeb()(_deps(), query="reachy mini")

    assert result["query"] == "reachy mini"
    assert result["results"] == [{"title": "Example", "snippet": "A snippet", "url": "https://example.com"}]


@pytest.mark.asyncio
async def test_search_web_rejects_empty_query() -> None:
    """An empty query returns an error without searching."""
    result = await SearchWeb()(_deps(), query="   ")

    assert "error" in result
