from __future__ import annotations

import pytest


pytest.importorskip("mcp_types")

from mcp_types import Tool, TextContent, CallToolResult, ListToolsResult

from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    RemoteToolCallResponse,
    validate_http_mcp_url,
    build_namespaced_tool_name,
)


def test_validate_http_mcp_url_rejects_non_http_scheme() -> None:
    """Only HTTP(S) MCP endpoints are supported."""
    with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
        validate_http_mcp_url("stdio://local-server")


def test_validate_http_mcp_url_rejects_non_local_plain_http() -> None:
    """Remote servers must use HTTPS unless they are local development endpoints."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_http_mcp_url("http://example.com/mcp")


def test_build_namespaced_tool_name_normalizes_tool_segment() -> None:
    """Remote tool names are normalized into app-safe tool IDs."""
    assert build_namespaced_tool_name("gradio_docs", "search-docs") == "gradio_docs__search_docs"


def test_remote_tool_spec_translates_to_function_spec() -> None:
    """Discovered MCP tools should translate into app function specs."""
    tool = Tool(
        name="search-docs",
        description="Search the docs",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    spec = RemoteToolSpec.from_mcp_tool("gradio_docs", tool)

    assert spec.remote_name == "search-docs"
    assert spec.namespaced_name == "gradio_docs__search_docs"
    assert spec.to_function_spec() == {
        "type": "function",
        "name": "gradio_docs__search_docs",
        "description": "Search the docs",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def test_remote_tool_error_result_maps_to_app_payload() -> None:
    """Remote tool errors should remain visible after response mapping."""
    result = CallToolResult(
        content=[TextContent(type="text", text="Search backend unavailable")],
        structured_content=None,
        is_error=True,
    )

    payload = RemoteToolCallResponse.from_call_tool_result(
        server_alias="gradio_docs",
        remote_tool_name="search-docs",
        result=result,
    ).to_tool_result()

    assert payload["status"] == "error"
    assert payload["namespaced_tool_name"] == "gradio_docs__search_docs"
    assert payload["text"] == "Search backend unavailable"


def test_remote_tool_spec_reads_fields_the_sdk_actually_exposes() -> None:
    """Pins the v1→v2 rename: the wire names are pydantic *aliases*, not attributes.

    Reading `tool.inputSchema` returns nothing on v2, which silently degrades every
    remote tool to an empty parameters schema instead of raising anywhere.
    """
    tool = Tool(
        name="search-docs",
        description="Search the docs",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    assert not hasattr(tool, "inputSchema")

    spec = RemoteToolSpec.from_mcp_tool("gradio_docs", tool)
    assert spec.parameters_schema["properties"] == {"query": {"type": "string"}}
    assert spec.parameters_schema["required"] == ["query"]


def test_remote_tool_call_response_reads_fields_the_sdk_actually_exposes() -> None:
    """Same rename on results: `isError`/`structuredContent` are aliases, so an error would read as success."""
    result = CallToolResult(
        content=[TextContent(type="text", text="done")],
        structured_content={"rows": 2},
        is_error=True,
    )
    assert not hasattr(result, "isError")
    assert not hasattr(result, "structuredContent")

    response = RemoteToolCallResponse.from_call_tool_result(
        server_alias="gradio_docs",
        remote_tool_name="search-docs",
        result=result,
    )
    assert response.status == "error"
    assert response.structured_content == {"rows": 2}


@pytest.mark.asyncio
async def test_list_all_tools_follows_the_pagination_cursor() -> None:
    """Pins the v1→v2 rename on the *paging* field, which fails silently.

    `nextCursor` is a pydantic alias on v2, so a client still reading it gets
    None on a page that genuinely has a successor: discovery stops after page
    one and every tool beyond it simply never exists, with no error raised
    anywhere. Servers with a small tool count hide this, which is why it
    survives casual testing.
    """
    first = ListToolsResult(
        tools=[Tool(name="alpha", description="first page", input_schema={"type": "object"})],
        nextCursor="page-2",
    )
    last = ListToolsResult(
        tools=[Tool(name="omega", description="second page", input_schema={"type": "object"})],
    )
    assert not hasattr(first, "nextCursor")

    pages = [first, last]
    seen_cursors: list[str | None] = []

    class _PagingClient:
        async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
            seen_cursors.append(cursor)
            return pages[len(seen_cursors) - 1]

    client = RemoteMcpToolClient(RemoteMcpServerConfig(alias="gradio_docs", url="https://example.invalid/mcp"))
    tools = await client._list_all_tools(_PagingClient())

    assert [tool.name for tool in tools] == ["alpha", "omega"]
    assert seen_cursors == [None, "page-2"]
