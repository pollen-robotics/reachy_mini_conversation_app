from __future__ import annotations
import asyncio
from typing import Any
from collections.abc import AsyncIterator

import pytest
import uvicorn
import pytest_asyncio
from starlette.responses import PlainTextResponse


pytest.importorskip("mcp.server")

from mcp.server import MCPServer

import reachy_mini_conversation_app.mcp_client as mcp_client_module
from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    McpTransportError,
    McpToolTimeoutError,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
)


class _BearerAuthMiddleware:
    def __init__(self, app: object, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            headers = {key.decode("latin-1"): value.decode("latin-1") for key, value in scope.get("headers", [])}
            if headers.get("authorization") != f"Bearer {self.token}":
                response = PlainTextResponse("Unauthorized", status_code=401)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


def _build_local_mcp_app(required_token: str, *, json_response: bool = True) -> object:
    mcp_server = MCPServer("Reachy Test MCP")

    @mcp_server.tool()
    def echo_text(message: str) -> dict[str, str]:
        """Echo text as structured JSON."""
        return {"echo": message}

    @mcp_server.tool()
    async def slow_echo(message: str, delay_s: float = 0.2) -> dict[str, str]:
        """Delay, then echo text as structured JSON."""
        await asyncio.sleep(delay_s)
        return {"echo": message}

    # Gradio Spaces answer over SSE (json_response=False), where the SDK caps the size of one event.
    app = mcp_server.streamable_http_app(json_response=json_response, stateless_http=True)
    return _BearerAuthMiddleware(app, required_token)


async def _wait_for_server(host: str, port: int) -> None:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            await asyncio.sleep(0.05)
            continue

        writer.close()
        await writer.wait_closed()
        return

    raise RuntimeError(f"Timed out waiting for local MCP server on {host}:{port}")


@pytest_asyncio.fixture(params=[True, False], ids=["json", "sse"])
async def local_mcp_server(unused_tcp_port: int, request: pytest.FixtureRequest) -> AsyncIterator[tuple[str, str]]:
    """Run a local HTTP MCP server that requires a bearer token, answering in JSON or over SSE."""
    token = "local-test-token"
    app = _build_local_mcp_app(token, json_response=request.param)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=unused_tcp_port,
            log_level="error",
            access_log=False,
            ws="none",
        )
    )

    task = asyncio.create_task(server.serve())
    await _wait_for_server("127.0.0.1", unused_tcp_port)

    try:
        yield (f"http://127.0.0.1:{unused_tcp_port}/mcp", token)
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_remote_mcp_tool_client_calls_known_tool_without_discovery(
    local_mcp_server: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached tool spec should let the first call skip remote discovery."""
    server_url, token = local_mcp_server
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="gradio_docs",
            url=server_url,
            headers={"Authorization": f"Bearer {token}"},
            request_timeout_s=2.0,
            tool_timeout_s=1.0,
        ),
        known_tools=[
            RemoteToolSpec(
                server_alias="gradio_docs",
                remote_name="echo_text",
                namespaced_name="gradio_docs__echo_text",
                description="Echo text",
                parameters_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
        ],
    )

    async def _fail_list_tool_specs() -> list[RemoteToolSpec]:
        raise AssertionError("discovery should not run")

    monkeypatch.setattr(client, "list_tool_specs", _fail_list_tool_specs)

    result = await client.call_tool("gradio_docs__echo_text", {"message": "hello"})

    assert result["status"] == "ok"
    assert result["structured_content"] == {"echo": "hello"}


@pytest.mark.asyncio
async def test_remote_mcp_tool_client_discovers_calls_and_handles_timeout(
    local_mcp_server: tuple[str, str],
) -> None:
    """The client should discover tools, map schemas, invoke tools, and handle timeouts/auth."""
    server_url, token = local_mcp_server
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="gradio_docs",
            url=server_url,
            headers={"Authorization": f"Bearer {token}"},
            request_timeout_s=2.0,
            tool_timeout_s=1.0,
        )
    )

    function_specs = await client.list_function_specs()
    function_names = sorted(spec["name"] for spec in function_specs)
    assert function_names == ["gradio_docs__echo_text", "gradio_docs__slow_echo"]

    echo_spec = next(spec for spec in function_specs if spec["name"] == "gradio_docs__echo_text")
    assert echo_spec["parameters"]["properties"]["message"]["type"] == "string"

    result = await client.call_tool("gradio_docs__echo_text", {"message": "hello"})
    assert result["status"] == "ok"
    assert result["structured_content"] == {"echo": "hello"}
    assert result["namespaced_tool_name"] == "gradio_docs__echo_text"

    timeout_client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="gradio_docs",
            url=server_url,
            headers={"Authorization": f"Bearer {token}"},
            request_timeout_s=2.0,
            tool_timeout_s=0.05,
        )
    )
    with pytest.raises(McpToolTimeoutError, match="Timed out calling MCP tool"):
        await timeout_client.call_tool(
            "gradio_docs__slow_echo",
            {"message": "hello", "delay_s": 0.2},
        )

    unauthorized_client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="gradio_docs",
            url=server_url,
            request_timeout_s=2.0,
            tool_timeout_s=1.0,
        )
    )
    with pytest.raises(McpTransportError, match="Failed to discover MCP tools"):
        await unauthorized_client.list_tool_specs()


@pytest.mark.asyncio
async def test_remote_mcp_tool_client_passes_its_event_size_cap_to_the_transport(
    local_mcp_server: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_event_bytes reaches the SDK transport, whose 1 MiB default refuses a Space's inline audio.

    The SDK only enforces the cap while an event arrives in several network reads, which a local
    server sends in one, so this checks the wiring rather than the refusal.
    """
    server_url, token = local_mcp_server
    client_cls, streamable_http_client = mcp_client_module._load_mcp_sdk()
    seen_limits: list[int | None] = []

    def _recording_transport(url: str, **kwargs: Any) -> Any:
        seen_limits.append(kwargs.get("max_sse_event_size"))
        return streamable_http_client(url, **kwargs)

    monkeypatch.setattr(mcp_client_module, "_load_mcp_sdk", lambda: (client_cls, _recording_transport))
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="media",
            url=server_url,
            headers={"Authorization": f"Bearer {token}"},
            max_event_bytes=32 * 1024 * 1024,
        )
    )

    result = await client.call_tool("media__echo_text", {"message": "hello"})

    assert result["status"] == "ok"
    assert seen_limits and set(seen_limits) == {32 * 1024 * 1024}
