from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import pytest
import uvicorn
import pytest_asyncio
from starlette.routing import Mount
from starlette.responses import PlainTextResponse
from starlette.applications import Starlette


pytest.importorskip("mcp.server.fastmcp")

from mcp.server.fastmcp import FastMCP

from reachy_mini_conversation_app.mcp_client import (
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


def _build_local_mcp_app(required_token: str) -> object:
    mcp_server = FastMCP("Reachy Test MCP", stateless_http=True, json_response=True)

    @mcp_server.tool()
    def echo_text(message: str) -> dict[str, str]:
        """Echo text as structured JSON."""
        return {"echo": message}

    @mcp_server.tool()
    async def slow_echo(message: str, delay_s: float = 0.2) -> dict[str, str]:
        """Delay, then echo text as structured JSON."""
        await asyncio.sleep(delay_s)
        return {"echo": message}

    mcp_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with mcp_server.session_manager.run():
            yield

    app = Starlette(routes=[Mount("/", app=mcp_app)], lifespan=lifespan)
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


@pytest_asyncio.fixture
async def local_mcp_server(unused_tcp_port: int) -> AsyncIterator[tuple[str, str]]:
    """Run a local HTTP MCP server that requires a bearer token."""
    token = "local-test-token"
    app = _build_local_mcp_app(token)
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
