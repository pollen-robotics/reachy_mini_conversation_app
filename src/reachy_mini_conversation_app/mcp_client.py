"""Helpers for consuming remote MCP tools over HTTP(S).

This module validates remote endpoints, discovers tools, and maps calls/results
into the app's tool interface without mutating the local project environment or
downloading third-party Python code.
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Mapping, AsyncIterator
from datetime import timedelta
from contextlib import asynccontextmanager
from dataclasses import field, dataclass
from urllib.parse import urlparse


if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as McpTool
    from mcp.types import CallToolResult as McpCallToolResult


_NAME_SEGMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_NORMALIZER_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
_LOCAL_HTTP_HOSTS = {"127.0.0.1", "localhost", "::1"}
_NAMESPACE_SEPARATOR = "__"


class McpClientError(RuntimeError):
    """Base error for the MCP client."""


class McpDependencyError(McpClientError):
    """Raised when the optional MCP SDK is not installed."""


class McpTransportError(McpClientError):
    """Raised when discovery fails before a remote tool runs."""


class McpToolInvocationError(McpClientError):
    """Raised when a remote tool call fails at the transport layer."""


class McpToolTimeoutError(McpToolInvocationError):
    """Raised when a remote tool call exceeds the configured timeout."""


def _require_name_segment(label: str, value: str) -> str:
    candidate = value.strip()
    if _NAME_SEGMENT_PATTERN.fullmatch(candidate) is None:
        raise ValueError(f"Invalid {label} '{value}'. Expected pattern '[A-Za-z_][A-Za-z0-9_]*'.")
    return candidate


def apply_name_normalization(value: str) -> str:
    """Replace non-identifier characters with underscores and collapse runs."""
    normalized = _NAME_NORMALIZER_PATTERN.sub("_", value).strip("_")
    return re.sub(r"_+", "_", normalized)


def _normalize_name_segment(label: str, value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError(f"{label.capitalize()} cannot be empty.")

    normalized = apply_name_normalization(raw)
    if not normalized:
        raise ValueError(f"{label.capitalize()} '{value}' cannot be normalized into a valid tool identifier.")
    if normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return _require_name_segment(label, normalized)


def validate_http_mcp_url(url: str) -> str:
    """Validate that the MCP endpoint uses HTTP(S)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported MCP URL scheme '{parsed.scheme}'. Use http:// or https://.")
    if not parsed.netloc:
        raise ValueError(f"Invalid MCP URL '{url}'. Missing host.")

    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host not in _LOCAL_HTTP_HOSTS:
        raise ValueError("Remote MCP servers must use HTTPS. Plain HTTP is only allowed for localhost.")
    return url


def build_namespaced_tool_name(server_alias: str, tool_name: str) -> str:
    """Build a local tool name for a remote MCP tool."""
    alias = _require_name_segment("server alias", server_alias)
    tool_segment = _normalize_name_segment("tool name", tool_name)
    return f"{alias}{_NAMESPACE_SEPARATOR}{tool_segment}"


def _dump_content_block(block: object) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        dumped = block.model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {"type": getattr(block, "type", "unknown")}


def _join_text_content(content_blocks: list[dict[str, Any]]) -> str | None:
    text_parts = [
        block["text"] for block in content_blocks if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def _exception_contains_timeout(exc: BaseException) -> bool:
    timeout_exception = _httpx_timeout_exception_type()
    if isinstance(exc, timeout_exception):
        return True

    if "timed out" in str(exc).lower() or "deadline exceeded" in str(exc).lower():
        return True

    nested: list[BaseException] = []
    grouped_exceptions = getattr(exc, "exceptions", None)
    if isinstance(grouped_exceptions, tuple):
        nested.extend(grouped_exceptions)
    if exc.__cause__ is not None:
        nested.append(exc.__cause__)
    if exc.__context__ is not None:
        nested.append(exc.__context__)

    return any(_exception_contains_timeout(item) for item in nested)


def _load_mcp_sdk() -> tuple[type["ClientSession"], Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise McpDependencyError(
            "Remote MCP tools require the optional 'remote_tools' dependencies. "
            "Install the project with the 'remote_tools' extra before using this module."
        ) from exc
    return ClientSession, streamable_http_client


def _load_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise McpDependencyError(
            "Remote MCP tools require the optional 'remote_tools' dependencies. "
            "Install the project with the 'remote_tools' extra before using this module."
        ) from exc
    return httpx


def _httpx_timeout_exception_type() -> tuple[type[BaseException], ...]:
    try:
        timeout_exception = _load_httpx().TimeoutException
    except McpDependencyError:
        return (TimeoutError,)
    return (TimeoutError, timeout_exception)


@dataclass(frozen=True)
class RemoteMcpServerConfig:
    """Allowlisted MCP server configuration."""

    alias: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    request_timeout_s: float = 10.0
    tool_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration once the dataclass has been created."""
        object.__setattr__(self, "alias", _require_name_segment("server alias", self.alias))
        object.__setattr__(self, "url", validate_http_mcp_url(self.url))
        object.__setattr__(self, "headers", {str(k): str(v) for k, v in self.headers.items()})
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be greater than zero.")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be greater than zero.")


@dataclass(frozen=True)
class RemoteToolSpec:
    """App-facing representation of a remote MCP tool."""

    server_alias: str
    remote_name: str
    namespaced_name: str
    description: str
    parameters_schema: dict[str, Any]

    @classmethod
    def from_mcp_tool(cls, server_alias: str, tool: "McpTool") -> "RemoteToolSpec":
        """Build an app-facing spec from an MCP SDK tool descriptor."""
        description = (getattr(tool, "description", None) or "").strip()
        parameters_schema = getattr(tool, "inputSchema", None)
        if not isinstance(parameters_schema, dict):
            parameters_schema = {"type": "object", "properties": {}, "required": []}

        remote_name = str(getattr(tool, "name", "")).strip()
        if not remote_name:
            raise ValueError("Remote MCP tool is missing a name.")

        return cls(
            server_alias=server_alias,
            remote_name=remote_name,
            namespaced_name=build_namespaced_tool_name(server_alias, remote_name),
            description=description or f"Remote MCP tool '{remote_name}' from server '{server_alias}'.",
            parameters_schema=dict(parameters_schema),
        )

    def to_function_spec(self) -> dict[str, Any]:
        """Translate to the app's function-calling shape."""
        return {
            "type": "function",
            "name": self.namespaced_name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


@dataclass(frozen=True)
class RemoteToolCallResponse:
    """Mapped result for a remote MCP tool call."""

    server_alias: str
    remote_tool_name: str
    namespaced_tool_name: str
    status: str
    content_blocks: list[dict[str, Any]]
    text: str | None
    structured_content: Any | None

    @classmethod
    def from_call_tool_result(
        cls,
        *,
        server_alias: str,
        remote_tool_name: str,
        result: "McpCallToolResult",
    ) -> "RemoteToolCallResponse":
        """Convert an MCP SDK tool result into the app's result envelope."""
        content_blocks = [_dump_content_block(block) for block in getattr(result, "content", [])]
        return cls(
            server_alias=server_alias,
            remote_tool_name=remote_tool_name,
            namespaced_tool_name=build_namespaced_tool_name(server_alias, remote_tool_name),
            status="error" if bool(getattr(result, "isError", False)) else "ok",
            content_blocks=content_blocks,
            text=_join_text_content(content_blocks),
            structured_content=getattr(result, "structuredContent", None),
        )

    def to_tool_result(self) -> dict[str, Any]:
        """Return a dict shaped like the app's tool results."""
        payload: dict[str, Any] = {
            "status": self.status,
            "server_alias": self.server_alias,
            "remote_tool_name": self.remote_tool_name,
            "namespaced_tool_name": self.namespaced_tool_name,
            "content_blocks": self.content_blocks,
        }
        if self.text is not None:
            payload["text"] = self.text
        if self.structured_content is not None:
            payload["structured_content"] = self.structured_content
        return payload


class RemoteMcpToolClient:
    """Minimal async client for allowlisted remote MCP tool servers."""

    def __init__(self, server: RemoteMcpServerConfig) -> None:
        """Store one allowlisted server configuration and an in-memory tool cache."""
        self.server = server
        self._tool_index: dict[str, RemoteToolSpec] = {}

    async def list_tool_specs(self) -> list[RemoteToolSpec]:
        """Discover remote tools and translate them into app-facing specs."""
        try:
            async with self._session() as session:
                discovered = await self._list_all_tools(session)
        except McpDependencyError:
            raise
        except Exception as exc:
            raise McpTransportError(
                f"Failed to discover MCP tools from '{self.server.alias}' at {self.server.url}: {exc}"
            ) from exc

        specs = [RemoteToolSpec.from_mcp_tool(self.server.alias, tool) for tool in discovered]
        self._tool_index = _index_remote_tools(specs)
        return specs

    async def list_function_specs(self) -> list[dict[str, Any]]:
        """Discover tools and translate them into function-calling specs."""
        return [spec.to_function_spec() for spec in await self.list_tool_specs()]

    async def call_tool(self, namespaced_tool_name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a remote MCP tool by its namespaced local ID."""
        spec = await self._resolve_tool_spec(namespaced_tool_name)
        timeout_exception = _httpx_timeout_exception_type()

        try:
            async with self._session() as session:
                result = await session.call_tool(
                    spec.remote_name,
                    arguments=dict(arguments or {}),
                    read_timeout_seconds=timedelta(seconds=self.server.tool_timeout_s),
                )
        except McpDependencyError:
            raise
        except timeout_exception as exc:
            raise McpToolTimeoutError(
                f"Timed out calling MCP tool '{namespaced_tool_name}' from '{self.server.alias}'."
            ) from exc
        except Exception as exc:
            if _exception_contains_timeout(exc):
                raise McpToolTimeoutError(
                    f"Timed out calling MCP tool '{namespaced_tool_name}' from '{self.server.alias}'."
                ) from exc
            raise McpToolInvocationError(
                f"Failed to call MCP tool '{namespaced_tool_name}' from '{self.server.alias}': {exc}"
            ) from exc

        return RemoteToolCallResponse.from_call_tool_result(
            server_alias=self.server.alias,
            remote_tool_name=spec.remote_name,
            result=result,
        ).to_tool_result()

    async def _resolve_tool_spec(self, namespaced_tool_name: str) -> RemoteToolSpec:
        spec = self._tool_index.get(namespaced_tool_name)
        if spec is not None:
            return spec

        await self.list_tool_specs()
        spec = self._tool_index.get(namespaced_tool_name)
        if spec is None:
            raise ValueError(f"Unknown remote MCP tool '{namespaced_tool_name}' for server '{self.server.alias}'.")
        return spec

    async def _list_all_tools(self, session: "ClientSession") -> list["McpTool"]:
        tools: list[McpTool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = getattr(page, "nextCursor", None)
            if cursor is None:
                return tools

    @asynccontextmanager
    async def _session(self) -> AsyncIterator["ClientSession"]:
        client_session_cls, streamable_http_client = _load_mcp_sdk()
        httpx = _load_httpx()
        client_timeout = max(self.server.request_timeout_s, self.server.tool_timeout_s)

        async with httpx.AsyncClient(
            headers=self.server.headers,
            follow_redirects=False,
            timeout=client_timeout,
        ) as http_client:
            async with streamable_http_client(self.server.url, http_client=http_client) as transport:
                read_stream, write_stream, _ = transport
                async with client_session_cls(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session


def _index_remote_tools(specs: list[RemoteToolSpec]) -> dict[str, RemoteToolSpec]:
    index: dict[str, RemoteToolSpec] = {}
    collisions: dict[str, list[str]] = {}

    for spec in specs:
        existing = index.get(spec.namespaced_name)
        if existing is None:
            index[spec.namespaced_name] = spec
            continue

        collisions.setdefault(spec.namespaced_name, [existing.remote_name]).append(spec.remote_name)

    if collisions:
        details = "; ".join(
            f"{tool_name}: {sorted(remote_names)}" for tool_name, remote_names in sorted(collisions.items())
        )
        raise ValueError(f"Remote MCP tool names collide after local namespacing/normalization. Conflicts: {details}")

    return index
