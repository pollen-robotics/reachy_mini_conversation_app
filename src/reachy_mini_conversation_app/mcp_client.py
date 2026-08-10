"""Helpers for consuming remote MCP tools over HTTP(S).

This module validates remote endpoints, discovers tools, and maps calls/results
into the app's tool interface without mutating the local project environment or
downloading third-party Python code.
"""

from __future__ import annotations
import re
import ipaddress
from typing import TYPE_CHECKING, Any, Mapping, Sequence, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import field, dataclass
from urllib.parse import urlparse


if TYPE_CHECKING:
    from mcp import Client
    from mcp_types import Tool as McpTool
    from mcp_types import CallToolResult as McpCallToolResult


_NAME_SEGMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_NORMALIZER_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
_LOCAL_HTTP_HOSTS = {"127.0.0.1", "localhost", "::1"}
_NAMESPACE_SEPARATOR = "__"


class McpClientError(RuntimeError):
    """Base error for the MCP client."""


class McpDependencyError(McpClientError):
    """Raised when a required MCP client dependency is not installed."""


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


def _require_alias_segment(label: str, value: str) -> str:
    candidate = _require_name_segment(label, value)
    # '__' separates the alias from the tool segment in namespaced tool names, so an
    # alias containing '__' or ending in '_' makes prefix matches (e.g. profile
    # cleanup on remove) claim a sibling alias's tools.
    if _NAMESPACE_SEPARATOR in candidate or candidate.endswith("_"):
        raise ValueError(
            f"Invalid {label} '{value}'. Aliases cannot contain '{_NAMESPACE_SEPARATOR}' or end with '_'."
        )
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


def _is_loopback_name(host: str) -> bool:
    """Whether the hostname itself pins to loopback. '*.localhost' subdomains do not count: RFC 6761 pinning is only a SHOULD, so a resolver may send them off-machine."""
    return host in _LOCAL_HTTP_HOSTS


def _parse_host_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse a URL host into an IP address, tolerating IPv6 brackets and zone ids."""
    candidate = host.strip("[]").split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        # Classify by the embedded IPv4 address: before CPython 3.11.10/3.12.4
        # (CVE-2024-4032) is_private is true for the whole ::ffff:0:0/96 range,
        # which would let plain HTTP through to mapped *public* addresses.
        return ip.ipv4_mapped
    return ip


def _is_local_http_host(host: str) -> bool:
    """Return whether plain HTTP is acceptable for this host (local network only)."""
    if not host:
        return False
    if _is_loopback_name(host):
        return True
    if host.endswith((".local", ".localhost")):  # mDNS names, and *.localhost dev hosts
        return True
    ip = _parse_host_ip(host)
    return ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local)


def is_loopback_mcp_url(url: str) -> bool:
    """Return whether the MCP URL points at a loopback host, where cleartext credentials never leave the machine."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if _is_loopback_name(host):
        return True
    ip = _parse_host_ip(host)
    return ip is not None and ip.is_loopback


def is_plaintext_remote_url(url: str) -> bool:
    """Return whether the URL sends traffic in cleartext beyond the local machine (plain HTTP to a non-loopback host). Single home of the classification both credential guards rely on."""
    return url.lower().startswith("http://") and not is_loopback_mcp_url(url)


def validate_http_mcp_url(url: str) -> str:
    """Validate that the MCP endpoint uses HTTP(S)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported MCP URL scheme '{parsed.scheme}'. Use http:// or https://.")
    if not parsed.netloc:
        raise ValueError(f"Invalid MCP URL '{url}'. Missing host.")

    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and not _is_local_http_host(host):
        raise ValueError(
            "Remote MCP servers must use HTTPS. Plain HTTP is only allowed for loopback, "
            "private, or link-local hosts (and *.local mDNS names)."
        )
    return url


def build_namespaced_tool_name(server_alias: str, tool_name: str) -> str:
    """Build a local tool name for a remote MCP tool."""
    alias = _require_alias_segment("server alias", server_alias)
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


def _is_timeout_mcp_error(exc: BaseException) -> bool:
    """Whether this is the SDK's request-timeout error (JSON-RPC -32001)."""
    try:
        from mcp import MCPError
        from mcp_types import REQUEST_TIMEOUT
    except ImportError:
        return False
    return isinstance(exc, MCPError) and getattr(exc, "code", None) == REQUEST_TIMEOUT


def _exception_contains_timeout(exc: BaseException) -> bool:
    timeout_exception = _httpx_timeout_exception_type()
    if isinstance(exc, timeout_exception):
        return True
    if _is_timeout_mcp_error(exc):
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


def _load_mcp_sdk() -> tuple[type["Client"], Any]:
    try:
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise McpDependencyError(
            "Remote MCP tools require the app's MCP client dependencies. Reinstall or update the app environment."
        ) from exc
    return Client, streamable_http_client


def _load_httpx2() -> Any:
    try:
        import httpx2
    except ImportError as exc:
        raise McpDependencyError(
            "Remote MCP tools require the app's HTTP client dependencies. Reinstall or update the app environment."
        ) from exc
    return httpx2


def _httpx_timeout_exception_type() -> tuple[type[BaseException], ...]:
    try:
        timeout_exception = _load_httpx2().TimeoutException
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
    # Explicit opt-in to sending credential headers over plain HTTP to a non-loopback host.
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        """Validate configuration once the dataclass has been created."""
        object.__setattr__(self, "alias", _require_alias_segment("server alias", self.alias))
        object.__setattr__(self, "url", validate_http_mcp_url(self.url))
        object.__setattr__(self, "headers", {str(k): str(v) for k, v in self.headers.items()})
        has_credentials = any(k.lower() == "authorization" for k in self.headers)
        if has_credentials and is_plaintext_remote_url(self.url) and not self.allow_insecure_http:
            raise ValueError(
                f"MCP server '{self.alias}' would send credentials over plain HTTP ({self.url}), "
                "exposing them to anyone on the network. Use HTTPS or a loopback address."
            )
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
        parameters_schema = getattr(tool, "input_schema", None)
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
            status="error" if bool(getattr(result, "is_error", False)) else "ok",
            content_blocks=content_blocks,
            text=_join_text_content(content_blocks),
            structured_content=getattr(result, "structured_content", None),
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

    def __init__(self, server: RemoteMcpServerConfig, known_tools: Sequence[RemoteToolSpec] = ()) -> None:
        """Store one allowlisted server configuration and an in-memory tool cache."""
        self.server = server
        self._tool_index = _index_remote_tools(list(known_tools))

    async def list_tool_specs(self) -> list[RemoteToolSpec]:
        """Discover remote tools and translate them into app-facing specs."""
        try:
            async with self._connected_client() as client:
                discovered = await self._list_all_tools(client)
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
            async with self._connected_client() as client:
                result = await client.call_tool(
                    spec.remote_name,
                    arguments=dict(arguments or {}),
                    read_timeout_seconds=self.server.tool_timeout_s,
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

    async def _list_all_tools(self, client: "Client") -> list["McpTool"]:
        tools: list[McpTool] = []
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = getattr(page, "next_cursor", None)
            if cursor is None:
                return tools

    @asynccontextmanager
    async def _connected_client(self) -> AsyncIterator["Client"]:
        client_cls, streamable_http_client = _load_mcp_sdk()
        httpx2 = _load_httpx2()
        client_timeout = max(self.server.request_timeout_s, self.server.tool_timeout_s)

        async with httpx2.AsyncClient(
            headers=self.server.headers,
            follow_redirects=False,
            timeout=client_timeout,
        ) as http_client:
            transport = streamable_http_client(self.server.url, http_client=http_client)
            # The default mode='auto' probes `server/discover` and falls back to the
            # `initialize` handshake, so one client speaks every protocol revision.
            async with client_cls(transport) as client:
                yield client


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
