"""Source-agnostic building blocks shared by the remote tool-source manifests.

Both installed Hugging Face Spaces (``tool_spaces``) and generic MCP servers
(``mcp_servers``) persist the tools discovered at install time, rebuild clients
from that cache at startup, and wire their tool IDs into profile toolsets. This
module holds the pieces the generic MCP-server code needs so it does not depend
on Space-specific machinery.
"""

import os
import json
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
)


logger = logging.getLogger(__name__)

# Where terminal mode (no managed app instance) keeps manifests and downloads.
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")

MCP_SERVERS_FILENAME = "mcp_servers.json"


def manifest_path(instance_path: str | Path | None, filename: str) -> Path:
    """Return a source manifest's path: the app instance dir, or external_content/ in terminal mode."""
    if instance_path is not None:
        return Path(instance_path) / filename
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / filename


def read_manifest_envelope(path: Path, entries_key: str) -> tuple[list[object], int] | None:
    """Read and validate a manifest's shared JSON envelope, returning (raw entries, version) or None when absent."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid payload in {path}: expected a JSON object.")
    entries = payload.get(entries_key, [])
    if not isinstance(entries, list):
        raise RuntimeError(f"Invalid payload in {path}: '{entries_key}' must be a list.")
    version = payload.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise RuntimeError(f"Invalid payload in {path}: 'version' must be an int.")
    return entries, version


def write_manifest_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a manifest payload, replacing the file atomically so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
        temporary_path.replace(path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove temporary manifest %s: %s", temporary_path, exc)
    return path


def configured_server_aliases(instance_path: str | Path | None) -> set[str]:
    """Aliases claimed in the MCP servers manifest; raises RuntimeError so alias-collision checks fail closed."""
    # Raw envelope read rather than mcp_servers.read_mcp_servers: this shared module
    # must not import mcp_servers back (import cycle), and counting even invalid
    # entries' aliases is the conservative choice for a collision check.
    envelope = read_manifest_envelope(manifest_path(instance_path, MCP_SERVERS_FILENAME), "servers")
    if envelope is None:
        return set()
    entries, _version = envelope
    return {str(entry["alias"]) for entry in entries if isinstance(entry, dict) and entry.get("alias")}


@dataclass(frozen=True)
class CachedRemoteTool:
    """App-facing metadata for one remote tool cached in a source's manifest."""

    local_name: str
    client_tool_name: str
    remote_name: str
    description: str
    parameters_schema: dict[str, Any]


def parse_cached_tools(raw_tools: object) -> list[CachedRemoteTool]:
    """Parse a manifest entry's cached-tools list, skipping malformed items."""
    if not isinstance(raw_tools, list):
        return []
    return [
        CachedRemoteTool(
            local_name=str(tool["local_name"]),
            client_tool_name=str(tool["client_tool_name"]),
            remote_name=str(tool.get("remote_name", "")),
            description=str(tool.get("description", "")),
            parameters_schema=dict(tool.get("parameters_schema") or {}),
        )
        for tool in raw_tools
        if isinstance(tool, dict)
        and tool.get("local_name")
        and tool.get("client_tool_name")
        and isinstance(tool.get("parameters_schema") or {}, dict)
    ]


def build_cached_tools_client(
    server_config: RemoteMcpServerConfig,
    cached_tools: Sequence[CachedRemoteTool],
) -> RemoteMcpToolClient:
    """Build an MCP client from a transport config and manifest-cached tool records."""
    return RemoteMcpToolClient(
        server_config,
        known_tools=[
            RemoteToolSpec(
                server_alias=server_config.alias,
                remote_name=tool.remote_name,
                namespaced_name=tool.client_tool_name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
            for tool in cached_tools
            if tool.remote_name
        ],
    )
