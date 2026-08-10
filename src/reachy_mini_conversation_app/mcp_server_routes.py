"""JSON-RPC methods for managing custom remote MCP server tools."""

import asyncio
import logging
from typing import Any
from pathlib import Path
from collections.abc import Callable

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import LOCKED_PROFILE, config
from reachy_mini_conversation_app.mcp_servers import (
    BEARER_AUTH_TYPE,
    McpServerAuth,
    McpServerManifestError,
    InstalledMcpServersManifest,
    McpServerAliasConflictError,
    McpServerNotConfiguredError,
    McpServerProfileUpdateError,
    read_mcp_servers,
    remove_mcp_server,
    install_mcp_server,
    find_server_token_env,
    list_token_requirements,
)
from reachy_mini_conversation_app.profile_store import canonical_profile_name
from reachy_mini_conversation_app.tool_settings import (
    RestartCallback,
    apply_tool_change,
    raise_tool_settings_error,
)
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names


logger = logging.getLogger(__name__)

# Persists env values and reports whether they reached the instance `.env`:
# "persisted", "session" (nothing to write to), or "failed".
PersistEnvCallback = Callable[[dict[str, str]], str]


def _server_settings_payload(
    manifest: InstalledMcpServersManifest,
    token_set_by_alias: dict[str, bool],
) -> dict[str, object]:
    return {
        "servers": [
            {
                "alias": server.alias,
                "url": server.url,
                "tool_count": len(server.tools),
                "token_env": server.auth.token_env if server.auth is not None else None,
                "token_set": token_set_by_alias.get(server.alias, False),
            }
            for server in sorted(manifest.servers, key=lambda entry: entry.alias)
        ],
        "editable": LOCKED_PROFILE is None,
    }


def _settings_payload(instance_path: str | Path | None) -> dict[str, object]:
    """Build the settings payload, folding in which servers currently have their token set."""
    manifest = read_mcp_servers(instance_path)
    token_set_by_alias = {req.alias: req.token_set for req in list_token_requirements(instance_path)}
    return _server_settings_payload(manifest, token_set_by_alias)


def _error_detail(error: BaseException) -> str:
    return str(error).strip() or type(error).__name__


def _required_string(params: dict[str, Any], key: str, reason: str, message: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise_tool_settings_error(reason, message)
    return value.strip()


def _optional_float(params: dict[str, Any], key: str) -> float | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise_tool_settings_error("invalid_mcp_timeout", "Timeouts must be a number of seconds greater than zero.")
    return float(value)


def _auth_from_params(params: dict[str, Any]) -> McpServerAuth | None:
    """Build the auth descriptor from request params, or None when the server needs no token."""
    token_env = params.get("token_env")
    if token_env is None or (isinstance(token_env, str) and not token_env.strip()):
        return None
    if not isinstance(token_env, str):
        raise_tool_settings_error("invalid_mcp_token_env", "Enter the name of the environment variable to read.")
    allow_insecure_http = params.get("allow_insecure_http", False)
    if not isinstance(allow_insecure_http, bool):
        raise_tool_settings_error("invalid_mcp_auth", "The insecure-HTTP opt-in must be true or false.")
    try:
        return McpServerAuth(
            type=BEARER_AUTH_TYPE,
            token_env=token_env.strip(),
            allow_insecure_http=allow_insecure_http,
        )
    except ValueError as exc:
        raise_tool_settings_error("invalid_mcp_token_env", _error_detail(exc))


def register_mcp_server_methods(
    rpc: JsonRpcServer,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    restart_conversation: RestartCallback,
    persist_env_values: PersistEnvCallback,
    *,
    instance_path: str | Path | None,
) -> None:
    """Register custom MCP server management methods."""

    async def _list_mcp_servers(_params: dict[str, Any]) -> dict[str, object]:
        try:
            return await asyncio.to_thread(_settings_payload, instance_path)
        except Exception as exc:
            logger.exception("Failed to list configured MCP servers")
            raise_tool_settings_error("mcp_servers_unavailable", _error_detail(exc))

    async def _add_mcp_server(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "MCP server editing is locked.")
        alias = _required_string(params, "alias", "invalid_mcp_alias", "Enter a short alias, e.g. my_server.")
        url = _required_string(params, "url", "invalid_mcp_url", "Enter the server's MCP endpoint URL.")
        auth = _auth_from_params(params)

        try:
            result = await asyncio.to_thread(
                install_mcp_server,
                alias,
                url,
                instance_path,
                auth=auth,
                request_timeout_s=_optional_float(params, "request_timeout_s"),
                tool_timeout_s=_optional_float(params, "tool_timeout_s"),
                install_only=True,
            )
        except McpServerAliasConflictError as exc:
            logger.warning("MCP server alias conflict: %s", exc)
            raise_tool_settings_error("mcp_server_alias_conflict", _error_detail(exc))
        except McpServerManifestError as exc:
            logger.error("Refusing to rewrite a damaged MCP servers manifest: %s", exc)
            raise_tool_settings_error("mcp_servers_manifest_damaged", _error_detail(exc))
        except ValueError as exc:
            logger.warning("Invalid MCP server configuration %r: %s", alias, exc)
            raise_tool_settings_error("invalid_mcp_server", _error_detail(exc))
        except RuntimeError as exc:
            logger.error("Failed to configure MCP server %r: %s", alias, exc)
            raise_tool_settings_error("mcp_server_add_failed", _error_detail(exc))
        except Exception as exc:
            logger.exception("Unexpected failure configuring MCP server %r", alias)
            raise_tool_settings_error("mcp_server_add_failed", _error_detail(exc))

        active_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        try:
            active_tools = await asyncio.to_thread(read_profile_tool_names, active_profile, instance_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Configured MCP server but could not inspect active profile %r: %s", active_profile, exc)
            active_tools = []
        apply_detail = (
            await asyncio.to_thread(
                apply_tool_change,
                instance_path,
                get_loop,
                restart_conversation,
                "mcp_servers_changed",
            )
            if any(tool_name.startswith(f"{result.resolved_server.alias}__") for tool_name in active_tools)
            else "The server is ready to assign to personalities."
        )

        action = "Refreshed" if result.refreshed else "Added"
        tool_count = len(result.resolved_server.tools)
        tool_label = "tool" if tool_count == 1 else "tools"
        message = (
            f"{action} {result.resolved_server.alias} with {tool_count} {tool_label}. "
            "Choose which personalities can use them in Tool access. "
            f"{apply_detail}"
        )
        return {
            **await asyncio.to_thread(_settings_payload, instance_path),
            "message": message,
        }

    async def _remove_mcp_server(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "MCP server editing is locked.")
        alias = _required_string(params, "alias", "invalid_mcp_alias", "Enter the alias of a configured server.")
        try:
            result = await asyncio.to_thread(remove_mcp_server, alias, instance_path)
            disabled_profiles = result.disabled_profiles
        except McpServerNotConfiguredError as exc:
            logger.warning("Cannot remove MCP server %r: %s", alias, exc)
            raise_tool_settings_error("mcp_server_not_configured", _error_detail(exc))
        except McpServerProfileUpdateError as exc:
            logger.error("Failed to disable removed MCP server tools: %s", exc)
            raise_tool_settings_error("profile_disable_failed", _error_detail(exc))
        except McpServerManifestError as exc:
            logger.error("Refusing to rewrite a damaged MCP servers manifest: %s", exc)
            raise_tool_settings_error("mcp_servers_manifest_damaged", _error_detail(exc))
        except ValueError as exc:
            logger.warning("Invalid MCP server alias %r: %s", alias, exc)
            raise_tool_settings_error("invalid_mcp_alias", _error_detail(exc))
        except RuntimeError as exc:
            logger.error("Failed to remove MCP server %r: %s", alias, exc)
            raise_tool_settings_error("mcp_server_remove_failed", _error_detail(exc))
        except Exception as exc:
            logger.exception("Unexpected failure removing MCP server %r", alias)
            raise_tool_settings_error("mcp_server_remove_failed", _error_detail(exc))

        active_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        apply_detail = (
            await asyncio.to_thread(
                apply_tool_change,
                instance_path,
                get_loop,
                restart_conversation,
                "mcp_servers_changed",
            )
            if any(profile_name == active_profile for profile_name, _ in disabled_profiles)
            else "No active conversation restart is needed."
        )

        disabled_tool_count = sum(len(tool_ids) for _, tool_ids in disabled_profiles)
        tool_label = "tool" if disabled_tool_count == 1 else "tools"
        message = (
            f"Removed {result.removed_server.alias}. "
            f"Disabled {disabled_tool_count} {tool_label} across personalities. {apply_detail}"
        )
        return {
            **await asyncio.to_thread(_settings_payload, instance_path),
            "message": message,
        }

    async def _save_mcp_server_token(params: dict[str, Any]) -> dict[str, object]:
        if LOCKED_PROFILE is not None:
            raise_tool_settings_error("profile_locked", "MCP server editing is locked.")
        alias = _required_string(params, "alias", "invalid_mcp_alias", "Enter the alias of a configured server.")
        token = params.get("token")
        if not isinstance(token, str) or not token.strip():
            raise_tool_settings_error("empty_token", "Enter a token first.")

        try:
            token_env = await asyncio.to_thread(find_server_token_env, instance_path, alias)
        except RuntimeError as exc:
            logger.warning("Could not read the MCP servers manifest: %s", exc)
            raise_tool_settings_error("mcp_servers_unavailable", _error_detail(exc))
        if token_env is None:
            raise_tool_settings_error("unknown_mcp_server", "That MCP server is no longer configured.")

        try:
            persist_state = await asyncio.to_thread(persist_env_values, {token_env: token.strip()})
        except ValueError as exc:
            # The value cannot round-trip through the instance `.env`.
            raise_tool_settings_error("invalid_token", _error_detail(exc))
        except Exception as exc:
            logger.exception("Failed to store the token for MCP server %r", alias)
            raise_tool_settings_error("token_save_failed", _error_detail(exc))

        # The client reads the token when it is built, so tools that were skipped
        # for a missing token only appear after the registry is rebuilt.
        apply_detail = await asyncio.to_thread(
            apply_tool_change,
            instance_path,
            get_loop,
            restart_conversation,
            "mcp_server_token_changed",
        )
        if persist_state == "failed":
            message = (
                f"Saved the token for {alias} for this session only: writing the instance .env failed, "
                f"so it will not survive a restart. {apply_detail}"
            )
        elif persist_state == "session":
            message = f"Saved the token for {alias} for this session. {apply_detail}"
        else:
            message = f"Saved the token for {alias}. {apply_detail}"
        return {
            **await asyncio.to_thread(_settings_payload, instance_path),
            "message": message,
        }

    rpc.register("mcp_servers.list", _list_mcp_servers)
    rpc.register("mcp_servers.add", _add_mcp_server)
    rpc.register("mcp_servers.remove", _remove_mcp_server)
    rpc.register("mcp_servers.save_token", _save_mcp_server_token)
