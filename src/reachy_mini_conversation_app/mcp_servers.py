"""Manage generic remote MCP server tool sources for the conversation app.

Unlike ``tool_spaces`` (which is specific to Hugging Face Gradio Spaces), this
module lets the app talk to any HTTP(S) MCP server given a URL and an optional
auth token. The token value is never persisted: only the *name* of the
environment variable that holds it is stored in the manifest, and the value is
read from the environment at runtime.

Tools discovered at add time are cached in the manifest, so startup builds
clients from the cache without any network discovery (matching the installed
tool-spaces behavior). Unlike Space tools, generic server tools keep their raw
namespaced name ``{alias}__{tool}`` with no redundant-prefix cleaning:
arbitrary servers have no naming convention to strip.
"""

import os
import re
import asyncio
import logging
import argparse
import threading
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, replace, dataclass
from collections.abc import Sequence

# Importing config loads the .env file, so an auth token placed there is
# available when resolving servers from the standalone CLI.
from reachy_mini_conversation_app.config import USER_PERSONALITIES_DIRNAME, config
from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    validate_http_mcp_url,
    _require_alias_segment,
    is_plaintext_remote_url,
)
from reachy_mini_conversation_app.tool_spaces import read_installed_tool_spaces
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME, list_profile_names
from reachy_mini_conversation_app.profile_toolsets import (
    ProfileToolsets,
    enable_profile_tools,
    read_profile_toolsets,
    write_profile_toolsets,
    read_profile_tool_names,
    get_profile_toolsets_path,
    profile_toolsets_transaction,
    disable_profile_tools_by_prefix,
)
from reachy_mini_conversation_app.remote_tool_sources import (
    MCP_SERVERS_FILENAME,
    CachedRemoteTool,
    manifest_path,
    parse_cached_tools,
    read_manifest_envelope,
    write_manifest_payload,
    build_cached_tools_client,
)


logger = logging.getLogger(__name__)

MCP_SERVERS_VERSION = 1
BEARER_AUTH_TYPE = "bearer"
# POSIX-style environment variable name: leading letter/underscore, then
# letters/digits/underscores. Names outside this set can't round-trip through the
# instance `.env` (e.g. a name with a space writes a line python-dotenv won't parse).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_AUTH_TYPES = {BEARER_AUTH_TYPE}
_MANIFEST_LOCK = threading.RLock()


@dataclass(frozen=True)
class McpServerAuth:
    """Auth descriptor for an MCP server. Stores the env-var name, never the secret."""

    type: str
    token_env: str
    # Explicit opt-in to sending the token over plain HTTP to a non-loopback host.
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        """Validate the auth descriptor."""
        auth_type = self.type.strip().lower()
        if auth_type not in _SUPPORTED_AUTH_TYPES:
            raise ValueError(
                f"Unsupported MCP auth type '{self.type}'. Expected one of: {sorted(_SUPPORTED_AUTH_TYPES)}."
            )
        object.__setattr__(self, "type", auth_type)
        token_env = self.token_env.strip()
        if not token_env:
            raise ValueError("MCP auth 'token_env' (the environment variable name) cannot be empty.")
        if not _ENV_VAR_NAME_RE.match(token_env):
            raise ValueError(
                f"Invalid MCP auth 'token_env' name '{token_env}'. Use a valid environment variable "
                "name: a letter or underscore followed by letters, digits, or underscores "
                "(e.g. MCP_SERVER_TOKEN)."
            )
        object.__setattr__(self, "token_env", token_env)


@dataclass(frozen=True)
class InstalledMcpServer:
    """Persisted record for one configured MCP server and the tools discovered at add time."""

    alias: str
    url: str
    auth: McpServerAuth | None = None
    request_timeout_s: float = 10.0
    tool_timeout_s: float = 30.0
    tools: list[CachedRemoteTool] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate alias, URL and timeouts once the dataclass is created."""
        object.__setattr__(self, "alias", _require_alias_segment("server alias", self.alias))
        object.__setattr__(self, "url", validate_http_mcp_url(self.url))
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be greater than zero.")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be greater than zero.")


@dataclass(frozen=True)
class InstalledMcpServersManifest:
    """Persisted manifest of configured MCP servers."""

    version: int = MCP_SERVERS_VERSION
    servers: list[InstalledMcpServer] = field(default_factory=list)
    # Entries the reader dropped as invalid. A rewrite from this manifest would
    # permanently delete them, so add/remove refuse while it is non-zero.
    skipped_entries: int = 0


class McpServerAliasConflictError(RuntimeError):
    """Raised when a server alias is already claimed by another tool source."""


class McpServerNotConfiguredError(RuntimeError):
    """Raised when removing a server that is not configured."""


class McpServerProfileUpdateError(RuntimeError):
    """Raised when configured MCP server tools cannot be updated in profiles."""


class McpServerManifestError(RuntimeError):
    """Raised when the manifest holds entries a rewrite would destroy."""


@dataclass(frozen=True)
class McpServerInstallResult:
    """Result of configuring or refreshing one MCP server."""

    resolved_server: InstalledMcpServer
    manifest: InstalledMcpServersManifest
    manifest_path: Path
    refreshed: bool
    enabled_profile: str | None
    added_tool_ids: list[str]


@dataclass(frozen=True)
class McpServerRemovalResult:
    """Result of removing one configured MCP server."""

    removed_server: InstalledMcpServer
    manifest: InstalledMcpServersManifest
    disabled_profiles: list[tuple[str, list[str]]]


def get_mcp_servers_path(instance_path: str | Path | None) -> Path:
    """Return the MCP servers manifest path for the current mode."""
    return manifest_path(instance_path, MCP_SERVERS_FILENAME)


def _parse_auth(raw_auth: object, alias: str, path: Path) -> McpServerAuth | None:
    if raw_auth is None:
        return None
    if not isinstance(raw_auth, dict):
        raise RuntimeError(f"Invalid 'auth' for MCP server '{alias}' in {path}: expected an object.")
    allow_insecure_http = raw_auth.get("allow_insecure_http", False)
    if not isinstance(allow_insecure_http, bool):
        # Refuse truthy strings like "false": a mis-typed value must never grant the insecure opt-in.
        raise RuntimeError(
            f"Invalid 'auth' for MCP server '{alias}' in {path}: 'allow_insecure_http' must be true or false."
        )
    try:
        return McpServerAuth(
            type=str(raw_auth.get("type", "")),
            token_env=str(raw_auth.get("token_env", "")),
            allow_insecure_http=allow_insecure_http,
        )
    except ValueError as exc:
        raise RuntimeError(f"Invalid 'auth' for MCP server '{alias}' in {path}: {exc}") from exc


def _parse_server_entry(raw_server: object, path: Path) -> InstalledMcpServer:
    if not isinstance(raw_server, dict):
        raise RuntimeError(f"Invalid MCP servers entry in {path}: expected an object.")
    alias = str(raw_server.get("alias", ""))
    auth = _parse_auth(raw_server.get("auth"), alias, path)
    try:
        return InstalledMcpServer(
            alias=alias,
            url=str(raw_server.get("url", "")),
            auth=auth,
            request_timeout_s=float(raw_server.get("request_timeout_s", 10.0)),
            tool_timeout_s=float(raw_server.get("tool_timeout_s", 30.0)),
            tools=parse_cached_tools(raw_server.get("tools", [])),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid MCP server entry in {path}: {exc}") from exc


def read_mcp_servers(instance_path: str | Path | None) -> InstalledMcpServersManifest:
    """Read the configured MCP servers manifest if present, skipping invalid entries so one bad server cannot disable the rest."""
    with _MANIFEST_LOCK:
        path = get_mcp_servers_path(instance_path)
        envelope = read_manifest_envelope(path, "servers")
    if envelope is None:
        return InstalledMcpServersManifest()
    raw_servers, version = envelope

    servers: list[InstalledMcpServer] = []
    seen_aliases: set[str] = set()
    skipped = 0
    for raw_server in raw_servers:
        try:
            server = _parse_server_entry(raw_server, path)
            if server.alias in seen_aliases:
                raise RuntimeError(f"Duplicate MCP server alias '{server.alias}' found in {path}.")
        except RuntimeError as exc:
            logger.warning("Skipping invalid MCP server entry: %s", exc)
            skipped += 1
            continue
        seen_aliases.add(server.alias)
        servers.append(server)

    return InstalledMcpServersManifest(version=version, servers=servers, skipped_entries=skipped)


def write_mcp_servers(instance_path: str | Path | None, manifest: InstalledMcpServersManifest) -> Path:
    """Persist the MCP servers manifest. The token value is never stored, only token_env."""
    servers_payload: list[dict[str, Any]] = []
    for server in manifest.servers:
        entry: dict[str, Any] = {
            "alias": server.alias,
            "url": server.url,
            "request_timeout_s": server.request_timeout_s,
            "tool_timeout_s": server.tool_timeout_s,
            "tools": [asdict(tool) for tool in server.tools],
        }
        if server.auth is not None:
            entry["auth"] = {"type": server.auth.type, "token_env": server.auth.token_env}
            if server.auth.allow_insecure_http:
                entry["auth"]["allow_insecure_http"] = True
        servers_payload.append(entry)

    payload = {"version": manifest.version, "servers": servers_payload}
    with _MANIFEST_LOCK:
        return write_manifest_payload(get_mcp_servers_path(instance_path), payload)


def configured_aliases(instance_path: str | Path | None) -> set[str]:
    """Aliases currently claimed by configured MCP servers."""
    return {server.alias for server in read_mcp_servers(instance_path).servers}


def _resolve_auth_headers(server: InstalledMcpServer) -> dict[str, str]:
    """Build request headers for a server, reading the secret from the environment."""
    if server.auth is None:
        return {}
    if server.auth.type == BEARER_AUTH_TYPE:
        token = (os.environ.get(server.auth.token_env) or "").strip()
        if not token:
            raise RuntimeError(
                f"Env var '{server.auth.token_env}' for MCP server '{server.alias}' is not set or empty."
            )
        if is_plaintext_remote_url(server.url):
            if not server.auth.allow_insecure_http:
                raise RuntimeError(
                    f"MCP server '{server.alias}' would send its bearer token over plain HTTP ({server.url}), "
                    "exposing it to anyone on the network. Use HTTPS, a loopback address, or opt in "
                    "explicitly with 'mcp-servers add ... --allow-insecure-token'."
                )
            logger.warning(
                "MCP server '%s' sends its bearer token over plain HTTP (%s) per --allow-insecure-token; "
                "the token is visible to anyone on the local network.",
                server.alias,
                server.url,
            )
        return {"Authorization": f"Bearer {token}"}
    # Unreachable: McpServerAuth validates the type, but keep this defensive.
    raise RuntimeError(f"Unsupported MCP auth type '{server.auth.type}' for server '{server.alias}'.")


def build_server_config(server: InstalledMcpServer) -> RemoteMcpServerConfig:
    """Build a transport config for a server, resolving auth headers from the environment."""
    return RemoteMcpServerConfig(
        alias=server.alias,
        url=server.url,
        headers=_resolve_auth_headers(server),
        request_timeout_s=server.request_timeout_s,
        tool_timeout_s=server.tool_timeout_s,
        allow_insecure_http=server.auth.allow_insecure_http if server.auth is not None else False,
    )


def build_generic_remote_client(server: InstalledMcpServer) -> RemoteMcpToolClient:
    """Build an MCP client from cached tools, raising RuntimeError when auth cannot be resolved."""
    return build_cached_tools_client(build_server_config(server), server.tools)


@dataclass(frozen=True)
class McpTokenRequirement:
    """One configured MCP server's auth-token requirement, for the settings UI."""

    alias: str
    token_env: str
    token_set: bool


def list_token_requirements(instance_path: str | Path | None) -> list[McpTokenRequirement]:
    """Return the configured servers' auth-token requirements for the settings UI."""
    requirements: list[McpTokenRequirement] = []
    for server in read_mcp_servers(instance_path).servers:
        if server.auth is not None and server.auth.type == BEARER_AUTH_TYPE:
            token_set = bool((os.environ.get(server.auth.token_env) or "").strip())
            requirements.append(
                McpTokenRequirement(alias=server.alias, token_env=server.auth.token_env, token_set=token_set)
            )
    return requirements


def find_server_token_env(instance_path: str | Path | None, alias: str) -> str | None:
    """Return the token env-var name for a configured MCP server alias, or None."""
    return next((req.token_env for req in list_token_requirements(instance_path) if req.alias == alias), None)


def _build_generic_server_tools(remote_specs: Sequence[RemoteToolSpec]) -> list[CachedRemoteTool]:
    """Map discovered remote specs to app-facing tools without HF-specific name cleaning."""
    return [
        CachedRemoteTool(
            local_name=spec.namespaced_name,
            client_tool_name=spec.namespaced_name,
            remote_name=spec.remote_name,
            description=spec.description,
            parameters_schema=dict(spec.parameters_schema),
        )
        for spec in remote_specs
    ]


async def resolve_mcp_server(server: InstalledMcpServer) -> InstalledMcpServer:
    """Connect to a configured MCP server and return it with freshly discovered tools."""
    client = RemoteMcpToolClient(build_server_config(server))
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{server.alias}': {exc}") from exc

    return replace(server, tools=_build_generic_server_tools(remote_specs))


def resolve_mcp_server_sync(server: InstalledMcpServer) -> InstalledMcpServer:
    """Resolve one configured MCP server synchronously."""
    return asyncio.run(resolve_mcp_server(server))


def _restore_profile_toolsets(
    instance_path: str | Path | None,
    toolsets: ProfileToolsets,
    settings_existed: bool,
) -> None:
    settings_path = get_profile_toolsets_path(instance_path)
    if settings_existed:
        write_profile_toolsets(instance_path, toolsets)
    else:
        settings_path.unlink(missing_ok=True)


def _require_rewritable_manifest(manifest: InstalledMcpServersManifest, instance_path: str | Path | None) -> None:
    if manifest.skipped_entries:
        raise McpServerManifestError(
            f"The MCP servers manifest contains {manifest.skipped_entries} invalid entry(ies) that a rewrite "
            f"would permanently delete. Fix or remove them in {get_mcp_servers_path(instance_path)} first."
        )


def _space_aliases(instance_path: str | Path | None) -> set[str]:
    """Aliases claimed by installed Spaces. Raises so alias-collision checks fail closed."""
    return {space.alias for space in read_installed_tool_spaces(instance_path).spaces}


def install_mcp_server(
    alias: str,
    url: str,
    instance_path: str | Path | None,
    *,
    auth: McpServerAuth | None = None,
    request_timeout_s: float | None = None,
    tool_timeout_s: float | None = None,
    install_only: bool = False,
    profile: str | None = None,
) -> McpServerInstallResult:
    """Configure or refresh one MCP server and optionally enable its tools in a profile.

    Re-running an add is the documented cache-refresh flow, so options that are
    not repeated keep their stored values instead of resetting to defaults.
    """
    target_profile = profile or config.REACHY_MINI_CUSTOM_PROFILE or DEFAULT_PROFILE_NAME
    if not install_only:
        try:
            read_profile_tool_names(target_profile, instance_path)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            raise McpServerProfileUpdateError(
                f"Cannot enable MCP server tools in profile '{target_profile}': {exc}"
            ) from exc

    with _MANIFEST_LOCK:
        manifest = read_mcp_servers(instance_path)
        _require_rewritable_manifest(manifest, instance_path)
        requested_alias = _require_alias_segment("server alias", alias)
        existing = next((entry for entry in manifest.servers if entry.alias == requested_alias), None)

        server = InstalledMcpServer(
            alias=requested_alias,
            url=url,
            auth=auth if auth is not None else (existing.auth if existing is not None else None),
            request_timeout_s=(
                request_timeout_s
                if request_timeout_s is not None
                else existing.request_timeout_s
                if existing is not None
                else 10.0
            ),
            tool_timeout_s=(
                tool_timeout_s
                if tool_timeout_s is not None
                else existing.tool_timeout_s
                if existing is not None
                else 30.0
            ),
        )

        if existing is not None and existing.url != server.url:
            raise McpServerAliasConflictError(
                f"MCP server alias '{server.alias}' is already configured for {existing.url}. "
                "Remove it first to point it elsewhere."
            )
        if existing is None:
            # Fail closed: an unnoticed collision crashes tool registration at boot,
            # so refuse the add when the other manifest cannot be checked.
            try:
                space_aliases = _space_aliases(instance_path)
            except RuntimeError as exc:
                raise McpServerAliasConflictError(
                    f"Cannot add MCP server '{server.alias}': the installed tool-spaces manifest is unreadable, "
                    f"so the alias cannot be checked for collisions. Fix it first: {exc}"
                ) from exc
            if server.alias in space_aliases:
                raise McpServerAliasConflictError(
                    f"Cannot add MCP server '{server.alias}': its alias collides with an installed tool space. "
                    "Choose a different alias."
                )

    # Resolve outside the manifest lock: discovery is a network round-trip, and
    # failing before anything is persisted is the point (bad URL, unreachable
    # server, missing token). Discovered tools are cached so startup needs no
    # network; re-running add refreshes the cache.
    resolved = resolve_mcp_server_sync(server)

    with _MANIFEST_LOCK, profile_toolsets_transaction():
        manifest = read_mcp_servers(instance_path)
        _require_rewritable_manifest(manifest, instance_path)
        refreshed = any(entry.alias == resolved.alias for entry in manifest.servers)
        updated_manifest = InstalledMcpServersManifest(
            version=manifest.version,
            servers=sorted(
                [entry for entry in manifest.servers if entry.alias != resolved.alias] + [resolved],
                key=lambda entry: entry.alias,
            ),
        )

        enabled_profile: str | None = None
        added_tool_ids: list[str] = []
        profile_toolsets = ProfileToolsets()
        profile_settings_existed = False
        if not install_only:
            profile_toolsets = read_profile_toolsets(instance_path)
            profile_settings_existed = get_profile_toolsets_path(instance_path).exists()
            tool_ids = [tool.local_name for tool in resolved.tools]
            try:
                added_tool_ids = enable_profile_tools(target_profile, tool_ids, instance_path)
            except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
                raise McpServerProfileUpdateError(
                    f"Could not enable '{resolved.alias}' in profile '{target_profile}': {exc}"
                ) from exc
            enabled_profile = target_profile

        try:
            written_path = write_mcp_servers(instance_path, updated_manifest)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            if not install_only:
                try:
                    _restore_profile_toolsets(instance_path, profile_toolsets, profile_settings_existed)
                except (OSError, RuntimeError, UnicodeError, ValueError) as rollback_exc:
                    raise McpServerProfileUpdateError(
                        f"Could not persist '{resolved.alias}', and restoring the previous profile tool "
                        f"settings also failed: {rollback_exc}"
                    ) from rollback_exc
            raise RuntimeError(f"Could not persist MCP server '{resolved.alias}': {exc}") from exc

    return McpServerInstallResult(
        resolved_server=resolved,
        manifest=updated_manifest,
        manifest_path=written_path,
        refreshed=refreshed,
        enabled_profile=enabled_profile,
        added_tool_ids=added_tool_ids,
    )


def remove_mcp_server(alias: str, instance_path: str | Path | None) -> McpServerRemovalResult:
    """Remove one configured MCP server and disable its tools in all profiles."""
    validated_alias = _require_alias_segment("server alias", alias)
    with _MANIFEST_LOCK, profile_toolsets_transaction():
        manifest = read_mcp_servers(instance_path)
        _require_rewritable_manifest(manifest, instance_path)
        removed_server = next((entry for entry in manifest.servers if entry.alias == validated_alias), None)
        if removed_server is None:
            raise McpServerNotConfiguredError(f"MCP server not configured: {validated_alias}")

        updated_manifest = InstalledMcpServersManifest(
            version=manifest.version,
            servers=[entry for entry in manifest.servers if entry.alias != validated_alias],
        )
        profile_toolsets = read_profile_toolsets(instance_path)
        profile_settings_existed = get_profile_toolsets_path(instance_path).exists()
        profile_names = [DEFAULT_PROFILE_NAME, *list_profile_names(config.PROFILES_DIRECTORY)]
        profile_names.extend(
            f"{USER_PERSONALITIES_DIRNAME}/{name}" for name in list_profile_names(config.user_personalities_root())
        )
        profile_names.extend(profile_toolsets.profiles)
        try:
            disabled_profiles = disable_profile_tools_by_prefix(
                profile_names,
                f"{removed_server.alias}__",
                instance_path,
            )
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            raise McpServerProfileUpdateError(
                f"Could not remove '{validated_alias}' because its profile tool access could not be updated: {exc}"
            ) from exc
        try:
            write_mcp_servers(instance_path, updated_manifest)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            try:
                _restore_profile_toolsets(instance_path, profile_toolsets, profile_settings_existed)
            except (OSError, RuntimeError, UnicodeError, ValueError) as rollback_exc:
                raise McpServerProfileUpdateError(
                    f"Could not persist removal of '{validated_alias}', and restoring the previous profile tool "
                    f"settings also failed: {rollback_exc}"
                ) from rollback_exc
            raise RuntimeError(f"Could not persist removal of MCP server '{validated_alias}': {exc}") from exc

    return McpServerRemovalResult(
        removed_server=removed_server,
        manifest=updated_manifest,
        disabled_profiles=disabled_profiles,
    )


def format_mcp_server_listing(server: InstalledMcpServer) -> str:
    """Format one configured MCP server for terminal output (no secrets)."""
    lines = [
        f"{server.alias}",
        f"  MCP endpoint: {server.url}",
    ]
    if server.tools:
        lines.append("  Tools:")
        lines.extend([f"    - {tool.local_name}" for tool in server.tools])
    else:
        lines.append("  Tools: none discovered")
    return "\n".join(lines)


def _auth_from_args(args: argparse.Namespace, existing_auth: McpServerAuth | None) -> McpServerAuth | None:
    """Build the auth descriptor for an add, keeping stored values for flags that were not repeated."""
    token_env = (getattr(args, "token_env", None) or "").strip()
    stored_allow_insecure = existing_auth is not None and existing_auth.allow_insecure_http
    requested_allow_insecure = bool(getattr(args, "allow_insecure_token", False))
    # The insecure-HTTP opt-in sticks until 'remove', like every other stored flag.
    allow_insecure_token = requested_allow_insecure or stored_allow_insecure

    if not token_env:
        if allow_insecure_token and not stored_allow_insecure:
            logger.warning("--allow-insecure-token has no effect without --token-env.")
        if existing_auth is not None:
            logger.info("Keeping stored auth (token env '%s').", existing_auth.token_env)
        return existing_auth

    if stored_allow_insecure and not requested_allow_insecure:
        logger.info("Keeping the stored --allow-insecure-token opt-in.")
    return McpServerAuth(
        type=BEARER_AUTH_TYPE,
        token_env=token_env,
        allow_insecure_http=allow_insecure_token,
    )


def handle_mcp_servers_command(args: argparse.Namespace, *, instance_path: str | Path | None = None) -> int:
    """Handle mcp-servers subcommands from the main CLI."""
    command = getattr(args, "mcp_servers_command", None)
    if command == "add":
        try:
            existing = next(
                (entry for entry in read_mcp_servers(instance_path).servers if entry.alias == args.alias.strip()),
                None,
            )
            auth = _auth_from_args(args, existing.auth if existing is not None else None)
            install_result = install_mcp_server(
                args.alias,
                args.url,
                instance_path,
                auth=auth,
                request_timeout_s=getattr(args, "request_timeout", None),
                tool_timeout_s=getattr(args, "tool_timeout", None),
                install_only=bool(getattr(args, "install_only", False)),
                profile=getattr(args, "profile", None),
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1

        action = "Refreshed" if install_result.refreshed else "Configured"
        logger.info("%s MCP server: %s", action, install_result.resolved_server.alias)
        logger.info("Manifest: %s", install_result.manifest_path)
        logger.info("%s", format_mcp_server_listing(install_result.resolved_server))

        if install_result.enabled_profile is None:
            logger.info("Server configured. Select its tools under Tool access to enable them.")
            return 0
        if install_result.added_tool_ids:
            logger.info("Enabled in profile '%s': %s", install_result.enabled_profile, install_result.added_tool_ids)
        else:
            logger.info("All tool IDs already present in profile '%s'.", install_result.enabled_profile)
        return 0

    if command == "remove":
        try:
            removal_result = remove_mcp_server(args.alias, instance_path)
        except McpServerNotConfiguredError as exc:
            logger.warning("%s", exc)
            return 1
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1

        logger.info("Removed MCP server: %s", removal_result.removed_server.alias)
        for profile_name, disabled_tool_ids in removal_result.disabled_profiles:
            logger.info("Disabled in profile '%s': %s", profile_name, disabled_tool_ids)
        return 0

    if command == "list":
        manifest = read_mcp_servers(instance_path)
        logger.info("Manifest: %s", get_mcp_servers_path(instance_path))
        if not manifest.servers:
            logger.info("No configured MCP servers.")
            return 0
        for server in manifest.servers:
            logger.info("%s", format_mcp_server_listing(server))
        return 0

    raise RuntimeError(f"Unknown mcp-servers command: {command}")
