"""Manage installed Hugging Face Space tool sources for the conversation app."""

import os
import re
import json
import asyncio
import logging
import argparse
import threading
from typing import Any
from pathlib import Path
from collections import Counter
from dataclasses import field, asdict, dataclass
from urllib.parse import urlsplit
from collections.abc import Sequence

from huggingface_hub import HfApi, SpaceInfo, get_token
from huggingface_hub.errors import RepositoryNotFoundError

from reachy_mini_conversation_app.config import USER_PERSONALITIES_DIRNAME, config
from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    apply_name_normalization,
    build_namespaced_tool_name,
)
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


logger = logging.getLogger(__name__)

INSTALLED_TOOL_SPACES_FILENAME = "installed_tool_spaces.json"
INSTALLED_TOOL_SPACES_VERSION = 2
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
_MANIFEST_LOCK = threading.RLock()
# Bundled Pollen Spaces seeded when no manifest exists, so startup needs no Hugging Face discovery.
PREINSTALLED_TOOL_SPACE_SPECS = {
    "pollen-robotics/reachy-mini-search-tool": (
        RemoteToolSpec(
            server_alias="pollen_robotics_reachy_mini_search_tool",
            remote_name="reachy_mini_search_tool_search_web",
            namespaced_name=build_namespaced_tool_name(
                "pollen_robotics_reachy_mini_search_tool", "reachy_mini_search_tool_search_web"
            ),
            description=(
                "Search the web for current information and return a short list of results (title, snippet, url). "
                "Call this directly whenever the user asks to search, check the web, look something up, "
                "find today's events, or learn what is happening now. Do not just say you'll look it up."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["query"],
            },
        ),
    ),
    "pollen-robotics/reachy-mini-time-tool": (
        RemoteToolSpec(
            server_alias="pollen_robotics_reachy_mini_time_tool",
            remote_name="reachy_mini_time_tool_get_time",
            namespaced_name=build_namespaced_tool_name(
                "pollen_robotics_reachy_mini_time_tool", "reachy_mini_time_tool_get_time"
            ),
            description=(
                "Get the current date and time for an IANA timezone, and optionally the difference to a second timezone. "
                "Call this directly whenever the user asks what time it is or the time somewhere. Pass an IANA name like "
                "'Europe/Paris' or 'Asia/Tokyo' for a named place (derive it from the place), or leave the timezone empty "
                "for the user's own local time. Do not ask for their city and do not just say you'll check."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "default": "",
                        "description": "IANA timezone like 'Europe/Paris'. Empty resolves the user's local time.",
                    },
                    "compare_timezone": {
                        "type": "string",
                        "default": "",
                        "description": "Optional second IANA timezone to compare against.",
                    },
                },
                "required": [],
            },
        ),
    ),
    "pollen-robotics/reachy-mini-weather-tool": (
        RemoteToolSpec(
            server_alias="pollen_robotics_reachy_mini_weather_tool",
            remote_name="reachy_mini_weather_tool_get_weather",
            namespaced_name=build_namespaced_tool_name(
                "pollen_robotics_reachy_mini_weather_tool", "reachy_mini_weather_tool_get_weather"
            ),
            description=(
                "Get today's weather for a place: current conditions, high and low temperature, and rain chance. "
                "Call this directly whenever the user asks about the weather, forecast, or temperature for "
                "somewhere. Do not just say you'll check."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Place name for the weather lookup."},
                },
                "required": ["location"],
            },
        ),
    ),
}
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class InstalledToolSpaceTool:
    """App-facing metadata for one remote tool exposed by an installed Space."""

    local_name: str
    client_tool_name: str
    remote_name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass(frozen=True)
class InstalledToolSpace:
    """Metadata for one Space and its discovered tools."""

    slug: str
    alias: str
    mcp_url: str
    private: bool
    tools: list[InstalledToolSpaceTool] = field(default_factory=list)


@dataclass(frozen=True)
class InstalledToolSpacesManifest:
    """Persisted manifest of installed Space tool sources."""

    version: int = INSTALLED_TOOL_SPACES_VERSION
    spaces: list[InstalledToolSpace] = field(default_factory=list)


class ToolSpaceAliasConflictError(RuntimeError):
    """Raised when two Space slugs resolve to the same local alias."""


class ToolSpaceNotInstalledError(RuntimeError):
    """Raised when removing a Space that is not installed."""


class ToolSpaceProfileUpdateError(RuntimeError):
    """Raised when installed Space tools cannot be updated in profiles."""


@dataclass(frozen=True)
class ToolSpaceInstallResult:
    """Result of installing or refreshing one Space tool source."""

    resolved_space: InstalledToolSpace
    manifest: InstalledToolSpacesManifest
    manifest_path: Path
    refreshed: bool
    enabled_profile: str | None
    added_tool_ids: list[str]


@dataclass(frozen=True)
class ToolSpaceRemovalResult:
    """Result of removing one installed Space tool source."""

    removed_space: InstalledToolSpace
    manifest: InstalledToolSpacesManifest
    disabled_profiles: list[tuple[str, list[str]]]


def get_installed_tool_spaces_path(instance_path: str | Path | None) -> Path:
    """Return the installed tool-spaces manifest path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / INSTALLED_TOOL_SPACES_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / INSTALLED_TOOL_SPACES_FILENAME


def _preinstalled_installed_spaces() -> list[InstalledToolSpace]:
    """Build the bundled Pollen Spaces as manifest entries with their tools cached from static specs."""
    spaces: list[InstalledToolSpace] = []
    for slug, remote_specs in PREINSTALLED_TOOL_SPACE_SPECS.items():
        alias = normalize_space_alias(slug)
        spaces.append(
            InstalledToolSpace(
                slug=slug,
                alias=alias,
                mcp_url=f"https://{slug.replace('/', '-')}.hf.space/gradio_api/mcp/",
                private=False,
                tools=_build_installed_tool_space_tools(slug=slug, alias=alias, remote_specs=list(remote_specs)),
            )
        )
    return spaces


def read_installed_tool_spaces(instance_path: str | Path | None) -> InstalledToolSpacesManifest:
    """Read the installed tool-spaces manifest, or seed the bundled Pollen Spaces when none exists."""
    with _MANIFEST_LOCK:
        manifest_path = get_installed_tool_spaces_path(instance_path)
        if not manifest_path.exists():
            return InstalledToolSpacesManifest(spaces=_preinstalled_installed_spaces())

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read installed tool spaces from {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: expected a JSON object.")

    expected_manifest_fields = {"version", "spaces"}
    if set(payload) != expected_manifest_fields:
        invalid_fields = sorted(set(payload) ^ expected_manifest_fields)
        raise RuntimeError(
            f"Invalid installed tool spaces payload in {manifest_path}: missing or unknown fields: "
            f"{', '.join(invalid_fields)}."
        )

    version = payload["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != INSTALLED_TOOL_SPACES_VERSION:
        raise RuntimeError(
            f"Unsupported installed tool spaces version in {manifest_path}: expected {INSTALLED_TOOL_SPACES_VERSION}."
        )

    raw_spaces = payload["spaces"]
    if not isinstance(raw_spaces, list):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: 'spaces' must be a list.")

    spaces: list[InstalledToolSpace] = []
    seen_slugs: set[str] = set()
    seen_aliases: set[str] = set()
    for raw_space in raw_spaces:
        if not isinstance(raw_space, dict):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: expected an object.")

        expected_space_fields = {"slug", "alias", "mcp_url", "private", "tools"}
        if set(raw_space) != expected_space_fields:
            invalid_fields = sorted(set(raw_space) ^ expected_space_fields)
            raise RuntimeError(
                f"Invalid installed tool spaces entry in {manifest_path}: missing or unknown fields: "
                f"{', '.join(invalid_fields)}."
            )
        if not isinstance(raw_space["slug"], str):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: 'slug' must be a string.")
        if not isinstance(raw_space["alias"], str):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: 'alias' must be a string.")
        if not isinstance(raw_space["mcp_url"], str):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: 'mcp_url' must be a string.")
        if not isinstance(raw_space["private"], bool):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: 'private' must be a boolean.")
        if not isinstance(raw_space["tools"], list):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: 'tools' must be a list.")

        slug = validate_space_slug(raw_space["slug"])
        alias = normalize_space_alias(slug)
        if raw_space["alias"] != alias:
            raise RuntimeError(f"Invalid installed tool space '{slug}' in {manifest_path}: alias must be '{alias}'.")
        if slug in seen_slugs:
            raise RuntimeError(f"Duplicate installed tool space '{slug}' found in {manifest_path}.")
        if alias in seen_aliases:
            raise RuntimeError(
                f"Installed tool spaces manifest contains alias collision '{alias}' in {manifest_path}. "
                "Remove one of the conflicting spaces with 'tool-spaces remove'."
            )
        mcp_url = validate_space_mcp_url(raw_space["mcp_url"])
        cached_tools: list[InstalledToolSpaceTool] = []
        seen_local_names: set[str] = set()
        seen_client_tool_names: set[str] = set()
        for raw_tool in raw_space["tools"]:
            if not isinstance(raw_tool, dict):
                raise RuntimeError(
                    f"Invalid cached tool for installed Space '{slug}' in {manifest_path}: expected an object."
                )
            expected_tool_fields = {
                "local_name",
                "client_tool_name",
                "remote_name",
                "description",
                "parameters_schema",
            }
            if set(raw_tool) != expected_tool_fields:
                invalid_fields = sorted(set(raw_tool) ^ expected_tool_fields)
                raise RuntimeError(
                    f"Invalid cached tool for installed Space '{slug}' in {manifest_path}: missing or unknown "
                    f"fields: {', '.join(invalid_fields)}."
                )
            if not all(
                isinstance(raw_tool[field_name], str)
                for field_name in ("local_name", "client_tool_name", "remote_name", "description")
            ):
                raise RuntimeError(
                    f"Invalid cached tool for installed Space '{slug}' in {manifest_path}: tool names and "
                    "descriptions must be strings."
                )
            if not isinstance(raw_tool["parameters_schema"], dict):
                raise RuntimeError(
                    f"Invalid cached tool for installed Space '{slug}' in {manifest_path}: "
                    "'parameters_schema' must be an object."
                )

            local_name = raw_tool["local_name"].strip()
            client_tool_name = raw_tool["client_tool_name"].strip()
            remote_name = raw_tool["remote_name"].strip()
            if not local_name.startswith(f"{alias}__") or not client_tool_name.startswith(f"{alias}__"):
                raise RuntimeError(
                    f"Invalid cached tool for installed Space '{slug}' in {manifest_path}: tool names must use "
                    f"the '{alias}__' prefix."
                )
            if not remote_name:
                raise RuntimeError(
                    f"Invalid cached tool for installed Space '{slug}' in {manifest_path}: remote_name is empty."
                )
            if local_name in seen_local_names or client_tool_name in seen_client_tool_names:
                raise RuntimeError(f"Duplicate cached tool found for installed Space '{slug}' in {manifest_path}.")
            seen_local_names.add(local_name)
            seen_client_tool_names.add(client_tool_name)
            cached_tools.append(
                InstalledToolSpaceTool(
                    local_name=local_name,
                    client_tool_name=client_tool_name,
                    remote_name=remote_name,
                    description=raw_tool["description"],
                    parameters_schema=dict(raw_tool["parameters_schema"]),
                )
            )
        seen_slugs.add(slug)
        seen_aliases.add(alias)
        spaces.append(
            InstalledToolSpace(
                slug=slug,
                alias=alias,
                mcp_url=mcp_url,
                private=raw_space["private"],
                tools=cached_tools,
            )
        )
    return InstalledToolSpacesManifest(version=version, spaces=spaces)


def write_installed_tool_spaces(
    instance_path: str | Path | None,
    manifest: InstalledToolSpacesManifest,
) -> Path:
    """Persist the installed tool-spaces manifest."""
    with _MANIFEST_LOCK:
        manifest_path = get_installed_tool_spaces_path(instance_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": manifest.version,
            "spaces": [asdict(space) for space in manifest.spaces],
        }
        temporary_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
            temporary_path.replace(manifest_path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove temporary Tool Space manifest %s: %s", temporary_path, exc)
        return manifest_path


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


def validate_space_slug(slug: str) -> str:
    """Validate a public HF Space slug."""
    candidate = slug.strip()
    if _SLUG_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            f"Invalid Space slug '{slug}'. Expected the form 'owner/space-name' with alnum, '.', '_' or '-'."
        )
    return candidate


def validate_space_mcp_url(mcp_url: str) -> str:
    """Validate a standard HTTPS MCP endpoint hosted by Hugging Face Spaces."""
    candidate = mcp_url.strip()
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid Hugging Face Space MCP URL '{mcp_url}'.") from exc
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".hf.space")
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.path != "/gradio_api/mcp/"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Invalid Hugging Face Space MCP URL '{mcp_url}'.")
    return candidate


def normalize_space_alias(slug: str) -> str:
    """Derive a local alias from a Space slug."""
    normalized = apply_name_normalization(slug)
    if not normalized:
        raise ValueError(f"Space slug '{slug}' cannot be normalized into a local alias.")
    if normalized[0].isdigit():
        normalized = f"space_{normalized}"
    return normalized


def _normalize_segment(value: str) -> str:
    normalized = apply_name_normalization(value)
    if not normalized:
        return "tool"
    if normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return normalized


def _clean_space_tool_name(slug: str, alias: str, remote_name: str) -> str:
    normalized_remote_name = _normalize_segment(remote_name)
    space_name = slug.split("/", maxsplit=1)[1]
    normalized_space_name = _normalize_segment(space_name)
    redundant_prefix = f"{normalized_space_name}_"

    if normalized_remote_name.startswith(redundant_prefix):
        cleaned_name = normalized_remote_name[len(redundant_prefix) :]
        if cleaned_name:
            return f"{alias}__{cleaned_name}"
    return f"{alias}__{normalized_remote_name}"


def _build_installed_tool_space_tools(
    *,
    slug: str,
    alias: str,
    remote_specs: Sequence[RemoteToolSpec],
) -> list[InstalledToolSpaceTool]:
    cleaned_names = [_clean_space_tool_name(slug, alias, spec.remote_name) for spec in remote_specs]
    collisions = {name for name, count in Counter(cleaned_names).items() if count > 1}

    tools: list[InstalledToolSpaceTool] = []
    for remote_spec, cleaned_name in zip(remote_specs, cleaned_names, strict=True):
        local_name = remote_spec.namespaced_name if cleaned_name in collisions else cleaned_name
        tools.append(
            InstalledToolSpaceTool(
                local_name=local_name,
                client_tool_name=remote_spec.namespaced_name,
                remote_name=remote_spec.remote_name,
                description=remote_spec.description,
                parameters_schema=dict(remote_spec.parameters_schema),
            )
        )
    return tools


def _build_space_mcp_url(space_info: SpaceInfo, slug: str) -> str:
    host = (space_info.host or "").strip()
    if host:
        if host.startswith("http://") or host.startswith("https://"):
            return f"{host.rstrip('/')}/gradio_api/mcp/"
        return f"https://{host.rstrip('/')}/gradio_api/mcp/"

    subdomain = (space_info.subdomain or "").strip()
    if subdomain:
        return f"https://{subdomain}.hf.space/gradio_api/mcp/"

    slug_host = slug.replace("/", "-")
    return f"https://{slug_host}.hf.space/gradio_api/mcp/"


def _validate_space_info(slug: str, space_info: SpaceInfo) -> None:
    if bool(space_info.disabled):
        raise RuntimeError(f"Space '{slug}' is disabled and cannot be installed.")
    if (space_info.sdk or "").strip().lower() != "gradio":
        raise RuntimeError(f"Space '{slug}' is not a Gradio Space and cannot expose the standard MCP endpoint.")


def build_remote_client(
    alias: str,
    mcp_url: str,
    *,
    private: bool,
    cached_tools: Sequence[InstalledToolSpaceTool] = (),
) -> RemoteMcpToolClient:
    """Build an MCP client for an installed Space, sending the HF token only to private Spaces."""
    validated_mcp_url = validate_space_mcp_url(mcp_url)
    token = (config.HF_TOKEN or get_token()) if private else None
    headers = {"Authorization": f"Bearer {token}"} if private and token else {}
    return RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=alias,
            url=validated_mcp_url,
            headers=headers,
            request_timeout_s=10.0,
            tool_timeout_s=30.0,
        ),
        known_tools=[
            RemoteToolSpec(
                server_alias=alias,
                remote_name=tool.remote_name,
                namespaced_name=tool.client_tool_name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
            for tool in cached_tools
            if tool.remote_name
        ],
    )


async def resolve_tool_space(slug: str) -> InstalledToolSpace:
    """Validate and discover tools from one HF Space, authenticating private Spaces with the HF token."""
    validated_slug = validate_space_slug(slug)
    alias = normalize_space_alias(validated_slug)
    token = config.HF_TOKEN or get_token()
    try:
        space_info = HfApi().space_info(validated_slug, timeout=10.0, token=token or False)
    except RepositoryNotFoundError as exc:
        if token is None:
            raise RuntimeError(
                f"Space '{validated_slug}' was not found. If it is private, set HF_TOKEN "
                "or run 'hf auth login' for an account that can access it."
            ) from exc
        raise RuntimeError(
            f"Space '{validated_slug}' was not found, or the current Hugging Face token cannot access it."
        ) from exc
    _validate_space_info(validated_slug, space_info)

    mcp_url = _build_space_mcp_url(space_info, validated_slug)
    private = bool(space_info.private)
    try:
        client = build_remote_client(alias, mcp_url, private=private)
    except ValueError as exc:
        raise RuntimeError(f"Space '{validated_slug}' returned an unsupported MCP endpoint: {exc}") from exc
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{validated_slug}': {exc}") from exc

    return InstalledToolSpace(
        slug=validated_slug,
        alias=alias,
        mcp_url=mcp_url,
        private=private,
        tools=_build_installed_tool_space_tools(slug=validated_slug, alias=alias, remote_specs=remote_specs),
    )


def resolve_tool_space_sync(slug: str) -> InstalledToolSpace:
    """Resolve one Space synchronously."""
    return asyncio.run(resolve_tool_space(slug))


def install_tool_space(
    slug: str,
    instance_path: str | Path | None,
    *,
    install_only: bool = False,
    profile: str | None = None,
) -> ToolSpaceInstallResult:
    """Install or refresh one Space and optionally enable its tools in a profile."""
    target_profile = profile or config.REACHY_MINI_CUSTOM_PROFILE or "default"
    if not install_only:
        try:
            read_profile_tool_names(target_profile, instance_path)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            raise ToolSpaceProfileUpdateError(
                f"Cannot enable Space tools in profile '{target_profile}': {exc}"
            ) from exc

    resolved_space = resolve_tool_space_sync(slug)
    with _MANIFEST_LOCK, profile_toolsets_transaction():
        manifest = read_installed_tool_spaces(instance_path)
        alias_conflict = next(
            (
                installed_space
                for installed_space in manifest.spaces
                if installed_space.slug != resolved_space.slug and installed_space.alias == resolved_space.alias
            ),
            None,
        )
        if alias_conflict is not None:
            raise ToolSpaceAliasConflictError(
                f"Cannot install '{resolved_space.slug}': its local alias '{resolved_space.alias}' conflicts with "
                f"already-installed '{alias_conflict.slug}'. Rename one Space on Hugging Face to get a distinct alias."
            )

        refreshed = any(installed_space.slug == resolved_space.slug for installed_space in manifest.spaces)
        updated_manifest = InstalledToolSpacesManifest(
            version=INSTALLED_TOOL_SPACES_VERSION,
            spaces=sorted(
                [space for space in manifest.spaces if space.slug != resolved_space.slug] + [resolved_space],
                key=lambda space: space.slug,
            ),
        )
        enabled_profile: str | None = None
        added_tool_ids: list[str] = []
        profile_toolsets = ProfileToolsets()
        profile_settings_existed = False
        if not install_only:
            profile_toolsets = read_profile_toolsets(instance_path)
            profile_settings_existed = get_profile_toolsets_path(instance_path).exists()
            tool_ids = [tool.local_name for tool in resolved_space.tools]
            try:
                added_tool_ids = enable_profile_tools(target_profile, tool_ids, instance_path)
            except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
                raise ToolSpaceProfileUpdateError(
                    f"Could not enable '{resolved_space.slug}' in profile '{target_profile}': {exc}"
                ) from exc
            enabled_profile = target_profile

        try:
            manifest_path = write_installed_tool_spaces(instance_path, updated_manifest)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            if not install_only:
                try:
                    _restore_profile_toolsets(instance_path, profile_toolsets, profile_settings_existed)
                except (OSError, RuntimeError, UnicodeError, ValueError) as rollback_exc:
                    raise ToolSpaceProfileUpdateError(
                        f"Could not persist '{resolved_space.slug}', and restoring the previous profile tool "
                        f"settings also failed: {rollback_exc}"
                    ) from rollback_exc
            raise RuntimeError(f"Could not persist installed Tool Space '{resolved_space.slug}': {exc}") from exc

    return ToolSpaceInstallResult(
        resolved_space=resolved_space,
        manifest=updated_manifest,
        manifest_path=manifest_path,
        refreshed=refreshed,
        enabled_profile=enabled_profile,
        added_tool_ids=added_tool_ids,
    )


def remove_tool_space(
    slug: str,
    instance_path: str | Path | None,
) -> ToolSpaceRemovalResult:
    """Remove one installed Space and disable its tools in all profiles."""
    validated_slug = validate_space_slug(slug)
    with _MANIFEST_LOCK, profile_toolsets_transaction():
        manifest = read_installed_tool_spaces(instance_path)
        removed_space = next((space for space in manifest.spaces if space.slug == validated_slug), None)
        if removed_space is None:
            raise ToolSpaceNotInstalledError(f"Space not installed: {validated_slug}")

        updated_manifest = InstalledToolSpacesManifest(
            version=INSTALLED_TOOL_SPACES_VERSION,
            spaces=[space for space in manifest.spaces if space.slug != validated_slug],
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
                f"{removed_space.alias}__",
                instance_path,
            )
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            raise ToolSpaceProfileUpdateError(
                f"Could not remove '{validated_slug}' because its profile tool access could not be updated: {exc}"
            ) from exc
        try:
            write_installed_tool_spaces(instance_path, updated_manifest)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            try:
                _restore_profile_toolsets(instance_path, profile_toolsets, profile_settings_existed)
            except (OSError, RuntimeError, UnicodeError, ValueError) as rollback_exc:
                raise ToolSpaceProfileUpdateError(
                    f"Could not persist removal of '{validated_slug}', and restoring the previous profile tool "
                    f"settings also failed: {rollback_exc}"
                ) from rollback_exc
            raise RuntimeError(f"Could not persist removal of Tool Space '{validated_slug}': {exc}") from exc

    return ToolSpaceRemovalResult(
        removed_space=removed_space,
        manifest=updated_manifest,
        disabled_profiles=disabled_profiles,
    )


def format_space_tool_listing(space: InstalledToolSpace) -> str:
    """Format one installed or resolved Space for terminal output."""
    lines = [
        f"{space.slug} ({space.alias})",
        f"  MCP endpoint: {space.mcp_url}",
    ]
    if space.tools:
        lines.append("  Tools:")
        lines.extend([f"    - {tool.local_name}" for tool in space.tools])
    else:
        lines.append("  Tools: none discovered")
    return "\n".join(lines)


def handle_tool_spaces_command(args: argparse.Namespace, *, instance_path: str | Path | None = None) -> int:
    """Handle tool-spaces subcommands from the main CLI."""
    command = getattr(args, "tool_spaces_command", None)
    if command == "add":
        target_profile = args.profile
        if target_profile is None and not args.install_only:
            target_profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
        try:
            install_result = install_tool_space(
                args.space_slug,
                instance_path,
                install_only=args.install_only,
                profile=target_profile,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1

        action = "Refreshed" if install_result.refreshed else "Installed"
        logger.info("%s Space tool source: %s", action, install_result.resolved_space.slug)
        logger.info("Manifest: %s", install_result.manifest_path)
        logger.info("%s", format_space_tool_listing(install_result.resolved_space))

        if args.install_only:
            logger.info("Tools installed. Select them under Tool access to enable them.")
            return 0

        if install_result.added_tool_ids:
            logger.info("Enabled in profile '%s': %s", install_result.enabled_profile, install_result.added_tool_ids)
        else:
            logger.info("All tool IDs already present in profile '%s'.", install_result.enabled_profile)
        return 0

    if command == "remove":
        try:
            removal_result = remove_tool_space(args.space_slug, instance_path)
        except ToolSpaceNotInstalledError as exc:
            logger.warning("%s", exc)
            return 1
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1

        logger.info("Removed Space tool source: %s", removal_result.removed_space.slug)
        for profile_name, disabled_tool_ids in removal_result.disabled_profiles:
            logger.info("Disabled in profile '%s': %s", profile_name, disabled_tool_ids)
        return 0

    if command == "list":
        manifest = read_installed_tool_spaces(instance_path)
        manifest_path = get_installed_tool_spaces_path(instance_path)
        logger.info("Manifest: %s", manifest_path)
        if not manifest.spaces:
            logger.info("No installed Space tool sources.")
            return 0

        for installed_space in manifest.spaces:
            logger.info("%s", format_space_tool_listing(installed_space))
        return 0

    raise RuntimeError(f"Unknown tool-spaces command: {command}")
