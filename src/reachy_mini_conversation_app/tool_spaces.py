"""Manage installed Hugging Face Space tool sources for the conversation app."""

from __future__ import annotations
import re
import json
import asyncio
import logging
import argparse
from typing import Any
from pathlib import Path
from collections import Counter
from dataclasses import field, asdict, dataclass
from collections.abc import Sequence

from huggingface_hub import HfApi, SpaceInfo, get_token
from huggingface_hub.errors import RepositoryNotFoundError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    apply_name_normalization,
)


logger = logging.getLogger(__name__)

INSTALLED_TOOL_SPACES_FILENAME = "installed_tool_spaces.json"
INSTALLED_TOOL_SPACES_VERSION = 2
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
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
    """Persisted record for one installed Space and the tools discovered at install time."""

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


@dataclass(frozen=True)
class ResolvedInstalledToolSpace:
    """Runtime description of an installed Space."""

    slug: str
    alias: str
    mcp_url: str
    private: bool
    tags: list[str]
    tools: list[InstalledToolSpaceTool]
    client: RemoteMcpToolClient


def get_installed_tool_spaces_path(instance_path: str | Path | None) -> Path:
    """Return the installed tool-spaces manifest path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / INSTALLED_TOOL_SPACES_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / INSTALLED_TOOL_SPACES_FILENAME


def read_installed_tool_spaces(instance_path: str | Path | None) -> InstalledToolSpacesManifest:
    """Read the installed tool-spaces manifest if present."""
    manifest_path = get_installed_tool_spaces_path(instance_path)
    if not manifest_path.exists():
        return InstalledToolSpacesManifest()

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read installed tool spaces from {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: expected a JSON object.")

    raw_spaces = payload.get("spaces", [])
    if not isinstance(raw_spaces, list):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: 'spaces' must be a list.")

    spaces: list[InstalledToolSpace] = []
    seen_slugs: set[str] = set()
    seen_aliases: set[str] = set()
    for raw_space in raw_spaces:
        if not isinstance(raw_space, dict):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: expected an object.")

        slug = validate_space_slug(str(raw_space.get("slug", "")))
        alias = normalize_space_alias(slug)
        if slug in seen_slugs:
            raise RuntimeError(f"Duplicate installed tool space '{slug}' found in {manifest_path}.")
        if alias in seen_aliases:
            raise RuntimeError(
                f"Installed tool spaces manifest contains alias collision '{alias}' in {manifest_path}. "
                "Remove one of the conflicting spaces with 'tool-spaces remove'."
            )
        mcp_url = str(raw_space.get("mcp_url", "")).strip()
        if not mcp_url:
            logger.warning(
                "Installed Space '%s' predates cached tool metadata and will be skipped. Re-run 'tool-spaces add %s'.",
                slug,
                slug,
            )
            continue
        cached_tools = [
            InstalledToolSpaceTool(
                local_name=str(tool["local_name"]),
                client_tool_name=str(tool["client_tool_name"]),
                remote_name=str(tool.get("remote_name", "")),
                description=str(tool.get("description", "")),
                parameters_schema=dict(tool.get("parameters_schema") or {}),
            )
            for tool in raw_space.get("tools", [])
            if isinstance(tool, dict) and tool.get("local_name") and tool.get("client_tool_name")
        ]
        seen_slugs.add(slug)
        seen_aliases.add(alias)
        spaces.append(
            InstalledToolSpace(
                slug=slug,
                alias=alias,
                mcp_url=mcp_url,
                private=bool(raw_space.get("private", False)),
                tools=cached_tools,
            )
        )

    version = payload.get("version", 1)
    if not isinstance(version, int):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: 'version' must be an int.")
    return InstalledToolSpacesManifest(version=version, spaces=spaces)


def write_installed_tool_spaces(
    instance_path: str | Path | None,
    manifest: InstalledToolSpacesManifest,
) -> Path:
    """Persist the installed tool-spaces manifest."""
    manifest_path = get_installed_tool_spaces_path(instance_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "spaces": [asdict(space) for space in manifest.spaces],
    }
    manifest_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return manifest_path


def _append_tools_to_profile(profile: str, tool_ids: list[str]) -> list[str]:
    """Append tool IDs to a profile's tools.txt. Returns the IDs that were added."""
    tools_txt = config.resolve_profile_dir(profile) / "tools.txt"
    if not tools_txt.parent.is_dir():
        raise RuntimeError(
            f"Profile '{profile}' not found at {tools_txt.parent}. Use --install-only to skip profile wiring."
        )

    existing_content = tools_txt.read_text(encoding="utf-8") if tools_txt.exists() else ""
    existing: set[str] = set()
    for line in existing_content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            existing.add(stripped)

    to_add = [tid for tid in tool_ids if tid not in existing]
    if to_add:
        with tools_txt.open("a", encoding="utf-8") as f:
            if existing_content and not existing_content.endswith("\n"):
                f.write("\n")
            for tid in to_add:
                f.write(f"{tid}\n")
    return to_add


def _disable_space_tools_in_profiles(alias: str) -> list[tuple[str, list[str]]]:
    """Strip a Space's tool IDs from every profile's tools.txt. Returns (profile, removed IDs) per profile touched."""
    prefix = f"{alias}__"
    removed_by_profile: list[tuple[str, list[str]]] = []
    seen: set[Path] = set()
    for root in (config.PROFILES_DIRECTORY, config.user_personalities_root()):
        for tools_txt in sorted(root.glob("*/tools.txt")):
            resolved = tools_txt.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            lines = tools_txt.read_text(encoding="utf-8").splitlines()
            removed = [line.strip() for line in lines if line.strip().startswith(prefix)]
            if not removed:
                continue
            kept = [line for line in lines if not line.strip().startswith(prefix)]
            tools_txt.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
            removed_by_profile.append((tools_txt.parent.name, removed))
    return removed_by_profile


def validate_space_slug(slug: str) -> str:
    """Validate a public HF Space slug."""
    candidate = slug.strip()
    if _SLUG_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            f"Invalid Space slug '{slug}'. Expected the form 'owner/space-name' with alnum, '.', '_' or '-'."
        )
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
    token = config.HF_TOKEN or get_token()
    headers = {"Authorization": f"Bearer {token}"} if private and token else {}
    return RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=alias,
            url=mcp_url,
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


async def resolve_tool_space(slug: str) -> ResolvedInstalledToolSpace:
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
    client = build_remote_client(alias, mcp_url, private=private)
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{validated_slug}': {exc}") from exc

    return ResolvedInstalledToolSpace(
        slug=validated_slug,
        alias=alias,
        mcp_url=mcp_url,
        private=private,
        tags=sorted(space_info.tags or []),
        tools=_build_installed_tool_space_tools(slug=validated_slug, alias=alias, remote_specs=remote_specs),
        client=client,
    )


def resolve_tool_space_sync(slug: str) -> ResolvedInstalledToolSpace:
    """Resolve one Space synchronously."""
    return asyncio.run(resolve_tool_space(slug))


def format_space_tool_listing(space: ResolvedInstalledToolSpace) -> str:
    """Format one resolved Space for terminal output."""
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
        try:
            resolved_space = resolve_tool_space_sync(args.space_slug)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        manifest = read_installed_tool_spaces(instance_path)
        already_installed = any(space.slug == resolved_space.slug for space in manifest.spaces)
        if already_installed:
            logger.info("Space already installed: %s", resolved_space.slug)
            logger.info("%s", format_space_tool_listing(resolved_space))
        else:
            alias_conflict = next((s for s in manifest.spaces if s.alias == resolved_space.alias), None)
            if alias_conflict:
                logger.error(
                    "Cannot install '%s': its local alias '%s' conflicts with already-installed '%s'. "
                    "Rename one Space on Hugging Face to get a distinct alias.",
                    resolved_space.slug,
                    resolved_space.alias,
                    alias_conflict.slug,
                )
                return 1

            installed = InstalledToolSpace(
                slug=resolved_space.slug,
                alias=resolved_space.alias,
                mcp_url=resolved_space.mcp_url,
                private=resolved_space.private,
                tools=resolved_space.tools,
            )
            updated_spaces = sorted(
                [*manifest.spaces, installed],
                key=lambda space: space.slug,
            )
            manifest_path = write_installed_tool_spaces(
                instance_path,
                InstalledToolSpacesManifest(version=INSTALLED_TOOL_SPACES_VERSION, spaces=updated_spaces),
            )
            logger.info("Installed Space tool source: %s", resolved_space.slug)
            logger.info("Manifest: %s", manifest_path)
            logger.info("%s", format_space_tool_listing(resolved_space))

        if args.install_only:
            logger.info("Tools installed. Add tool IDs to a profile's tools.txt to enable them.")
            return 0

        target_profile = args.profile
        if target_profile is None:
            target_profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"

        tool_ids = [tool.local_name for tool in resolved_space.tools]
        try:
            added = _append_tools_to_profile(target_profile, tool_ids)
        except RuntimeError as exc:
            logger.error("Cannot enable tools: %s", exc)
            return 1
        if added:
            logger.info("Enabled in profile '%s': %s", target_profile, added)
        else:
            logger.info("All tool IDs already present in profile '%s'.", target_profile)
        return 0

    if command == "remove":
        validated_slug = validate_space_slug(args.space_slug)
        manifest = read_installed_tool_spaces(instance_path)
        remaining_spaces = [space for space in manifest.spaces if space.slug != validated_slug]
        if len(remaining_spaces) == len(manifest.spaces):
            logger.warning("Space not installed: %s", validated_slug)
            return 1

        write_installed_tool_spaces(
            instance_path,
            InstalledToolSpacesManifest(version=manifest.version, spaces=remaining_spaces),
        )
        logger.info("Removed Space tool source: %s", validated_slug)
        for profile_name, disabled_tool_ids in _disable_space_tools_in_profiles(normalize_space_alias(validated_slug)):
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
            try:
                resolved_space = resolve_tool_space_sync(installed_space.slug)
            except Exception as exc:
                logger.warning("Space '%s' (%s) is unavailable: %s", installed_space.slug, installed_space.alias, exc)
                continue
            logger.info("%s", format_space_tool_listing(resolved_space))
        return 0

    raise RuntimeError(f"Unknown tool-spaces command: {command}")
