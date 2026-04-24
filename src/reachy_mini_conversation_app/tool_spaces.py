"""Manage installed Hugging Face Space tool sources for the conversation app."""

from __future__ import annotations
import re
import json
import asyncio
import logging
from typing import Any, Sequence
from pathlib import Path
from dataclasses import field, asdict, dataclass

from huggingface_hub import HfApi, SpaceInfo

from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
)


logger = logging.getLogger(__name__)

INSTALLED_TOOL_SPACES_FILENAME = "installed_tool_spaces.json"
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_NAME_NORMALIZER_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class InstalledToolSpace:
    """Persisted record for one installed public Space."""

    slug: str
    alias: str


@dataclass(frozen=True)
class InstalledToolSpacesManifest:
    """Persisted manifest of installed public Space tool sources."""

    version: int = 1
    spaces: list[InstalledToolSpace] = field(default_factory=list)


@dataclass(frozen=True)
class InstalledToolSpaceTool:
    """App-facing metadata for one remote tool exposed by an installed Space."""

    local_name: str
    client_tool_name: str
    remote_name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass(frozen=True)
class ResolvedInstalledToolSpace:
    """Runtime description of an installed public Space."""

    slug: str
    alias: str
    mcp_url: str
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
    for raw_space in raw_spaces:
        if not isinstance(raw_space, dict):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: expected an object.")

        slug = validate_space_slug(str(raw_space.get("slug", "")))
        alias = normalize_space_alias(slug)
        if slug in seen_slugs:
            raise RuntimeError(f"Duplicate installed tool space '{slug}' found in {manifest_path}.")
        seen_slugs.add(slug)
        spaces.append(InstalledToolSpace(slug=slug, alias=alias))

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
    normalized = _NAME_NORMALIZER_PATTERN.sub("_", slug).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ValueError(f"Space slug '{slug}' cannot be normalized into a local alias.")
    if normalized[0].isdigit():
        normalized = f"space_{normalized}"
    return normalized


def _normalize_name_segment(value: str) -> str:
    normalized = _NAME_NORMALIZER_PATTERN.sub("_", value).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        return "tool"
    if normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return normalized


def _clean_space_tool_name(slug: str, alias: str, remote_name: str) -> str:
    normalized_remote_name = _normalize_name_segment(remote_name)
    space_name = slug.split("/", maxsplit=1)[1]
    normalized_space_name = _normalize_name_segment(space_name)
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
    collisions = {name for name in cleaned_names if cleaned_names.count(name) > 1}

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


def _build_public_space_mcp_url(space_info: SpaceInfo, slug: str) -> str:
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


def _validate_public_space_info(slug: str, space_info: SpaceInfo) -> None:
    if bool(space_info.private):
        raise RuntimeError(f"Space '{slug}' is not public and cannot be installed in this v1 flow.")
    if bool(space_info.disabled):
        raise RuntimeError(f"Space '{slug}' is disabled and cannot be installed.")
    if (space_info.sdk or "").strip().lower() != "gradio":
        raise RuntimeError(f"Space '{slug}' is not a Gradio Space and cannot expose the standard MCP endpoint.")


async def resolve_public_tool_space(slug: str) -> ResolvedInstalledToolSpace:
    """Validate and discover tools from one public HF Space."""
    validated_slug = validate_space_slug(slug)
    alias = normalize_space_alias(validated_slug)
    space_info = HfApi().space_info(validated_slug, timeout=10.0, token=False)
    _validate_public_space_info(validated_slug, space_info)

    mcp_url = _build_public_space_mcp_url(space_info, validated_slug)
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=alias,
            url=mcp_url,
            request_timeout_s=10.0,
            tool_timeout_s=30.0,
        )
    )
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{validated_slug}': {exc}") from exc

    return ResolvedInstalledToolSpace(
        slug=validated_slug,
        alias=alias,
        mcp_url=mcp_url,
        tags=sorted(space_info.tags or []),
        tools=_build_installed_tool_space_tools(slug=validated_slug, alias=alias, remote_specs=remote_specs),
        client=client,
    )


def resolve_public_tool_space_sync(slug: str) -> ResolvedInstalledToolSpace:
    """Resolve one public Space synchronously."""
    try:
        previous_loop = asyncio.get_running_loop()
    except RuntimeError:
        previous_loop = None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(resolve_public_tool_space(slug))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        if previous_loop is not None and not previous_loop.is_closed():
            asyncio.set_event_loop(previous_loop)
        else:
            asyncio.set_event_loop(None)


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


def handle_tool_spaces_command(args: Any, *, instance_path: str | Path | None = None) -> int:
    """Handle tool-spaces subcommands from the main CLI."""
    command = getattr(args, "tool_spaces_command", None)
    if command == "add":
        resolved_space = resolve_public_tool_space_sync(args.space_slug)
        manifest = read_installed_tool_spaces(instance_path)
        if any(space.slug == resolved_space.slug for space in manifest.spaces):
            print(f"Space already installed: {resolved_space.slug}")
            print(format_space_tool_listing(resolved_space))
            print("Next step: add the tool IDs you want to use to the desired profile's tools.txt.")
            return 0

        updated_spaces = sorted(
            [*manifest.spaces, InstalledToolSpace(slug=resolved_space.slug, alias=resolved_space.alias)],
            key=lambda space: space.slug,
        )
        manifest_path = write_installed_tool_spaces(
            instance_path,
            InstalledToolSpacesManifest(version=manifest.version, spaces=updated_spaces),
        )
        print(f"Installed Space tool source: {resolved_space.slug}")
        print(f"Manifest: {manifest_path}")
        print(format_space_tool_listing(resolved_space))
        print("Next step: add the tool IDs you want to use to the desired profile's tools.txt.")
        return 0

    if command == "remove":
        validated_slug = validate_space_slug(args.space_slug)
        manifest = read_installed_tool_spaces(instance_path)
        remaining_spaces = [space for space in manifest.spaces if space.slug != validated_slug]
        if len(remaining_spaces) == len(manifest.spaces):
            print(f"Space not installed: {validated_slug}")
            return 1

        try:
            removed_space: ResolvedInstalledToolSpace | None = resolve_public_tool_space_sync(validated_slug)
        except Exception as exc:
            removed_space = None
            logger.warning("Could not refresh tools for '%s' before removal: %s", validated_slug, exc)

        write_installed_tool_spaces(
            instance_path,
            InstalledToolSpacesManifest(version=manifest.version, spaces=remaining_spaces),
        )
        print(f"Removed Space tool source: {validated_slug}")
        if removed_space is not None:
            print(format_space_tool_listing(removed_space))
        return 0

    if command == "list":
        manifest = read_installed_tool_spaces(instance_path)
        manifest_path = get_installed_tool_spaces_path(instance_path)
        print(f"Manifest: {manifest_path}")
        if not manifest.spaces:
            print("No installed Space tool sources.")
            return 0

        for installed_space in manifest.spaces:
            try:
                resolved_space = resolve_public_tool_space_sync(installed_space.slug)
            except Exception as exc:
                print(f"{installed_space.slug} ({installed_space.alias})")
                print(f"  Unavailable: {exc}")
                continue
            print(format_space_tool_listing(resolved_space))
        return 0

    raise RuntimeError(f"Unknown tool-spaces command: {command}")
