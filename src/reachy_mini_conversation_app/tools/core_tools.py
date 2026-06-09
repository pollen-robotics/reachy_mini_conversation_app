from __future__ import annotations
import re
import abc
import sys
import json
import asyncio
import inspect
import logging
import importlib
import importlib.util
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Callable, ClassVar, Sequence
from pathlib import Path
from dataclasses import dataclass

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY as DEFAULT_PROFILES_PATH

# Import config to ensure .env is loaded before reading REACHY_MINI_CUSTOM_PROFILE
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.tool_constants import SystemTool


if TYPE_CHECKING:
    from reachy_mini_conversation_app.mcp_client import RemoteMcpToolClient
    from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


class MissingToolFileError(FileNotFoundError):
    """Raised when a requested tool file is absent on disk."""


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any  # MovementManager from moves.py
    # Optional deps
    camera_worker: Any | None = None  # CameraWorker for frame buffering
    vision_processor: Any | None = None
    motion_duration_s: float = 1.0


class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema
    """

    _auto_register: ClassVar[bool] = True

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError


ALL_TOOLS: Dict[str, Tool] = {}
ALL_TOOL_SPECS: List[Dict[str, Any]] = []
_TOOLS_INITIALIZED = False
_TOOLS_SIGNATURE: tuple[str, str, str | None, bool, str | None] | None = None
_TOOLS_INSTANCE_PATH: str | Path | None = None
_LOADED_TOOL_CLASS_CACHE: Dict[tuple[str, str], List[type[Tool]]] = {}


class RemoteMcpTool(Tool):
    """Adapter exposing one remote MCP tool through the local Tool interface."""

    _auto_register: ClassVar[bool] = False

    def __init__(
        self,
        *,
        slug: str,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        client_tool_name: str,
        client: "RemoteMcpToolClient",
    ) -> None:
        """Store the resolved local/remote names and the shared MCP client."""
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self._space_slug = slug
        self._client_tool_name = client_tool_name
        self._client = client
        self._registry_source = f"space:{slug}:{client_tool_name}"

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Invoke the underlying remote MCP tool."""
        result = await self._client.call_tool(self._client_tool_name, kwargs)
        payload = dict(result)
        if payload.get("namespaced_tool_name") == self._client_tool_name:
            payload["namespaced_tool_name"] = self.name
        payload.setdefault("tool_space_slug", self._space_slug)
        return payload


_LOADED_REMOTE_TOOL_CACHE: Dict[tuple[str, str], RemoteMcpTool] = {}


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    """Load a Python module from a file path."""
    if not file_path.is_file():
        raise MissingToolFileError(f"tool file not found at {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not (spec and spec.loader):
        raise ModuleNotFoundError(f"Cannot create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid leaving a partially initialised module registered on failure
        sys.modules.pop(module_name, None)
        raise
    return module


def _format_error(error: Exception) -> str:
    """Format an exception for logging."""
    if isinstance(error, FileNotFoundError):
        return f"Tool file not found: {error}"
    if isinstance(error, ModuleNotFoundError):
        return f"Missing dependency: {error}"
    if isinstance(error, ImportError):
        return f"Import error: {error}"
    return f"{type(error).__name__}: {error}"


def _normalize_signature_path(value: str | Path | None) -> str | None:
    """Normalize a path-like value for registry invalidation and cache keys."""
    if value is None:
        return None
    try:
        return str(Path(value).expanduser().resolve())
    except Exception:
        return str(value)


def _cache_key_for_file(file_path: Path) -> tuple[str, str]:
    """Return the cache key for a file-backed tool source."""
    return ("file", _normalize_signature_path(file_path) or str(file_path))


def _cache_key_for_module(module_path: str) -> tuple[str, str]:
    """Return the cache key for an importable module-backed tool source."""
    return ("module", module_path)


def _tool_classes_from_module(module: ModuleType) -> List[type[Tool]]:
    """Return auto-registerable Tool classes defined directly in module."""
    tool_classes: List[type[Tool]] = []
    seen_class_ids: set[int] = set()
    for value in vars(module).values():
        if not inspect.isclass(value) or value.__module__ != module.__name__:
            continue
        try:
            is_tool_class = issubclass(value, Tool)
        except TypeError:
            continue
        if value is Tool or not is_tool_class or inspect.isabstract(value) or not value._auto_register:
            continue
        cls_id = id(value)
        if cls_id in seen_class_ids:
            continue
        seen_class_ids.add(cls_id)
        tool_classes.append(value)
    return tool_classes


def _load_cached_tool_classes(
    cache_key: tuple[str, str],
    load_module: Callable[[], ModuleType],
) -> tuple[List[type[Tool]], bool]:
    """Load tool classes once per source and return whether the cache was reused."""
    cached_classes = _LOADED_TOOL_CLASS_CACHE.get(cache_key)
    if cached_classes is not None:
        return cached_classes, True

    module = load_module()
    tool_classes = _tool_classes_from_module(module)
    _LOADED_TOOL_CLASS_CACHE[cache_key] = tool_classes
    return tool_classes, False


def _try_load_tool_classes(
    tool_name: str,
    module_path: str,
    fallback_directory: Path | None,
    file_subpath: str,
) -> tuple[str, List[type[Tool]], bool]:
    """Try to load tool classes: first via importlib, then from a configured external file."""
    try:
        return (
            "module",
            *_load_cached_tool_classes(
                _cache_key_for_module(module_path),
                lambda: importlib.import_module(module_path),
            ),
        )
    except ModuleNotFoundError:
        if fallback_directory is None:
            raise
        tool_file = fallback_directory / file_subpath
        return (
            "file",
            *_load_cached_tool_classes(
                _cache_key_for_file(tool_file),
                lambda: _load_module_from_file(tool_name, tool_file),
            ),
        )


def _build_tool_registry(
    tool_classes: List[type[Tool]],
    extra_tools: Sequence[Tool] | None = None,
) -> Dict[str, Tool]:
    """Instantiate tools and fail if duplicate Tool.name values are detected."""
    unique_classes: List[type[Tool]] = []
    seen_class_ids: set[int] = set()
    for cls in tool_classes:
        cls_id = id(cls)
        if cls_id in seen_class_ids:
            continue
        seen_class_ids.add(cls_id)
        unique_classes.append(cls)

    tool_instances: list[Tool] = []
    tool_instances.extend(cls() for cls in unique_classes)
    if extra_tools:
        tool_instances.extend(extra_tools)

    name_to_sources: Dict[str, List[str]] = {}
    for tool in tool_instances:
        source = getattr(tool, "_registry_source", f"{tool.__class__.__module__}.{tool.__class__.__name__}")
        name_to_sources.setdefault(tool.name, []).append(source)

    collisions = {tool_name: sources for tool_name, sources in name_to_sources.items() if len(sources) > 1}
    if collisions:
        details = "; ".join(f"{tool_name}: {sources}" for tool_name, sources in sorted(collisions.items()))
        raise RuntimeError(
            f"Duplicate Tool.name values detected while loading tools. Tool.name must be unique. Conflicts: {details}"
        )

    return {tool.name: tool for tool in tool_instances}


def _tool_registry_signature(instance_path: str | Path | None) -> tuple[str, str, str | None, bool, str | None]:
    """Return the runtime inputs that determine the active tool registry."""
    return (
        config.REACHY_MINI_CUSTOM_PROFILE or "default",
        _normalize_signature_path(config.PROFILES_DIRECTORY) or "",
        _normalize_signature_path(config.TOOLS_DIRECTORY),
        bool(config.AUTOLOAD_EXTERNAL_TOOLS),
        _normalize_signature_path(instance_path),
    )


# Registry & specs (dynamic)
def _resolve_profile_tools_txt_path() -> tuple[str, Path]:
    """Resolve the active profile's tools.txt with fallback to the built-in default."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
    logger.info(f"Loading tools for profile: {profile}")

    profile_dir = config.PROFILES_DIRECTORY / profile
    tools_txt_path = profile_dir / "tools.txt"
    default_tools_txt_path = DEFAULT_PROFILES_PATH / "default" / "tools.txt"

    if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_PATH:
        logger.info(
            "Loading external profile '%s' from %s",
            profile,
            profile_dir,
        )

    if not tools_txt_path.exists():
        if profile != "default" and default_tools_txt_path.exists():
            logger.warning(
                "tools.txt not found for profile '%s' at %s. Falling back to default profile tools at %s",
                profile,
                tools_txt_path,
                default_tools_txt_path,
            )
            return profile, default_tools_txt_path
        logger.error(f"✗ tools.txt not found at {tools_txt_path}")
        sys.exit(1)

    return profile, tools_txt_path


def _read_profile_tool_names() -> list[str]:
    """Read enabled tool names from the active profile's tools.txt file."""
    _, tools_txt_path = _resolve_profile_tools_txt_path()

    try:
        lines = tools_txt_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        logger.error(f"✗ Failed to read tools.txt: {e}")
        sys.exit(1)

    tool_names: list[str] = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        tool_names.append(stripped_line)

    tool_names.extend({tool.value for tool in SystemTool})

    if config.AUTOLOAD_EXTERNAL_TOOLS and config.TOOLS_DIRECTORY and config.TOOLS_DIRECTORY.is_dir():
        discovered_external_tools: List[str] = []
        for tool_file in sorted(config.TOOLS_DIRECTORY.glob("*.py")):
            if tool_file.name.startswith("_"):
                continue
            candidate_name = tool_file.stem
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate_name):
                logger.warning("Skipping external tool with invalid name: %s", tool_file.name)
                continue
            discovered_external_tools.append(candidate_name)

        extra_tools = [name for name in discovered_external_tools if name not in tool_names]
        if extra_tools:
            tool_names.extend(extra_tools)
            logger.info(
                "AUTOLOAD_EXTERNAL_TOOLS enabled: added %d external tool(s): %s",
                len(extra_tools),
                extra_tools,
            )

    logger.info(f"Found {len(tool_names)} tools to load: {tool_names}")
    return tool_names


def _resolve_remote_tools(tool_names: list[str], instance_path: str | Path | None) -> list[RemoteMcpTool]:
    """Resolve installed public Space tools enabled by the active profile."""
    from reachy_mini_conversation_app.tool_spaces import read_installed_tool_spaces, resolve_public_tool_space_sync

    remote_tools: list[RemoteMcpTool] = []
    for installed_space in read_installed_tool_spaces(instance_path).spaces:
        enabled_space_tools = sorted(name for name in tool_names if name.startswith(f"{installed_space.alias}__"))
        if not enabled_space_tools:
            logger.debug(
                "Installed Space '%s' has no enabled tools in the active profile; skipping discovery.",
                installed_space.slug,
            )
            continue

        try:
            resolved_space = resolve_public_tool_space_sync(installed_space.slug)
        except Exception as exc:
            logger.warning(
                "Space '%s' is unavailable, skipping its tools (%s): %s",
                installed_space.slug,
                ", ".join(enabled_space_tools),
                exc,
            )
            continue

        enabled_tool_names = set(enabled_space_tools)
        discovered_tool_names = {tool.local_name for tool in resolved_space.tools}
        missing_tool_names = sorted(enabled_tool_names - discovered_tool_names)
        if missing_tool_names:
            logger.warning(
                "Enabled tools from '%s' not found in Space and will be skipped: %s",
                installed_space.slug,
                ", ".join(missing_tool_names),
            )

        for remote_tool in resolved_space.tools:
            if remote_tool.local_name not in enabled_tool_names:
                continue
            cache_key = ("remote", f"{resolved_space.slug}:{remote_tool.local_name}:{remote_tool.client_tool_name}")
            cached_tool = _LOADED_REMOTE_TOOL_CACHE.get(cache_key)
            if cached_tool is None:
                cached_tool = RemoteMcpTool(
                    slug=resolved_space.slug,
                    name=remote_tool.local_name,
                    description=remote_tool.description,
                    parameters_schema=remote_tool.parameters_schema,
                    client_tool_name=remote_tool.client_tool_name,
                    client=resolved_space.client,
                )
                _LOADED_REMOTE_TOOL_CACHE[cache_key] = cached_tool
            remote_tools.append(cached_tool)

    return remote_tools


def _load_profile_tools(tool_names: list[str], remote_tool_names: set[str]) -> List[type[Tool]]:
    """Load local profile/shared tools while skipping resolved remote tool IDs."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
    loaded_tool_classes: List[type[Tool]] = []

    for tool_name in tool_names:
        if tool_name in remote_tool_names:
            logger.info("✓ Registered remote tool: %s", tool_name)
            continue

        loaded = False
        profile_error = None
        profile_tool_file = config.PROFILES_DIRECTORY / profile / f"{tool_name}.py"

        try:
            tool_classes, reused_cache = _load_cached_tool_classes(
                _cache_key_for_file(profile_tool_file),
                lambda: _load_module_from_file(tool_name, profile_tool_file),
            )
            loaded_tool_classes.extend(tool_classes)
            profile_scope = "external" if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_PATH else "built-in"
            action = "Reused" if reused_cache else "Loaded"
            logger.info("✓ %s %s profile tool: %s", action, profile_scope, tool_name)
            loaded = True
        except MissingToolFileError:
            logger.debug("No profile-local tool file for '%s' at %s", tool_name, profile_tool_file)
        except FileNotFoundError as e:
            profile_error = _format_error(e)
            logger.error(f"❌ Failed to load profile tool '{tool_name}': {profile_error}")
        except Exception as e:
            profile_error = _format_error(e)
            logger.error(f"❌ Failed to load profile tool '{tool_name}': {profile_error}")

        if not loaded:
            shared_module_path = f"reachy_mini_conversation_app.tools.{tool_name}"
            try:
                source, tool_classes, reused_cache = _try_load_tool_classes(
                    tool_name,
                    module_path=shared_module_path,
                    fallback_directory=config.TOOLS_DIRECTORY,
                    file_subpath=f"{tool_name}.py",
                )
                loaded_tool_classes.extend(tool_classes)
                action = "Reused" if reused_cache else "Loaded"
                if source == "file":
                    logger.info("✓ %s external tool: %s", action, tool_name)
                else:
                    logger.info("✓ %s core tool: %s", action, tool_name)
            except (ModuleNotFoundError, FileNotFoundError):
                if profile_error:
                    logger.error(f"❌ Tool '{tool_name}' also not found in shared tools")
                else:
                    logger.warning(f"⚠️ Tool '{tool_name}' not found in profile or shared tools")
            except Exception as e:
                logger.error(f"❌ Failed to load shared tool '{tool_name}': {_format_error(e)}")
                logger.error(f"  Module path: {shared_module_path}")

    return loaded_tool_classes


def initialize_tools(instance_path: str | Path | None = None, *, force: bool = False) -> None:
    """Populate or refresh the active-profile tool registry.

    When ``force`` is true, file-backed tools are re-executed, while importable
    tool modules still follow normal ``importlib``/``sys.modules`` caching.
    """
    global ALL_TOOLS, ALL_TOOL_SPECS, _TOOLS_INITIALIZED, _TOOLS_SIGNATURE, _TOOLS_INSTANCE_PATH

    if force:
        _LOADED_TOOL_CLASS_CACHE.clear()
        _LOADED_REMOTE_TOOL_CACHE.clear()

    if instance_path is not None:
        _TOOLS_INSTANCE_PATH = instance_path
    effective_instance_path = _TOOLS_INSTANCE_PATH
    signature = _tool_registry_signature(effective_instance_path)

    if _TOOLS_INITIALIZED and not force and signature == _TOOLS_SIGNATURE:
        logger.debug("Tools already initialized for active profile; skipping reinitialization.")
        return
    if _TOOLS_INITIALIZED:
        logger.info("Reloading tool registry for active profile/configuration change.")

    tool_names = _read_profile_tool_names()
    remote_tools = _resolve_remote_tools(tool_names, effective_instance_path)
    remote_tool_names = {tool.name for tool in remote_tools}
    loaded_tool_classes = _load_profile_tools(tool_names, remote_tool_names)

    ALL_TOOLS = _build_tool_registry(
        loaded_tool_classes,
        extra_tools=remote_tools,
    )
    ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]

    for tool_name, tool in ALL_TOOLS.items():
        logger.info(f"tool registered: {tool_name} - {tool.description}")

    _TOOLS_INITIALIZED = True
    _TOOLS_SIGNATURE = signature


def get_tool_specs(exclusion_list: list[str] | None = None) -> list[Dict[str, Any]]:
    """Get tool specs, optionally excluding some tools."""
    initialize_tools()
    exclusion_list = exclusion_list or []
    return [spec for spec in ALL_TOOL_SPECS if spec.get("name") not in exclusion_list]


def get_active_tool_specs(deps: ToolDependencies) -> list[Dict[str, Any]]:
    """Get tool specs filtered by what the current session deps support."""
    exclusion_list: list[str] = []
    if not (deps.camera_worker and deps.camera_worker.head_tracker):
        exclusion_list.append("head_tracking")
    return get_tool_specs(exclusion_list)


# Dispatcher
def _safe_load_obj(args_json: str) -> Dict[str, Any]:
    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def _dispatch_tool_call(tool_name: str, args: Dict[str, Any], deps: ToolDependencies) -> Dict[str, Any]:
    initialize_tools()
    tool = ALL_TOOLS.get(tool_name)
    if not tool:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        return await tool(deps, **args)
    except asyncio.CancelledError:
        logger.info("Tool cancelled: %s", tool_name)
        return {"error": "Tool cancelled"}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", tool_name, msg)
        return {"error": msg}


async def dispatch_tool_call(tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
    """Dispatch a tool call by name with JSON args and dependencies."""
    return await _dispatch_tool_call(tool_name, _safe_load_obj(args_json), deps)


async def dispatch_tool_call_with_manager(
    tool_name: str, args_json: str, deps: ToolDependencies, tool_manager: "BackgroundToolManager"
) -> Dict[str, Any]:
    """Dispatch a tool call, injecting a BackgroundToolManager into the args."""
    args = _safe_load_obj(args_json)
    args["tool_manager"] = tool_manager
    return await _dispatch_tool_call(tool_name, args, deps)
