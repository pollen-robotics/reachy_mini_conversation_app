import abc
import sys
import json
import asyncio
import inspect
import logging
import importlib
import threading
import importlib.util
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Callable, ClassVar, Sequence, TypedDict
from pathlib import Path
from dataclasses import dataclass

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.config import config, list_tool_module_names
from reachy_mini_conversation_app.mcp_client import McpToolTimeoutError, McpToolInvocationError
from reachy_mini_conversation_app.tool_spaces import build_remote_client, read_installed_tool_spaces
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
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
    instance_path: str | Path | None = None
    camera_enabled: bool = False
    motion_duration_s: float = 1.0
    go_to_sleep: Callable[[], dict[str, Any]] | None = None


class ToolSpec(TypedDict):
    """Function-calling spec for a tool, in the OpenAI-compatible shape."""

    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, Any]  # arbitrary JSON Schema


class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema

    Tools may override:
      - needs_response: bool = True  # set False to skip the spoken follow-up after this tool runs
    """

    _auto_register: ClassVar[bool] = True
    needs_response: ClassVar[bool] = True

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> ToolSpec:
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
_TOOLS_SIGNATURE: tuple[str, str, str | None, bool, str | None] | None = None
_TOOLS_INSTANCE_PATH: str | Path | None = None
_LOADED_TOOL_CLASS_CACHE: Dict[tuple[str, str], List[type[Tool]]] = {}
_REMOTE_TOOL_RETRY_DELAY_S = 0.25
_TOOLS_LOCK = threading.RLock()
_EXTERNAL_TOOL_MODULE_NAMESPACE = "reachy_mini_conversation_app._external_tools"


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
        try:
            result = await self._client.call_tool(self._client_tool_name, kwargs)
        except McpToolTimeoutError:
            # Timeout subclasses the retryable error, but retrying it would just double the wait.
            raise
        except McpToolInvocationError as exc:
            logger.warning("Remote MCP tool failed once; retrying %s from %s: %s", self.name, self._space_slug, exc)
            await asyncio.sleep(_REMOTE_TOOL_RETRY_DELAY_S)
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
                ("module", module_path),
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
                ("file", _normalize_signature_path(tool_file) or str(tool_file)),
                lambda: _load_module_from_file(f"{_EXTERNAL_TOOL_MODULE_NAMESPACE}.{tool_name}", tool_file),
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
def _read_profile_tool_names(instance_path: str | Path | None) -> list[str]:
    """Read enabled tool names from the active profile's effective toolset."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
    logger.info("Loading tools for profile: %s", profile)
    try:
        tool_names = read_profile_tool_names(profile, instance_path)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Failed to read tools for profile %r: %s", profile, exc)
        raise RuntimeError(f"Failed to read tools for profile {profile!r}") from exc

    tool_names.extend(tool.value for tool in SystemTool if tool.value not in tool_names)

    if config.AUTOLOAD_EXTERNAL_TOOLS:
        discovered_external_tools = list_tool_module_names(config.TOOLS_DIRECTORY)
        extra_tools = [name for name in discovered_external_tools if name not in tool_names]
        if extra_tools:
            tool_names.extend(extra_tools)
            logger.info(
                "AUTOLOAD_EXTERNAL_TOOLS enabled: added %d external tool(s): %s",
                len(extra_tools),
                extra_tools,
            )

    logger.info("Found %d tools to load: %s", len(tool_names), tool_names)
    return tool_names


def _resolve_remote_tools(tool_names: list[str], instance_path: str | Path | None) -> list[RemoteMcpTool]:
    """Build Space tools enabled by the active profile from the cached install manifest, without any network calls."""
    remote_tools: list[RemoteMcpTool] = []
    for installed_space in read_installed_tool_spaces(instance_path).spaces:
        enabled_tool_names = {name for name in tool_names if name.startswith(f"{installed_space.alias}__")}
        if not enabled_tool_names:
            continue

        discovered_tool_names = {tool.local_name for tool in installed_space.tools}
        missing_tool_names = sorted(enabled_tool_names - discovered_tool_names)
        if missing_tool_names:
            logger.warning(
                "Tools enabled from '%s' are missing from the install manifest and will be skipped: %s. "
                "Re-run 'tool-spaces add %s' to refresh.",
                installed_space.slug,
                ", ".join(missing_tool_names),
                installed_space.slug,
            )

        client = build_remote_client(
            installed_space.alias,
            installed_space.mcp_url,
            private=installed_space.private,
            cached_tools=installed_space.tools,
        )
        for remote_tool in installed_space.tools:
            if remote_tool.local_name not in enabled_tool_names:
                continue
            cache_key = ("remote", f"{installed_space.slug}:{remote_tool.local_name}:{remote_tool.client_tool_name}")
            cached_tool = _LOADED_REMOTE_TOOL_CACHE.get(cache_key)
            if cached_tool is None:
                cached_tool = RemoteMcpTool(
                    slug=installed_space.slug,
                    name=remote_tool.local_name,
                    description=remote_tool.description,
                    parameters_schema=remote_tool.parameters_schema,
                    client_tool_name=remote_tool.client_tool_name,
                    client=client,
                )
                _LOADED_REMOTE_TOOL_CACHE[cache_key] = cached_tool
            remote_tools.append(cached_tool)

    return remote_tools


def _load_enabled_tools(tool_names: list[str], remote_tool_names: set[str]) -> List[type[Tool]]:
    """Load shared and external tools while skipping resolved remote tool IDs."""
    loaded_tool_classes: List[type[Tool]] = []

    for tool_name in tool_names:
        if tool_name in remote_tool_names:
            logger.info("✓ Registered remote tool: %s", tool_name)
            continue

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
            logger.warning("⚠️ Tool '%s' not found in shared or external tools", tool_name)
        except Exception as e:
            logger.error("❌ Failed to load tool '%s': %s", tool_name, _format_error(e))
            logger.error("  Module path: %s", shared_module_path)

    return loaded_tool_classes


def initialize_tools(instance_path: str | Path | None = None, *, force: bool = False) -> None:
    """Populate or refresh the active-profile tool registry.

    When ``force`` is true, file-backed tools are re-executed, while importable
    tool modules still follow normal ``importlib``/``sys.modules`` caching.
    """
    global ALL_TOOLS, _TOOLS_SIGNATURE, _TOOLS_INSTANCE_PATH

    with _TOOLS_LOCK:
        if force:
            _LOADED_TOOL_CLASS_CACHE.clear()
            _LOADED_REMOTE_TOOL_CACHE.clear()

        if instance_path is not None:
            _TOOLS_INSTANCE_PATH = instance_path
        effective_instance_path = _TOOLS_INSTANCE_PATH
        signature = _tool_registry_signature(effective_instance_path)

        if _TOOLS_SIGNATURE is not None and not force and signature == _TOOLS_SIGNATURE:
            logger.debug("Tools already initialized for active profile; skipping reinitialization.")
            return
        if _TOOLS_SIGNATURE is not None:
            logger.info("Reloading tool registry for active profile/configuration change.")

        tool_names = _read_profile_tool_names(effective_instance_path)
        remote_tools = _resolve_remote_tools(tool_names, effective_instance_path)
        remote_tool_names = {tool.name for tool in remote_tools}
        loaded_tool_classes = _load_enabled_tools(tool_names, remote_tool_names)
        tools = _build_tool_registry(
            loaded_tool_classes,
            extra_tools=remote_tools,
        )
        ALL_TOOLS = tools
        _TOOLS_SIGNATURE = signature

        for tool_name, tool in tools.items():
            logger.info("tool registered: %s - %s", tool_name, tool.description)


def get_tool_specs(exclusion_list: list[str] | None = None) -> list[ToolSpec]:
    """Get tool specs, optionally excluding some tools."""
    initialize_tools()
    exclusion_list = exclusion_list or []
    with _TOOLS_LOCK:
        return [tool.spec() for tool in ALL_TOOLS.values() if tool.name not in exclusion_list]


def get_tools() -> dict[str, Tool]:
    """Return a shallow snapshot of the active tool registry."""
    initialize_tools()
    with _TOOLS_LOCK:
        return dict(ALL_TOOLS)


# Dispatcher
def _safe_load_obj(args_json: str) -> Dict[str, Any]:
    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def _dispatch_tool_call(tool_name: str, args: Dict[str, Any], deps: ToolDependencies) -> Dict[str, Any]:
    tool = get_tools().get(tool_name)
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
