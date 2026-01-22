from __future__ import annotations
import abc
import sys
import json
import inspect
import logging
import importlib
from typing import Any, Dict, List
from pathlib import Path
from dataclasses import dataclass
import importlib.util
from reachy_mini import ReachyMini
# Import config to ensure .env is loaded before reading REACHY_MINI_CUSTOM_PROFILE
from reachy_mini_conversation_app.config import config  # noqa: F401


logger = logging.getLogger(__name__)


DEFAULT_PROFILES_DIRECTORY = "reachy_mini_conversation_app.profiles"


def _load_module_from_file(module_name: str, file_path: Path) -> None:
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not (spec and spec.loader):
        raise ModuleNotFoundError(f"Cannot create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _try_load_tool(
    tool_name: str,
    module_path: str,
    fallback_directory: Path | None,
    file_subpath: str,
) -> bool:
    """Try to load a tool: first via importlib, then from file if fallback is configured."""
    try:
        importlib.import_module(module_path)
        return True
    except ModuleNotFoundError:
        if fallback_directory is None:
            raise
        tool_file = fallback_directory / file_subpath
        _load_module_from_file(tool_name, tool_file)
        return True


def _format_error(error: Exception) -> str:
    """Format an exception for logging."""
    if isinstance(error, (ModuleNotFoundError, FileNotFoundError)):
        return f"Missing dependency: {error}"
    if isinstance(error, ImportError):
        return f"Import error: {error}"
    return f"{type(error).__name__}: {error}"


if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


ALL_TOOLS: Dict[str, "Tool"] = {}
ALL_TOOL_SPECS: List[Dict[str, Any]] = []
_TOOLS_INITIALIZED = False



def get_concrete_subclasses(base: type[Tool]) -> List[type[Tool]]:
    """Recursively find all concrete (non-abstract) subclasses of a base class."""
    result: List[type[Tool]] = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        # recurse into subclasses
        result.extend(get_concrete_subclasses(cls))
    return result


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any  # MovementManager from moves.py
    # Optional deps
    camera_worker: Any | None = None  # CameraWorker for frame buffering
    vision_manager: Any | None = None
    head_wobbler: Any | None = None  # HeadWobbler for audio-reactive motion
    motion_duration_s: float = 1.0


# Tool base class
class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema
    """

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


# Registry & specs (dynamic)
def _load_profile_tools() -> None:
    """Load tools based on profile's tools.txt file."""
    # Determine which profile to use
    profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"
    logger.info(f"Loading tools for profile: {profile}")

    # Build path to tools.txt
    # Get the profile directory path
    profile_module_path = config.PROFILES_DIRECTORY / profile
    tools_txt_path = profile_module_path / "tools.txt"

    if not tools_txt_path.exists():
        logger.error(f"✗ tools.txt not found at {tools_txt_path}")
        sys.exit(1)

    # Read and parse tools.txt
    try:
        with open(tools_txt_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"✗ Failed to read tools.txt: {e}")
        sys.exit(1)

    # Parse tool names (skip comments and blank lines)
    tool_names = []
    for line in lines:
        line = line.strip()
        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue
        tool_names.append(line)

    logger.info(f"Found {len(tool_names)} tools to load: {tool_names}")

    # Import each tool
    for tool_name in tool_names:
        loaded = False
        profile_error = None
        profile_import_path = f"{DEFAULT_PROFILES_DIRECTORY}.{profile}.{tool_name}"

        # Try profile tool first
        try:
            _try_load_tool(
                tool_name,
                module_path=profile_import_path,
                fallback_directory=config.PROFILES_DIRECTORY,
                file_subpath=f"{profile}/{tool_name}.py",
            )
            logger.info(f"✓ Loaded profile tool: {tool_name}")
            loaded = True
        except (ModuleNotFoundError, FileNotFoundError) as e:
            if tool_name not in str(e):
                profile_error = _format_error(e)
                logger.error(f"❌ Failed to load profile tool '{tool_name}': {profile_error}")
                logger.error(f"  Module path: {profile_import_path}")
        except Exception as e:
            profile_error = _format_error(e)
            logger.error(f"❌ Failed to load profile tool '{tool_name}': {profile_error}")
            logger.error(f"  Module path: {profile_import_path}")

        # Try tools directory if not found in profile
        if not loaded:
            shared_module_path = f"reachy_mini_conversation_app.tools.{tool_name}"
            try:
                _try_load_tool(
                    tool_name,
                    module_path=shared_module_path,
                    fallback_directory=config.TOOLS_DIRECTORY,
                    file_subpath=f"{tool_name}.py",
                )
                logger.info(f"✓ Loaded shared tool: {tool_name}")
            except (ModuleNotFoundError, FileNotFoundError):
                if profile_error:
                    logger.error(f"❌ Tool '{tool_name}' also not found in shared tools")
                else:
                    logger.warning(f"⚠️ Tool '{tool_name}' not found in profile or shared tools")
            except Exception as e:
                logger.error(f"❌ Failed to load shared tool '{tool_name}': {_format_error(e)}")
                logger.error(f"  Module path: {shared_module_path}")



def _initialize_tools() -> None:
    """Populate registry once, even if module is imported repeatedly."""
    global ALL_TOOLS, ALL_TOOL_SPECS, _TOOLS_INITIALIZED

    if _TOOLS_INITIALIZED:
        logger.debug("Tools already initialized; skipping reinitialization.")
        return

    _load_profile_tools()

    ALL_TOOLS = {cls.name: cls() for cls in get_concrete_subclasses(Tool)}  # type: ignore[type-abstract]
    ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]

    for tool_name, tool in ALL_TOOLS.items():
        logger.info(f"tool registered: {tool_name} - {tool.description}")

    _TOOLS_INITIALIZED = True


_initialize_tools()


def get_tool_specs(exclusion_list: list[str] = []) -> list[Dict[str, Any]]:
    """Get tool specs, optionally excluding some tools."""
    return [spec for spec in ALL_TOOL_SPECS if spec.get("name") not in exclusion_list]


# Dispatcher
def _safe_load_obj(args_json: str) -> Dict[str, Any]:
    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def dispatch_tool_call(tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
    """Dispatch a tool call by name with JSON args and dependencies."""
    tool = ALL_TOOLS.get(tool_name)

    if not tool:
        return {"error": f"unknown tool: {tool_name}"}

    args = _safe_load_obj(args_json)
    try:
        return await tool(deps, **args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", tool_name, msg)
        return {"error": msg}
