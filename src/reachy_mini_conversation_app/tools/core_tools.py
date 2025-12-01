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

from reachy_mini import ReachyMini
# Import config to ensure .env is loaded before reading REACHY_MINI_CUSTOM_PROFILE
from reachy_mini_conversation_app.config import config  # noqa: F401


logger = logging.getLogger(__name__)


PROFILES_DIRECTORY = "reachy_mini_conversation_app.profiles"

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
    profile_module_path = Path(__file__).parent.parent / "profiles" / profile
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

        # Try profile-local tool first
        try:
            profile_tool_module = f"{PROFILES_DIRECTORY}.{profile}.{tool_name}"
            importlib.import_module(profile_tool_module)
            logger.info(f"✓ Loaded profile-local tool: {tool_name}")
            loaded = True
        except ModuleNotFoundError as e:
            # Check if it's the tool module itself that's missing (expected) or a dependency
            if tool_name in str(e):
                pass  # Tool not in profile directory, try shared tools
            else:
                # Missing import dependency within the tool file
                profile_error = f"Missing dependency: {e}"
                logger.error(f"❌ Failed to load profile-local tool '{tool_name}': {profile_error}")
                logger.error(f"  Module path: {profile_tool_module}")
        except ImportError as e:
            profile_error = f"Import error: {e}"
            logger.error(f"❌ Failed to load profile-local tool '{tool_name}': {profile_error}")
            logger.error(f"  Module path: {profile_tool_module}")
        except Exception as e:
            profile_error = f"{type(e).__name__}: {e}"
            logger.error(f"❌ Failed to load profile-local tool '{tool_name}': {profile_error}")
            logger.error(f"  Module path: {profile_tool_module}")

        # Try shared tools library if not found in profile
        if not loaded:
            try:
                shared_tool_module = f"reachy_mini_conversation_app.tools.{tool_name}"
                importlib.import_module(shared_tool_module)
                logger.info(f"✓ Loaded shared tool: {tool_name}")
                loaded = True
            except ModuleNotFoundError:
                if profile_error:
                    # Already logged error from profile attempt
                    logger.error(f"❌ Tool '{tool_name}' also not found in shared tools")
                else:
                    logger.warning(f"⚠️ Tool '{tool_name}' not found in profile or shared tools")
            except ImportError as e:
                logger.error(f"❌ Failed to load shared tool '{tool_name}': Import error: {e}")
                logger.error(f"  Module path: {shared_tool_module}")
            except Exception as e:
                logger.error(f"❌ Failed to load shared tool '{tool_name}': {type(e).__name__}: {e}")
                logger.error(f"  Module path: {shared_tool_module}")


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
