from __future__ import annotations
import abc
import sys
import json
import inspect
import logging
import importlib
from typing import Any, Dict, List, ClassVar
from pathlib import Path
from dataclasses import dataclass

from reachy_mini import ReachyMini
# Import config to ensure .env is loaded before reading REACHY_MINI_CUSTOM_PROFILE
from reachy_mini_conversation_app.config import config  # noqa: F401


logger = logging.getLogger(__name__)

# Configure the parent logger for all tools so that logs propagate correctly
_tools_logger = logging.getLogger("reachy_mini_conversation_app.tools")
if not _tools_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s")
    handler.setFormatter(formatter)
    _tools_logger.addHandler(handler)
    _tools_logger.setLevel(logging.DEBUG)


PROFILES_DIRECTORY = "reachy_mini_conversation_app.profiles"

# Re-export imported modules that tests need to access for patching
__all__ = [
    "EnvVar",
    "Tool",
    "ToolDependencies",
    "ALL_TOOLS",
    "ALL_TOOL_SPECS",
    "BASE_CONFIG_VARS",
    "get_concrete_subclasses",
    "get_tool_specs",
    "dispatch_tool_call",
    "collect_all_env_vars",
    "get_config_vars",
    "_safe_load_obj",
    "_load_profile_tools",
    "_initialize_tools",
    "_TOOLS_INITIALIZED",
    # Re-exported for test patching
    "config",
    "Path",
    "importlib",
]

ALL_TOOLS: Dict[str, "Tool"] = {}
ALL_TOOL_SPECS: List[Dict[str, Any]] = []
_TOOLS_INITIALIZED = False

# Base configuration variables that are always needed (not tool-specific)
# These are shown in the UI regardless of which tools are loaded
# Format: EnvVar instances for consistency
BASE_CONFIG_VARS: List["EnvVar"] = []  # Populated after EnvVar is defined



def get_concrete_subclasses(base: type) -> List[type[Tool]]:
    """Recursively find all concrete (non-abstract) subclasses of a base class."""
    result: List[type[Tool]] = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        # recurse into subclasses
        result.extend(get_concrete_subclasses(cls))
    return result


@dataclass
class EnvVar:
    """Declaration of an environment variable required by a tool.

    Tools can declare their required environment variables by adding a
    `required_env_vars` class attribute with a list of EnvVar instances.
    These declarations are collected at startup and used to dynamically
    populate the configuration UI.

    Example:
        class MyTool(Tool):
            name = "my_tool"
            required_env_vars = [
                EnvVar("MY_API_KEY", is_secret=True, description="API key for service"),
                EnvVar("MY_MODEL", default="gpt-4", description="Model to use"),
            ]

    """

    name: str  # Environment variable name (e.g., "ANTHROPIC_API_KEY")
    is_secret: bool = False  # If True, value is masked in UI
    description: str = ""  # Human-readable description for UI
    default: str | None = None  # Default value if not set
    required: bool = True  # If True, tool may fail without this var

    def to_config_tuple(self) -> tuple[str, str, bool, str]:
        """Convert to CONFIG_VARS tuple format for backward compatibility.

        Returns:
            Tuple of (env_var_name, config_attr_name, is_secret, description)

        """
        return (self.name, self.name, self.is_secret, self.description)


# Populate BASE_CONFIG_VARS now that EnvVar is defined
BASE_CONFIG_VARS.extend([
    EnvVar("OPENAI_API_KEY", is_secret=True, description="OpenAI API key (required for voice)"),
    EnvVar("MODEL_NAME", is_secret=False, description="OpenAI model name", required=False),
    EnvVar("HF_TOKEN", is_secret=True, description="Hugging Face token (optional, for vision)", required=False),
    EnvVar("HF_HOME", is_secret=False, description="Hugging Face cache directory", required=False),
    EnvVar("LOCAL_VISION_MODEL", is_secret=False, description="Local vision model path", required=False),
    EnvVar(
        "REACHY_MINI_CUSTOM_PROFILE",
        is_secret=False,
        description="Custom profile name",
        required=False,
    ),
])


def collect_all_env_vars() -> List[EnvVar]:
    """Collect all environment variables from base config and loaded tools.

    Returns a deduplicated list of EnvVar instances, with base config vars
    first, followed by tool-specific vars. If the same variable is declared
    by multiple tools, only the first occurrence is kept and a warning is logged.

    Returns:
        List of unique EnvVar instances.

    """
    seen: Dict[str, EnvVar] = {}
    result: List[EnvVar] = []

    # Add base config vars first
    for env_var in BASE_CONFIG_VARS:
        if env_var.name not in seen:
            seen[env_var.name] = env_var
            result.append(env_var)

    # Collect from all loaded tools
    for tool_name, tool in ALL_TOOLS.items():
        for env_var in getattr(tool, "required_env_vars", []):
            if env_var.name in seen:
                existing = seen[env_var.name]
                # Log warning if declarations differ
                if (
                    existing.is_secret != env_var.is_secret
                    or existing.description != env_var.description
                ):
                    logger.warning(
                        f"EnvVar '{env_var.name}' declared differently by tool '{tool_name}'. "
                        f"Using first declaration."
                    )
            else:
                seen[env_var.name] = env_var
                result.append(env_var)

    return result


def get_config_vars() -> List[tuple[str, str, bool, str]]:
    """Get configuration variables in CONFIG_VARS tuple format.

    This function provides backward compatibility with the existing CONFIG_VARS
    format used by headless_personality_ui.py.

    Returns:
        List of tuples: (env_var_name, config_attr_name, is_secret, description)

    """
    return [env_var.to_config_tuple() for env_var in collect_all_env_vars()]


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
    background_task_manager: Any | None = None  # BackgroundTaskManager for async tasks


# Tool base class
class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema

    Optional:
      - supports_background: bool  # Whether tool can run in background (default: False)
      - required_env_vars: List[EnvVar]  # Environment variables this tool needs
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    supports_background: bool = False  # Override in subclass to enable background execution
    required_env_vars: ClassVar[List[EnvVar]] = []  # Override to declare env var dependencies

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

    ALL_TOOLS = {cls.name: cls() for cls in get_concrete_subclasses(Tool)}
    ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]

    for tool_name, tool in ALL_TOOLS.items():
        logger.info(f"tool registered: {tool_name} - {tool.description}")

    _TOOLS_INITIALIZED = True


_initialize_tools()


def get_tool_specs(exclusion_list: list[str] = []) -> list[Dict[str, Any]]:
    """Get tool specs, optionally excluding some tools."""
    return [spec for spec in ALL_TOOL_SPECS if spec.get("name") not in exclusion_list]


# Dispatcher
def _safe_load_obj(args_json: str | None) -> Dict[str, Any]:
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
