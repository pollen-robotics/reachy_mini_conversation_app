"""ReachyMiniScript - A kid-friendly programming language for Reachy Mini."""

import inspect
import logging
from typing import Any, Dict
from pathlib import Path

from reachy_mini_conversation_app.rmscript.errors import CompiledTool, CompilationError
from reachy_mini_conversation_app.rmscript.compiler import ReachyMiniScriptCompiler


logger = logging.getLogger(__name__)


def create_tool_from_rmscript(filename: str) -> type:
    """Dynamically create a Tool class from an rmscript file.

    This factory function compiles a .rmscript file and creates a Tool subclass
    that can be automatically discovered by the tool collection system. The class
    inherits from Tool and implements all required methods.

    Args:
        filename: Relative path to .rmscript file from the caller's directory
                 (e.g., "look_around.rmscript")

    Returns:
        A Tool subclass that will be automatically registered in ALL_TOOLS

    Example:
        # In your profile's __init__.py:
        from reachy_mini_conversation_app.rmscript import create_tool_from_rmscript

        # This creates and registers the tool automatically
        LookAround = create_tool_from_rmscript("look_around.rmscript")

        # Or even simpler (no variable needed):
        create_tool_from_rmscript("look_around.rmscript")

    """
    # Import here to avoid circular dependency
    from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

    # Get caller's directory using inspect
    caller_frame = inspect.currentframe()
    if caller_frame is None:
        raise RuntimeError("Cannot determine caller's context")

    caller_frame = caller_frame.f_back
    if caller_frame is None:
        raise RuntimeError("Cannot determine caller's context")

    caller_file = caller_frame.f_globals.get("__file__")
    if caller_file is None:
        raise RuntimeError("Cannot determine caller's file path")

    caller_dir = Path(caller_file).parent
    script_path = caller_dir / filename

    # Compile the rmscript file
    logger.debug(f"Compiling rmscript from {script_path}")
    compiler = ReachyMiniScriptCompiler()

    try:
        compiled = compiler.compile_file(str(script_path))
    except Exception as e:
        logger.error(f"Failed to read or compile {script_path}: {e}")
        raise

    # Log compilation results
    if not compiled.success:
        logger.warning(f"rmscript '{filename}' compiled with errors:")
        for error in compiled.errors:
            logger.warning(f"  {error}")
    elif compiled.warnings:
        logger.info(f"rmscript '{filename}' compiled with warnings:")
        for warning in compiled.warnings:
            logger.info(f"  {warning}")
    else:
        logger.info(f"✓ rmscript '{filename}' compiled successfully as '{compiled.name}'")

    # Create the __call__ method
    if compiled.success:

        async def __call__(
            self: Tool, deps: ToolDependencies, **kwargs: Any
        ) -> Dict[str, Any]:
            """Execute the rmscript tool."""
            return compiled.execute_queued(deps)

    else:
        # Create a __call__ that returns compilation errors
        async def __call__(
            self: Tool, deps: ToolDependencies, **kwargs: Any
        ) -> Dict[str, Any]:
            """Return compilation errors."""
            error_messages = "\n".join(str(e) for e in compiled.errors)
            return {"error": f"Tool compilation failed:\n{error_messages}"}

    # Use tool name from script, or filename as fallback
    tool_name = compiled.name if compiled.name else Path(filename).stem

    # Create the dynamic Tool class using type()
    ToolClass = type(
        tool_name,  # Class name
        (Tool,),  # Base classes
        {
            "name": tool_name,
            "description": compiled.description or f"Tool from {filename}",
            "parameters_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
            "__call__": __call__,
            "__module__": caller_frame.f_globals.get("__name__", "__main__"),
            "_compiled_tool": compiled,  # Store for debugging
        },
    )

    logger.debug(
        f"Created Tool class '{tool_name}' from {filename} in module {ToolClass.__module__}"
    )

    return ToolClass


__all__ = [
    "ReachyMiniScriptCompiler",
    "CompilationError",
    "CompiledTool",
    "create_tool_from_rmscript",
]
