"""Execute code tool with safety measures."""

import os
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where code is generated
CODE_OUTPUT_DIR = Path.home() / "reachy_code"

# Allowed extensions for execution
ALLOWED_EXTENSIONS = {".py", ".sh"}

# Maximum execution time in seconds
MAX_EXECUTION_TIME = 30


class ExecuteCodeTool(Tool):
    """Execute generated code with user confirmation."""

    name = "execute_code"
    description = (
        "Execute a previously generated code file. "
        "IMPORTANT: Always ask user for confirmation before calling this tool. "
        "Only Python (.py) and Shell (.sh) scripts can be executed. "
        "The code runs in a sandboxed environment with a timeout."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "The full path to the code file to execute",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm execution. Always ask user first.",
            },
        },
        "required": ["filepath", "confirmed"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the specified code file."""
        filepath = kwargs.get("filepath", "")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: execute_code - filepath='{filepath}', confirmed={confirmed}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "Execution not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        # Validate filepath
        if not filepath:
            return {"error": "No filepath provided."}

        path = Path(filepath)

        # Security check: must be in CODE_OUTPUT_DIR
        try:
            path.resolve().relative_to(CODE_OUTPUT_DIR.resolve())
        except ValueError:
            return {
                "error": f"Security error: Can only execute files in {CODE_OUTPUT_DIR}",
                "filepath": filepath,
            }

        # Check file exists
        if not path.exists():
            return {"error": f"File not found: {filepath}"}

        # Check extension
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return {
                "error": f"Cannot execute files with extension '{path.suffix}'. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            }

        # Determine command based on extension
        # Note: Only .py and .sh are in ALLOWED_EXTENSIONS, checked above
        if path.suffix.lower() == ".py":
            cmd = ["python3", str(path)]
        else:  # .sh - only other allowed extension
            cmd = ["bash", str(path)]

        try:
            # Execute with timeout and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=MAX_EXECUTION_TIME,
                cwd=CODE_OUTPUT_DIR,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            output = result.stdout
            error_output = result.stderr
            return_code = result.returncode

            # Truncate output if too long
            max_output = 2000
            if len(output) > max_output:
                output = output[:max_output] + "\n... (output truncated)"
            if len(error_output) > max_output:
                error_output = error_output[:max_output] + "\n... (error output truncated)"

            if return_code == 0:
                return {
                    "status": "success",
                    "return_code": return_code,
                    "output": output or "(no output)",
                    "stderr": error_output if error_output else None,
                }
            else:
                return {
                    "status": "error",
                    "return_code": return_code,
                    "output": output,
                    "stderr": error_output or "(no error message)",
                }

        except subprocess.TimeoutExpired:
            return {
                "error": f"Execution timed out after {MAX_EXECUTION_TIME} seconds.",
                "hint": "The code may have an infinite loop or is taking too long.",
            }
        except PermissionError:
            return {"error": "Permission denied. Cannot execute this file."}
        except Exception as e:
            logger.exception(f"Error executing code: {e}")
            return {"error": f"Execution failed: {str(e)}"}
