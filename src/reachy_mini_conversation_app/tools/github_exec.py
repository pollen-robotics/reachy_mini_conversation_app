"""GitHub exec tool - execute commands in a local repository."""

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any, Dict, List

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"

# Allowed commands whitelist (for safety)
ALLOWED_COMMANDS = {
    # Package managers
    "npm", "yarn", "pnpm", "pip", "uv", "poetry", "cargo", "go",
    # Build tools
    "make", "cmake", "gradle", "mvn",
    # Testing
    "pytest", "jest", "mocha", "cargo test", "go test",
    # Linting/formatting
    "ruff", "black", "isort", "prettier", "eslint", "flake8", "mypy",
    # Other dev tools
    "python", "node", "ruby", "php",
    "cat", "ls", "head", "tail", "grep", "find", "wc",
    "docker", "docker-compose",
}

# Blocked patterns (dangerous commands)
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "> /dev/",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # Fork bomb
    "chmod -R 777 /",
    "curl | sh",
    "wget | sh",
    "curl | bash",
    "wget | bash",
]


class GitHubExecTool(Tool):
    """Execute commands in a local repository."""

    name = "github_exec"
    description = (
        "Execute a shell command in a local repository directory. "
        "Useful for running tests, builds, linting, or other development commands. "
        "Commands are executed in the repository's root directory. "
        "Common dev commands are allowed by default. Other commands require confirmed=true."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "command": {
                "type": "string",
                "description": "Command to execute (e.g., 'pytest', 'npm test', 'make build')",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 300, max: 600)",
            },
            "env": {
                "type": "object",
                "description": "Additional environment variables to set",
                "additionalProperties": {"type": "string"},
            },
            "confirmed": {
                "type": "boolean",
                "description": "Required for non-whitelisted commands. Always ask user first for unknown commands.",
            },
        },
        "required": ["repo", "command"],
    }

    def _check_command(self, command: str) -> tuple[str, str]:
        """
        Check command safety status.
        Returns: (status, reason)
        - status: "allowed" (whitelisted), "blocked" (dangerous), "requires_confirmation" (not whitelisted)
        """
        cmd_lower = command.lower().strip()

        # Check for blocked patterns (always blocked, even with confirmation)
        for pattern in BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return "blocked", f"Dangerous pattern detected: {pattern}"

        # Get the base command (first word)
        try:
            parts = shlex.split(command)
            if not parts:
                return "blocked", "Empty command"
            base_cmd = parts[0].split("/")[-1]  # Handle full paths
        except ValueError:
            return "blocked", "Invalid command syntax"

        # Check if base command is in whitelist
        if base_cmd in ALLOWED_COMMANDS:
            return "allowed", ""

        # Check for compound commands with allowed base
        for allowed in ALLOWED_COMMANDS:
            if cmd_lower.startswith(allowed + " "):
                return "allowed", ""

        # Not in whitelist, but can be run with confirmation
        return "requires_confirmation", f"Command '{base_cmd}' is not in the whitelist"

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute a command in a repository."""
        repo_name = kwargs.get("repo", "")
        command = kwargs.get("command", "")
        timeout = min(kwargs.get("timeout", 300), 600)
        env_vars: Dict[str, str] = kwargs.get("env", {})
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_exec - repo='{repo_name}', command='{command}', confirmed={confirmed}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not command:
            return {"error": "Command is required."}

        # Check command safety
        status, reason = self._check_command(command)

        if status == "blocked":
            # Dangerous commands are always blocked
            return {
                "error": f"Command blocked: {reason}",
                "hint": "This command pattern is blocked for safety reasons.",
            }

        if status == "requires_confirmation" and not confirmed:
            # Non-whitelisted commands need confirmation
            return {
                "status": "confirmation_required",
                "message": f"{reason}. User confirmation required to execute.",
                "command": command,
                "hint": "Ask the user for confirmation, then set confirmed=true to execute.",
            }

        # Repository path
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {"error": f"Repository not found: {local_name}"}

        if not repo_path.is_dir():
            return {"error": f"'{local_name}' is not a directory."}

        try:
            # Prepare environment
            import os
            env = os.environ.copy()
            env.update(env_vars)

            # Execute command with bash -l to load .bashrc/.profile
            # Using bash -l -c to run as login shell (loads ~/.profile, ~/.bashrc)
            escaped_command = command.replace("'", "'\"'\"'")
            bash_command = f"bash -l -c '{escaped_command}'"

            process = await asyncio.create_subprocess_shell(
                bash_command,
                cwd=str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "error": f"Command timed out after {timeout} seconds.",
                    "command": command,
                    "repo": local_name,
                    "hint": "Increase timeout or simplify the command.",
                }

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate if too long
            max_output = 20000
            stdout_truncated = False
            stderr_truncated = False

            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... [output truncated]"
                stdout_truncated = True

            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... [output truncated]"
                stderr_truncated = True

            result: Dict[str, Any] = {
                "status": "success" if process.returncode == 0 else "failed",
                "exit_code": process.returncode,
                "command": command,
                "repo": local_name,
                "cwd": str(repo_path),
            }

            if stdout_str.strip():
                result["stdout"] = stdout_str
                if stdout_truncated:
                    result["stdout_truncated"] = True

            if stderr_str.strip():
                result["stderr"] = stderr_str
                if stderr_truncated:
                    result["stderr_truncated"] = True

            if process.returncode != 0:
                result["message"] = f"Command failed with exit code {process.returncode}"
            else:
                result["message"] = "Command executed successfully"

            return result

        except Exception as e:
            logger.exception(f"Error executing command: {e}")
            return {"error": f"Failed to execute command: {str(e)}"}
