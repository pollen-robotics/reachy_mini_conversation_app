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
        "IMPORTANT: For safety, only common development commands are allowed."
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
        },
        "required": ["repo", "command"],
    }

    def _is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check if command is allowed."""
        cmd_lower = command.lower().strip()

        # Check for blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return False, f"Blocked pattern detected: {pattern}"

        # Get the base command (first word)
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command"
            base_cmd = parts[0].split("/")[-1]  # Handle full paths
        except ValueError:
            return False, "Invalid command syntax"

        # Check if base command is in whitelist
        if base_cmd in ALLOWED_COMMANDS:
            return True, ""

        # Check for compound commands with allowed base
        for allowed in ALLOWED_COMMANDS:
            if cmd_lower.startswith(allowed + " "):
                return True, ""

        return False, f"Command '{base_cmd}' not in allowed list. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute a command in a repository."""
        repo_name = kwargs.get("repo", "")
        command = kwargs.get("command", "")
        timeout = min(kwargs.get("timeout", 300), 600)
        env_vars: Dict[str, str] = kwargs.get("env", {})

        logger.info(f"Tool call: github_exec - repo='{repo_name}', command='{command}'")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not command:
            return {"error": "Command is required."}

        # Check if command is allowed
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return {
                "error": f"Command not allowed: {reason}",
                "hint": "Only common development commands are permitted for safety.",
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
