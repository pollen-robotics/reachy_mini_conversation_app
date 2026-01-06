"""GitHub write file tool - write content directly to a file in a repository."""

import logging
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubWriteFileTool(Tool):
    """Write content directly to a file in a repository."""

    name = "github_write_file"
    description = (
        "Write content directly to a file in a local repository. "
        "Use this tool to CREATE new files or MODIFY existing files. "
        "For modifying existing files: first read the file with github_read_file, "
        "then write the modified content with this tool. "
        "DO NOT use the 'code' tool to modify existing files - use this tool instead."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "path": {
                "type": "string",
                "description": "Path to the file within the repo (e.g., 'src/main.py')",
            },
            "content": {
                "type": "string",
                "description": "The full content to write to the file",
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, overwrite existing file. If false, fail if file exists. Default: true",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "If true, create parent directories if they don't exist. Default: true",
            },
        },
        "required": ["repo", "path", "content"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Write content to a file."""
        repo_name = kwargs.get("repo", "")
        file_path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        overwrite = kwargs.get("overwrite", True)
        create_dirs = kwargs.get("create_dirs", True)

        logger.info(f"Tool call: github_write_file - repo='{repo_name}', path='{file_path}'")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not file_path:
            return {"error": "File path is required."}
        if content is None:
            return {"error": "Content is required (can be empty string)."}

        # Repository path
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {
                "error": f"Repository not found: {local_name}",
                "hint": "Use github_clone to clone the repository first.",
            }

        # Build full file path
        full_path = repo_path / file_path

        # Security check: ensure path is within repo
        try:
            full_path.resolve().relative_to(repo_path.resolve())
        except ValueError:
            return {"error": "Invalid path: cannot write outside the repository."}

        # Check if file exists
        file_exists = full_path.exists()
        if file_exists and not overwrite:
            return {
                "error": f"File already exists: {file_path}",
                "hint": "Set overwrite=true to replace it.",
            }

        # Create parent directories if needed
        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        elif not full_path.parent.exists():
            return {
                "error": f"Parent directory does not exist: {full_path.parent.relative_to(repo_path)}",
                "hint": "Set create_dirs=true to create parent directories.",
            }

        try:
            # Write the file
            full_path.write_text(content, encoding="utf-8")

            # Count lines
            lines = len(content.splitlines())

            action = "modified" if file_exists else "created"
            logger.info(f"File {action}: {full_path}")

            return {
                "status": "success",
                "action": action,
                "repo": local_name,
                "path": file_path,
                "lines": lines,
                "message": f"File {action} successfully: {file_path}",
                "hint": "Use github_add to stage the file, then github_commit to commit.",
            }

        except Exception as e:
            logger.exception(f"Error writing file: {e}")
            return {"error": f"Failed to write file: {str(e)}"}
