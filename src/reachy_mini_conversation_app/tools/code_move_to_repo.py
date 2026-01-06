"""Move generated code to a repository."""

import shutil
import logging
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directories
CODE_OUTPUT_DIR = Path.home() / "reachy_code"
REPOS_DIR = Path.home() / "reachy_repos"


class CodeMoveToRepoTool(Tool):
    """Move generated code file to a repository."""

    name = "code_move_to_repo"
    description = (
        "Move a generated code file from ~/reachy_code/ to a cloned repository in ~/reachy_repos/. "
        "Use this after generating code to add it to a repository."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The filename in ~/reachy_code/ to move (e.g., '20240101_120000_fibonacci.py')",
            },
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "dest_path": {
                "type": "string",
                "description": "Destination path within the repo (e.g., 'src/utils/fibonacci.py'). If just a directory, the original filename is kept.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, overwrite existing file (default: false)",
            },
        },
        "required": ["filename", "repo", "dest_path"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move a generated code file to a repository."""
        filename = kwargs.get("filename", "")
        repo_name = kwargs.get("repo", "")
        dest_path = kwargs.get("dest_path", "")
        overwrite = kwargs.get("overwrite", False)

        logger.info(f"Tool call: code_move_to_repo - file='{filename}', repo='{repo_name}', dest='{dest_path}'")

        if not filename:
            return {"error": "Filename is required."}
        if not repo_name:
            return {"error": "Repository name is required."}
        if not dest_path:
            return {"error": "Destination path is required."}

        # Source file
        source_file = CODE_OUTPUT_DIR / filename
        if not source_file.exists():
            # Try to find a matching file
            matches = list(CODE_OUTPUT_DIR.glob(f"*{filename}*"))
            if matches:
                return {
                    "error": f"File not found: {filename}",
                    "hint": f"Did you mean one of: {[m.name for m in matches[:5]]}",
                }
            return {"error": f"File not found: {filename}"}

        # Validate source is in CODE_OUTPUT_DIR
        try:
            source_file.resolve().relative_to(CODE_OUTPUT_DIR.resolve())
        except ValueError:
            return {"error": "Invalid source file path."}

        # Repository
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {
                "error": f"Repository not found: {local_name}",
                "hint": "Use github_clone to clone the repository first.",
            }

        # Destination
        dest_full = repo_path / dest_path

        # If dest_path is a directory, append the original filename
        if dest_full.exists() and dest_full.is_dir():
            dest_full = dest_full / source_file.name
        elif not dest_full.suffix:
            # No extension, treat as directory
            dest_full.mkdir(parents=True, exist_ok=True)
            dest_full = dest_full / source_file.name

        # Validate destination is within repo
        try:
            dest_full.resolve().relative_to(repo_path.resolve())
        except ValueError:
            return {"error": "Invalid destination: cannot write outside the repository."}

        # Check if destination exists
        if dest_full.exists() and not overwrite:
            return {
                "error": f"Destination file already exists: {dest_path}",
                "hint": "Set overwrite=true to replace it.",
            }

        try:
            # Create parent directories if needed
            dest_full.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            shutil.move(str(source_file), str(dest_full))

            logger.info(f"Moved {source_file} to {dest_full}")

            return {
                "status": "success",
                "message": f"File moved successfully!",
                "source": str(source_file),
                "destination": str(dest_full),
                "repo": local_name,
                "relative_path": str(dest_full.relative_to(repo_path)),
                "hint": "Use github_commit to commit this change.",
            }

        except Exception as e:
            logger.exception(f"Error moving file: {e}")
            return {"error": f"Failed to move file: {str(e)}"}
