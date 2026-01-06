"""GitHub list files tool."""

import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubListFilesTool(Tool):
    """List files in a cloned repository."""

    name = "github_list_files"
    description = (
        "List files and directories in a cloned GitHub repository. "
        "Use this to explore the structure of a repository."
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
                "description": "Optional subdirectory path within the repo (e.g., 'src/components')",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, list files recursively (default: false)",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for recursive listing (default: 3)",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """List files in a repository."""
        repo_name = kwargs.get("repo", "")
        subpath = kwargs.get("path", "")
        recursive = kwargs.get("recursive", False)
        max_depth = kwargs.get("max_depth", 3)

        logger.info(f"Tool call: github_list_files - repo='{repo_name}', path='{subpath}'")

        if not repo_name:
            return {"error": "Repository name is required."}

        # Handle owner/repo format
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {
                "error": f"Repository not found at {repo_path}",
                "hint": "Use github_clone to clone the repository first.",
            }

        # Build target path
        target_path = repo_path / subpath if subpath else repo_path

        if not target_path.exists():
            return {"error": f"Path not found: {subpath}"}

        if not target_path.is_dir():
            return {"error": f"'{subpath}' is not a directory."}

        try:
            files: List[Dict[str, Any]] = []

            if recursive:
                # Use find command for recursive listing
                cmd = ["find", str(target_path), "-maxdepth", str(max_depth), "-type", "f", "-o", "-type", "d"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line and line != str(target_path):
                            rel_path = Path(line).relative_to(repo_path)
                            is_dir = Path(line).is_dir()
                            files.append({
                                "path": str(rel_path),
                                "type": "directory" if is_dir else "file",
                            })
            else:
                # List immediate children
                for item in sorted(target_path.iterdir()):
                    # Skip hidden files and .git
                    if item.name.startswith("."):
                        continue
                    rel_path = item.relative_to(repo_path)
                    files.append({
                        "path": str(rel_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })

            # Limit results
            max_results = 100
            truncated = len(files) > max_results
            if truncated:
                files = files[:max_results]

            return {
                "status": "success",
                "repo": local_name,
                "path": subpath or "/",
                "count": len(files),
                "truncated": truncated,
                "files": files,
            }

        except subprocess.TimeoutExpired:
            return {"error": "Listing timed out. Try a smaller path or reduce max_depth."}
        except Exception as e:
            logger.exception(f"Error listing files: {e}")
            return {"error": f"Failed to list files: {str(e)}"}
