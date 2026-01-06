"""GitHub pull tool."""

import subprocess
import logging
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubPullTool(Tool):
    """Pull latest changes from GitHub."""

    name = "github_pull"
    description = (
        "Pull the latest changes from a GitHub repository. "
        "Use this when the user wants to update a local repository with remote changes."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Pull latest changes from remote."""
        repo_name = kwargs.get("repo", "")

        logger.info(f"Tool call: github_pull - repo='{repo_name}'")

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

        if not (repo_path / ".git").exists():
            return {"error": f"'{repo_path}' is not a git repository."}

        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_path,
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if "Already up to date" in output:
                    return {
                        "status": "up_to_date",
                        "message": "Repository is already up to date.",
                        "path": str(repo_path),
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Repository updated successfully!",
                        "path": str(repo_path),
                        "details": output,
                    }
            else:
                error_msg = result.stderr.strip()
                if "conflict" in error_msg.lower():
                    return {
                        "error": "Pull failed due to conflicts.",
                        "hint": "Resolve conflicts manually or reset the repository.",
                    }
                return {"error": f"Pull failed: {error_msg}"}

        except subprocess.TimeoutExpired:
            return {"error": "Pull timed out."}
        except Exception as e:
            logger.exception(f"Error pulling repo: {e}")
            return {"error": f"Failed to pull: {str(e)}"}
