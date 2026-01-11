"""GitHub pull tool using GitPython."""

import logging
from typing import Any, Dict
from pathlib import Path

from git import Repo, GitCommandError, InvalidGitRepositoryError

from .github_env_vars import GITHUB_ENV_VARS
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
    required_env_vars = GITHUB_ENV_VARS
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
        """Pull latest changes from remote using GitPython."""
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

        try:
            # Open repository with GitPython
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            return {"error": f"'{repo_path}' is not a git repository."}

        try:
            # Get current branch
            current_branch = repo.active_branch.name

            # Pull with fast-forward only
            origin = repo.remotes.origin
            fetch_info = origin.pull(ff_only=True)

            # Check if there were updates
            if fetch_info:
                # Get the first fetch info
                info = fetch_info[0]
                if info.flags & info.HEAD_UPTODATE:
                    return {
                        "status": "up_to_date",
                        "message": "Repository is already up to date.",
                        "path": str(repo_path),
                        "branch": current_branch,
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Repository updated successfully!",
                        "path": str(repo_path),
                        "branch": current_branch,
                        "commit": repo.head.commit.hexsha[:7],
                    }
            else:
                return {
                    "status": "up_to_date",
                    "message": "Repository is already up to date.",
                    "path": str(repo_path),
                    "branch": current_branch,
                }

        except GitCommandError as e:
            error_msg = str(e)
            if "conflict" in error_msg.lower():
                return {
                    "error": "Pull failed due to conflicts.",
                    "hint": "Resolve conflicts manually or reset the repository.",
                }
            elif "diverged" in error_msg.lower() or "non-fast-forward" in error_msg.lower():
                return {
                    "error": "Pull failed: local and remote have diverged.",
                    "hint": "You may need to merge or rebase manually.",
                }
            return {"error": f"Pull failed: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error pulling repo: {e}")
            return {"error": f"Failed to pull: {str(e)}"}
