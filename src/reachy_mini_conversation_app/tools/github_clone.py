"""GitHub repository clone tool using GitPython."""

import logging
from pathlib import Path
from typing import Any, Dict

from git import Repo, GitCommandError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubCloneTool(Tool):
    """Clone GitHub repositories."""

    name = "github_clone"
    description = (
        "Clone a GitHub repository to the local machine. "
        "Use this when the user wants to download or clone a repository."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "branch": {
                "type": "string",
                "description": "Optional branch to clone (defaults to main branch)",
            },
        },
        "required": ["repo"],
    }

    def _get_full_repo_name(self, repo: str) -> str:
        """Get full repo name with owner."""
        if "/" in repo:
            return repo
        if config.GITHUB_DEFAULT_OWNER:
            return f"{config.GITHUB_DEFAULT_OWNER}/{repo}"
        raise ValueError(
            f"Repository '{repo}' must include owner (e.g., 'owner/repo') "
            "or set GITHUB_DEFAULT_OWNER in environment."
        )

    def _get_clone_url(self, full_repo: str) -> str:
        """Build clone URL with token if available."""
        token = config.GITHUB_TOKEN
        if token:
            return f"https://{token}@github.com/{full_repo}.git"
        return f"https://github.com/{full_repo}.git"

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Clone a GitHub repository using GitPython."""
        repo_name = kwargs.get("repo", "")
        branch = kwargs.get("branch")

        logger.info(f"Tool call: github_clone - repo='{repo_name}', branch={branch}")

        # Validate inputs
        if not repo_name:
            return {"error": "Repository name is required."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        # Create repos directory
        REPOS_DIR.mkdir(parents=True, exist_ok=True)

        # Extract repo name for local directory
        local_name = full_repo.split("/")[-1]
        local_path = REPOS_DIR / local_name

        # Check if already cloned
        if local_path.exists():
            return {
                "status": "exists",
                "message": f"Repository already exists at {local_path}",
                "path": str(local_path),
                "hint": "Use github_pull to update it, or delete the folder to re-clone.",
            }

        # Build clone URL
        clone_url = self._get_clone_url(full_repo)
        token = config.GITHUB_TOKEN

        try:
            # Clone using GitPython
            clone_kwargs: Dict[str, Any] = {
                "depth": 1,  # Shallow clone for speed
            }
            if branch:
                clone_kwargs["branch"] = branch

            repo = Repo.clone_from(clone_url, local_path, **clone_kwargs)

            # Configure git user if GITHUB_DEFAULT_OWNER is set
            owner = config.GITHUB_DEFAULT_OWNER
            if owner:
                email = config.GITHUB_OWNER_EMAIL or f"{owner}@users.noreply.github.com"
                with repo.config_writer() as git_config:
                    git_config.set_value("user", "name", owner)
                    git_config.set_value("user", "email", email)

            logger.info(f"Cloned {full_repo} to {local_path}")
            return {
                "status": "success",
                "message": "Repository cloned successfully!",
                "repo": full_repo,
                "path": str(local_path),
                "branch": branch or "default",
            }

        except GitCommandError as e:
            error_msg = str(e)
            # Hide token from error messages
            if token:
                error_msg = error_msg.replace(token, "***")

            if "not found" in error_msg.lower() or "404" in error_msg:
                return {"error": f"Repository '{full_repo}' not found or not accessible."}
            elif "authentication" in error_msg.lower() or "403" in error_msg:
                return {"error": "Authentication failed. Check your GITHUB_TOKEN."}
            else:
                return {"error": f"Clone failed: {error_msg}"}

        except Exception as e:
            error_msg = str(e)
            if token:
                error_msg = error_msg.replace(token, "***")
            logger.exception(f"Error cloning repo: {e}")
            return {"error": f"Failed to clone repository: {error_msg}"}
