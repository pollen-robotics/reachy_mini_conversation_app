"""GitHub push tool using GitPython."""

import logging
from pathlib import Path
from typing import Any, Dict

from git import Repo, InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubPushTool(Tool):
    """Push local changes to GitHub."""

    name = "github_push"
    description = (
        "Push local commits to GitHub. "
        "IMPORTANT: Always ask user for confirmation before calling this tool. "
        "Use this when the user wants to push their local changes to the remote repository."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm push. Always ask user first.",
            },
        },
        "required": ["repo", "confirmed"],
    }

    def _get_authenticated_url(self, repo: Repo) -> str | None:
        """Get remote URL with token authentication."""
        token = config.GITHUB_TOKEN
        if not token:
            return None

        try:
            origin = repo.remotes.origin
            url = origin.url

            # Convert SSH or HTTPS URL to authenticated HTTPS
            if url.startswith("git@github.com:"):
                # SSH format: git@github.com:owner/repo.git
                repo_path = url.replace("git@github.com:", "").replace(".git", "")
                return f"https://{token}@github.com/{repo_path}.git"
            elif "github.com" in url:
                # HTTPS format: https://github.com/owner/repo.git
                # or already has token: https://token@github.com/owner/repo.git
                if "@github.com" in url:
                    # Remove existing credentials
                    url = url.split("@github.com")[1]
                    url = f"https://{token}@github.com{url}"
                else:
                    url = url.replace("https://github.com", f"https://{token}@github.com")
                return url
        except Exception:
            pass
        return None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Push local commits to remote using GitPython."""
        repo_name = kwargs.get("repo", "")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_push - repo='{repo_name}', confirmed={confirmed}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "Push not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}

        # Handle owner/repo format
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {"error": f"Repository not found at {repo_path}"}

        # Check for token (needed for push)
        token = config.GITHUB_TOKEN
        if not token:
            return {
                "error": "GITHUB_TOKEN is required for push operations. "
                "Please set it in your .env file."
            }

        try:
            # Open repository with GitPython
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            return {"error": f"'{repo_path}' is not a git repository."}

        try:
            # Get current branch
            current_branch = repo.active_branch.name

            # Check if there are commits to push
            origin = repo.remotes.origin

            # Check if branch has a tracking branch (upstream)
            tracking_branch = repo.active_branch.tracking_branch()
            needs_upstream = tracking_branch is None

            if not needs_upstream:
                # Fetch to compare with remote
                try:
                    origin.fetch()
                    commits_ahead = list(repo.iter_commits(f"{tracking_branch.name}..HEAD"))
                    if not commits_ahead:
                        return {
                            "status": "nothing_to_push",
                            "message": "No local commits to push.",
                            "path": str(repo_path),
                            "branch": current_branch,
                        }
                except GitCommandError:
                    # Remote branch might not exist yet
                    needs_upstream = True

            # Get authenticated URL and temporarily set it
            auth_url = self._get_authenticated_url(repo)
            original_url = origin.url

            if auth_url:
                # Temporarily change remote URL to authenticated version
                origin.set_url(auth_url)

            try:
                # Push to remote (with --set-upstream if needed)
                if needs_upstream:
                    # Push with upstream tracking for new branches
                    push_info = origin.push(refspec=f"{current_branch}:{current_branch}", set_upstream=True)
                else:
                    push_info = origin.push()

                # Check push result
                if push_info:
                    info = push_info[0]
                    if info.flags & info.ERROR:
                        error_msg = info.summary if hasattr(info, "summary") else "Push failed"
                        if token:
                            error_msg = error_msg.replace(token, "***")
                        return {"error": f"Push failed: {error_msg}"}
                    elif info.flags & info.REJECTED:
                        return {
                            "error": "Push rejected. Remote has changes not present locally.",
                            "hint": "Use github_pull first to get the latest changes.",
                        }

                result = {
                    "status": "success",
                    "message": "Changes pushed successfully!",
                    "path": str(repo_path),
                    "branch": current_branch,
                }

                if needs_upstream:
                    result["upstream_set"] = True
                    result["message"] = f"Branch '{current_branch}' pushed and upstream set!"

                return result

            finally:
                # Restore original URL
                if auth_url:
                    origin.set_url(original_url)

        except GitCommandError as e:
            error_msg = str(e)
            # Hide token from error messages
            if token:
                error_msg = error_msg.replace(token, "***")

            if "rejected" in error_msg.lower():
                return {
                    "error": "Push rejected. Remote has changes not present locally.",
                    "hint": "Use github_pull first to get the latest changes.",
                }
            elif "permission" in error_msg.lower() or "403" in error_msg:
                return {"error": "Permission denied. Check your GITHUB_TOKEN permissions."}
            return {"error": f"Push failed: {error_msg}"}

        except Exception as e:
            error_msg = str(e)
            if token:
                error_msg = error_msg.replace(token, "***")
            logger.exception(f"Error pushing repo: {e}")
            return {"error": f"Failed to push: {error_msg}"}
