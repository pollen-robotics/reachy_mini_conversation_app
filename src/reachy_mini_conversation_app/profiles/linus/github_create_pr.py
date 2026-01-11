"""GitHub create pull request tool using PyGithub."""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

from git import Repo, InvalidGitRepositoryError
from github import Github, GithubException

from .github_env_vars import GITHUB_ENV_VARS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubCreatePRTool(Tool):
    """Create a pull request on GitHub."""

    name = "github_create_pr"
    description = (
        "Create a pull request on GitHub from the current branch to a target branch. "
        "The current branch must be pushed to remote before creating the PR. "
        "IMPORTANT: Always ask user for confirmation before calling this tool."
    )
    required_env_vars = GITHUB_ENV_VARS
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "title": {
                "type": "string",
                "description": "Title of the pull request",
            },
            "body": {
                "type": "string",
                "description": "Description/body of the pull request (supports markdown)",
            },
            "base": {
                "type": "string",
                "description": "Target branch to merge into (e.g., 'main', 'develop'). Defaults to repo's default branch.",
            },
            "head": {
                "type": "string",
                "description": "Source branch with changes. Defaults to current branch.",
            },
            "draft": {
                "type": "boolean",
                "description": "Create as draft PR (default: false)",
            },
            "issue_number": {
                "type": "integer",
                "description": "Optional issue number to link (will add 'Closes #N' to body)",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm PR creation. Always ask user first.",
            },
        },
        "required": ["repo", "title", "confirmed"],
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

    def _get_current_branch(self, repo_name: str) -> Optional[str]:
        """Get current branch from local repo."""
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return None

        try:
            repo = Repo(repo_path)
            return repo.active_branch.name
        except (InvalidGitRepositoryError, Exception):
            return None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Create a pull request on GitHub."""
        repo_name = kwargs.get("repo", "")
        title = kwargs.get("title", "")
        body = kwargs.get("body", "")
        base = kwargs.get("base")
        head = kwargs.get("head")
        draft = kwargs.get("draft", False)
        issue_number = kwargs.get("issue_number")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_create_pr - repo='{repo_name}', title='{title}'")

        # Check confirmation
        if not confirmed:
            return {
                "error": "PR creation not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}
        if not title:
            return {"error": "PR title is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to create pull requests."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        # Get current branch if head not specified
        if not head:
            head = self._get_current_branch(repo_name)
            if not head:
                return {
                    "error": "Could not determine current branch. Please specify 'head' parameter.",
                }

        # Add issue reference to body if provided
        pr_body = body or ""
        if issue_number:
            if pr_body:
                pr_body += f"\n\nCloses #{issue_number}"
            else:
                pr_body = f"Closes #{issue_number}"

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)

            # Get default branch if base not specified
            if not base:
                base = gh_repo.default_branch

            # Check if head branch exists on remote
            try:
                gh_repo.get_branch(head)
            except GithubException as e:
                if e.status == 404:
                    return {
                        "error": f"Branch '{head}' not found on remote.",
                        "hint": "Push the branch first using github_push.",
                    }
                raise

            # Check if PR already exists
            existing_prs = gh_repo.get_pulls(state="open", head=f"{gh_repo.owner.login}:{head}", base=base)
            for pr in existing_prs:
                return {
                    "error": f"A pull request already exists for '{head}' -> '{base}'.",
                    "existing_pr": {
                        "number": pr.number,
                        "title": pr.title,
                        "url": pr.html_url,
                    },
                }

            # Create the pull request
            pr = gh_repo.create_pull(
                title=title,
                body=pr_body,
                base=base,
                head=head,
                draft=draft,
            )

            result = {
                "status": "success",
                "message": f"Pull request #{pr.number} created successfully!",
                "pr_number": pr.number,
                "title": pr.title,
                "url": pr.html_url,
                "base": base,
                "head": head,
                "draft": draft,
            }

            if issue_number:
                result["linked_issue"] = issue_number

            return result

        except GithubException as e:
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)

            if "No commits between" in error_msg:
                return {
                    "error": f"No commits between '{base}' and '{head}'.",
                    "hint": "Make sure your branch has commits that differ from the base branch.",
                }
            elif e.status == 422:
                return {
                    "error": f"Could not create PR: {error_msg}",
                    "hint": "Check that the branches exist and have different commits.",
                }
            elif e.status == 404:
                return {"error": f"Repository '{full_repo}' not found or not accessible."}

            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error creating PR: {e}")
            return {"error": f"Failed to create pull request: {str(e)}"}
