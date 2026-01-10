"""GitHub issue creation tool."""

import logging
from typing import Any, Dict, Optional

from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from .github_env_vars import GITHUB_ENV_VARS


logger = logging.getLogger(__name__)


class GitHubIssueTool(Tool):
    """Create GitHub issues."""

    name = "github_issue"
    description = (
        "Create a new issue on a GitHub repository. "
        "Use this when the user wants to report a bug, request a feature, "
        "or create any kind of issue on GitHub."
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
                "description": "Issue title",
            },
            "body": {
                "type": "string",
                "description": "Issue body/description (supports markdown)",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of labels to add to the issue",
            },
        },
        "required": ["repo", "title", "body"],
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

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Create a GitHub issue."""
        repo_name = kwargs.get("repo", "")
        title = kwargs.get("title", "")
        body = kwargs.get("body", "")
        labels: Optional[list[str]] = kwargs.get("labels")

        logger.info(f"Tool call: github_issue - repo='{repo_name}', title='{title[:50]}...'")

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {
                "error": "GITHUB_TOKEN is not configured. "
                "Please set it in your .env file to use GitHub features."
            }

        # Validate inputs
        if not repo_name:
            return {"error": "Repository name is required."}
        if not title:
            return {"error": "Issue title is required."}
        if not body:
            return {"error": "Issue body is required."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            # Connect to GitHub
            g = Github(token)
            repo = g.get_repo(full_repo)

            # Create issue
            issue = repo.create_issue(
                title=title,
                body=body,
                labels=labels or [],
            )

            logger.info(f"Created issue #{issue.number} on {full_repo}")

            return {
                "status": "success",
                "message": "Issue created successfully!",
                "issue_number": issue.number,
                "issue_url": issue.html_url,
                "title": issue.title,
                "repo": full_repo,
            }

        except GithubException as e:
            if e.status == 401:
                return {"error": "GitHub authentication failed. Check your GITHUB_TOKEN."}
            elif e.status == 403:
                return {"error": "Permission denied. Your token may not have 'repo' scope."}
            elif e.status == 404:
                return {"error": f"Repository '{full_repo}' not found or not accessible."}
            else:
                logger.exception(f"GitHub API error: {e}")
                return {"error": f"GitHub API error: {e.data.get('message', str(e))}"}
        except Exception as e:
            logger.exception(f"Error creating issue: {e}")
            return {"error": f"Failed to create issue: {str(e)}"}
