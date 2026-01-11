"""GitHub comment issue tool using PyGithub."""

import logging
from typing import Any, Dict

from github import Github, GithubException

from .github_env_vars import GITHUB_ENV_VARS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubCommentIssueTool(Tool):
    """Add a comment to an issue on GitHub."""

    name = "github_comment_issue"
    description = (
        "Add a comment to an existing issue on GitHub. "
        "Supports markdown formatting in the comment body."
    )
    required_env_vars = GITHUB_ENV_VARS
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number to comment on",
            },
            "body": {
                "type": "string",
                "description": "Comment body (supports markdown)",
            },
        },
        "required": ["repo", "issue_number", "body"],
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
        """Add a comment to an issue on GitHub."""
        repo_name = kwargs.get("repo", "")
        issue_number = kwargs.get("issue_number")
        body = kwargs.get("body", "")

        logger.info(f"Tool call: github_comment_issue - repo='{repo_name}', issue=#{issue_number}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not issue_number:
            return {"error": "Issue number is required."}
        if not body:
            return {"error": "Comment body is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to comment on issues."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            issue = gh_repo.get_issue(issue_number)

            # Create comment
            comment = issue.create_comment(body)

            return {
                "status": "success",
                "message": f"Comment added to issue #{issue_number}!",
                "issue_number": issue_number,
                "issue_title": issue.title,
                "comment_id": comment.id,
                "comment_url": comment.html_url,
            }

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Issue #{issue_number} not found in '{full_repo}'."}
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error commenting on issue: {e}")
            return {"error": f"Failed to comment on issue: {str(e)}"}
