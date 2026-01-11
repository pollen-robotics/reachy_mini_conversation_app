"""GitHub pull request comment tool."""

import logging
from typing import Any, Dict

from github import Github, GithubException

from .github_env_vars import GITHUB_ENV_VARS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubPRCommentTool(Tool):
    """Comment on GitHub pull requests."""

    name = "github_pr_comment"
    description = (
        "Add a comment to a GitHub pull request. "
        "Use this when the user wants to review, comment, or provide feedback on a PR."
    )
    required_env_vars = GITHUB_ENV_VARS
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "pr_number": {
                "type": "integer",
                "description": "Pull request number",
            },
            "comment": {
                "type": "string",
                "description": "Comment text (supports markdown)",
            },
        },
        "required": ["repo", "pr_number", "comment"],
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
        """Add a comment to a pull request."""
        repo_name = kwargs.get("repo", "")
        pr_number = kwargs.get("pr_number")
        comment = kwargs.get("comment", "")

        logger.info(f"Tool call: github_pr_comment - repo='{repo_name}', pr=#{pr_number}")

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
        if not pr_number:
            return {"error": "Pull request number is required."}
        if not comment:
            return {"error": "Comment text is required."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            # Connect to GitHub
            g = Github(token)
            repo = g.get_repo(full_repo)

            # Get the pull request
            pr = repo.get_pull(pr_number)

            # Add comment (using issue comment API, which works for PRs)
            issue_comment = pr.create_issue_comment(comment)

            logger.info(f"Added comment to PR #{pr_number} on {full_repo}")

            return {
                "status": "success",
                "message": f"Comment added to PR #{pr_number}",
                "pr_number": pr_number,
                "pr_title": pr.title,
                "comment_url": issue_comment.html_url,
                "repo": full_repo,
            }

        except GithubException as e:
            if e.status == 401:
                return {"error": "GitHub authentication failed. Check your GITHUB_TOKEN."}
            elif e.status == 403:
                return {"error": "Permission denied. Your token may not have 'repo' scope."}
            elif e.status == 404:
                return {"error": f"PR #{pr_number} not found in '{full_repo}'."}
            else:
                logger.exception(f"GitHub API error: {e}")
                return {"error": f"GitHub API error: {e.data.get('message', str(e))}"}
        except Exception as e:
            logger.exception(f"Error commenting on PR: {e}")
            return {"error": f"Failed to comment on PR: {str(e)}"}
