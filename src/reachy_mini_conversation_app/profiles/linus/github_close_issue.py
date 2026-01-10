"""GitHub close issue tool using PyGithub."""

import logging
from typing import Any, Dict

from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubCloseIssueTool(Tool):
    """Close or reopen an issue on GitHub."""

    name = "github_close_issue"
    description = (
        "Close or reopen an issue on GitHub. "
        "Can optionally add a comment when closing. "
        "IMPORTANT: Always ask user for confirmation before closing an issue."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number to close or reopen",
            },
            "action": {
                "type": "string",
                "enum": ["close", "reopen"],
                "description": "Action to perform: close or reopen the issue",
            },
            "reason": {
                "type": "string",
                "enum": ["completed", "not_planned"],
                "description": "Reason for closing (only for close action). 'completed' for resolved issues, 'not_planned' for won't fix.",
            },
            "comment": {
                "type": "string",
                "description": "Optional comment to add when closing/reopening",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm action. Always ask user first.",
            },
        },
        "required": ["repo", "issue_number", "action", "confirmed"],
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
        """Close or reopen an issue on GitHub."""
        repo_name = kwargs.get("repo", "")
        issue_number = kwargs.get("issue_number")
        action = kwargs.get("action", "close")
        reason = kwargs.get("reason", "completed")
        comment = kwargs.get("comment")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_close_issue - repo='{repo_name}', issue=#{issue_number}, action={action}")

        # Check confirmation
        if not confirmed:
            return {
                "error": f"Issue {action} not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}
        if not issue_number:
            return {"error": "Issue number is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to close/reopen issues."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            issue = gh_repo.get_issue(issue_number)

            # Add comment if provided
            if comment:
                issue.create_comment(comment)

            if action == "close":
                # Close the issue with reason
                if reason == "not_planned":
                    issue.edit(state="closed", state_reason="not_planned")
                else:
                    issue.edit(state="closed", state_reason="completed")

                return {
                    "status": "success",
                    "message": f"Issue #{issue_number} closed successfully!",
                    "issue_number": issue_number,
                    "title": issue.title,
                    "url": issue.html_url,
                    "reason": reason,
                    "comment_added": comment is not None,
                }

            elif action == "reopen":
                issue.edit(state="open")

                return {
                    "status": "success",
                    "message": f"Issue #{issue_number} reopened successfully!",
                    "issue_number": issue_number,
                    "title": issue.title,
                    "url": issue.html_url,
                    "comment_added": comment is not None,
                }

            else:
                return {"error": f"Unknown action: {action}. Use 'close' or 'reopen'."}

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Issue #{issue_number} not found in '{full_repo}'."}
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error closing/reopening issue: {e}")
            return {"error": f"Failed to {action} issue: {str(e)}"}
