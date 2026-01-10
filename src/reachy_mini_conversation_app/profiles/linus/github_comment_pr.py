"""GitHub comment pull request tool using PyGithub."""

import logging
from typing import Any, Dict, Optional

from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from .github_env_vars import GITHUB_ENV_VARS


logger = logging.getLogger(__name__)


class GitHubCommentPRTool(Tool):
    """Add a comment to a pull request on GitHub."""

    name = "github_comment_pr"
    description = (
        "Add a comment to an existing pull request on GitHub. "
        "Can add general comments or review comments on specific files/lines. "
        "Supports markdown formatting."
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
                "description": "Pull request number to comment on",
            },
            "body": {
                "type": "string",
                "description": "Comment body (supports markdown)",
            },
            "comment_type": {
                "type": "string",
                "enum": ["general", "review"],
                "description": "Type of comment: 'general' for issue-style comment, 'review' for code review comment",
            },
            "path": {
                "type": "string",
                "description": "File path for review comment (required if comment_type is 'review')",
            },
            "line": {
                "type": "integer",
                "description": "Line number for review comment (required if comment_type is 'review')",
            },
            "side": {
                "type": "string",
                "enum": ["LEFT", "RIGHT"],
                "description": "Side of the diff for review comment: LEFT (deletion) or RIGHT (addition). Default: RIGHT",
            },
            "start_line": {
                "type": "integer",
                "description": "Start line for multi-line review comment (optional)",
            },
        },
        "required": ["repo", "pr_number", "body"],
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
        """Add a comment to a pull request on GitHub."""
        repo_name = kwargs.get("repo", "")
        pr_number = kwargs.get("pr_number")
        body = kwargs.get("body", "")
        comment_type = kwargs.get("comment_type", "general")
        path: Optional[str] = kwargs.get("path")
        line: Optional[int] = kwargs.get("line")
        side = kwargs.get("side", "RIGHT")
        start_line: Optional[int] = kwargs.get("start_line")

        logger.info(f"Tool call: github_comment_pr - repo='{repo_name}', pr=#{pr_number}, type={comment_type}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not pr_number:
            return {"error": "Pull request number is required."}
        if not body:
            return {"error": "Comment body is required."}

        # Validate review comment parameters
        if comment_type == "review":
            if not path:
                return {"error": "File path is required for review comments."}
            if not line:
                return {"error": "Line number is required for review comments."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to comment on pull requests."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            pr = gh_repo.get_pull(pr_number)

            if comment_type == "general":
                # Add general comment (like issue comment)
                comment = pr.create_issue_comment(body)

                return {
                    "status": "success",
                    "message": f"Comment added to PR #{pr_number}!",
                    "pr_number": pr_number,
                    "pr_title": pr.title,
                    "comment_id": comment.id,
                    "comment_url": comment.html_url,
                }

            elif comment_type == "review":
                # Add review comment on specific file/line
                # Get the latest commit
                commit = pr.get_commits().reversed[0]

                # Build review comment parameters
                review_params = {
                    "body": body,
                    "commit": commit,
                    "path": path,
                    "line": line,
                    "side": side,
                }

                # Add start_line for multi-line comments
                if start_line is not None:
                    review_params["start_line"] = start_line
                    review_params["start_side"] = side

                # Create review comment
                review_comment = pr.create_review_comment(**review_params)

                result: Dict[str, Any] = {
                    "status": "success",
                    "message": f"Review comment added to PR #{pr_number}!",
                    "pr_number": pr_number,
                    "pr_title": pr.title,
                    "comment_id": review_comment.id,
                    "path": path,
                    "line": line,
                }

                if start_line:
                    result["start_line"] = start_line

                return result

            else:
                return {"error": f"Unknown comment type: {comment_type}. Use 'general' or 'review'."}

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Pull request #{pr_number} not found in '{full_repo}'."}
            elif e.status == 422:
                error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
                return {
                    "error": f"Invalid review comment: {error_msg}",
                    "hint": "Ensure the file path and line number exist in the PR diff.",
                }
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error commenting on pull request: {e}")
            return {"error": f"Failed to comment on pull request: {str(e)}"}
