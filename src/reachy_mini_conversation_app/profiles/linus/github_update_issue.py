"""GitHub update issue tool using PyGithub."""

import logging
from typing import Any, Dict, List, Optional

from github import Github, GithubException

from .github_env_vars import GITHUB_ENV_VARS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubUpdateIssueTool(Tool):
    """Update an existing issue on GitHub."""

    name = "github_update_issue"
    description = (
        "Update an existing issue on GitHub. "
        "Can modify title, body, labels, assignees, and milestone."
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
                "description": "Issue number to update",
            },
            "title": {
                "type": "string",
                "description": "New title for the issue (optional)",
            },
            "body": {
                "type": "string",
                "description": "New body/description for the issue (optional)",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New list of labels (replaces existing labels)",
            },
            "assignees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New list of assignees (replaces existing assignees)",
            },
            "milestone": {
                "type": "integer",
                "description": "Milestone number to assign (use 0 to remove milestone)",
            },
        },
        "required": ["repo", "issue_number"],
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
        """Update an issue on GitHub."""
        repo_name = kwargs.get("repo", "")
        issue_number = kwargs.get("issue_number")
        title = kwargs.get("title")
        body = kwargs.get("body")
        labels: Optional[List[str]] = kwargs.get("labels")
        assignees: Optional[List[str]] = kwargs.get("assignees")
        milestone = kwargs.get("milestone")

        logger.info(f"Tool call: github_update_issue - repo='{repo_name}', issue=#{issue_number}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not issue_number:
            return {"error": "Issue number is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to update issues."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            issue = gh_repo.get_issue(issue_number)

            # Track what was updated
            updates = []

            # Update title
            if title is not None:
                issue.edit(title=title)
                updates.append("title")

            # Update body
            if body is not None:
                issue.edit(body=body)
                updates.append("body")

            # Update labels
            if labels is not None:
                issue.set_labels(*labels)
                updates.append("labels")

            # Update assignees
            if assignees is not None:
                # Clear existing and add new
                for assignee in issue.assignees:
                    issue.remove_from_assignees(assignee)
                if assignees:
                    issue.add_to_assignees(*assignees)
                updates.append("assignees")

            # Update milestone
            if milestone is not None:
                if milestone == 0:
                    issue.edit(milestone=None)
                else:
                    ms = gh_repo.get_milestone(milestone)
                    issue.edit(milestone=ms)
                updates.append("milestone")

            if not updates:
                return {
                    "status": "no_changes",
                    "message": "No updates provided.",
                    "issue_number": issue_number,
                    "url": issue.html_url,
                }

            return {
                "status": "success",
                "message": f"Issue #{issue_number} updated successfully!",
                "issue_number": issue_number,
                "title": issue.title,
                "url": issue.html_url,
                "updated_fields": updates,
            }

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Issue #{issue_number} not found in '{full_repo}'."}
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error updating issue: {e}")
            return {"error": f"Failed to update issue: {str(e)}"}
