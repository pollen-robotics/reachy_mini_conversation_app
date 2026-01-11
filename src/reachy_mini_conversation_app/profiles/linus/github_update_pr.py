"""GitHub update pull request tool using PyGithub."""

import logging
from typing import Any, Dict, List, Optional

from github import Github, GithubException

from .github_env_vars import GITHUB_ENV_VARS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubUpdatePRTool(Tool):
    """Update an existing pull request on GitHub."""

    name = "github_update_pr"
    description = (
        "Update an existing pull request on GitHub. "
        "Can modify title, body, labels, assignees, reviewers, and milestone."
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
                "description": "Pull request number to update",
            },
            "title": {
                "type": "string",
                "description": "New title for the PR (optional)",
            },
            "body": {
                "type": "string",
                "description": "New body/description for the PR (optional)",
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
            "reviewers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Request review from these users (adds to existing)",
            },
            "milestone": {
                "type": "integer",
                "description": "Milestone number to assign (use 0 to remove milestone)",
            },
            "draft": {
                "type": "boolean",
                "description": "Convert to draft (true) or ready for review (false)",
            },
        },
        "required": ["repo", "pr_number"],
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
        """Update a pull request on GitHub."""
        repo_name = kwargs.get("repo", "")
        pr_number = kwargs.get("pr_number")
        title = kwargs.get("title")
        body = kwargs.get("body")
        labels: Optional[List[str]] = kwargs.get("labels")
        assignees: Optional[List[str]] = kwargs.get("assignees")
        reviewers: Optional[List[str]] = kwargs.get("reviewers")
        milestone = kwargs.get("milestone")
        draft = kwargs.get("draft")

        logger.info(f"Tool call: github_update_pr - repo='{repo_name}', pr=#{pr_number}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not pr_number:
            return {"error": "Pull request number is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to update pull requests."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            pr = gh_repo.get_pull(pr_number)

            # Track what was updated
            updates = []

            # Update title
            if title is not None:
                pr.edit(title=title)
                updates.append("title")

            # Update body
            if body is not None:
                pr.edit(body=body)
                updates.append("body")

            # Update labels (PR uses issue labels)
            if labels is not None:
                pr.set_labels(*labels)
                updates.append("labels")

            # Update assignees
            if assignees is not None:
                # Get the issue associated with the PR to manage assignees
                issue = gh_repo.get_issue(pr_number)
                for assignee in issue.assignees:
                    issue.remove_from_assignees(assignee)
                if assignees:
                    issue.add_to_assignees(*assignees)
                updates.append("assignees")

            # Request reviewers
            if reviewers is not None and reviewers:
                pr.create_review_request(reviewers=reviewers)
                updates.append("reviewers")

            # Update milestone
            if milestone is not None:
                issue = gh_repo.get_issue(pr_number)
                if milestone == 0:
                    issue.edit(milestone=None)
                else:
                    ms = gh_repo.get_milestone(milestone)
                    issue.edit(milestone=ms)
                updates.append("milestone")

            # Convert draft status
            if draft is not None:
                if draft and not pr.draft:
                    # Convert to draft - requires GraphQL API, not available in PyGithub
                    return {
                        "error": "Converting to draft requires GitHub GraphQL API, not supported.",
                        "hint": "Create the PR as draft initially using draft=true in github_create_pr.",
                    }
                elif not draft and pr.draft:
                    # Mark as ready for review
                    # This also requires GraphQL API in PyGithub
                    return {
                        "error": "Marking ready for review requires GitHub GraphQL API, not supported.",
                        "hint": "Use the GitHub web interface to mark as ready for review.",
                    }

            if not updates:
                return {
                    "status": "no_changes",
                    "message": "No updates provided.",
                    "pr_number": pr_number,
                    "url": pr.html_url,
                }

            return {
                "status": "success",
                "message": f"Pull request #{pr_number} updated successfully!",
                "pr_number": pr_number,
                "title": pr.title,
                "url": pr.html_url,
                "updated_fields": updates,
            }

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Pull request #{pr_number} not found in '{full_repo}'."}
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error updating pull request: {e}")
            return {"error": f"Failed to update pull request: {str(e)}"}
