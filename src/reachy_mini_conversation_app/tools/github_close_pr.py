"""GitHub close pull request tool using PyGithub."""

import logging
from typing import Any, Dict

from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubClosePRTool(Tool):
    """Close or merge a pull request on GitHub."""

    name = "github_close_pr"
    description = (
        "Close or merge a pull request on GitHub. "
        "Can close without merging, or merge with different strategies. "
        "IMPORTANT: Always ask user for confirmation before closing or merging."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "pr_number": {
                "type": "integer",
                "description": "Pull request number to close or merge",
            },
            "action": {
                "type": "string",
                "enum": ["close", "merge", "reopen"],
                "description": "Action to perform: close (without merge), merge, or reopen",
            },
            "merge_method": {
                "type": "string",
                "enum": ["merge", "squash", "rebase"],
                "description": "Merge method (only for merge action). Default: merge",
            },
            "commit_title": {
                "type": "string",
                "description": "Custom commit title for merge (optional)",
            },
            "commit_message": {
                "type": "string",
                "description": "Custom commit message for merge (optional)",
            },
            "comment": {
                "type": "string",
                "description": "Optional comment to add when closing/merging",
            },
            "delete_branch": {
                "type": "boolean",
                "description": "Delete the head branch after merge (default: false)",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm action. Always ask user first.",
            },
        },
        "required": ["repo", "pr_number", "action", "confirmed"],
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
        """Close or merge a pull request on GitHub."""
        repo_name = kwargs.get("repo", "")
        pr_number = kwargs.get("pr_number")
        action = kwargs.get("action", "close")
        merge_method = kwargs.get("merge_method", "merge")
        commit_title = kwargs.get("commit_title")
        commit_message = kwargs.get("commit_message")
        comment = kwargs.get("comment")
        delete_branch = kwargs.get("delete_branch", False)
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_close_pr - repo='{repo_name}', pr=#{pr_number}, action={action}")

        # Check confirmation
        if not confirmed:
            return {
                "error": f"PR {action} not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}
        if not pr_number:
            return {"error": "Pull request number is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to close/merge pull requests."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            pr = gh_repo.get_pull(pr_number)

            # Add comment if provided
            if comment:
                pr.create_issue_comment(comment)

            if action == "close":
                # Close without merging
                pr.edit(state="closed")

                return {
                    "status": "success",
                    "message": f"Pull request #{pr_number} closed without merging.",
                    "pr_number": pr_number,
                    "title": pr.title,
                    "url": pr.html_url,
                    "comment_added": comment is not None,
                }

            elif action == "merge":
                # Check if mergeable
                if not pr.mergeable:
                    return {
                        "error": f"Pull request #{pr_number} cannot be merged.",
                        "mergeable_state": pr.mergeable_state,
                        "hint": "Resolve conflicts or address failing checks first.",
                    }

                # Merge the PR
                merge_result = pr.merge(
                    commit_title=commit_title,
                    commit_message=commit_message,
                    merge_method=merge_method,
                )

                result = {
                    "status": "success",
                    "message": f"Pull request #{pr_number} merged successfully!",
                    "pr_number": pr_number,
                    "title": pr.title,
                    "url": pr.html_url,
                    "merge_method": merge_method,
                    "sha": merge_result.sha,
                    "comment_added": comment is not None,
                }

                # Delete branch if requested
                if delete_branch:
                    try:
                        head_ref = pr.head.ref
                        # Only delete if it's in the same repo (not a fork)
                        if pr.head.repo.full_name == full_repo:
                            ref = gh_repo.get_git_ref(f"heads/{head_ref}")
                            ref.delete()
                            result["branch_deleted"] = head_ref
                    except GithubException as e:
                        result["branch_delete_error"] = str(e)

                return result

            elif action == "reopen":
                pr.edit(state="open")

                return {
                    "status": "success",
                    "message": f"Pull request #{pr_number} reopened successfully!",
                    "pr_number": pr_number,
                    "title": pr.title,
                    "url": pr.html_url,
                    "comment_added": comment is not None,
                }

            else:
                return {"error": f"Unknown action: {action}. Use 'close', 'merge', or 'reopen'."}

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Pull request #{pr_number} not found in '{full_repo}'."}
            elif e.status == 405:
                return {
                    "error": "Merge not allowed.",
                    "hint": "Check if the PR is already merged, closed, or has merge restrictions.",
                }
            elif e.status == 409:
                return {
                    "error": "Merge conflict.",
                    "hint": "Resolve conflicts before merging.",
                }
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error closing/merging pull request: {e}")
            return {"error": f"Failed to {action} pull request: {str(e)}"}
