"""GitHub merge pull request tool using PyGithub."""

import logging
from typing import Any, Dict

from github import Github, GithubException

from .github_env_vars import GITHUB_ENV_VARS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubMergePRTool(Tool):
    """Merge a pull request on GitHub."""

    name = "github_merge_pr"
    description = (
        "Merge a pull request on GitHub. Supports merge, squash, and rebase strategies. "
        "The PR must be in a mergeable state (no conflicts, checks passed if required). "
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
            "pr_number": {
                "type": "integer",
                "description": "Pull request number to merge",
            },
            "merge_method": {
                "type": "string",
                "enum": ["merge", "squash", "rebase"],
                "description": "Merge strategy: 'merge' (default), 'squash', or 'rebase'",
            },
            "commit_title": {
                "type": "string",
                "description": "Custom commit title (for merge/squash). Defaults to PR title.",
            },
            "commit_message": {
                "type": "string",
                "description": "Custom commit message (for merge/squash). Defaults to PR body.",
            },
            "delete_branch": {
                "type": "boolean",
                "description": "Delete the head branch after merge (default: true)",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm merge. Always ask user first.",
            },
        },
        "required": ["repo", "pr_number", "confirmed"],
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
        """Merge a pull request on GitHub."""
        repo_name = kwargs.get("repo", "")
        pr_number = kwargs.get("pr_number")
        merge_method = kwargs.get("merge_method", "merge")
        commit_title = kwargs.get("commit_title")
        commit_message = kwargs.get("commit_message")
        delete_branch = kwargs.get("delete_branch", True)
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_merge_pr - repo='{repo_name}', pr=#{pr_number}, method={merge_method}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "Merge not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}
        if not pr_number:
            return {"error": "PR number is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to merge pull requests."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)

            # Get the pull request
            try:
                pr = gh_repo.get_pull(pr_number)
            except GithubException as e:
                if e.status == 404:
                    return {"error": f"Pull request #{pr_number} not found."}
                raise

            # Check PR state
            if pr.state != "open":
                return {
                    "error": f"Pull request #{pr_number} is not open (state: {pr.state}).",
                    "hint": "Only open pull requests can be merged.",
                }

            # Check if PR is mergeable
            if pr.mergeable is False:
                return {
                    "error": f"Pull request #{pr_number} has merge conflicts.",
                    "hint": "Resolve conflicts before merging.",
                    "mergeable_state": pr.mergeable_state,
                }

            # Check if checks are blocking (if mergeable_state is 'blocked')
            if pr.mergeable_state == "blocked":
                return {
                    "error": f"Pull request #{pr_number} is blocked.",
                    "hint": "Required checks may be failing or pending. Use github_pr_checks to view status.",
                    "mergeable_state": pr.mergeable_state,
                }

            # Prepare merge parameters
            merge_kwargs: Dict[str, Any] = {
                "merge_method": merge_method,
            }

            if commit_title:
                merge_kwargs["commit_title"] = commit_title
            if commit_message:
                merge_kwargs["commit_message"] = commit_message

            # Perform the merge
            merge_result = pr.merge(**merge_kwargs)

            if not merge_result.merged:
                return {
                    "error": f"Failed to merge PR #{pr_number}.",
                    "message": merge_result.message,
                }

            result: Dict[str, Any] = {
                "status": "success",
                "message": f"Pull request #{pr_number} merged successfully!",
                "pr_number": pr_number,
                "title": pr.title,
                "merge_method": merge_method,
                "sha": merge_result.sha,
                "merged_by": merge_result.message,
            }

            # Delete branch if requested
            if delete_branch:
                try:
                    # Get the head branch ref
                    head_ref = pr.head.ref
                    # Only delete if it's not the default branch
                    if head_ref != gh_repo.default_branch:
                        ref = gh_repo.get_git_ref(f"heads/{head_ref}")
                        ref.delete()
                        result["branch_deleted"] = head_ref
                        logger.info(f"Deleted branch: {head_ref}")
                except GithubException as e:
                    # Branch deletion is not critical, just log warning
                    logger.warning(f"Could not delete branch: {e}")
                    result["branch_delete_warning"] = f"Could not delete branch: {str(e)}"

            return result

        except GithubException as e:
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)

            if e.status == 404:
                return {"error": f"Repository '{full_repo}' not found or not accessible."}
            elif e.status == 405:
                return {
                    "error": f"Merge method '{merge_method}' not allowed for this repository.",
                    "hint": "The repository may have restrictions on merge methods.",
                }
            elif e.status == 409:
                return {
                    "error": "Merge conflict or head branch was modified.",
                    "hint": "Pull the latest changes and try again.",
                }

            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error merging PR: {e}")
            return {"error": f"Failed to merge pull request: {str(e)}"}
