"""GitHub PR checks tool - get CI status and errors using PyGithub."""

import logging
from typing import Any, Dict, List

from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from .github_env_vars import GITHUB_ENV_VARS


logger = logging.getLogger(__name__)


class GitHubPRChecksTool(Tool):
    """Get CI check status and errors for a pull request."""

    name = "github_pr_checks"
    description = (
        "Get the CI/CD check status and errors for a pull request. "
        "Shows all check runs (GitHub Actions, etc.) with their status, conclusion, and error details. "
        "Useful for debugging failed CI pipelines."
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
                "description": "Pull request number to get checks for",
            },
            "filter_status": {
                "type": "string",
                "enum": ["all", "failed", "pending", "success"],
                "description": "Filter checks by status. Default: 'all'",
            },
            "include_logs": {
                "type": "boolean",
                "description": "Include log URLs for failed checks (default: true)",
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
        """Get CI check status and errors for a pull request."""
        repo_name = kwargs.get("repo", "")
        pr_number = kwargs.get("pr_number")
        filter_status = kwargs.get("filter_status", "all")
        include_logs = kwargs.get("include_logs", True)

        logger.info(f"Tool call: github_pr_checks - repo='{repo_name}', pr=#{pr_number}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not pr_number:
            return {"error": "Pull request number is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to get PR checks."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)
            pr = gh_repo.get_pull(pr_number)

            # Get the head commit of the PR
            head_sha = pr.head.sha
            commit = gh_repo.get_commit(head_sha)

            # Get check runs for this commit
            check_runs = commit.get_check_runs()

            # Also get commit statuses (for external CI systems)
            combined_status = commit.get_combined_status()

            checks: List[Dict[str, Any]] = []
            failed_checks: List[Dict[str, Any]] = []
            pending_checks: List[Dict[str, Any]] = []
            successful_checks: List[Dict[str, Any]] = []

            # Process check runs (GitHub Actions, etc.)
            for run in check_runs:
                check_info: Dict[str, Any] = {
                    "name": run.name,
                    "status": run.status,  # queued, in_progress, completed
                    "conclusion": run.conclusion,  # success, failure, neutral, cancelled, skipped, timed_out, action_required
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                }

                # Add details URL for logs
                if include_logs and run.details_url:
                    check_info["details_url"] = run.details_url

                # Add HTML URL
                if run.html_url:
                    check_info["url"] = run.html_url

                # Extract error output if available and failed
                if run.conclusion in ["failure", "timed_out", "action_required"]:
                    if run.output:
                        output = run.output
                        if output.title:
                            check_info["error_title"] = output.title
                        if output.summary:
                            check_info["error_summary"] = output.summary[:500]  # Limit length
                        if output.text:
                            check_info["error_text"] = output.text[:1000]  # Limit length
                    failed_checks.append(check_info)
                elif run.status != "completed":
                    pending_checks.append(check_info)
                else:
                    successful_checks.append(check_info)

                checks.append(check_info)

            # Process commit statuses (external CI like Jenkins, CircleCI, etc.)
            for status in combined_status.statuses:
                status_info: Dict[str, Any] = {
                    "name": status.context,
                    "status": "completed" if status.state in ["success", "failure", "error"] else "pending",
                    "conclusion": status.state,  # pending, success, failure, error
                    "description": status.description,
                }

                if include_logs and status.target_url:
                    status_info["details_url"] = status.target_url

                if status.state in ["failure", "error"]:
                    failed_checks.append(status_info)
                elif status.state == "pending":
                    pending_checks.append(status_info)
                else:
                    successful_checks.append(status_info)

                checks.append(status_info)

            # Filter based on requested status
            if filter_status == "failed":
                filtered_checks = failed_checks
            elif filter_status == "pending":
                filtered_checks = pending_checks
            elif filter_status == "success":
                filtered_checks = successful_checks
            else:
                filtered_checks = checks

            # Determine overall status
            if failed_checks:
                overall_status = "failing"
            elif pending_checks:
                overall_status = "pending"
            elif successful_checks:
                overall_status = "passing"
            else:
                overall_status = "no_checks"

            result: Dict[str, Any] = {
                "status": "success",
                "pr_number": pr_number,
                "pr_title": pr.title,
                "head_sha": head_sha[:8],
                "overall_status": overall_status,
                "summary": {
                    "total": len(checks),
                    "failed": len(failed_checks),
                    "pending": len(pending_checks),
                    "successful": len(successful_checks),
                },
                "checks": filtered_checks,
            }

            # Add failed checks separately for easy access
            if failed_checks and filter_status == "all":
                result["failed_checks"] = failed_checks

            return result

        except GithubException as e:
            if e.status == 404:
                return {"error": f"Pull request #{pr_number} not found in '{full_repo}'."}
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except Exception as e:
            logger.exception(f"Error getting PR checks: {e}")
            return {"error": f"Failed to get PR checks: {str(e)}"}
