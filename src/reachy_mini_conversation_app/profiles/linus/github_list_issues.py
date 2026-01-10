"""GitHub list issues tool."""

import logging
from typing import Any, Dict, List

from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubListIssuesTool(Tool):
    """List issues on a GitHub repository."""

    name = "github_list_issues"
    description = (
        "List issues on a GitHub repository. "
        "Use this to see open bugs, feature requests, or tasks."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "state": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "Filter by issue state (default: 'open')",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by labels (e.g., ['bug', 'enhancement'])",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of issues to return (default: 20, max: 50)",
            },
        },
        "required": ["repo"],
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
        """List issues on a repository."""
        repo_name = kwargs.get("repo", "")
        state = kwargs.get("state", "open")
        labels = kwargs.get("labels", [])
        limit = min(kwargs.get("limit", 20), 50)

        logger.info(f"Tool call: github_list_issues - repo='{repo_name}', state={state}")

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {
                "error": "GITHUB_TOKEN is not configured. "
                "Please set it in your .env file to use GitHub features."
            }

        if not repo_name:
            return {"error": "Repository name is required."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            repo = g.get_repo(full_repo)

            # Get issues (note: PRs are also issues in GitHub API)
            issues_iter = repo.get_issues(state=state, labels=labels)

            issues: List[Dict[str, Any]] = []
            count = 0
            for issue in issues_iter:
                # Skip pull requests (they appear as issues)
                if issue.pull_request is not None:
                    continue

                issues.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "author": issue.user.login if issue.user else None,
                    "labels": [label.name for label in issue.labels],
                    "created_at": issue.created_at.isoformat() if issue.created_at else None,
                    "comments": issue.comments,
                    "url": issue.html_url,
                })

                count += 1
                if count >= limit:
                    break

            return {
                "status": "success",
                "repo": full_repo,
                "state_filter": state,
                "count": len(issues),
                "issues": issues,
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
            logger.exception(f"Error listing issues: {e}")
            return {"error": f"Failed to list issues: {str(e)}"}
