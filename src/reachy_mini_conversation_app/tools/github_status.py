"""GitHub status tool - show repository status using GitPython."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from git import Repo, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubStatusTool(Tool):
    """Show the status of a repository."""

    name = "github_status"
    description = (
        "Show the current status of a local repository. "
        "Shows staged files, modified files, untracked files, and current branch info."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Show repository status."""
        repo_name = kwargs.get("repo", "")

        logger.info(f"Tool call: github_status - repo='{repo_name}'")

        if not repo_name:
            return {"error": "Repository name is required."}

        # Repository path
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {"error": f"Repository not found: {local_name}"}

        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            return {"error": f"'{local_name}' is not a git repository."}

        try:
            # Current branch
            try:
                current_branch = repo.active_branch.name
            except TypeError:
                # Detached HEAD
                current_branch = f"HEAD detached at {repo.head.commit.hexsha[:8]}"

            # Tracking info
            tracking_info = {}
            try:
                tracking_branch = repo.active_branch.tracking_branch()
                if tracking_branch:
                    tracking_info["tracking"] = tracking_branch.name
                    # Commits ahead/behind
                    commits_ahead = list(repo.iter_commits(f"{tracking_branch.name}..HEAD"))
                    commits_behind = list(repo.iter_commits(f"HEAD..{tracking_branch.name}"))
                    tracking_info["ahead"] = len(commits_ahead)
                    tracking_info["behind"] = len(commits_behind)
            except Exception:
                pass

            # Staged files (changes to be committed)
            staged_output = repo.git.diff("--cached", "--name-status")
            staged_files: List[Dict[str, str]] = []
            if staged_output.strip():
                for line in staged_output.strip().split("\n"):
                    if line:
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            status, filepath = parts
                            status_map = {
                                "A": "added",
                                "M": "modified",
                                "D": "deleted",
                                "R": "renamed",
                                "C": "copied",
                            }
                            staged_files.append({
                                "file": filepath,
                                "status": status_map.get(status[0], status),
                            })

            # Modified files (not staged)
            modified_output = repo.git.diff("--name-status")
            modified_files: List[Dict[str, str]] = []
            if modified_output.strip():
                for line in modified_output.strip().split("\n"):
                    if line:
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            status, filepath = parts
                            status_map = {
                                "M": "modified",
                                "D": "deleted",
                            }
                            modified_files.append({
                                "file": filepath,
                                "status": status_map.get(status[0], status),
                            })

            # Untracked files
            untracked_files = repo.untracked_files

            # Build result
            result: Dict[str, Any] = {
                "status": "success",
                "repo": local_name,
                "branch": current_branch,
                "path": str(repo_path),
            }

            if tracking_info:
                result["tracking"] = tracking_info

            # Clean or dirty?
            is_clean = not staged_files and not modified_files and not untracked_files
            result["clean"] = is_clean

            if is_clean:
                result["message"] = "Working tree clean"
            else:
                result["message"] = "Changes detected"

            if staged_files:
                result["staged"] = staged_files
                result["staged_count"] = len(staged_files)

            if modified_files:
                result["modified"] = modified_files
                result["modified_count"] = len(modified_files)

            if untracked_files:
                # Limit display
                result["untracked"] = untracked_files[:30]
                result["untracked_count"] = len(untracked_files)
                if len(untracked_files) > 30:
                    result["untracked_truncated"] = True

            # Summary
            summary = []
            if staged_files:
                summary.append(f"{len(staged_files)} staged")
            if modified_files:
                summary.append(f"{len(modified_files)} modified")
            if untracked_files:
                summary.append(f"{len(untracked_files)} untracked")
            if summary:
                result["summary"] = ", ".join(summary)

            return result

        except Exception as e:
            logger.exception(f"Error getting status: {e}")
            return {"error": f"Failed to get status: {str(e)}"}
