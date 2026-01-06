"""GitHub add tool - stage files using GitPython."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from git import Repo, InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubAddTool(Tool):
    """Stage files for commit."""

    name = "github_add"
    description = (
        "Stage files for the next commit (git add). "
        "Can stage specific files, all changes, or use patterns."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of files to stage (relative paths). Use '.' to stage all changes.",
            },
            "all": {
                "type": "boolean",
                "description": "Stage all changes including untracked files (git add -A). Default: false",
            },
            "update": {
                "type": "boolean",
                "description": "Only stage modified and deleted files, not new files (git add -u). Default: false",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stage files for commit."""
        repo_name = kwargs.get("repo", "")
        files: List[str] = kwargs.get("files", [])
        add_all = kwargs.get("all", False)
        update_only = kwargs.get("update", False)

        logger.info(f"Tool call: github_add - repo='{repo_name}', files={files}, all={add_all}")

        if not repo_name:
            return {"error": "Repository name is required."}

        if not files and not add_all and not update_only:
            return {"error": "Specify files to stage, or use all=true or update=true."}

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
            # Get status before staging
            staged_before = set(repo.git.diff("--cached", "--name-only").strip().split("\n"))
            staged_before = {f for f in staged_before if f}

            # Stage files
            if add_all:
                repo.git.add(A=True)
            elif update_only:
                repo.git.add(u=True)
            else:
                for file in files:
                    if file == ".":
                        repo.git.add(A=True)
                    else:
                        file_path = repo_path / file
                        if file_path.exists():
                            repo.index.add([file])
                        else:
                            # File was deleted, stage the deletion
                            try:
                                repo.index.remove([file], working_tree=False)
                            except GitCommandError:
                                return {"error": f"File not found and not tracked: {file}"}

            # Get status after staging
            staged_after = repo.git.diff("--cached", "--name-only").strip().split("\n")
            staged_after = [f for f in staged_after if f]

            # Determine newly staged files
            newly_staged = [f for f in staged_after if f not in staged_before]

            # Get unstaged changes remaining
            unstaged = repo.git.diff("--name-only").strip().split("\n")
            unstaged = [f for f in unstaged if f]

            # Get untracked files
            untracked = repo.untracked_files

            result: Dict[str, Any] = {
                "status": "success",
                "repo": local_name,
                "staged_files": staged_after,
                "staged_count": len(staged_after),
            }

            if newly_staged:
                result["newly_staged"] = newly_staged
                result["message"] = f"Staged {len(newly_staged)} file(s)."
            else:
                result["message"] = "No new files staged (already staged or no changes)."

            if unstaged:
                result["unstaged_remaining"] = unstaged
                result["unstaged_count"] = len(unstaged)

            if untracked:
                result["untracked"] = untracked[:20]  # Limit display
                result["untracked_count"] = len(untracked)
                if len(untracked) > 20:
                    result["untracked_truncated"] = True

            result["hint"] = "Use github_commit to commit staged changes."

            return result

        except GitCommandError as e:
            return {"error": f"Git command failed: {str(e)}"}
        except Exception as e:
            logger.exception(f"Error staging files: {e}")
            return {"error": f"Failed to stage files: {str(e)}"}
