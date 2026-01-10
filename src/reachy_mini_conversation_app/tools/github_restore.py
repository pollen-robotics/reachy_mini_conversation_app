"""GitHub restore tool - restore files using git restore command."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from git import Repo, InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubRestoreTool(Tool):
    """Restore files in a repository using git restore."""

    name = "github_restore"
    description = (
        "Restore files in a local repository using git restore. "
        "Use --staged to unstage files (remove from staging area without losing changes). "
        "Use --worktree to discard working tree changes (restore to staged or committed state). "
        "Use both to fully restore files to committed state. "
        "IMPORTANT: Restoring worktree changes is IRREVERSIBLE. Ask user for confirmation."
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
                "description": "List of files to restore (relative paths). Use '.' for all files.",
            },
            "staged": {
                "type": "boolean",
                "description": "If true, unstage files (git restore --staged). Removes from staging area without losing changes.",
            },
            "worktree": {
                "type": "boolean",
                "description": "If true, restore working tree (git restore --worktree). IRREVERSIBLE - discards local changes!",
            },
            "source": {
                "type": "string",
                "description": "Restore from this commit/branch (e.g., 'HEAD~1', 'main'). Default: HEAD for staged, index for worktree.",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true when using --worktree. Always ask user first - worktree restore is IRREVERSIBLE!",
            },
        },
        "required": ["repo", "files"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Restore files using git restore."""
        repo_name = kwargs.get("repo", "")
        files: List[str] = kwargs.get("files", [])
        staged = kwargs.get("staged", False)
        worktree = kwargs.get("worktree", False)
        source = kwargs.get("source")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_restore - repo='{repo_name}', files={files}, staged={staged}, worktree={worktree}")

        if not repo_name:
            return {"error": "Repository name is required."}

        if not files:
            return {"error": "Files list is required. Use ['.'] for all files."}

        # Default behavior: if neither staged nor worktree specified, assume staged (safer option)
        if not staged and not worktree:
            staged = True

        # Check confirmation for worktree restore (irreversible)
        if worktree and not confirmed:
            return {
                "error": "Worktree restore not confirmed. This action is IRREVERSIBLE!",
                "hint": "Set confirmed=true after getting user approval, or use staged=true to only unstage files.",
            }

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
            # Build restore command arguments
            # Note: after line 79-80, at least one of staged/worktree is always True
            restore_args = []

            if staged and worktree:
                restore_args.extend(["--staged", "--worktree"])
            elif staged:
                restore_args.append("--staged")
            else:  # worktree must be True here
                restore_args.append("--worktree")

            if source:
                restore_args.extend(["--source", source])

            # Get list of affected files before restore
            affected_files = []

            if staged:
                # Get staged files
                staged_output = repo.git.diff("--cached", "--name-only")
                staged_files = [f for f in staged_output.strip().split("\n") if f]

                if "." in files:
                    affected_files.extend(staged_files)
                else:
                    affected_files.extend([f for f in files if f in staged_files])

            if worktree:
                # Get modified files in worktree
                modified_output = repo.git.diff("--name-only")
                modified_files = [f for f in modified_output.strip().split("\n") if f]

                if "." in files:
                    affected_files.extend([f for f in modified_files if f not in affected_files])
                else:
                    affected_files.extend([f for f in files if f in modified_files and f not in affected_files])

            # Add files to restore
            restore_args.append("--")
            restore_args.extend(files)

            # Execute git restore
            repo.git.restore(*restore_args)

            # Build result
            result: Dict[str, Any] = {
                "status": "success",
                "repo": local_name,
            }

            actions = []
            if staged:
                actions.append("unstaged")
            if worktree:
                actions.append("restored worktree")

            if "." in files:
                result["message"] = f"Successfully {' and '.join(actions)} all files."
                result["files"] = affected_files if affected_files else "all applicable files"
            else:
                result["message"] = f"Successfully {' and '.join(actions)} specified files."
                result["files"] = files

            if source:
                result["source"] = source

            result["staged"] = staged
            result["worktree"] = worktree

            # Hint for next steps (at least one of staged/worktree is always True)
            if staged and not worktree:
                result["hint"] = "Files are now unstaged. Use github_add to re-stage them if needed."
            else:  # worktree is True
                result["hint"] = "Working tree changes have been discarded."

            return result

        except GitCommandError as e:
            error_msg = str(e)
            if "did not match any file" in error_msg.lower():
                return {"error": "No matching files found to restore.", "hint": "Check file paths and ensure files have changes."}
            return {"error": f"Git restore failed: {error_msg}"}
        except Exception as e:
            logger.exception(f"Error restoring files: {e}")
            return {"error": f"Failed to restore files: {str(e)}"}
