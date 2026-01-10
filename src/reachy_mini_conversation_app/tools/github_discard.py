"""GitHub discard tool - discard unstaged changes using GitPython."""

import logging
from typing import Any, Dict, List
from pathlib import Path

from git import Repo, GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubDiscardTool(Tool):
    """Discard unstaged changes in a repository."""

    name = "github_discard"
    description = (
        "Discard unstaged changes in a local repository. "
        "Can restore specific files to their last committed state, or remove untracked files. "
        "IMPORTANT: This action is IRREVERSIBLE. Always ask user for confirmation before calling."
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
                "description": "List of files to discard changes for (relative paths). Use '.' for all files.",
            },
            "untracked": {
                "type": "boolean",
                "description": "If true, also remove untracked files (git clean -fd).",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm discard. Always ask user first - this is IRREVERSIBLE!",
            },
        },
        "required": ["repo", "confirmed"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Discard unstaged changes."""
        repo_name = kwargs.get("repo", "")
        files: List[str] = kwargs.get("files", ["."])
        untracked = kwargs.get("untracked", False)
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_discard - repo='{repo_name}', files={files}, untracked={untracked}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "Discard not confirmed. This action is IRREVERSIBLE!",
                "hint": "Set confirmed=true after getting user approval.",
            }

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
            discarded_files = []
            cleaned_files = []

            # Get list of modified files before discard
            modified_output = repo.git.diff("--name-only")
            modified_files = [f for f in modified_output.strip().split("\n") if f]

            # Discard changes (git checkout -- <files>)
            if files:
                for file in files:
                    if file == ".":
                        # Discard all changes
                        repo.git.checkout("--", ".")
                        discarded_files = modified_files
                    else:
                        try:
                            repo.git.checkout("--", file)
                            discarded_files.append(file)
                        except GitCommandError as e:
                            logger.warning(f"Could not discard {file}: {e}")

            # Remove untracked files if requested
            if untracked:
                # Get list of untracked files before clean
                untracked_files = repo.untracked_files

                if untracked_files:
                    # git clean -fd (force, directories)
                    repo.git.clean("-fd")
                    cleaned_files = untracked_files

            result: Dict[str, Any] = {
                "status": "success",
                "message": "Changes discarded successfully!",
                "repo": local_name,
            }

            if discarded_files:
                result["discarded_files"] = discarded_files
                result["discarded_count"] = len(discarded_files)

            if cleaned_files:
                result["cleaned_files"] = cleaned_files
                result["cleaned_count"] = len(cleaned_files)

            if not discarded_files and not cleaned_files:
                result["message"] = "No changes to discard."

            return result

        except GitCommandError as e:
            return {"error": f"Git command failed: {str(e)}"}
        except Exception as e:
            logger.exception(f"Error discarding changes: {e}")
            return {"error": f"Failed to discard changes: {str(e)}"}
