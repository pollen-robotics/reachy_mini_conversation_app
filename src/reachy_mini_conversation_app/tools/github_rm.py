"""GitHub rm tool - remove files from a repository."""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from git import Repo, InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubRmTool(Tool):
    """Remove files or directories from a repository."""

    name = "github_rm"
    description = (
        "Remove files or directories from a local repository. "
        "Can remove from git tracking only or delete from filesystem entirely. "
        "IMPORTANT: This action is IRREVERSIBLE. Always ask user for confirmation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file or directory paths to remove (relative to repo root)",
            },
            "git_only": {
                "type": "boolean",
                "description": "If true, only remove from git tracking (keep files on disk). Default: false",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, remove directories recursively. Default: false",
            },
            "force": {
                "type": "boolean",
                "description": "If true, force removal even if files have local modifications. Default: false",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm removal. Always ask user first - this is IRREVERSIBLE!",
            },
        },
        "required": ["repo", "paths", "confirmed"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Remove files or directories from a repository."""
        repo_name = kwargs.get("repo", "")
        paths: List[str] = kwargs.get("paths", [])
        git_only = kwargs.get("git_only", False)
        recursive = kwargs.get("recursive", False)
        force = kwargs.get("force", False)
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_rm - repo='{repo_name}', paths={paths}, git_only={git_only}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "File removal not confirmed. This action is IRREVERSIBLE!",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}
        if not paths:
            return {"error": "At least one path is required."}

        # Repository path
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {"error": f"Repository not found: {local_name}"}

        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            return {"error": f"'{local_name}' is not a git repository."}

        removed_files: List[str] = []
        removed_dirs: List[str] = []
        errors: List[Dict[str, str]] = []

        for path in paths:
            full_path = repo_path / path

            # Security: ensure path is within repo
            try:
                full_path.resolve().relative_to(repo_path.resolve())
            except ValueError:
                errors.append({"path": path, "error": "Cannot remove files outside the repository"})
                continue

            if not full_path.exists():
                # Check if it's tracked by git even if not on disk
                try:
                    repo.git.ls_files("--error-unmatch", path)
                    # File is tracked but doesn't exist - remove from index
                    repo.index.remove([path], working_tree=False)
                    removed_files.append(path)
                    continue
                except GitCommandError:
                    errors.append({"path": path, "error": "File or directory not found"})
                    continue

            try:
                if full_path.is_dir():
                    if not recursive:
                        errors.append({
                            "path": path,
                            "error": "Is a directory. Set recursive=true to remove directories."
                        })
                        continue

                    # Remove directory
                    if git_only:
                        # Remove from git tracking only
                        repo.index.remove([path], working_tree=False, r=True)
                    else:
                        # Remove from git and filesystem
                        try:
                            repo.index.remove([path], working_tree=True, r=True, force=force)
                        except GitCommandError:
                            # May not be tracked, just remove from filesystem
                            shutil.rmtree(full_path)

                    removed_dirs.append(path)

                else:
                    # Remove file
                    if git_only:
                        # Remove from git tracking only (keeps file on disk)
                        try:
                            repo.index.remove([path], working_tree=False)
                        except GitCommandError:
                            errors.append({"path": path, "error": "File not tracked by git"})
                            continue
                    else:
                        # Remove from git and filesystem
                        try:
                            repo.index.remove([path], working_tree=True, force=force)
                        except GitCommandError:
                            # May not be tracked, just remove from filesystem
                            full_path.unlink()

                    removed_files.append(path)

            except GitCommandError as e:
                error_msg = str(e)
                if "has local modifications" in error_msg or "has changes staged" in error_msg:
                    errors.append({
                        "path": path,
                        "error": "File has local modifications. Set force=true to remove anyway."
                    })
                else:
                    errors.append({"path": path, "error": str(e)})
            except Exception as e:
                errors.append({"path": path, "error": str(e)})

        # Build result
        result: Dict[str, Any] = {
            "repo": local_name,
            "git_only": git_only,
        }

        if removed_files or removed_dirs:
            result["status"] = "success"
            result["message"] = "Files removed successfully!"

            if removed_files:
                result["removed_files"] = removed_files
                result["files_count"] = len(removed_files)

            if removed_dirs:
                result["removed_dirs"] = removed_dirs
                result["dirs_count"] = len(removed_dirs)

            if git_only:
                result["hint"] = "Files removed from git tracking but kept on disk. Use github_commit to commit."
            else:
                result["hint"] = "Files deleted. Use github_commit to commit the removal."

        elif errors:
            result["status"] = "failed"
            result["message"] = "No files were removed."

        if errors:
            result["errors"] = errors

        return result
