"""GitHub commit tool with semantic-release format."""

import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"

# Semantic release commit types
COMMIT_TYPES = {
    "feat": "A new feature",
    "fix": "A bug fix",
    "docs": "Documentation only changes",
    "style": "Changes that do not affect the meaning of the code",
    "refactor": "A code change that neither fixes a bug nor adds a feature",
    "perf": "A code change that improves performance",
    "test": "Adding missing tests or correcting existing tests",
    "build": "Changes that affect the build system or external dependencies",
    "ci": "Changes to CI configuration files and scripts",
    "chore": "Other changes that don't modify src or test files",
    "revert": "Reverts a previous commit",
}


class GitHubCommitTool(Tool):
    """Stage and commit changes with semantic-release format."""

    name = "github_commit"
    description = (
        "Stage files and create a commit in a local repository using semantic-release format. "
        "IMPORTANT: Always ask user for confirmation before calling this tool. "
        "Commit types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "type": {
                "type": "string",
                "enum": list(COMMIT_TYPES.keys()),
                "description": "Commit type (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)",
            },
            "scope": {
                "type": "string",
                "description": "Optional scope of the change (e.g., 'api', 'ui', 'auth')",
            },
            "message": {
                "type": "string",
                "description": "Short description of the change (imperative mood, e.g., 'add user login')",
            },
            "body": {
                "type": "string",
                "description": "Optional longer description of the change",
            },
            "breaking": {
                "type": "boolean",
                "description": "If true, marks this as a breaking change (adds ! after type)",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of files to stage (relative paths). Use '.' to stage all changes.",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm commit. Always ask user first.",
            },
        },
        "required": ["repo", "type", "message", "files", "confirmed"],
    }

    def _build_commit_message(
        self,
        commit_type: str,
        message: str,
        scope: Optional[str] = None,
        body: Optional[str] = None,
        breaking: bool = False,
    ) -> str:
        """Build a semantic-release compliant commit message."""
        # Build the header: type(scope)!: message
        header = commit_type
        if scope:
            header += f"({scope})"
        if breaking:
            header += "!"
        header += f": {message}"

        # Full message
        full_message = header
        if body:
            full_message += f"\n\n{body}"
        if breaking:
            full_message += "\n\nBREAKING CHANGE: This is a breaking change."

        return full_message

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stage and commit changes."""
        repo_name = kwargs.get("repo", "")
        commit_type = kwargs.get("type", "")
        scope = kwargs.get("scope")
        message = kwargs.get("message", "")
        body = kwargs.get("body")
        breaking = kwargs.get("breaking", False)
        files: List[str] = kwargs.get("files", [])
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_commit - repo='{repo_name}', type={commit_type}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "Commit not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}
        if not commit_type:
            return {"error": f"Commit type is required. Valid types: {list(COMMIT_TYPES.keys())}"}
        if commit_type not in COMMIT_TYPES:
            return {"error": f"Invalid commit type '{commit_type}'. Valid types: {list(COMMIT_TYPES.keys())}"}
        if not message:
            return {"error": "Commit message is required."}
        if not files:
            return {"error": "At least one file must be specified to stage."}

        # Repository path
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {"error": f"Repository not found: {local_name}"}

        if not (repo_path / ".git").exists():
            return {"error": f"'{local_name}' is not a git repository."}

        try:
            # Stage files
            for file in files:
                if file == ".":
                    # Stage all changes
                    result = subprocess.run(
                        ["git", "add", "-A"],
                        capture_output=True,
                        text=True,
                        cwd=repo_path,
                    )
                else:
                    result = subprocess.run(
                        ["git", "add", file],
                        capture_output=True,
                        text=True,
                        cwd=repo_path,
                    )

                if result.returncode != 0:
                    return {"error": f"Failed to stage '{file}': {result.stderr.strip()}"}

            # Check if there are staged changes
            status_result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )

            staged_files = [f for f in status_result.stdout.strip().split("\n") if f]
            if not staged_files:
                return {
                    "status": "nothing_to_commit",
                    "message": "No changes to commit.",
                    "hint": "Make sure you've made changes and specified the correct files.",
                }

            # Build commit message
            commit_msg = self._build_commit_message(
                commit_type=commit_type,
                message=message,
                scope=scope,
                body=body,
                breaking=breaking,
            )

            # Create commit
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                if "nothing to commit" in error_msg.lower():
                    return {"status": "nothing_to_commit", "message": "No changes to commit."}
                return {"error": f"Commit failed: {error_msg}"}

            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            commit_hash = hash_result.stdout.strip()

            return {
                "status": "success",
                "message": "Commit created successfully!",
                "repo": local_name,
                "commit_hash": commit_hash,
                "commit_type": commit_type,
                "commit_message": commit_msg.split("\n")[0],  # Just the header
                "files_committed": staged_files,
                "hint": "Use github_push to push this commit to remote.",
            }

        except Exception as e:
            logger.exception(f"Error creating commit: {e}")
            return {"error": f"Failed to create commit: {str(e)}"}
