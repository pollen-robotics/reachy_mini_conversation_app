"""GitHub reset tool - reset commits using GitPython."""

import logging
from pathlib import Path
from typing import Any, Dict

from git import Repo, InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubResetTool(Tool):
    """Reset commits in a repository."""

    name = "github_reset"
    description = (
        "Reset commits in a local repository. "
        "Can undo commits with different modes: soft (keep changes staged), "
        "mixed (keep changes unstaged), or hard (discard all changes). "
        "IMPORTANT: Hard reset is IRREVERSIBLE. Always ask user for confirmation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "mode": {
                "type": "string",
                "enum": ["soft", "mixed", "hard"],
                "description": "Reset mode: 'soft' (keep staged), 'mixed' (keep unstaged, default), 'hard' (discard all)",
            },
            "target": {
                "type": "string",
                "description": "Reset target: commit SHA, 'HEAD~N' (N commits back), branch name, or tag. Default: HEAD~1",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true for hard reset. Always ask user first - hard reset is IRREVERSIBLE!",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Reset commits in a repository."""
        repo_name = kwargs.get("repo", "")
        mode = kwargs.get("mode", "mixed")
        target = kwargs.get("target", "HEAD~1")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_reset - repo='{repo_name}', mode={mode}, target={target}")

        if not repo_name:
            return {"error": "Repository name is required."}

        # Hard reset requires confirmation
        if mode == "hard" and not confirmed:
            return {
                "error": "Hard reset not confirmed. This will PERMANENTLY discard changes!",
                "hint": "Set confirmed=true after getting user approval, or use 'soft'/'mixed' mode.",
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
            # Get current state before reset
            current_branch = repo.active_branch.name
            current_commit = repo.head.commit
            current_sha = current_commit.hexsha[:8]
            current_message = current_commit.message.strip().split("\n")[0][:50]

            # Get commits that will be reset
            try:
                target_commit = repo.commit(target)
            except Exception:
                return {
                    "error": f"Invalid target: {target}",
                    "hint": "Use a commit SHA, 'HEAD~N', branch name, or tag.",
                }

            target_sha = target_commit.hexsha[:8]
            target_message = target_commit.message.strip().split("\n")[0][:50]

            # Count commits being reset
            commits_to_reset = list(repo.iter_commits(f"{target}..HEAD"))
            num_commits = len(commits_to_reset)

            if num_commits == 0:
                return {
                    "status": "no_change",
                    "message": "No commits to reset (already at target).",
                    "current_commit": current_sha,
                    "target": target,
                }

            # Perform the reset
            if mode == "soft":
                repo.head.reset(target, index=False, working_tree=False)
            elif mode == "mixed":
                repo.head.reset(target, index=True, working_tree=False)
            elif mode == "hard":
                repo.head.reset(target, index=True, working_tree=True)

            # Get new state after reset
            new_commit = repo.head.commit
            new_sha = new_commit.hexsha[:8]

            result: Dict[str, Any] = {
                "status": "success",
                "message": f"Reset {num_commits} commit(s) successfully!",
                "repo": local_name,
                "branch": current_branch,
                "mode": mode,
                "previous_commit": {
                    "sha": current_sha,
                    "message": current_message,
                },
                "current_commit": {
                    "sha": new_sha,
                    "message": target_message,
                },
                "commits_reset": num_commits,
            }

            # Add mode-specific info
            if mode == "soft":
                result["hint"] = "Changes are staged. Use github_commit to recommit or github_discard to remove."
            elif mode == "mixed":
                result["hint"] = "Changes are unstaged. Use github_commit to stage and commit, or github_discard to remove."
            elif mode == "hard":
                result["warning"] = "All changes have been permanently discarded."

            # List reset commits
            result["reset_commits"] = [
                {"sha": c.hexsha[:8], "message": c.message.strip().split("\n")[0][:50]}
                for c in commits_to_reset[:10]  # Limit to 10
            ]
            if num_commits > 10:
                result["reset_commits_truncated"] = True

            return result

        except GitCommandError as e:
            return {"error": f"Git reset failed: {str(e)}"}
        except Exception as e:
            logger.exception(f"Error resetting: {e}")
            return {"error": f"Failed to reset: {str(e)}"}
