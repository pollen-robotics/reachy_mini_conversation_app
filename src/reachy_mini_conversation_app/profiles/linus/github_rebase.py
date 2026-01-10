"""GitHub rebase tool - rebase branches using GitPython."""

import logging
from typing import Any, Dict
from pathlib import Path

from git import Repo, GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from .github_env_vars import GITHUB_ENV_VARS


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubRebaseTool(Tool):
    """Rebase a branch onto another branch."""

    name = "github_rebase"
    description = (
        "Rebase the current branch onto another branch. "
        "This replays commits from current branch on top of the target branch. "
        "Can also abort or continue a rebase in progress. "
        "IMPORTANT: Rebase rewrites history. Always ask user for confirmation."
    )
    required_env_vars = GITHUB_ENV_VARS
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "action": {
                "type": "string",
                "enum": ["start", "continue", "abort", "skip"],
                "description": "Rebase action: 'start' new rebase, 'continue' after resolving conflicts, 'abort' to cancel, 'skip' to skip current commit",
            },
            "onto": {
                "type": "string",
                "description": "Branch to rebase onto (e.g., 'main', 'develop'). Required for 'start' action.",
            },
            "update_remote": {
                "type": "boolean",
                "description": "Fetch the target branch from remote before rebasing (default: true)",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm rebase. Rebase rewrites history!",
            },
        },
        "required": ["repo", "action"],
    }

    def _get_authenticated_url(self, repo: Repo) -> str | None:
        """Get remote URL with token authentication."""
        token = config.GITHUB_TOKEN
        if not token:
            return None

        try:
            origin = repo.remotes.origin
            url = origin.url

            if url.startswith("git@github.com:"):
                repo_path = url.replace("git@github.com:", "").replace(".git", "")
                return f"https://{token}@github.com/{repo_path}.git"
            elif "github.com" in url:
                if "@github.com" in url:
                    url = url.split("@github.com")[1]
                    url = f"https://{token}@github.com{url}"
                else:
                    url = url.replace("https://github.com", f"https://{token}@github.com")
                return url
        except Exception:
            pass
        return None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Rebase branches."""
        repo_name = kwargs.get("repo", "")
        action = kwargs.get("action", "start")
        onto = kwargs.get("onto", "")
        update_remote = kwargs.get("update_remote", True)
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_rebase - repo='{repo_name}', action={action}, onto={onto}")

        if not repo_name:
            return {"error": "Repository name is required."}

        # Confirmation required for start action
        if action == "start" and not confirmed:
            return {
                "error": "Rebase not confirmed. Rebase rewrites commit history!",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if action == "start" and not onto:
            return {"error": "Target branch (onto) is required for starting a rebase."}

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
            current_branch = repo.active_branch.name

            # Check if rebase is in progress
            rebase_in_progress = (repo_path / ".git" / "rebase-merge").exists() or \
                                 (repo_path / ".git" / "rebase-apply").exists()

            if action == "start":
                if rebase_in_progress:
                    return {
                        "error": "A rebase is already in progress.",
                        "hint": "Use action='continue', 'abort', or 'skip' to handle the current rebase.",
                    }

                # Fetch remote if requested
                if update_remote:
                    try:
                        origin = repo.remotes.origin
                        auth_url = self._get_authenticated_url(repo)
                        original_url = origin.url

                        if auth_url:
                            origin.set_url(auth_url)

                        try:
                            origin.fetch()
                        finally:
                            if auth_url:
                                origin.set_url(original_url)
                    except Exception as e:
                        logger.warning(f"Failed to fetch: {e}")

                # Get commits that will be rebased
                try:
                    # Check if onto branch exists
                    if onto not in [b.name for b in repo.branches]:
                        # Try remote branch
                        remote_onto = f"origin/{onto}"
                        if remote_onto not in [ref.name for ref in repo.remotes.origin.refs]:
                            return {"error": f"Branch '{onto}' not found locally or on remote."}
                        onto_ref = remote_onto
                    else:
                        onto_ref = onto

                    commits_to_rebase = list(repo.iter_commits(f"{onto_ref}..HEAD"))
                    num_commits = len(commits_to_rebase)

                    if num_commits == 0:
                        return {
                            "status": "no_change",
                            "message": f"Branch '{current_branch}' is already up to date with '{onto}'.",
                            "branch": current_branch,
                            "onto": onto,
                        }

                except Exception as e:
                    return {"error": f"Failed to analyze rebase: {str(e)}"}

                # Perform the rebase
                try:
                    repo.git.rebase(onto_ref)

                    return {
                        "status": "success",
                        "message": f"Rebased {num_commits} commit(s) onto '{onto}' successfully!",
                        "repo": local_name,
                        "branch": current_branch,
                        "onto": onto,
                        "commits_rebased": num_commits,
                        "warning": "History has been rewritten. Use force push (github_push with force=true) if branch was already pushed.",
                    }

                except GitCommandError as e:
                    if "conflict" in str(e).lower() or "could not apply" in str(e).lower():
                        # Get conflicted files
                        conflicted = repo.index.unmerged_blobs()
                        conflict_files = [str(path) for path in conflicted.keys()] if conflicted else []

                        return {
                            "status": "conflict",
                            "message": "Rebase paused due to conflicts.",
                            "repo": local_name,
                            "branch": current_branch,
                            "onto": onto,
                            "conflicted_files": conflict_files[:20],
                            "hint": "Resolve conflicts, then use action='continue'. Or use action='abort' to cancel.",
                        }
                    raise

            elif action == "continue":
                if not rebase_in_progress:
                    return {"error": "No rebase in progress to continue."}

                try:
                    repo.git.rebase("--continue")

                    return {
                        "status": "success",
                        "message": "Rebase continued successfully!",
                        "repo": local_name,
                        "branch": current_branch,
                    }

                except GitCommandError as e:
                    if "conflict" in str(e).lower():
                        conflicted = repo.index.unmerged_blobs()
                        conflict_files = [str(path) for path in conflicted.keys()] if conflicted else []

                        return {
                            "status": "conflict",
                            "message": "More conflicts encountered.",
                            "conflicted_files": conflict_files[:20],
                            "hint": "Resolve conflicts, then use action='continue' again.",
                        }
                    raise

            elif action == "abort":
                if not rebase_in_progress:
                    return {"error": "No rebase in progress to abort."}

                repo.git.rebase("--abort")

                return {
                    "status": "success",
                    "message": "Rebase aborted successfully.",
                    "repo": local_name,
                    "branch": current_branch,
                }

            elif action == "skip":
                if not rebase_in_progress:
                    return {"error": "No rebase in progress to skip."}

                repo.git.rebase("--skip")

                return {
                    "status": "success",
                    "message": "Skipped current commit and continued rebase.",
                    "repo": local_name,
                    "branch": current_branch,
                    "warning": "A commit was skipped. Some changes may be lost.",
                }

            else:
                return {"error": f"Unknown action: {action}"}

        except GitCommandError as e:
            error_msg = str(e)
            return {"error": f"Git rebase failed: {error_msg}"}
        except Exception as e:
            logger.exception(f"Error rebasing: {e}")
            return {"error": f"Failed to rebase: {str(e)}"}
