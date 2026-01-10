"""GitHub branch tool - create and switch branches using GitPython."""

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


class GitHubBranchTool(Tool):
    """Create, switch, and list branches in a repository."""

    name = "github_branch"
    description = (
        "Manage git branches in a local repository. "
        "Can create new branches, switch between branches, or list all branches."
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
                "enum": ["create", "switch", "list", "delete"],
                "description": "Action to perform: create, switch, list, or delete a branch.",
            },
            "branch": {
                "type": "string",
                "description": "Branch name (required for create, switch, delete actions).",
            },
            "from_branch": {
                "type": "string",
                "description": "Base branch to create from (optional, defaults to current branch).",
            },
            "push": {
                "type": "boolean",
                "description": "Push the new branch to remote and set upstream (for create action only).",
            },
            "force": {
                "type": "boolean",
                "description": "Force delete branch even if not merged (for delete action only).",
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
        """Manage git branches."""
        repo_name = kwargs.get("repo", "")
        action = kwargs.get("action", "")
        branch = kwargs.get("branch", "")
        from_branch = kwargs.get("from_branch")
        push_branch = kwargs.get("push", False)
        force = kwargs.get("force", False)

        logger.info(f"Tool call: github_branch - repo='{repo_name}', action={action}, branch={branch}")

        if not repo_name:
            return {"error": "Repository name is required."}

        if not action:
            return {"error": "Action is required (create, switch, list, delete)."}

        if action in ["create", "switch", "delete"] and not branch:
            return {"error": f"Branch name is required for '{action}' action."}

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

            if action == "list":
                # List all branches
                local_branches = [b.name for b in repo.branches]
                remote_branches = []
                for remote in repo.remotes:
                    for ref in remote.refs:
                        remote_branches.append(ref.name)

                return {
                    "status": "success",
                    "repo": local_name,
                    "current_branch": current_branch,
                    "local_branches": local_branches,
                    "remote_branches": remote_branches,
                }

            elif action == "create":
                # Check if branch already exists
                if branch in [b.name for b in repo.branches]:
                    return {
                        "error": f"Branch '{branch}' already exists.",
                        "hint": "Use action='switch' to switch to it, or choose a different name.",
                    }

                # Determine base branch
                if from_branch:
                    if from_branch not in [b.name for b in repo.branches]:
                        return {"error": f"Base branch '{from_branch}' does not exist."}
                    base = repo.branches[from_branch]
                else:
                    base = repo.active_branch

                # Create new branch
                new_branch = repo.create_head(branch, base)

                # Switch to new branch
                new_branch.checkout()

                result = {
                    "status": "success",
                    "message": f"Created and switched to branch '{branch}'.",
                    "repo": local_name,
                    "branch": branch,
                    "from_branch": base.name,
                    "previous_branch": current_branch,
                }

                # Push to remote and set upstream if requested
                if push_branch:
                    token = config.GITHUB_TOKEN
                    if not token:
                        result["push_error"] = "GITHUB_TOKEN not set, branch not pushed."
                    else:
                        try:
                            origin = repo.remotes.origin
                            auth_url = self._get_authenticated_url(repo)
                            original_url = origin.url

                            if auth_url:
                                origin.set_url(auth_url)

                            try:
                                # Push with upstream tracking
                                origin.push(refspec=f"{branch}:{branch}", set_upstream=True)
                                result["pushed"] = True
                                result["upstream"] = f"origin/{branch}"
                                result["message"] = f"Created branch '{branch}', pushed to remote with upstream set."
                            finally:
                                if auth_url:
                                    origin.set_url(original_url)

                        except GitCommandError as e:
                            result["push_error"] = f"Failed to push: {str(e)}"

                return result

            elif action == "switch":
                # Check if branch exists locally
                if branch not in [b.name for b in repo.branches]:
                    # Try to find remote branch
                    remote_ref = None
                    for remote in repo.remotes:
                        for ref in remote.refs:
                            if ref.name.endswith(f"/{branch}"):
                                remote_ref = ref
                                break
                        if remote_ref:
                            break

                    if remote_ref:
                        # Create local branch from remote
                        new_branch = repo.create_head(branch, remote_ref)
                        new_branch.set_tracking_branch(remote_ref)
                        new_branch.checkout()
                        return {
                            "status": "success",
                            "message": f"Created local branch '{branch}' from remote and switched to it.",
                            "repo": local_name,
                            "branch": branch,
                            "tracking": remote_ref.name,
                            "previous_branch": current_branch,
                        }
                    else:
                        return {
                            "error": f"Branch '{branch}' does not exist locally or remotely.",
                            "hint": "Use action='create' to create a new branch.",
                        }

                # Switch to existing branch
                repo.branches[branch].checkout()

                return {
                    "status": "success",
                    "message": f"Switched to branch '{branch}'.",
                    "repo": local_name,
                    "branch": branch,
                    "previous_branch": current_branch,
                }

            elif action == "delete":
                # Cannot delete current branch
                if branch == current_branch:
                    return {
                        "error": f"Cannot delete the current branch '{branch}'.",
                        "hint": "Switch to another branch first.",
                    }

                # Check if branch exists
                if branch not in [b.name for b in repo.branches]:
                    return {"error": f"Branch '{branch}' does not exist."}

                # Delete branch
                if force:
                    repo.delete_head(branch, force=True)
                else:
                    try:
                        repo.delete_head(branch)
                    except GitCommandError as e:
                        if "not fully merged" in str(e).lower():
                            return {
                                "error": f"Branch '{branch}' is not fully merged.",
                                "hint": "Use force=true to force delete, or merge the branch first.",
                            }
                        raise

                return {
                    "status": "success",
                    "message": f"Deleted branch '{branch}'.",
                    "repo": local_name,
                    "deleted_branch": branch,
                    "current_branch": current_branch,
                }

            else:
                return {"error": f"Unknown action: {action}"}

        except GitCommandError as e:
            return {"error": f"Git command failed: {str(e)}"}
        except Exception as e:
            logger.exception(f"Error managing branch: {e}")
            return {"error": f"Failed to manage branch: {str(e)}"}
