"""GitHub branch tool - create and switch branches using GitPython."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from git import Repo, InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


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
            "force": {
                "type": "boolean",
                "description": "Force delete branch even if not merged (for delete action only).",
            },
        },
        "required": ["repo", "action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Manage git branches."""
        repo_name = kwargs.get("repo", "")
        action = kwargs.get("action", "")
        branch = kwargs.get("branch", "")
        from_branch = kwargs.get("from_branch")
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

                return {
                    "status": "success",
                    "message": f"Created and switched to branch '{branch}'.",
                    "repo": local_name,
                    "branch": branch,
                    "from_branch": base.name,
                    "previous_branch": current_branch,
                }

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
