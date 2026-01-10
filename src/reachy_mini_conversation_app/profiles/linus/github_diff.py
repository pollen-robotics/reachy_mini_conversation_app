"""GitHub diff tool - show file differences using GitPython."""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from git import Repo, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubDiffTool(Tool):
    """Show differences in a repository."""

    name = "github_diff"
    description = (
        "Show file differences in a local repository. "
        "Can show staged changes, unstaged changes, or diff between commits/branches."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "staged": {
                "type": "boolean",
                "description": "Show staged changes only (git diff --cached). Default: false (shows unstaged)",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific files to diff (optional, defaults to all)",
            },
            "commit": {
                "type": "string",
                "description": "Compare with a specific commit (SHA, branch, tag, HEAD~N)",
            },
            "compare": {
                "type": "string",
                "description": "Compare two refs: 'commit1..commit2' or 'branch1...branch2'",
            },
            "stat_only": {
                "type": "boolean",
                "description": "Show only stats (files changed, insertions, deletions). Default: false",
            },
            "name_only": {
                "type": "boolean",
                "description": "Show only file names. Default: false",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines around changes (default: 3)",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Show differences."""
        repo_name = kwargs.get("repo", "")
        staged = kwargs.get("staged", False)
        files: Optional[List[str]] = kwargs.get("files")
        commit = kwargs.get("commit")
        compare = kwargs.get("compare")
        stat_only = kwargs.get("stat_only", False)
        name_only = kwargs.get("name_only", False)
        context_lines = kwargs.get("context_lines", 3)

        logger.info(f"Tool call: github_diff - repo='{repo_name}', staged={staged}")

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
            # Build diff arguments
            diff_args = []

            # Output format
            if stat_only:
                diff_args.append("--stat")
            elif name_only:
                diff_args.append("--name-only")
            else:
                diff_args.extend([f"-U{context_lines}"])

            # What to diff
            if compare:
                # Compare two refs
                diff_args.append(compare)
                diff_type = f"compare: {compare}"
            elif commit:
                # Compare with specific commit
                diff_args.append(commit)
                diff_type = f"vs {commit}"
            elif staged:
                # Staged changes
                diff_args.append("--cached")
                diff_type = "staged"
            else:
                # Unstaged changes (default)
                diff_type = "unstaged"

            # Specific files
            if files:
                diff_args.append("--")
                diff_args.extend(files)

            # Execute diff
            diff_output = repo.git.diff(*diff_args)

            # Get summary stats
            if not stat_only and not name_only:
                stat_args = ["--stat"]
                if compare:
                    stat_args.append(compare)
                elif commit:
                    stat_args.append(commit)
                elif staged:
                    stat_args.append("--cached")
                if files:
                    stat_args.append("--")
                    stat_args.extend(files)
                stat_output = repo.git.diff(*stat_args)
            else:
                stat_output = None

            # Count changed files
            name_args = ["--name-only"]
            if compare:
                name_args.append(compare)
            elif commit:
                name_args.append(commit)
            elif staged:
                name_args.append("--cached")
            if files:
                name_args.append("--")
                name_args.extend(files)
            changed_files = [f for f in repo.git.diff(*name_args).strip().split("\n") if f]

            result: Dict[str, Any] = {
                "status": "success",
                "repo": local_name,
                "diff_type": diff_type,
                "files_changed": len(changed_files),
            }

            if not changed_files:
                result["message"] = "No differences found."
                result["diff"] = ""
                return result

            result["changed_files"] = changed_files[:50]
            if len(changed_files) > 50:
                result["files_truncated"] = True

            # Truncate diff if too large
            max_diff_size = 50000
            if len(diff_output) > max_diff_size:
                diff_output = diff_output[:max_diff_size] + "\n\n... [diff truncated, showing first 50KB]"
                result["diff_truncated"] = True

            result["diff"] = diff_output

            if stat_output:
                result["stat"] = stat_output

            return result

        except Exception as e:
            logger.exception(f"Error getting diff: {e}")
            return {"error": f"Failed to get diff: {str(e)}"}
