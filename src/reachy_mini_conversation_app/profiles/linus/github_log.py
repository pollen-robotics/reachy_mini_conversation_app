"""GitHub log tool - show commit history using GitPython."""

import logging
from typing import Any, Dict, List
from pathlib import Path

from git import Repo, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubLogTool(Tool):
    """Show commit history of a repository."""

    name = "github_log"
    description = (
        "Show the commit history of a local repository. "
        "Can filter by branch, author, date range, or file path. "
        "Useful for reviewing recent changes and understanding project history."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "count": {
                "type": "integer",
                "description": "Number of commits to show (default: 10, max: 100)",
            },
            "branch": {
                "type": "string",
                "description": "Branch to show history for (default: current branch)",
            },
            "author": {
                "type": "string",
                "description": "Filter commits by author name or email",
            },
            "since": {
                "type": "string",
                "description": "Show commits after this date (e.g., '2024-01-01', '1 week ago')",
            },
            "until": {
                "type": "string",
                "description": "Show commits before this date (e.g., '2024-12-31', 'yesterday')",
            },
            "path": {
                "type": "string",
                "description": "Show only commits that affect this file or directory",
            },
            "grep": {
                "type": "string",
                "description": "Filter commits by message containing this text",
            },
            "oneline": {
                "type": "boolean",
                "description": "Show condensed one-line format (default: false)",
            },
            "stat": {
                "type": "boolean",
                "description": "Include file change statistics (default: false)",
            },
        },
        "required": ["repo"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Show commit history."""
        repo_name = kwargs.get("repo", "")
        count = min(kwargs.get("count", 10), 100)
        branch = kwargs.get("branch")
        author = kwargs.get("author")
        since = kwargs.get("since")
        until = kwargs.get("until")
        path = kwargs.get("path")
        grep = kwargs.get("grep")
        oneline = kwargs.get("oneline", False)
        stat = kwargs.get("stat", False)

        logger.info(f"Tool call: github_log - repo='{repo_name}', count={count}, branch={branch}")

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
            # Build git log arguments
            log_args = [f"-n{count}"]

            if branch:
                log_args.append(branch)

            if author:
                log_args.append(f"--author={author}")

            if since:
                log_args.append(f"--since={since}")

            if until:
                log_args.append(f"--until={until}")

            if grep:
                log_args.append(f"--grep={grep}")

            if stat:
                log_args.append("--stat")

            if path:
                log_args.append("--")
                log_args.append(path)

            # Get current branch
            try:
                current_branch = repo.active_branch.name
            except TypeError:
                current_branch = f"HEAD detached at {repo.head.commit.hexsha[:8]}"

            # Execute git log with custom format
            if oneline:
                log_args.insert(0, "--oneline")
                log_output = repo.git.log(*log_args)

                # Parse oneline output
                commits: List[Dict[str, Any]] = []
                for line in log_output.strip().split("\n"):
                    if line:
                        parts = line.split(" ", 1)
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1] if len(parts) > 1 else "",
                        })
            else:
                # Use a format that's easy to parse
                format_str = "%H|%h|%an|%ae|%ai|%s"
                log_args.insert(0, f"--format={format_str}")
                log_output = repo.git.log(*log_args)

                commits = []
                for line in log_output.strip().split("\n"):
                    if line and "|" in line:
                        parts = line.split("|", 5)
                        if len(parts) >= 6:
                            commit_info: Dict[str, Any] = {
                                "hash": parts[0],
                                "short_hash": parts[1],
                                "author": parts[2],
                                "author_email": parts[3],
                                "date": parts[4],
                                "message": parts[5],
                            }
                            commits.append(commit_info)

                # Get stats separately if requested
                if stat and commits:
                    stat_args = [f"-n{count}", "--stat", "--format="]
                    if branch:
                        stat_args.append(branch)
                    if author:
                        stat_args.append(f"--author={author}")
                    if since:
                        stat_args.append(f"--since={since}")
                    if until:
                        stat_args.append(f"--until={until}")
                    if path:
                        stat_args.append("--")
                        stat_args.append(path)

                    try:
                        stat_output = repo.git.log(*stat_args)
                        # Add stats to result
                        if stat_output.strip():
                            # Parse stats per commit (separated by empty lines)
                            stat_blocks = stat_output.strip().split("\n\n")
                            for i, block in enumerate(stat_blocks):
                                if i < len(commits) and block.strip():
                                    commits[i]["stats"] = block.strip()
                    except Exception:
                        pass

            result: Dict[str, Any] = {
                "status": "success",
                "repo": local_name,
                "branch": branch or current_branch,
                "commit_count": len(commits),
                "commits": commits,
            }

            if not commits:
                result["message"] = "No commits found matching the criteria."

            # Add filter info
            filters = []
            if author:
                filters.append(f"author: {author}")
            if since:
                filters.append(f"since: {since}")
            if until:
                filters.append(f"until: {until}")
            if path:
                filters.append(f"path: {path}")
            if grep:
                filters.append(f"grep: {grep}")
            if filters:
                result["filters"] = filters

            return result

        except Exception as e:
            logger.exception(f"Error getting log: {e}")
            return {"error": f"Failed to get commit history: {str(e)}"}
