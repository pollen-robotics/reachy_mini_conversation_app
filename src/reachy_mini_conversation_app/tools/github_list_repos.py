"""GitHub list local repos tool."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubListReposTool(Tool):
    """List locally cloned repositories."""

    name = "github_list_repos"
    description = (
        "List all repositories that have been cloned locally to ~/reachy_repos/. "
        "Use this to see what repos are available for other operations."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """List locally cloned repositories."""
        logger.info("Tool call: github_list_repos")

        if not REPOS_DIR.exists():
            return {
                "status": "empty",
                "message": "No repositories cloned yet.",
                "repos": [],
                "hint": "Use github_clone to clone a repository.",
            }

        repos: List[Dict[str, Any]] = []

        try:
            for item in sorted(REPOS_DIR.iterdir()):
                if item.is_dir() and (item / ".git").exists():
                    # Get some basic info about the repo
                    repo_info: Dict[str, Any] = {
                        "name": item.name,
                        "path": str(item),
                    }

                    # Try to get the remote URL
                    try:
                        config_file = item / ".git" / "config"
                        if config_file.exists():
                            content = config_file.read_text()
                            for line in content.split("\n"):
                                if "url = " in line:
                                    url = line.split("url = ")[-1].strip()
                                    # Clean up token from URL if present
                                    if "@github.com" in url:
                                        url = "https://github.com" + url.split("github.com")[-1]
                                    repo_info["remote_url"] = url
                                    break
                    except Exception:
                        pass

                    # Get current branch
                    try:
                        head_file = item / ".git" / "HEAD"
                        if head_file.exists():
                            head = head_file.read_text().strip()
                            if head.startswith("ref: refs/heads/"):
                                repo_info["branch"] = head.replace("ref: refs/heads/", "")
                    except Exception:
                        pass

                    repos.append(repo_info)

            return {
                "status": "success",
                "count": len(repos),
                "repos": repos,
            }

        except Exception as e:
            logger.exception(f"Error listing repos: {e}")
            return {"error": f"Failed to list repos: {str(e)}"}
