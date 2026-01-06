"""GitHub read file tool."""

import logging
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"

# Maximum file size to read (in bytes)
MAX_FILE_SIZE = 100 * 1024  # 100 KB

# Text file extensions
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".md", ".txt", ".rst", ".html", ".css", ".scss", ".less",
    ".xml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".sh", ".bash", ".zsh", ".fish",
    ".c", ".cpp", ".h", ".hpp", ".java", ".kt", ".scala",
    ".go", ".rs", ".rb", ".php", ".pl", ".lua",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".gitignore", ".gitattributes",
    ".makefile", ".cmake",
}


class GitHubReadFileTool(Tool):
    """Read a file from a cloned repository."""

    name = "github_read_file"
    description = (
        "Read the contents of a file from a cloned GitHub repository. "
        "Use this to view source code, configuration files, or documentation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "path": {
                "type": "string",
                "description": "Path to the file within the repo (e.g., 'src/main.py')",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional starting line number (1-indexed)",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional ending line number (1-indexed)",
            },
        },
        "required": ["repo", "path"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Read a file from a repository."""
        repo_name = kwargs.get("repo", "")
        file_path = kwargs.get("path", "")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")

        logger.info(f"Tool call: github_read_file - repo='{repo_name}', path='{file_path}'")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not file_path:
            return {"error": "File path is required."}

        # Handle owner/repo format
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {
                "error": f"Repository not found at {repo_path}",
                "hint": "Use github_clone to clone the repository first.",
            }

        # Build full file path
        full_path = repo_path / file_path

        # Security check: ensure path is within repo
        try:
            full_path.resolve().relative_to(repo_path.resolve())
        except ValueError:
            return {"error": "Invalid path: cannot access files outside the repository."}

        if not full_path.exists():
            return {"error": f"File not found: {file_path}"}

        if full_path.is_dir():
            return {
                "error": f"'{file_path}' is a directory, not a file.",
                "hint": "Use github_list_files to explore directories.",
            }

        # Check file size
        file_size = full_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "error": f"File is too large ({file_size} bytes). Maximum size is {MAX_FILE_SIZE} bytes.",
                "hint": "Use start_line and end_line to read a portion of the file.",
            }

        # Check if it's a text file
        suffix = full_path.suffix.lower()
        if suffix not in TEXT_EXTENSIONS and suffix != "":
            # Try to read anyway but warn
            pass

        try:
            content = full_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            total_lines = len(lines)

            # Apply line range if specified
            if start_line or end_line:
                start_idx = (start_line - 1) if start_line and start_line > 0 else 0
                end_idx = end_line if end_line else total_lines
                lines = lines[start_idx:end_idx]
                content = "\n".join(lines)

            # Truncate if still too long
            max_chars = 50000
            truncated = False
            if len(content) > max_chars:
                content = content[:max_chars]
                truncated = True

            return {
                "status": "success",
                "repo": local_name,
                "path": file_path,
                "total_lines": total_lines,
                "lines_returned": len(lines),
                "truncated": truncated,
                "content": content,
            }

        except UnicodeDecodeError:
            return {"error": "Cannot read file: not a text file or unknown encoding."}
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return {"error": f"Failed to read file: {str(e)}"}
