"""GitHub read file tool with optional AI analysis."""

import logging
from typing import Any, Dict
from pathlib import Path

import openai
import anthropic

from reachy_mini_conversation_app.config import config
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

# Default analysis prompts
DEFAULT_ANALYSIS_PROMPT = (
    "Analyze this code file and provide:\n"
    "1. A brief summary of what the code does\n"
    "2. Key functions/classes and their purpose\n"
    "3. Any potential issues or improvements\n"
    "4. Dependencies used\n"
    "Be concise but thorough."
)


class GitHubReadFileTool(Tool):
    """Read a file from a cloned repository with optional AI analysis."""

    name = "github_read_file"
    description = (
        "Read the contents of a file from a cloned GitHub repository. "
        "Optionally analyze the file using Claude or OpenAI. "
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
            "analyze": {
                "type": "boolean",
                "description": "If true, analyze the file content using AI. Default: false",
            },
            "analyzer": {
                "type": "string",
                "enum": ["claude", "openai"],
                "description": "AI model to use for analysis: 'claude' or 'openai'. Default: 'claude'",
            },
            "analysis_prompt": {
                "type": "string",
                "description": "Custom prompt for the analysis. If not provided, uses a default code analysis prompt.",
            },
        },
        "required": ["repo", "path"],
    }

    async def _analyze_with_claude(
        self, content: str, file_path: str, prompt: str
    ) -> Dict[str, Any]:
        """Analyze file content using Claude."""
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY is not configured."}

        try:
            client = anthropic.Anthropic(api_key=api_key)
            model = config.ANTHROPIC_MODEL or "claude-sonnet-4-20250514"

            user_message = f"File: {file_path}\n\n```\n{content}\n```\n\n{prompt}"

            message = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": user_message}],
            )

            return {
                "analyzer": "claude",
                "model": model,
                "analysis": message.content[0].text,
            }

        except anthropic.AuthenticationError:
            return {"error": "Invalid ANTHROPIC_API_KEY."}
        except anthropic.RateLimitError:
            return {"error": "Claude rate limit exceeded. Try again later."}
        except Exception as e:
            logger.exception(f"Claude analysis error: {e}")
            return {"error": f"Claude analysis failed: {str(e)}"}

    async def _analyze_with_openai(
        self, content: str, file_path: str, prompt: str
    ) -> Dict[str, Any]:
        """Analyze file content using OpenAI."""
        api_key = config.OPENAI_API_KEY
        if not api_key:
            return {"error": "OPENAI_API_KEY is not configured."}

        try:
            client = openai.OpenAI(api_key=api_key)
            model = config.OPENAI_MODEL or "gpt-4o"

            user_message = f"File: {file_path}\n\n```\n{content}\n```\n\n{prompt}"

            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": user_message}],
            )

            return {
                "analyzer": "openai",
                "model": model,
                "analysis": response.choices[0].message.content,
            }

        except openai.AuthenticationError:
            return {"error": "Invalid OPENAI_API_KEY."}
        except openai.RateLimitError:
            return {"error": "OpenAI rate limit exceeded. Try again later."}
        except Exception as e:
            logger.exception(f"OpenAI analysis error: {e}")
            return {"error": f"OpenAI analysis failed: {str(e)}"}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Read a file from a repository with optional AI analysis."""
        repo_name = kwargs.get("repo", "")
        file_path = kwargs.get("path", "")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        analyze = kwargs.get("analyze", False)
        analyzer = kwargs.get("analyzer", "claude")
        analysis_prompt = kwargs.get("analysis_prompt", DEFAULT_ANALYSIS_PROMPT)

        logger.info(f"Tool call: github_read_file - repo='{repo_name}', path='{file_path}', analyze={analyze}")

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

            result: Dict[str, Any] = {
                "status": "success",
                "repo": local_name,
                "path": file_path,
                "total_lines": total_lines,
                "lines_returned": len(lines),
                "truncated": truncated,
                "content": content,
            }

            # Perform AI analysis if requested
            if analyze:
                if analyzer == "openai":
                    analysis_result = await self._analyze_with_openai(
                        content, file_path, analysis_prompt
                    )
                else:
                    # Default to Claude
                    analysis_result = await self._analyze_with_claude(
                        content, file_path, analysis_prompt
                    )

                if "error" in analysis_result:
                    result["analysis_error"] = analysis_result["error"]
                else:
                    result["analysis"] = analysis_result

            return result

        except UnicodeDecodeError:
            return {"error": "Cannot read file: not a text file or unknown encoding."}
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            return {"error": f"Failed to read file: {str(e)}"}
