"""Code generation tool using Claude API."""

import re
import logging
from typing import Any, Dict
from pathlib import Path
from datetime import datetime

import anthropic

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory to store generated code
CODE_OUTPUT_DIR = Path.home() / "reachy_code"
REPOS_DIR = Path.home() / "reachy_repos"


class CodeTool(Tool):
    """Generate code using Claude API."""

    name = "code"
    description = (
        "Generate code using Claude AI. "
        "Use this tool when the user asks you to write, create, or generate code. "
        "Can save to ~/reachy_code/ (default) or directly to a repository in ~/reachy_repos/."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The coding question or task to solve",
            },
            "language": {
                "type": "string",
                "description": "Programming language (e.g., python, javascript, rust). Defaults to python.",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for the generated code (without extension)",
            },
            "repo": {
                "type": "string",
                "description": "Repository name to write to (folder in ~/reachy_repos/). If set, writes directly to repo.",
            },
            "path": {
                "type": "string",
                "description": "Path within the repo (e.g., 'src/utils/helper.py'). Required if repo is set.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, overwrite existing file in repo. Default: false",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Generate code using Claude API."""
        question = kwargs.get("question", "")
        language = kwargs.get("language", "python")
        filename = kwargs.get("filename")
        repo = kwargs.get("repo")
        repo_path_str = kwargs.get("path")
        overwrite = kwargs.get("overwrite", False)

        logger.info(f"Tool call: code - question='{question[:50]}...', language={language}, repo={repo}")

        # Check for API key
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            return {
                "error": "ANTHROPIC_API_KEY is not configured. "
                "Please set it in your .env file to use code generation."
            }

        # Validate repo parameters
        if repo and not repo_path_str:
            return {"error": "Path within repo is required when repo is specified."}

        # Check repo exists if specified
        if repo:
            local_name = repo.split("/")[-1] if "/" in repo else repo
            repo_dir = REPOS_DIR / local_name
            if not repo_dir.exists():
                return {
                    "error": f"Repository not found: {local_name}",
                    "hint": "Use github_clone to clone the repository first.",
                }

            # Check destination
            dest_file = repo_dir / repo_path_str
            if dest_file.exists() and not overwrite:
                return {
                    "error": f"File already exists: {repo_path_str}",
                    "hint": "Set overwrite=true to replace it.",
                }

            # Validate path is within repo
            try:
                dest_file.resolve().relative_to(repo_dir.resolve())
            except ValueError:
                return {"error": "Invalid path: cannot write outside the repository."}

        # Create output directory if it doesn't exist (for non-repo mode)
        if not repo:
            CODE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Build the prompt for Claude
        system_prompt = (
            "You are a skilled programmer. Generate clean, well-documented code. "
            "Return ONLY the code without any explanation or markdown formatting. "
            "Include comments in the code to explain what it does. "
            "If the code needs to demonstrate output, include a main section or example usage."
        )

        user_prompt = f"Write {language} code for the following task:\n\n{question}"

        try:
            # Call Claude API
            client = anthropic.Anthropic(api_key=api_key)
            model = config.ANTHROPIC_MODEL or "claude-sonnet-4-20250514"

            message = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract code from response
            code_content = message.content[0].text

            # Clean up code if it's wrapped in markdown code blocks
            code_content = self._extract_code_from_markdown(code_content)

            # Generate a brief explanation
            explanation = self._generate_explanation(code_content, language)

            # Save to repo or to reachy_code
            if repo:
                # Write directly to repo
                local_name = repo.split("/")[-1] if "/" in repo else repo
                repo_dir = REPOS_DIR / local_name
                dest_file = repo_dir / repo_path_str

                # Create parent directories if needed
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Write file
                dest_file.write_text(code_content, encoding="utf-8")

                logger.info(f"Code saved to {dest_file}")

                return {
                    "status": "success",
                    "message": f"Code generated and saved to {repo_path_str} in {local_name}",
                    "filepath": str(dest_file),
                    "repo": local_name,
                    "relative_path": repo_path_str,
                    "language": language,
                    "explanation": explanation,
                    "lines": len(code_content.splitlines()),
                    "hint": "Use github_add to stage the file, then github_commit to commit.",
                }
            else:
                # Save to reachy_code (original behavior)
                if not filename:
                    filename = self._generate_filename(question)

                extension = self._get_extension(language)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_filename = f"{timestamp}_{filename}.{extension}"
                filepath = CODE_OUTPUT_DIR / full_filename

                filepath.write_text(code_content, encoding="utf-8")

                logger.info(f"Code saved to {filepath}")

                return {
                    "status": "success",
                    "message": f"Code generated and saved to {filepath}",
                    "filepath": str(filepath),
                    "filename": full_filename,
                    "language": language,
                    "explanation": explanation,
                    "lines": len(code_content.splitlines()),
                    "hint": "Use code_move_to_repo to move this file to a repository.",
                }

        except anthropic.AuthenticationError:
            return {"error": "Invalid ANTHROPIC_API_KEY. Please check your API key."}
        except anthropic.RateLimitError:
            return {"error": "Rate limit exceeded. Please try again later."}
        except Exception as e:
            logger.exception(f"Error generating code: {e}")
            return {"error": f"Failed to generate code: {str(e)}"}

    def _extract_code_from_markdown(self, content: str) -> str:
        """Extract code from markdown code blocks if present."""
        # Match ```language\n...code...\n```
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()
        return content.strip()

    def _generate_filename(self, question: str) -> str:
        """Generate a filename from the question."""
        # Extract key words from the question
        words = re.findall(r"\b\w+\b", question.lower())
        # Filter out common words
        stopwords = {"a", "an", "the", "to", "for", "of", "in", "on", "at", "is", "are", "write", "create", "make", "code", "function", "script", "program"}
        keywords = [w for w in words if w not in stopwords and len(w) > 2][:3]

        if keywords:
            return "_".join(keywords)
        return "generated_code"

    def _get_extension(self, language: str) -> str:
        """Get file extension for a language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "rust": "rs",
            "go": "go",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "ruby": "rb",
            "php": "php",
            "swift": "swift",
            "kotlin": "kt",
            "scala": "scala",
            "shell": "sh",
            "bash": "sh",
            "sql": "sql",
            "html": "html",
            "css": "css",
        }
        return extensions.get(language.lower(), "txt")

    def _generate_explanation(self, code: str, language: str) -> str:
        """Generate a brief explanation of the code."""
        lines = code.splitlines()
        num_lines = len(lines)

        # Count functions/classes
        if language.lower() == "python":
            functions = len(re.findall(r"^\s*def\s+\w+", code, re.MULTILINE))
            classes = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))
        else:
            functions = len(re.findall(r"\bfunction\s+\w+|\w+\s*\([^)]*\)\s*{", code))
            classes = len(re.findall(r"\bclass\s+\w+", code))

        parts = [f"{num_lines} lines of {language}"]
        if functions:
            parts.append(f"{functions} function(s)")
        if classes:
            parts.append(f"{classes} class(es)")

        return ", ".join(parts)
