"""Code generation tool using Claude API."""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import anthropic

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory to store generated code
CODE_OUTPUT_DIR = Path.home() / "reachy_code"


class CodeTool(Tool):
    """Generate code using Claude API."""

    name = "code"
    description = (
        "Generate code using Claude AI. "
        "Use this tool when the user asks you to write, create, or generate code. "
        "The generated code will be saved to a file and you can offer to execute it."
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
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Generate code using Claude API."""
        question = kwargs.get("question", "")
        language = kwargs.get("language", "python")
        filename = kwargs.get("filename")

        logger.info(f"Tool call: code - question='{question[:50]}...', language={language}")

        # Check for API key
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            return {
                "error": "ANTHROPIC_API_KEY is not configured. "
                "Please set it in your .env file to use code generation."
            }

        # Create output directory if it doesn't exist
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

            # Generate filename
            if not filename:
                filename = self._generate_filename(question)

            # Add extension based on language
            extension = self._get_extension(language)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_filename = f"{timestamp}_{filename}.{extension}"
            filepath = CODE_OUTPUT_DIR / full_filename

            # Save code to file
            filepath.write_text(code_content, encoding="utf-8")

            logger.info(f"Code saved to {filepath}")

            # Generate a brief explanation
            explanation = self._generate_explanation(code_content, language)

            return {
                "status": "success",
                "message": f"Code generated and saved to {filepath}",
                "filepath": str(filepath),
                "language": language,
                "explanation": explanation,
                "lines": len(code_content.splitlines()),
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
