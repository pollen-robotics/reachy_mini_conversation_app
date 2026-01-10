"""GitHub edit file tool - AI-assisted file editing with optional model file reference."""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

import openai
import anthropic
from anthropic.types import TextBlock

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"


class GitHubEditFileTool(Tool):
    """Edit a file using AI with optional model file reference."""

    name = "github_edit_file"
    description = (
        "Edit a file using AI assistance. Can optionally use a 'model' file as reference "
        "to follow its structure, style, or patterns. The AI will analyze both files and "
        "generate the appropriate modifications. "
        "Use cases: "
        "1. Modify a file following instructions (edit_prompt required) "
        "2. Modify a file to match the structure of another file (model_path required) "
        "3. Both: modify following instructions while using a model file as reference"
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
                "description": "Path to the file to edit within the repo (e.g., 'src/main.py')",
            },
            "edit_prompt": {
                "type": "string",
                "description": "Instructions for how to edit the file (e.g., 'add error handling', 'refactor to use async')",
            },
            "model_path": {
                "type": "string",
                "description": "Optional path to a 'model' file to use as reference for structure/style",
            },
            "model_repo": {
                "type": "string",
                "description": "Repository containing the model file (defaults to same repo as target file)",
            },
            "analyzer": {
                "type": "string",
                "enum": ["claude", "openai"],
                "description": "AI provider to use: 'claude' (default) or 'openai'",
            },
            "apply": {
                "type": "boolean",
                "description": "If true, apply the changes to the file. If false, only preview the changes. Default: false",
            },
        },
        "required": ["repo", "path"],
    }

    def _read_file(self, repo_path: Path, file_path: str) -> Optional[str]:
        """Read a file from the repository."""
        full_path = repo_path / file_path

        # Security check
        try:
            full_path.resolve().relative_to(repo_path.resolve())
        except ValueError:
            return None

        if not full_path.exists():
            return None

        try:
            return full_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _build_prompt(
        self,
        target_content: str,
        target_path: str,
        edit_prompt: Optional[str] = None,
        model_content: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """Build the prompt for AI-assisted editing."""
        prompt = f"""You are an expert code editor. Your task is to modify a file based on the given instructions.

## Target file to edit: {target_path}
```
{target_content}
```
"""

        if model_content and model_path:
            prompt += f"""
## Model/Reference file: {model_path}
Use this file as a reference for structure, style, patterns, and conventions:
```
{model_content}
```
"""

        if edit_prompt:
            prompt += f"""
## Edit instructions:
{edit_prompt}
"""
        elif model_content:
            prompt += """
## Edit instructions:
Modify the target file to follow the same structure, style, patterns, and conventions as the model file.
"""

        prompt += """
## Requirements:
1. Return ONLY the complete modified file content
2. Do not include any explanations, markdown code blocks, or comments about the changes
3. Preserve the original functionality unless explicitly asked to change it
4. Follow the coding style and patterns from the model file (if provided)
5. Ensure the output is valid, syntactically correct code

Return the complete modified file content now:
"""
        return prompt

    def _edit_with_claude(self, prompt: str) -> Dict[str, Any]:
        """Generate edits using Claude."""
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not configured."}

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL or "claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                system="You are an expert code editor. Output only the modified file content, no explanations.",
            )
            first_block = response.content[0]
            content_text = first_block.text if isinstance(first_block, TextBlock) else str(first_block)
            return {"content": content_text}
        except Exception as e:
            logger.exception(f"Claude API error: {e}")
            return {"error": f"Claude API error: {str(e)}"}

    def _edit_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate edits using OpenAI."""
        api_key = config.OPENAI_API_KEY
        if not api_key:
            return {"error": "OPENAI_API_KEY not configured."}

        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert code editor. Output only the modified file content, no explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=8000,
            )
            return {"content": response.choices[0].message.content}
        except Exception as e:
            logger.exception(f"OpenAI API error: {e}")
            return {"error": f"OpenAI API error: {str(e)}"}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Edit a file using AI assistance."""
        repo_name = kwargs.get("repo", "")
        file_path = kwargs.get("path", "")
        edit_prompt = kwargs.get("edit_prompt")
        model_path = kwargs.get("model_path")
        model_repo = kwargs.get("model_repo")
        analyzer = kwargs.get("analyzer", "claude")
        apply = kwargs.get("apply", False)

        logger.info(f"Tool call: github_edit_file - repo='{repo_name}', path='{file_path}', model='{model_path}'")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not file_path:
            return {"error": "File path is required."}
        if not edit_prompt and not model_path:
            return {
                "error": "Either 'edit_prompt' or 'model_path' (or both) is required.",
                "hint": "Provide instructions for editing or a model file to follow.",
            }

        # Target repository path
        local_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        repo_path = REPOS_DIR / local_name

        if not repo_path.exists():
            return {
                "error": f"Repository not found: {local_name}",
                "hint": "Use github_clone to clone the repository first.",
            }

        # Read target file
        target_content = self._read_file(repo_path, file_path)
        if target_content is None:
            return {"error": f"Target file not found or unreadable: {file_path}"}

        # Read model file if specified
        model_content = None
        if model_path:
            if model_repo:
                model_local_name = model_repo.split("/")[-1] if "/" in model_repo else model_repo
                model_repo_path = REPOS_DIR / model_local_name
            else:
                model_repo_path = repo_path

            if not model_repo_path.exists():
                return {"error": f"Model repository not found: {model_repo or local_name}"}

            model_content = self._read_file(model_repo_path, model_path)
            if model_content is None:
                return {"error": f"Model file not found or unreadable: {model_path}"}

        # Build prompt
        prompt = self._build_prompt(
            target_content=target_content,
            target_path=file_path,
            edit_prompt=edit_prompt,
            model_content=model_content,
            model_path=model_path,
        )

        # Generate edits
        if analyzer == "openai":
            result = self._edit_with_openai(prompt)
        else:
            result = self._edit_with_claude(prompt)

        if "error" in result:
            return result

        new_content = result["content"]

        # Clean up response if it contains markdown code blocks
        if new_content.startswith("```"):
            lines = new_content.split("\n")
            # Remove first line (```language)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_content = "\n".join(lines)

        # Calculate diff summary
        old_lines = len(target_content.splitlines())
        new_lines = len(new_content.splitlines())

        response: Dict[str, Any] = {
            "status": "preview" if not apply else "applied",
            "repo": local_name,
            "path": file_path,
            "analyzer": analyzer,
            "original_lines": old_lines,
            "new_lines": new_lines,
        }

        if model_path:
            response["model_file"] = model_path
            if model_repo:
                response["model_repo"] = model_repo

        if apply:
            # Apply the changes
            try:
                full_path = repo_path / file_path
                full_path.write_text(new_content, encoding="utf-8")
                response["message"] = f"File edited successfully: {file_path}"
                response["hint"] = "Use github_add to stage the file, then github_commit to commit."
            except Exception as e:
                logger.exception(f"Error writing file: {e}")
                return {"error": f"Failed to write file: {str(e)}"}
        else:
            response["new_content"] = new_content
            response["message"] = "Preview generated. Set apply=true to apply changes."
            response["hint"] = "Review the new_content and call again with apply=true to save."

        return response
