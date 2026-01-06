"""GitHub commit tool with semantic-release format using GitPython and OpenAI."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from git import Repo, InvalidGitRepositoryError, GitCommandError
from openai import OpenAI

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repos are cloned
REPOS_DIR = Path.home() / "reachy_repos"

# Semantic release commit types
COMMIT_TYPES = {
    "feat": "A new feature",
    "fix": "A bug fix",
    "docs": "Documentation only changes",
    "style": "Changes that do not affect the meaning of the code",
    "refactor": "A code change that neither fixes a bug nor adds a feature",
    "perf": "A code change that improves performance",
    "test": "Adding missing tests or correcting existing tests",
    "build": "Changes that affect the build system or external dependencies",
    "ci": "Changes to CI configuration files and scripts",
    "chore": "Other changes that don't modify src or test files",
    "revert": "Reverts a previous commit",
}


class GitHubCommitTool(Tool):
    """Stage and commit changes with semantic-release format."""

    name = "github_commit"
    description = (
        "Stage files (add/remove) and create a commit in a local repository using semantic-release format. "
        "Supports auto-generating commit messages using OpenAI based on diff analysis. "
        "IMPORTANT: Always ask user for confirmation before calling this tool. "
        "Commit types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (the folder name in ~/reachy_repos/)",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of files to stage (relative paths). Use '.' to stage all changes.",
            },
            "remove_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of files to remove from git (git rm). Use for deleted files.",
            },
            "type": {
                "type": "string",
                "enum": list(COMMIT_TYPES.keys()),
                "description": "Commit type. If not provided and auto_message=true, will be auto-detected.",
            },
            "scope": {
                "type": "string",
                "description": "Optional scope of the change (e.g., 'api', 'ui', 'auth')",
            },
            "message": {
                "type": "string",
                "description": "Short description. If not provided and auto_message=true, will be auto-generated.",
            },
            "body": {
                "type": "string",
                "description": "Optional longer description of the change",
            },
            "breaking": {
                "type": "boolean",
                "description": "If true, marks this as a breaking change (adds ! after type)",
            },
            "auto_message": {
                "type": "boolean",
                "description": "If true, auto-generate commit message using OpenAI based on diff analysis.",
            },
            "issue_context": {
                "type": "string",
                "description": "Optional issue context (title, description) to help generate better commit message.",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm commit. Always ask user first.",
            },
        },
        "required": ["repo", "confirmed"],
    }

    def _build_commit_message(
        self,
        commit_type: str,
        message: str,
        scope: Optional[str] = None,
        body: Optional[str] = None,
        breaking: bool = False,
    ) -> str:
        """Build a semantic-release compliant commit message."""
        header = commit_type
        if scope:
            header += f"({scope})"
        if breaking:
            header += "!"
        header += f": {message}"

        full_message = header
        if body:
            full_message += f"\n\n{body}"
        if breaking:
            full_message += "\n\nBREAKING CHANGE: This is a breaking change."

        return full_message

    def _generate_commit_message(
        self,
        diff: str,
        staged_files: List[str],
        issue_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate commit message using OpenAI based on diff analysis."""
        if not config.OPENAI_API_KEY:
            return {"error": "OPENAI_API_KEY not configured for auto-message generation."}

        try:
            client = OpenAI(api_key=config.OPENAI_API_KEY)

            # Truncate diff if too long
            max_diff_length = 8000
            if len(diff) > max_diff_length:
                diff = diff[:max_diff_length] + "\n... (diff truncated)"

            prompt = f"""Analyze the following git diff and generate a semantic-release commit message.

## Staged files:
{', '.join(staged_files)}

## Git diff:
```
{diff}
```
"""
            if issue_context:
                prompt += f"""
## Issue context:
{issue_context}
"""

            prompt += """
## Instructions:
Generate a commit message following semantic-release conventions.
Return ONLY a JSON object with these fields:
- "type": one of (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
- "scope": optional scope (short, lowercase, e.g., "api", "ui", "auth")
- "message": short description in imperative mood (e.g., "add user login", "fix null pointer")
- "body": optional longer description if the change is complex
- "breaking": boolean, true if this is a breaking change

Example response:
{"type": "feat", "scope": "auth", "message": "add OAuth2 login support", "body": null, "breaking": false}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates semantic-release commit messages. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            import json
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            result = json.loads(content)
            return {
                "type": result.get("type", "chore"),
                "scope": result.get("scope"),
                "message": result.get("message", "update code"),
                "body": result.get("body"),
                "breaking": result.get("breaking", False),
            }

        except Exception as e:
            logger.exception(f"Error generating commit message: {e}")
            return {"error": f"Failed to generate commit message: {str(e)}"}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stage and commit changes using GitPython."""
        repo_name = kwargs.get("repo", "")
        commit_type = kwargs.get("type")
        scope = kwargs.get("scope")
        message = kwargs.get("message")
        body = kwargs.get("body")
        breaking = kwargs.get("breaking", False)
        files: List[str] = kwargs.get("files", [])
        remove_files: List[str] = kwargs.get("remove_files", [])
        auto_message = kwargs.get("auto_message", False)
        issue_context = kwargs.get("issue_context")
        confirmed = kwargs.get("confirmed", False)

        logger.info(f"Tool call: github_commit - repo='{repo_name}', type={commit_type}, auto={auto_message}")

        # Check confirmation
        if not confirmed:
            return {
                "error": "Commit not confirmed. Please ask the user for confirmation first.",
                "hint": "Set confirmed=true after getting user approval.",
            }

        if not repo_name:
            return {"error": "Repository name is required."}

        # Check if we have files to stage or message requirements
        if not files and not remove_files:
            return {"error": "At least one file must be specified (files or remove_files)."}

        if not auto_message and (not commit_type or not message):
            return {
                "error": "Either provide 'type' and 'message', or set 'auto_message=true' for auto-generation.",
            }

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
            # Configure git user if GITHUB_DEFAULT_OWNER is set
            owner = config.GITHUB_DEFAULT_OWNER
            email = config.GITHUB_OWNER_EMAIL or (f"{owner}@users.noreply.github.com" if owner else None)
            if owner:
                with repo.config_writer() as git_config:
                    try:
                        repo.config_reader().get_value("user", "name")
                    except Exception:
                        git_config.set_value("user", "name", owner)
                    try:
                        repo.config_reader().get_value("user", "email")
                    except Exception:
                        if email:
                            git_config.set_value("user", "email", email)

            # Remove files (git rm)
            removed_files = []
            for file in remove_files:
                try:
                    repo.index.remove([file], working_tree=True)
                    removed_files.append(file)
                except GitCommandError as e:
                    # File might already be deleted, try to remove from index only
                    try:
                        repo.index.remove([file], working_tree=False)
                        removed_files.append(file)
                    except Exception:
                        logger.warning(f"Could not remove file {file}: {e}")

            # Stage files (git add)
            for file in files:
                if file == ".":
                    repo.git.add(A=True)
                else:
                    # Check if file exists, if not it might be deleted
                    file_path = repo_path / file
                    if file_path.exists():
                        repo.index.add([file])
                    else:
                        # File was deleted, remove from index
                        try:
                            repo.index.remove([file], working_tree=False)
                            removed_files.append(file)
                        except Exception:
                            pass

            # Get staged files
            staged_output = repo.git.diff("--cached", "--name-only")
            staged_files = [f for f in staged_output.strip().split("\n") if f]

            if not staged_files:
                return {
                    "status": "nothing_to_commit",
                    "message": "No changes to commit.",
                    "hint": "Make sure you've made changes and specified the correct files.",
                }

            # Auto-generate commit message if requested
            if auto_message:
                diff = repo.git.diff("--cached")
                generated = self._generate_commit_message(diff, staged_files, issue_context)

                if "error" in generated:
                    return generated

                commit_type = generated["type"]
                scope = generated.get("scope") or scope
                message = generated["message"]
                body = generated.get("body") or body
                breaking = generated.get("breaking", False) or breaking

            # Validate commit type and message
            if not commit_type:
                return {"error": f"Commit type is required. Valid types: {list(COMMIT_TYPES.keys())}"}
            if commit_type not in COMMIT_TYPES:
                return {"error": f"Invalid commit type '{commit_type}'. Valid types: {list(COMMIT_TYPES.keys())}"}
            if not message:
                return {"error": "Commit message is required."}

            # Build commit message
            commit_msg = self._build_commit_message(
                commit_type=commit_type,
                message=message,
                scope=scope,
                body=body,
                breaking=breaking,
            )

            # Create commit
            commit = repo.index.commit(commit_msg)
            commit_hash = commit.hexsha[:7]

            result = {
                "status": "success",
                "message": "Commit created successfully!",
                "repo": local_name,
                "commit_hash": commit_hash,
                "commit_type": commit_type,
                "commit_message": commit_msg.split("\n")[0],
                "files_committed": staged_files,
                "hint": "Use github_push to push this commit to remote.",
            }

            if removed_files:
                result["files_removed"] = removed_files

            if auto_message:
                result["auto_generated"] = True

            return result

        except GitCommandError as e:
            error_msg = str(e)
            if "nothing to commit" in error_msg.lower():
                return {"status": "nothing_to_commit", "message": "No changes to commit."}
            return {"error": f"Git command failed: {error_msg}"}
        except Exception as e:
            logger.exception(f"Error creating commit: {e}")
            return {"error": f"Failed to create commit: {str(e)}"}
