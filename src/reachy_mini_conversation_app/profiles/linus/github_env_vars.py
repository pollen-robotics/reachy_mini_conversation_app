"""Shared environment variable declarations for GitHub tools."""

from reachy_mini_conversation_app.tools.core_tools import EnvVar


# Environment variables required by GitHub tools
# These are shared across all GitHub-related tools to avoid duplication
GITHUB_ENV_VARS = [
    EnvVar(
        "GITHUB_TOKEN",
        is_secret=True,
        description="GitHub Personal Access Token for API operations",
    ),
    EnvVar(
        "GITHUB_DEFAULT_OWNER",
        is_secret=False,
        description="Default GitHub owner/organization for repositories",
        required=False,
    ),
    EnvVar(
        "GITHUB_OWNER_EMAIL",
        is_secret=False,
        description="Email for git commits (defaults to owner@users.noreply.github.com)",
        required=False,
    ),
]
