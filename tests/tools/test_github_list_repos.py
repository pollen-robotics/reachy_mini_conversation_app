"""Unit tests for the github_list_repos tool."""

from typing import Any
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_list_repos import GitHubListReposTool


class TestGitHubListReposToolAttributes:
    """Tests for GitHubListReposTool tool attributes."""

    def test_github_list_repos_has_correct_name(self) -> None:
        """Test GitHubListReposTool tool has correct name."""
        tool = GitHubListReposTool()
        assert tool.name == "github_list_repos"

    def test_github_list_repos_has_description(self) -> None:
        """Test GitHubListReposTool tool has description."""
        tool = GitHubListReposTool()
        assert "list" in tool.description.lower()
        assert "repositories" in tool.description.lower()

    def test_github_list_repos_has_parameters_schema(self) -> None:
        """Test GitHubListReposTool tool has correct parameters schema."""
        tool = GitHubListReposTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert schema["required"] == []

    def test_github_list_repos_spec(self) -> None:
        """Test GitHubListReposTool tool spec generation."""
        tool = GitHubListReposTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_list_repos"


class TestGitHubListReposToolExecution:
    """Tests for GitHubListReposTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_list_repos_dir_not_exists(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos when repos dir doesn't exist."""
        tool = GitHubListReposTool()

        nonexistent = tmp_path / "nonexistent"

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", nonexistent):
            result = await tool(mock_deps)

        assert result["status"] == "empty"
        assert result["repos"] == []
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_list_repos_empty(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos with empty repos directory."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["repos"] == []

    @pytest.mark.asyncio
    async def test_github_list_repos_with_repos(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos with some repos."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        # Create repo1 with .git
        repo1 = repos_dir / "repo1"
        repo1.mkdir()
        (repo1 / ".git").mkdir()

        # Create repo2 with .git and config
        repo2 = repos_dir / "repo2"
        repo2.mkdir()
        git_dir = repo2 / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text('[remote "origin"]\n\turl = https://github.com/owner/repo2.git')
        (git_dir / "HEAD").write_text("ref: refs/heads/main")

        # Create non-repo directory (no .git)
        non_repo = repos_dir / "not_a_repo"
        non_repo.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["count"] == 2

        repo_names = [r["name"] for r in result["repos"]]
        assert "repo1" in repo_names
        assert "repo2" in repo_names
        assert "not_a_repo" not in repo_names

    @pytest.mark.asyncio
    async def test_github_list_repos_extracts_remote_url(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos extracts remote URL from config."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text('[remote "origin"]\n\turl = https://github.com/owner/myrepo.git')

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        assert result["repos"][0]["remote_url"] == "https://github.com/owner/myrepo.git"

    @pytest.mark.asyncio
    async def test_github_list_repos_cleans_token_from_url(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos removes token from URL."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text('[remote "origin"]\n\turl = https://ghp_secret@github.com/owner/myrepo.git')

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        # Token should be removed
        assert "ghp_secret" not in result["repos"][0]["remote_url"]
        assert "github.com" in result["repos"][0]["remote_url"]

    @pytest.mark.asyncio
    async def test_github_list_repos_extracts_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos extracts current branch."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/develop")

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        assert result["repos"][0]["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_github_list_repos_handles_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos handles exceptions."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            with patch.object(Path, "iterdir", side_effect=PermissionError("Access denied")):
                result = await tool(mock_deps)

        assert "error" in result
        assert "Failed to list" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_repos_handles_bad_config(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos handles unreadable config."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        # Config exists but doesn't have url
        (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0")

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["repos"][0]["name"] == "myrepo"
        # No remote_url since none was found
        assert "remote_url" not in result["repos"][0]

    @pytest.mark.asyncio
    async def test_github_list_repos_handles_config_read_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos handles exception when reading config (lines 66-67)."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        config_file = git_dir / "config"
        config_file.write_text("dummy")
        # Make config unreadable - we need to mock read_text to raise an exception
        # But creating a directory with same name won't work, so use a mock

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            # Patch the Path.read_text method to raise for the config file
            original_read_text = Path.read_text

            def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
                if "config" in str(self):
                    raise PermissionError("Cannot read config")
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", mock_read_text):
                result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["repos"][0]["name"] == "myrepo"
        # Should still succeed but without remote_url
        assert "remote_url" not in result["repos"][0]

    @pytest.mark.asyncio
    async def test_github_list_repos_detached_head(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos handles detached HEAD state (branch 74->79)."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        # Detached HEAD - not starting with "ref: refs/heads/"
        (git_dir / "HEAD").write_text("abc1234567890def")

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["repos"][0]["name"] == "myrepo"
        # Branch should not be set in detached HEAD state
        assert "branch" not in result["repos"][0]

    @pytest.mark.asyncio
    async def test_github_list_repos_handles_head_read_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_repos handles exception when reading HEAD (lines 76-77)."""
        tool = GitHubListReposTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        repo = repos_dir / "myrepo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        head_file = git_dir / "HEAD"
        head_file.write_text("ref: refs/heads/main")

        with patch("reachy_mini_conversation_app.tools.github_list_repos.REPOS_DIR", repos_dir):
            # Patch to raise exception when reading HEAD
            original_read_text = Path.read_text

            def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
                if "HEAD" in str(self):
                    raise OSError("Cannot read HEAD")
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", mock_read_text):
                result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["repos"][0]["name"] == "myrepo"
        # Branch should not be set due to read error
        assert "branch" not in result["repos"][0]
