"""Unit tests for the github_clone tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_clone import GitHubCloneTool


class TestGitHubCloneToolAttributes:
    """Tests for GitHubCloneTool tool attributes."""

    def test_github_clone_has_correct_name(self) -> None:
        """Test GitHubCloneTool tool has correct name."""
        tool = GitHubCloneTool()
        assert tool.name == "github_clone"

    def test_github_clone_has_description(self) -> None:
        """Test GitHubCloneTool tool has description."""
        tool = GitHubCloneTool()
        assert "clone" in tool.description.lower()
        assert "repository" in tool.description.lower()

    def test_github_clone_has_parameters_schema(self) -> None:
        """Test GitHubCloneTool tool has correct parameters schema."""
        tool = GitHubCloneTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "branch" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_clone_spec(self) -> None:
        """Test GitHubCloneTool tool spec generation."""
        tool = GitHubCloneTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_clone"


class TestGitHubCloneToolHelpers:
    """Tests for GitHubCloneTool helper methods."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubCloneTool()
        result = tool._get_full_repo_name("owner/myrepo")
        assert result == "owner/myrepo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name uses GITHUB_DEFAULT_OWNER."""
        tool = GitHubCloneTool()

        with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default_owner"
            result = tool._get_full_repo_name("myrepo")

        assert result == "default_owner/myrepo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises without owner."""
        tool = GitHubCloneTool()

        with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError) as exc_info:
                tool._get_full_repo_name("myrepo")

        assert "must include owner" in str(exc_info.value)

    def test_get_clone_url_with_token(self) -> None:
        """Test _get_clone_url includes token when available."""
        tool = GitHubCloneTool()

        with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_clone_url("owner/repo")

        assert "ghp_test123@github.com" in result
        assert "owner/repo.git" in result

    def test_get_clone_url_without_token(self) -> None:
        """Test _get_clone_url works without token."""
        tool = GitHubCloneTool()

        with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = tool._get_clone_url("owner/repo")

        assert result == "https://github.com/owner/repo.git"


class TestGitHubCloneToolExecution:
    """Tests for GitHubCloneTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_clone_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_clone returns error when repo is missing."""
        tool = GitHubCloneTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_clone_invalid_repo_name(self, mock_deps: ToolDependencies) -> None:
        """Test github_clone returns error for invalid repo name."""
        tool = GitHubCloneTool()

        with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None

            result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_clone_already_exists(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone returns exists status when repo already cloned."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = None

                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "exists"
        assert "already exists" in result["message"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_clone_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone successfully clones repo."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        mock_repo = MagicMock()
        mock_config_writer = MagicMock()
        mock_repo.config_writer.return_value.__enter__ = MagicMock(return_value=mock_config_writer)
        mock_repo.config_writer.return_value.__exit__ = MagicMock(return_value=None)

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = "ghp_token"
                mock_config.GITHUB_OWNER_EMAIL = "owner@example.com"

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.return_value = mock_repo

                    result = await tool(mock_deps, repo="myrepo", branch="develop")

        assert result["status"] == "success"
        assert result["repo"] == "owner/myrepo"
        assert result["branch"] == "develop"
        mock_repo_class.clone_from.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_clone_success_no_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone without branch uses default."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        mock_repo = MagicMock()
        mock_config_writer = MagicMock()
        mock_repo.config_writer.return_value.__enter__ = MagicMock(return_value=mock_config_writer)
        mock_repo.config_writer.return_value.__exit__ = MagicMock(return_value=None)

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = None
                mock_config.GITHUB_TOKEN = None
                mock_config.GITHUB_OWNER_EMAIL = None

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.return_value = mock_repo

                    result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "success"
        assert result["branch"] == "default"

    @pytest.mark.asyncio
    async def test_github_clone_not_found_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone handles repository not found."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.side_effect = GitCommandError("clone", "404 not found")

                    result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_clone_auth_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone handles authentication error."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = "bad_token"

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.side_effect = GitCommandError("clone", "403 authentication failed")

                    result = await tool(mock_deps, repo="private")

        assert "error" in result
        assert "Authentication" in result["error"]

    @pytest.mark.asyncio
    async def test_github_clone_generic_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone handles generic git error."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = "secret_token"

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.side_effect = GitCommandError("clone", "some secret_token error")

                    result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Clone failed" in result["error"]
        # Token should be hidden
        assert "secret_token" not in result["error"]

    @pytest.mark.asyncio
    async def test_github_clone_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone handles generic exceptions."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = "my_token"

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.side_effect = RuntimeError("Unexpected my_token error")

                    result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to clone" in result["error"]
        # Token should be hidden
        assert "my_token" not in result["error"]

    @pytest.mark.asyncio
    async def test_github_clone_generic_exception_no_token(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone handles generic exceptions without token."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                mock_config.GITHUB_TOKEN = None  # No token

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.side_effect = RuntimeError("Unexpected error")

                    result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to clone" in result["error"]
        assert "Unexpected error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_clone_success_with_owner_no_custom_email(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_clone sets default email when GITHUB_OWNER_EMAIL is not set."""
        tool = GitHubCloneTool()

        repos_dir = tmp_path / "reachy_repos"

        mock_repo = MagicMock()
        mock_config_writer = MagicMock()
        mock_repo.config_writer.return_value.__enter__ = MagicMock(return_value=mock_config_writer)
        mock_repo.config_writer.return_value.__exit__ = MagicMock(return_value=None)

        with patch("reachy_mini_conversation_app.tools.github_clone.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_clone.config") as mock_config:
                mock_config.GITHUB_DEFAULT_OWNER = "myowner"
                mock_config.GITHUB_TOKEN = None
                mock_config.GITHUB_OWNER_EMAIL = None  # No custom email, will use default

                with patch("reachy_mini_conversation_app.tools.github_clone.Repo") as mock_repo_class:
                    mock_repo_class.clone_from.return_value = mock_repo

                    result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        # Check that git config was set with the default noreply email
        mock_config_writer.set_value.assert_any_call("user", "name", "myowner")
        mock_config_writer.set_value.assert_any_call("user", "email", "myowner@users.noreply.github.com")
