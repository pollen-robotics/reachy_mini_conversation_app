"""Unit tests for the github_pull tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_pull import GitHubPullTool


class TestGitHubPullToolAttributes:
    """Tests for GitHubPullTool tool attributes."""

    def test_github_pull_has_correct_name(self) -> None:
        """Test GitHubPullTool tool has correct name."""
        tool = GitHubPullTool()
        assert tool.name == "github_pull"

    def test_github_pull_has_description(self) -> None:
        """Test GitHubPullTool tool has description."""
        tool = GitHubPullTool()
        assert "pull" in tool.description.lower()

    def test_github_pull_has_parameters_schema(self) -> None:
        """Test GitHubPullTool tool has correct parameters schema."""
        tool = GitHubPullTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_pull_spec(self) -> None:
        """Test GitHubPullTool tool spec generation."""
        tool = GitHubPullTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_pull"


class TestGitHubPullToolExecution:
    """Tests for GitHubPullTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_pull_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_pull returns error when repo is missing."""
        tool = GitHubPullTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pull_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull returns error when repo not found."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_pull_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull returns error for non-git directory."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit")

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pull_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull success."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_fetch_info = MagicMock()
        mock_fetch_info.HEAD_UPTODATE = 4  # Define the constant
        mock_fetch_info.flags = 0  # Not HEAD_UPTODATE

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_origin = MagicMock()
        mock_origin.pull.return_value = [mock_fetch_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin
        mock_repo.head.commit = mock_commit

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert "updated" in result["message"].lower()
        assert result["branch"] == "main"
        assert result["commit"] == "abc1234"

    @pytest.mark.asyncio
    async def test_github_pull_up_to_date(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull when already up to date."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_fetch_info = MagicMock()
        mock_fetch_info.HEAD_UPTODATE = 4  # Define the constant
        mock_fetch_info.flags = 4  # Set to HEAD_UPTODATE

        mock_origin = MagicMock()
        mock_origin.pull.return_value = [mock_fetch_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "up_to_date"
        assert "up to date" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_pull_empty_fetch_info(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull with empty fetch info."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_origin = MagicMock()
        mock_origin.pull.return_value = []

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "up_to_date"

    @pytest.mark.asyncio
    async def test_github_pull_conflict(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull with conflict."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_origin = MagicMock()
        mock_origin.pull.side_effect = GitCommandError("pull", "CONFLICT in file.py")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "conflict" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_pull_diverged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull with diverged branches."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_origin = MagicMock()
        mock_origin.pull.side_effect = GitCommandError("pull", "non-fast-forward update rejected")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "diverged" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_pull_generic_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull with generic git error."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_origin = MagicMock()
        mock_origin.pull.side_effect = GitCommandError("pull", "some error")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Pull failed" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pull_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull handles generic exceptions."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_origin = MagicMock()
        mock_origin.pull.side_effect = RuntimeError("Unexpected error")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to pull" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pull_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_pull handles repo name with slash."""
        tool = GitHubPullTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_origin = MagicMock()
        mock_origin.pull.return_value = []

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_pull.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_pull.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "up_to_date"
