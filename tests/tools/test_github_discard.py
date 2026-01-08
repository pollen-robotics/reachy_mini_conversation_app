"""Unit tests for the github_discard tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.github_discard import GitHubDiscardTool, REPOS_DIR
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubDiscardToolAttributes:
    """Tests for GitHubDiscardTool tool attributes."""

    def test_github_discard_has_correct_name(self) -> None:
        """Test GitHubDiscardTool tool has correct name."""
        tool = GitHubDiscardTool()
        assert tool.name == "github_discard"

    def test_github_discard_has_description(self) -> None:
        """Test GitHubDiscardTool tool has description."""
        tool = GitHubDiscardTool()
        assert "discard" in tool.description.lower()

    def test_github_discard_has_parameters_schema(self) -> None:
        """Test GitHubDiscardTool tool has correct parameters schema."""
        tool = GitHubDiscardTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "files" in schema["properties"]
        assert "untracked" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_discard_spec(self) -> None:
        """Test GitHubDiscardTool tool spec generation."""
        tool = GitHubDiscardTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_discard"


class TestGitHubDiscardToolExecution:
    """Tests for GitHubDiscardTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_discard_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_discard returns error when not confirmed."""
        tool = GitHubDiscardTool()

        result = await tool(mock_deps, repo="myrepo", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_discard_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_discard returns error when repo is missing."""
        tool = GitHubDiscardTool()

        result = await tool(mock_deps, repo="", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_discard_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard returns error when repo not found."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", confirmed=True)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_discard_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard returns error for non-git directory."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", confirmed=True)

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_discard_all_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard discards all changes."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file1.txt\nfile2.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        assert "discarded_files" in result
        mock_git.checkout.assert_called_with("--", ".")

    @pytest.mark.asyncio
    async def test_github_discard_specific_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard discards specific files."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file1.txt\nfile2.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file1.txt"], confirmed=True)

        assert result["status"] == "success"
        mock_git.checkout.assert_called_with("--", "file1.txt")

    @pytest.mark.asyncio
    async def test_github_discard_with_untracked(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard with untracked files removal."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = ["new1.txt", "new2.txt"]

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", untracked=True, confirmed=True)

        assert result["status"] == "success"
        assert "cleaned_files" in result
        assert result["cleaned_count"] == 2
        mock_git.clean.assert_called_with("-fd")

    @pytest.mark.asyncio
    async def test_github_discard_no_changes(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard when no changes to discard."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = ""

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        assert "no changes" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_discard_file_error_logged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard logs error for individual file failure."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file1.txt\nfile2.txt"
        mock_git.checkout.side_effect = [GitCommandError("checkout", "error"), None]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                # With specific files, errors are logged but continue
                result = await tool(mock_deps, repo="myrepo", files=["file1.txt", "file2.txt"], confirmed=True)

        assert result["status"] == "success"
        # Should have processed file2.txt successfully
        assert "discarded_files" in result
        assert "file2.txt" in result["discarded_files"]

    @pytest.mark.asyncio
    async def test_github_discard_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard handles git error."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = GitCommandError("diff", "error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "git command failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_discard_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard handles generic exception."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = RuntimeError("Unexpected error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "Failed to discard" in result["error"]

    @pytest.mark.asyncio
    async def test_github_discard_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard handles repo name with slash."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo", confirmed=True)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_discard_only_untracked(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard with only untracked files (no modified)."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = ""  # No modified files

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = ["untracked.txt"]

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", untracked=True, confirmed=True)

        assert result["status"] == "success"
        assert "cleaned_files" in result
        assert "discarded_files" not in result or len(result.get("discarded_files", [])) == 0

    @pytest.mark.asyncio
    async def test_github_discard_default_files_is_dot(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_discard defaults to '.' for files."""
        tool = GitHubDiscardTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_discard.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_discard.Repo", return_value=mock_repo):
                # Not passing files parameter
                result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        # Should have called checkout with "." (all files)
        mock_git.checkout.assert_called_with("--", ".")
