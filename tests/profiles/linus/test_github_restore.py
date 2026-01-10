"""Unit tests for the github_restore tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_restore import GitHubRestoreTool


class TestGitHubRestoreToolAttributes:
    """Tests for GitHubRestoreTool tool attributes."""

    def test_github_restore_has_correct_name(self) -> None:
        """Test GitHubRestoreTool tool has correct name."""
        tool = GitHubRestoreTool()
        assert tool.name == "github_restore"

    def test_github_restore_has_description(self) -> None:
        """Test GitHubRestoreTool tool has description."""
        tool = GitHubRestoreTool()
        assert "restore" in tool.description.lower()

    def test_github_restore_has_parameters_schema(self) -> None:
        """Test GitHubRestoreTool tool has correct parameters schema."""
        tool = GitHubRestoreTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "files" in schema["properties"]
        assert "staged" in schema["properties"]
        assert "worktree" in schema["properties"]
        assert "source" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "files" in schema["required"]

    def test_github_restore_spec(self) -> None:
        """Test GitHubRestoreTool tool spec generation."""
        tool = GitHubRestoreTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_restore"


class TestGitHubRestoreToolExecution:
    """Tests for GitHubRestoreTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_restore_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_restore returns error when repo is missing."""
        tool = GitHubRestoreTool()

        result = await tool(mock_deps, repo="", files=["file.txt"])

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_restore_missing_files(self, mock_deps: ToolDependencies) -> None:
        """Test github_restore returns error when files is missing."""
        tool = GitHubRestoreTool()

        result = await tool(mock_deps, repo="myrepo", files=[])

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_restore_worktree_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_restore worktree requires confirmation."""
        tool = GitHubRestoreTool()

        result = await tool(mock_deps, repo="myrepo", files=["file.txt"], worktree=True, confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_restore_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore returns error when repo not found."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", files=["file.txt"])

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_restore_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore returns error for non-git directory."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", files=["file.txt"])

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_restore_staged_default(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore defaults to staged mode."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"])

        assert result["status"] == "success"
        assert result["staged"] is True
        assert result["worktree"] is False
        mock_git.restore.assert_called()

    @pytest.mark.asyncio
    async def test_github_restore_staged_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore staged mode success."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"], staged=True)

        assert result["status"] == "success"
        assert "unstaged" in result["message"]
        assert result["staged"] is True
        mock_git.restore.assert_called_with("--staged", "--", "file.txt")

    @pytest.mark.asyncio
    async def test_github_restore_worktree_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore worktree mode success."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged diff
            "file.txt",  # worktree diff
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"], worktree=True, confirmed=True)

        assert result["status"] == "success"
        assert "restored worktree" in result["message"]
        assert result["worktree"] is True
        mock_git.restore.assert_called_with("--worktree", "--", "file.txt")

    @pytest.mark.asyncio
    async def test_github_restore_both_staged_and_worktree(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore with both staged and worktree."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "file.txt",  # staged diff
            "file.txt",  # worktree diff
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"], staged=True, worktree=True, confirmed=True)

        assert result["status"] == "success"
        assert result["staged"] is True
        assert result["worktree"] is True
        mock_git.restore.assert_called_with("--staged", "--worktree", "--", "file.txt")

    @pytest.mark.asyncio
    async def test_github_restore_with_source(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore with source commit."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"], staged=True, source="HEAD~1")

        assert result["status"] == "success"
        assert result["source"] == "HEAD~1"
        mock_git.restore.assert_called_with("--staged", "--source", "HEAD~1", "--", "file.txt")

    @pytest.mark.asyncio
    async def test_github_restore_all_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore with '.' for all files."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file1.txt\nfile2.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["."], staged=True)

        assert result["status"] == "success"
        assert "all files" in result["message"]

    @pytest.mark.asyncio
    async def test_github_restore_no_matching_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore when no files match."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = ""
        mock_git.restore.side_effect = GitCommandError("restore", "did not match any file")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["nonexistent.txt"], staged=True)

        assert "error" in result
        assert "no matching files" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_restore_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore handles git error."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"
        mock_git.restore.side_effect = GitCommandError("restore", "error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"], staged=True)

        assert "error" in result
        assert "git restore failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_restore_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore handles generic exception."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = RuntimeError("Unexpected error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"], staged=True)

        assert "error" in result
        assert "Failed to restore" in result["error"]

    @pytest.mark.asyncio
    async def test_github_restore_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore handles repo name with slash."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo", files=["file.txt"])

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_restore_multiple_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore with multiple files."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file1.txt\nfile2.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file1.txt", "file2.txt"], staged=True)

        assert result["status"] == "success"
        mock_git.restore.assert_called_with("--staged", "--", "file1.txt", "file2.txt")

    @pytest.mark.asyncio
    async def test_github_restore_worktree_only_all_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_restore with worktree only and all files (.)."""
        tool = GitHubRestoreTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file1.txt\nfile2.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.profiles.linus.github_restore.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_restore.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["."], worktree=True, confirmed=True)

        assert result["status"] == "success"
        assert result["worktree"] is True
        # Should have the worktree hint
        assert "hint" in result
        assert "Working tree" in result["hint"]
        mock_git.restore.assert_called_with("--worktree", "--", ".")

