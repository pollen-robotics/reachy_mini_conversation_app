"""Unit tests for the github_add tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_add import GitHubAddTool


class TestGitHubAddToolAttributes:
    """Tests for GitHubAddTool tool attributes."""

    def test_github_add_has_correct_name(self) -> None:
        """Test GitHubAddTool tool has correct name."""
        tool = GitHubAddTool()
        assert tool.name == "github_add"

    def test_github_add_has_description(self) -> None:
        """Test GitHubAddTool tool has description."""
        tool = GitHubAddTool()
        assert "stage" in tool.description.lower()

    def test_github_add_has_parameters_schema(self) -> None:
        """Test GitHubAddTool tool has correct parameters schema."""
        tool = GitHubAddTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "files" in schema["properties"]
        assert "all" in schema["properties"]
        assert "update" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_add_spec(self) -> None:
        """Test GitHubAddTool tool spec generation."""
        tool = GitHubAddTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_add"


class TestGitHubAddToolExecution:
    """Tests for GitHubAddTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_add_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_add returns error when repo is missing."""
        tool = GitHubAddTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_add_no_files_or_options(self, mock_deps: ToolDependencies) -> None:
        """Test github_add returns error when no files or options specified."""
        tool = GitHubAddTool()

        result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "specify" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_add_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add returns error when repo not found."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", files=["file.txt"])

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_add_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add returns error for non-git directory."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", files=["file.txt"])

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_add_files_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add successfully stages files."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "file.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"])

        assert result["status"] == "success"
        assert "staged" in result["message"].lower()
        mock_index.add.assert_called_once_with(["file.txt"])

    @pytest.mark.asyncio
    async def test_github_add_all_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add with all=True stages all files."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "file1.txt\nfile2.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", **{"all": True})

        assert result["status"] == "success"
        mock_git.add.assert_called_once_with(A=True)

    @pytest.mark.asyncio
    async def test_github_add_update_only(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add with update=True stages only modified files."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "modified.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = ["new.txt"]

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", update=True)

        assert result["status"] == "success"
        mock_git.add.assert_called_once_with(u=True)

    @pytest.mark.asyncio
    async def test_github_add_dot_stages_all(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add with '.' stages all changes."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "file1.txt\nfile2.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["."])

        assert result["status"] == "success"
        mock_git.add.assert_called_once_with(A=True)

    @pytest.mark.asyncio
    async def test_github_add_deleted_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add stages deleted file."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()
        # Don't create deleted.txt - it's been deleted

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "deleted.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["deleted.txt"])

        assert result["status"] == "success"
        mock_index.remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_add_file_not_found_not_tracked(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add returns error for non-existent untracked file."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_index = MagicMock()
        mock_index.remove.side_effect = GitCommandError("remove", "not tracked")

        mock_git = MagicMock()
        mock_git.diff.return_value = ""

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["nonexistent.txt"])

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_add_no_new_files_staged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add when files already staged."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "file.txt",  # staged before (already staged)
            "file.txt",  # staged after (same)
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"])

        assert result["status"] == "success"
        assert "no new files" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_add_with_remaining_unstaged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add shows remaining unstaged files."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "file.txt",  # staged after
            "other.txt",  # unstaged remaining
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"])

        assert result["status"] == "success"
        assert "unstaged_remaining" in result
        assert "other.txt" in result["unstaged_remaining"]

    @pytest.mark.asyncio
    async def test_github_add_with_untracked_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add shows untracked files."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "file.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index
        mock_repo.untracked_files = ["new1.txt", "new2.txt"]

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file.txt"])

        assert result["status"] == "success"
        assert "untracked" in result
        assert result["untracked_count"] == 2

    @pytest.mark.asyncio
    async def test_github_add_truncates_many_untracked(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add truncates display of many untracked files."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # staged before
            "file.txt",  # staged after
            "",  # unstaged
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = [f"file{i}.txt" for i in range(30)]

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", **{"all": True})

        assert result["status"] == "success"
        assert len(result["untracked"]) == 20  # Truncated to 20
        assert result["untracked_truncated"] is True
        assert result["untracked_count"] == 30

    @pytest.mark.asyncio
    async def test_github_add_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add handles repo name with slash."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = ["", "", ""]

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo", **{"all": True})

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_add_git_command_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add handles git command error."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = GitCommandError("diff", "error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", **{"all": True})

        assert "error" in result
        assert "git command failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_add_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_add handles generic exception."""
        tool = GitHubAddTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = RuntimeError("Unexpected error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_add.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_add.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", **{"all": True})

        assert "error" in result
        assert "Failed to stage" in result["error"]
