"""Unit tests for the github_rm tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.github_rm import GitHubRmTool, REPOS_DIR
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubRmToolAttributes:
    """Tests for GitHubRmTool tool attributes."""

    def test_github_rm_has_correct_name(self) -> None:
        """Test GitHubRmTool tool has correct name."""
        tool = GitHubRmTool()
        assert tool.name == "github_rm"

    def test_github_rm_has_description(self) -> None:
        """Test GitHubRmTool tool has description."""
        tool = GitHubRmTool()
        assert "remove" in tool.description.lower()

    def test_github_rm_has_parameters_schema(self) -> None:
        """Test GitHubRmTool tool has correct parameters schema."""
        tool = GitHubRmTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "paths" in schema["properties"]
        assert "git_only" in schema["properties"]
        assert "recursive" in schema["properties"]
        assert "force" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "paths" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_rm_spec(self) -> None:
        """Test GitHubRmTool tool spec generation."""
        tool = GitHubRmTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_rm"


class TestGitHubRmToolExecution:
    """Tests for GitHubRmTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_rm_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_rm returns error when not confirmed."""
        tool = GitHubRmTool()

        result = await tool(mock_deps, repo="myrepo", paths=["file.txt"], confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_rm_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_rm returns error when repo is missing."""
        tool = GitHubRmTool()

        result = await tool(mock_deps, repo="", paths=["file.txt"], confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_missing_paths(self, mock_deps: ToolDependencies) -> None:
        """Test github_rm returns error when paths is missing."""
        tool = GitHubRmTool()

        result = await tool(mock_deps, repo="myrepo", paths=[], confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm returns error when repo not found."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", paths=["file.txt"], confirmed=True)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm returns error for non-git directory."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", paths=["file.txt"], confirmed=True)

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rm_file_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm successfully removes file."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["file.txt"], confirmed=True)

        assert result["status"] == "success"
        assert "file.txt" in result["removed_files"]
        mock_index.remove.assert_called()

    @pytest.mark.asyncio
    async def test_github_rm_file_git_only(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm with git_only keeps file on disk."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["file.txt"], git_only=True, confirmed=True)

        assert result["status"] == "success"
        assert result["git_only"] is True
        mock_index.remove.assert_called_with(["file.txt"], working_tree=False)

    @pytest.mark.asyncio
    async def test_github_rm_file_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm handles file not found."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_git = MagicMock()
        mock_git.ls_files.side_effect = GitCommandError("ls-files", "not found")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["nonexistent.txt"], confirmed=True)

        assert result["status"] == "failed"
        assert "errors" in result
        assert "not found" in result["errors"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_tracked_deleted_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm removes tracked file that was deleted from disk."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        # Don't create the file - it's been deleted

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.ls_files.return_value = ""  # File is tracked

        mock_repo = MagicMock()
        mock_repo.index = mock_index
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["deleted.txt"], confirmed=True)

        assert result["status"] == "success"
        mock_index.remove.assert_called_with(["deleted.txt"], working_tree=False)

    @pytest.mark.asyncio
    async def test_github_rm_directory_needs_recursive(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm requires recursive flag for directories."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "mydir").mkdir()

        mock_repo = MagicMock()

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["mydir"], confirmed=True)

        assert "errors" in result
        assert "recursive" in result["errors"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_directory_recursive(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm removes directory with recursive flag."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "mydir").mkdir()

        mock_index = MagicMock()
        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["mydir"], recursive=True, confirmed=True)

        assert result["status"] == "success"
        assert "mydir" in result["removed_dirs"]

    @pytest.mark.asyncio
    async def test_github_rm_path_outside_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm blocks paths outside repository."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_repo = MagicMock()

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["../../../etc/passwd"], confirmed=True)

        assert "errors" in result
        assert "outside" in result["errors"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_multiple_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm removes multiple files."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file1.txt").touch()
        (repo_path / "file2.txt").touch()

        mock_index = MagicMock()
        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["file1.txt", "file2.txt"], confirmed=True)

        assert result["status"] == "success"
        assert result["files_count"] == 2

    @pytest.mark.asyncio
    async def test_github_rm_file_fallback_to_unlink(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm falls back to filesystem unlink when git remove fails."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "modified.txt"
        test_file.touch()

        mock_index = MagicMock()
        # Git remove fails, so code falls back to filesystem unlink
        mock_index.remove.side_effect = GitCommandError("rm", "not tracked")

        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["modified.txt"], confirmed=True)

        # When git remove fails, it falls back to filesystem unlink
        assert result["status"] == "success"
        assert "modified.txt" in result["removed_files"]
        # File should be deleted from filesystem
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_github_rm_untracked_file_git_only_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm git_only fails for untracked files."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "untracked.txt").touch()

        mock_index = MagicMock()
        mock_index.remove.side_effect = GitCommandError("rm", "not tracked")

        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["untracked.txt"], git_only=True, confirmed=True)

        assert "errors" in result
        assert "not tracked" in result["errors"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rm_partial_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm with some files succeeding and some failing."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "good.txt").touch()
        # Don't create bad.txt

        mock_index = MagicMock()
        mock_git = MagicMock()
        mock_git.ls_files.side_effect = GitCommandError("ls-files", "not found")

        mock_repo = MagicMock()
        mock_repo.index = mock_index
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", paths=["good.txt", "bad.txt"], confirmed=True)

        assert result["status"] == "success"
        assert "good.txt" in result["removed_files"]
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_github_rm_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rm handles repo name with slash."""
        tool = GitHubRmTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").touch()

        mock_index = MagicMock()
        mock_repo = MagicMock()
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_rm.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_rm.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo", paths=["file.txt"], confirmed=True)

        assert result["status"] == "success"
