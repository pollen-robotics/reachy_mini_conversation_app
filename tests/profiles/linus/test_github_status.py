"""Unit tests for the github_status tool."""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from git import InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_status import GitHubStatusTool


class TestGitHubStatusToolAttributes:
    """Tests for GitHubStatusTool tool attributes."""

    def test_github_status_has_correct_name(self) -> None:
        """Test GitHubStatusTool tool has correct name."""
        tool = GitHubStatusTool()
        assert tool.name == "github_status"

    def test_github_status_has_description(self) -> None:
        """Test GitHubStatusTool tool has description."""
        tool = GitHubStatusTool()
        assert "status" in tool.description.lower()
        assert "repository" in tool.description.lower()

    def test_github_status_has_parameters_schema(self) -> None:
        """Test GitHubStatusTool tool has correct parameters schema."""
        tool = GitHubStatusTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_status_spec(self) -> None:
        """Test GitHubStatusTool tool spec generation."""
        tool = GitHubStatusTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_status"


class TestGitHubStatusToolExecution:
    """Tests for GitHubStatusTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_status_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_status returns error when repo is missing."""
        tool = GitHubStatusTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_status_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status returns error when repo not found."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_status_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status returns error for non-git directory."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit")

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_status_clean_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status with clean repository."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.return_value = ""
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["branch"] == "main"
        assert result["clean"] is True
        assert "Working tree clean" in result["message"]

    @pytest.mark.asyncio
    async def test_github_status_with_staged_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status with staged files."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.side_effect = lambda *args, **kwargs: (
            "A\tnew_file.py\nM\tmodified.py" if "--cached" in args else ""
        )
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["clean"] is False
        assert result["staged_count"] == 2
        assert any(f["status"] == "added" for f in result["staged"])
        assert any(f["status"] == "modified" for f in result["staged"])

    @pytest.mark.asyncio
    async def test_github_status_with_modified_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status with modified (not staged) files."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.side_effect = lambda *args, **kwargs: (
            "" if "--cached" in args else "M\tfile.py\nD\tdeleted.py"
        )
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["modified_count"] == 2
        assert any(f["status"] == "modified" for f in result["modified"])
        assert any(f["status"] == "deleted" for f in result["modified"])

    @pytest.mark.asyncio
    async def test_github_status_with_untracked_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status with untracked files."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.return_value = ""
        mock_repo.untracked_files = ["new1.py", "new2.py", "new3.py"]

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["untracked_count"] == 3
        assert "new1.py" in result["untracked"]

    @pytest.mark.asyncio
    async def test_github_status_truncates_untracked(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status truncates long untracked list."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.return_value = ""
        mock_repo.untracked_files = [f"file{i}.py" for i in range(50)]

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["untracked_count"] == 50
        assert len(result["untracked"]) == 30  # Truncated to 30
        assert result["untracked_truncated"] is True

    @pytest.mark.asyncio
    async def test_github_status_with_tracking_info(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status with tracking branch info."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.side_effect = [
            [MagicMock(), MagicMock()],  # 2 ahead
            [MagicMock()],  # 1 behind
        ]
        mock_repo.git.diff.return_value = ""
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["tracking"]["tracking"] == "origin/main"
        assert result["tracking"]["ahead"] == 2
        assert result["tracking"]["behind"] == 1

    @pytest.mark.asyncio
    async def test_github_status_detached_head(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status with detached HEAD."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        type(mock_repo).active_branch = PropertyMock(side_effect=TypeError("detached HEAD"))
        mock_repo.head.commit.hexsha = "abc12345678"
        mock_repo.git.diff.return_value = ""
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "HEAD detached" in result["branch"]
        assert "abc12345" in result["branch"]

    @pytest.mark.asyncio
    async def test_github_status_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status handles repo name with slash."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.return_value = ""
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_status_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status handles exceptions."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.side_effect = RuntimeError("Git error")

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to get status" in result["error"]

    @pytest.mark.asyncio
    async def test_github_status_summary(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status generates summary."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        mock_repo.git.diff.side_effect = lambda *args, **kwargs: (
            "A\tfile.py" if "--cached" in args else "M\tother.py"
        )
        mock_repo.untracked_files = ["new.py"]

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "summary" in result
        assert "1 staged" in result["summary"]
        assert "1 modified" in result["summary"]
        assert "1 untracked" in result["summary"]

    @pytest.mark.asyncio
    async def test_github_status_with_empty_lines_in_output(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status handles empty lines in git diff output."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        # Git output with empty lines mixed in
        mock_repo.git.diff.side_effect = lambda *args, **kwargs: (
            "A\tfile1.py\n\nM\tfile2.py\n" if "--cached" in args else "\nM\tother.py\n\n"
        )
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        # Should have parsed files correctly, ignoring empty lines
        assert result["staged_count"] == 2
        assert result["modified_count"] == 1

    @pytest.mark.asyncio
    async def test_github_status_with_malformed_lines(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status handles malformed lines (no tab separator)."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        # Git output with malformed lines (no tab separator)
        mock_repo.git.diff.side_effect = lambda *args, **kwargs: (
            "A\tvalid.py\nmalformed_no_tab\nM\tvalid2.py" if "--cached" in args else "M\tvalid.py\nno_tab_here"
        )
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        # Should have parsed valid files, skipping malformed lines
        assert result["staged_count"] == 2  # valid.py and valid2.py
        assert result["modified_count"] == 1  # valid.py only

    @pytest.mark.asyncio
    async def test_github_status_modified_with_empty_line_in_middle(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_status handles empty line in the middle of modified files output."""
        tool = GitHubStatusTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = None
        # Modified output has empty line in the middle (after strip, still has empty line between)
        mock_repo.git.diff.side_effect = lambda *args, **kwargs: (
            "" if "--cached" in args else "M\tfile1.py\n\nM\tfile2.py"
        )
        mock_repo.untracked_files = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_status.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_status.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        # Should parse both files, skipping the empty line
        assert result["modified_count"] == 2
