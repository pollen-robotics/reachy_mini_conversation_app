"""Unit tests for the github_diff tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_diff import GitHubDiffTool


class TestGitHubDiffToolAttributes:
    """Tests for GitHubDiffTool tool attributes."""

    def test_github_diff_has_correct_name(self) -> None:
        """Test GitHubDiffTool tool has correct name."""
        tool = GitHubDiffTool()
        assert tool.name == "github_diff"

    def test_github_diff_has_description(self) -> None:
        """Test GitHubDiffTool tool has description."""
        tool = GitHubDiffTool()
        assert "diff" in tool.description.lower()

    def test_github_diff_has_parameters_schema(self) -> None:
        """Test GitHubDiffTool tool has correct parameters schema."""
        tool = GitHubDiffTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "staged" in schema["properties"]
        assert "files" in schema["properties"]
        assert "commit" in schema["properties"]
        assert "compare" in schema["properties"]
        assert "stat_only" in schema["properties"]
        assert "name_only" in schema["properties"]
        assert "context_lines" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_diff_spec(self) -> None:
        """Test GitHubDiffTool tool spec generation."""
        tool = GitHubDiffTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_diff"


class TestGitHubDiffToolExecution:
    """Tests for GitHubDiffTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_diff_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_diff returns error when repo is missing."""
        tool = GitHubDiffTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_diff_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff returns error when repo not found."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_diff_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff returns error for non-git directory."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit")

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_diff_unstaged_default(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff shows unstaged changes by default."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new",  # diff output
            "1 file changed",  # stat output
            "file.txt",  # name output
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["diff_type"] == "unstaged"
        assert "file.txt" in result["changed_files"]

    @pytest.mark.asyncio
    async def test_github_diff_staged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff shows staged changes."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "staged diff output",
            "stat output",
            "staged_file.txt",
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", staged=True)

        assert result["status"] == "success"
        assert result["diff_type"] == "staged"
        # Verify --cached was used
        assert any("--cached" in str(call) for call in mock_git.diff.call_args_list)

    @pytest.mark.asyncio
    async def test_github_diff_with_commit(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff compares with specific commit."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "commit diff",
            "stat",
            "file.txt",
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", commit="HEAD~1")

        assert result["status"] == "success"
        assert result["diff_type"] == "vs HEAD~1"

    @pytest.mark.asyncio
    async def test_github_diff_compare_refs(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff compares two refs."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "compare diff",
            "stat",
            "file.txt",
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", compare="main..feature")

        assert result["status"] == "success"
        assert result["diff_type"] == "compare: main..feature"

    @pytest.mark.asyncio
    async def test_github_diff_stat_only(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff stat_only mode."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            " file.txt | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)",
            "file.txt",
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", stat_only=True)

        assert result["status"] == "success"
        assert "stat" not in result  # No separate stat when stat_only

    @pytest.mark.asyncio
    async def test_github_diff_name_only(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff name_only mode."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "file1.txt\nfile2.txt",  # name-only output
            "file1.txt\nfile2.txt",  # for counting
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", name_only=True)

        assert result["status"] == "success"
        assert result["files_changed"] == 2

    @pytest.mark.asyncio
    async def test_github_diff_specific_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff with specific files."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "diff for file1.txt",
            "stat",
            "file1.txt",
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", files=["file1.txt"])

        assert result["status"] == "success"
        # Verify -- and files were used
        assert any("--" in str(call) for call in mock_git.diff.call_args_list)

    @pytest.mark.asyncio
    async def test_github_diff_no_changes(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff with no changes."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "",  # no diff
            "",  # no stat
            "",  # no files
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["files_changed"] == 0
        assert "no differences" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_diff_truncates_large_diff(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff truncates large diff output."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        large_diff = "x" * 60000  # Larger than 50KB limit

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            large_diff,
            "stat",
            "file.txt",
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["diff_truncated"] is True
        assert len(result["diff"]) < 60000

    @pytest.mark.asyncio
    async def test_github_diff_truncates_many_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff truncates list when many files changed."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        many_files = "\n".join([f"file{i}.txt" for i in range(100)])

        mock_git = MagicMock()
        mock_git.diff.side_effect = [
            "diff",
            "stat",
            many_files,
        ]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert len(result["changed_files"]) <= 50
        assert result["files_truncated"] is True

    @pytest.mark.asyncio
    async def test_github_diff_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff handles repo name with slash."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = ["diff", "stat", "file.txt"]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_diff_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff handles generic exception."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = RuntimeError("Unexpected error")

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to get diff" in result["error"]

    @pytest.mark.asyncio
    async def test_github_diff_context_lines(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_diff with custom context lines."""
        tool = GitHubDiffTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.side_effect = ["diff", "stat", "file.txt"]

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_diff.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_diff.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", context_lines=5)

        assert result["status"] == "success"
        # Verify -U5 was used
        call_args = mock_git.diff.call_args_list[0]
        assert "-U5" in call_args[0]
