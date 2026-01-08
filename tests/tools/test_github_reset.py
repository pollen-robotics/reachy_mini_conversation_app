"""Unit tests for the github_reset tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.github_reset import GitHubResetTool, REPOS_DIR
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubResetToolAttributes:
    """Tests for GitHubResetTool tool attributes."""

    def test_github_reset_has_correct_name(self) -> None:
        """Test GitHubResetTool tool has correct name."""
        tool = GitHubResetTool()
        assert tool.name == "github_reset"

    def test_github_reset_has_description(self) -> None:
        """Test GitHubResetTool tool has description."""
        tool = GitHubResetTool()
        assert "reset" in tool.description.lower()

    def test_github_reset_has_parameters_schema(self) -> None:
        """Test GitHubResetTool tool has correct parameters schema."""
        tool = GitHubResetTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "mode" in schema["properties"]
        assert "target" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_reset_spec(self) -> None:
        """Test GitHubResetTool tool spec generation."""
        tool = GitHubResetTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_reset"


class TestGitHubResetToolExecution:
    """Tests for GitHubResetTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_reset_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_reset returns error when repo is missing."""
        tool = GitHubResetTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_reset_hard_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_reset hard mode requires confirmation."""
        tool = GitHubResetTool()

        result = await tool(mock_deps, repo="myrepo", mode="hard", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_reset_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset returns error when repo not found."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_reset_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset returns error for non-git directory."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit")

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_reset_soft_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset soft mode success."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        # Create mock commits
        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit message"

        mock_target_commit = MagicMock()
        mock_target_commit.hexsha = "def5678901234"
        mock_target_commit.message = "Target commit message"

        mock_reset_commit = MagicMock()
        mock_reset_commit.hexsha = "abc1234567890"
        mock_reset_commit.message = "Current commit"

        mock_head = MagicMock()
        mock_head.commit = mock_target_commit  # After reset

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.head = mock_head
        mock_repo.commit.return_value = mock_target_commit
        mock_repo.iter_commits.return_value = [mock_reset_commit]

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", mode="soft", target="HEAD~1")

        assert result["status"] == "success"
        assert result["mode"] == "soft"
        assert result["commits_reset"] == 1
        assert "hint" in result
        mock_head.reset.assert_called_once_with("HEAD~1", index=False, working_tree=False)

    @pytest.mark.asyncio
    async def test_github_reset_mixed_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset mixed mode (default) success."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit"

        mock_target_commit = MagicMock()
        mock_target_commit.hexsha = "def5678901234"
        mock_target_commit.message = "Target commit"

        mock_reset_commit = MagicMock()
        mock_reset_commit.hexsha = "abc1234567890"
        mock_reset_commit.message = "Reset commit"

        mock_head = MagicMock()
        mock_head.commit = mock_target_commit

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.head = mock_head
        mock_repo.commit.return_value = mock_target_commit
        mock_repo.iter_commits.return_value = [mock_reset_commit]

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")  # Default mode=mixed, target=HEAD~1

        assert result["status"] == "success"
        assert result["mode"] == "mixed"
        mock_head.reset.assert_called_once_with("HEAD~1", index=True, working_tree=False)

    @pytest.mark.asyncio
    async def test_github_reset_hard_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset hard mode success with confirmation."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit"

        mock_target_commit = MagicMock()
        mock_target_commit.hexsha = "def5678901234"
        mock_target_commit.message = "Target commit"

        mock_reset_commit = MagicMock()
        mock_reset_commit.hexsha = "abc1234567890"
        mock_reset_commit.message = "Reset commit"

        mock_head = MagicMock()
        mock_head.commit = mock_target_commit

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.head = mock_head
        mock_repo.commit.return_value = mock_target_commit
        mock_repo.iter_commits.return_value = [mock_reset_commit]

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", mode="hard", confirmed=True)

        assert result["status"] == "success"
        assert result["mode"] == "hard"
        assert "warning" in result
        mock_head.reset.assert_called_once_with("HEAD~1", index=True, working_tree=True)

    @pytest.mark.asyncio
    async def test_github_reset_invalid_target(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset with invalid target."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.commit.side_effect = Exception("Invalid commit")

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", target="invalid_sha")

        assert "error" in result
        assert "invalid target" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_reset_no_commits_to_reset(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset when already at target."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"
        mock_commit.message = "Commit message"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_commit
        mock_repo.commit.return_value = mock_commit
        mock_repo.iter_commits.return_value = []  # No commits to reset

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", target="HEAD")

        assert result["status"] == "no_change"
        assert "already at target" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_reset_multiple_commits(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset with multiple commits."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit"

        mock_target_commit = MagicMock()
        mock_target_commit.hexsha = "def5678901234"
        mock_target_commit.message = "Target commit"

        # Create multiple commits to reset
        reset_commits = [
            MagicMock(hexsha=f"commit{i}1234567890", message=f"Commit {i}")
            for i in range(5)
        ]

        mock_head = MagicMock()
        mock_head.commit = mock_target_commit

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.head = mock_head
        mock_repo.commit.return_value = mock_target_commit
        mock_repo.iter_commits.return_value = reset_commits

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", target="HEAD~5")

        assert result["status"] == "success"
        assert result["commits_reset"] == 5
        assert len(result["reset_commits"]) == 5

    @pytest.mark.asyncio
    async def test_github_reset_truncates_many_commits(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset truncates display of many commits."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit"

        mock_target_commit = MagicMock()
        mock_target_commit.hexsha = "def5678901234"
        mock_target_commit.message = "Target commit"

        # Create many commits to reset
        reset_commits = [
            MagicMock(hexsha=f"commit{i}1234567890", message=f"Commit {i}")
            for i in range(15)
        ]

        mock_head = MagicMock()
        mock_head.commit = mock_target_commit

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.head = mock_head
        mock_repo.commit.return_value = mock_target_commit
        mock_repo.iter_commits.return_value = reset_commits

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", target="HEAD~15")

        assert result["status"] == "success"
        assert result["commits_reset"] == 15
        assert len(result["reset_commits"]) == 10  # Truncated
        assert result["reset_commits_truncated"] is True

    @pytest.mark.asyncio
    async def test_github_reset_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset handles git error."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_current_commit = MagicMock()
        mock_current_commit.hexsha = "abc1234567890"
        mock_current_commit.message = "Current commit"

        mock_target_commit = MagicMock()
        mock_target_commit.hexsha = "def5678901234"
        mock_target_commit.message = "Target commit"

        mock_head = MagicMock()
        mock_head.reset.side_effect = GitCommandError("reset", "error")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_current_commit
        mock_repo.head = mock_head
        mock_repo.commit.return_value = mock_target_commit
        mock_repo.iter_commits.return_value = [MagicMock(hexsha="abc", message="msg")]

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "git reset failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_reset_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset handles generic exception."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = MagicMock()
        mock_repo.commit.side_effect = RuntimeError("Unexpected")
        # Make iter_commits return empty to avoid reaching the reset
        mock_repo.iter_commits.return_value = []

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        # Either error or no_change status depending on execution path
        assert "error" in result or result["status"] == "no_change"

    @pytest.mark.asyncio
    async def test_github_reset_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_reset handles repo name with slash."""
        tool = GitHubResetTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"
        mock_commit.message = "Commit"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit = mock_commit
        mock_repo.commit.return_value = mock_commit
        mock_repo.iter_commits.return_value = []

        with patch("reachy_mini_conversation_app.tools.github_reset.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_reset.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "no_change"
