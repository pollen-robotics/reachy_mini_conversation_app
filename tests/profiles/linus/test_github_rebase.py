"""Unit tests for the github_rebase tool."""

from typing import cast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError
from git.repo import Repo

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_rebase import GitHubRebaseTool


class TestGitHubRebaseToolAttributes:
    """Tests for GitHubRebaseTool tool attributes."""

    def test_github_rebase_has_correct_name(self) -> None:
        """Test GitHubRebaseTool tool has correct name."""
        tool = GitHubRebaseTool()
        assert tool.name == "github_rebase"

    def test_github_rebase_has_description(self) -> None:
        """Test GitHubRebaseTool tool has description."""
        tool = GitHubRebaseTool()
        assert "rebase" in tool.description.lower()

    def test_github_rebase_has_parameters_schema(self) -> None:
        """Test GitHubRebaseTool tool has correct parameters schema."""
        tool = GitHubRebaseTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "action" in schema["properties"]
        assert "onto" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "action" in schema["required"]

    def test_github_rebase_spec(self) -> None:
        """Test GitHubRebaseTool tool spec generation."""
        tool = GitHubRebaseTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_rebase"


class TestGitHubRebaseToolHelpers:
    """Tests for GitHubRebaseTool helper methods."""

    def test_get_authenticated_url_with_token(self) -> None:
        """Test _get_authenticated_url with token."""
        tool = GitHubRebaseTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://github.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result is not None
        assert "ghp_test123@github.com" in result

    def test_get_authenticated_url_ssh(self) -> None:
        """Test _get_authenticated_url with SSH URL."""
        tool = GitHubRebaseTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "git@github.com:owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result == "https://ghp_test123@github.com/owner/repo.git"

    def test_get_authenticated_url_without_token(self) -> None:
        """Test _get_authenticated_url without token."""
        tool = GitHubRebaseTool()

        mock_repo = MagicMock()

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    def test_get_authenticated_url_non_github_url(self) -> None:
        """Test _get_authenticated_url with non-github URL returns None."""
        tool = GitHubRebaseTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://gitlab.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    def test_get_authenticated_url_access_error(self) -> None:
        """Test _get_authenticated_url handles exception during URL access."""
        tool = GitHubRebaseTool()

        class MockOrigin:
            @property
            def url(self) -> str:
                raise RuntimeError("Cannot access URL")

        class MockRemotes:
            origin = MockOrigin()

        class MockRepo:
            remotes = MockRemotes()

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(cast(Repo, MockRepo()))

        assert result is None


class TestGitHubRebaseToolExecution:
    """Tests for GitHubRebaseTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_rebase_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_rebase returns error when repo is missing."""
        tool = GitHubRebaseTool()

        result = await tool(mock_deps, repo="", action="start", onto="main", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rebase_start_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_rebase start requires confirmation."""
        tool = GitHubRebaseTool()

        result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rebase_start_missing_onto(self, mock_deps: ToolDependencies) -> None:
        """Test github_rebase start requires onto branch."""
        tool = GitHubRebaseTool()

        result = await tool(mock_deps, repo="myrepo", action="start", onto="", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rebase_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase returns error when repo not found."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", action="start", onto="main", confirmed=True)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_rebase_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase returns error for non-git directory."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", action="start", onto="main", confirmed=True)

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_start_already_in_progress(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase start when rebase already in progress."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-merge").mkdir()  # Rebase in progress

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True)

        assert "error" in result
        assert "already in progress" in result["error"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_rebase_start_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase start success."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock(), MagicMock()]  # 2 commits to rebase
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True)

        assert result["status"] == "success"
        assert result["commits_rebased"] == 2
        assert "warning" in result
        mock_repo.git.rebase.assert_called_with("main")

    @pytest.mark.asyncio
    async def test_github_rebase_start_no_change(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase start when already up to date."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = []  # No commits to rebase
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True)

        assert result["status"] == "no_change"
        assert "up to date" in result["message"]

    @pytest.mark.asyncio
    async def test_github_rebase_start_onto_remote_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase start with remote branch."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_remote_ref = MagicMock()
        mock_remote_ref.name = "origin/develop"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.refs = [mock_remote_ref]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = []  # develop not local
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="develop", confirmed=True)

        assert result["status"] == "success"
        mock_repo.git.rebase.assert_called_with("origin/develop")

    @pytest.mark.asyncio
    async def test_github_rebase_start_onto_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase start with nonexistent target branch."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.refs = []

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = []
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="nonexistent", confirmed=True)

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_start_conflict(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase start with conflict."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin
        mock_repo.git.rebase.side_effect = GitCommandError("rebase", "CONFLICT in file.py")
        mock_repo.index.unmerged_blobs.return_value = {("file.py", 1): MagicMock()}

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True)

        assert result["status"] == "conflict"
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_rebase_continue_not_in_progress(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase continue when no rebase in progress."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="continue")

        assert "error" in result
        assert "No rebase in progress" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_continue_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase continue success."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-merge").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="continue")

        assert result["status"] == "success"
        assert "continued" in result["message"]
        mock_repo.git.rebase.assert_called_with("--continue")

    @pytest.mark.asyncio
    async def test_github_rebase_continue_more_conflicts(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase continue with more conflicts."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-merge").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.git.rebase.side_effect = GitCommandError("rebase", "CONFLICT")
        mock_repo.index.unmerged_blobs.return_value = {}

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="continue")

        assert result["status"] == "conflict"
        assert "More conflicts" in result["message"]

    @pytest.mark.asyncio
    async def test_github_rebase_abort_not_in_progress(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase abort when no rebase in progress."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="abort")

        assert "error" in result
        assert "No rebase in progress" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_abort_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase abort success."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-apply").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="abort")

        assert result["status"] == "success"
        assert "aborted" in result["message"]
        mock_repo.git.rebase.assert_called_with("--abort")

    @pytest.mark.asyncio
    async def test_github_rebase_skip_not_in_progress(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase skip when no rebase in progress."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="skip")

        assert "error" in result
        assert "No rebase in progress" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_skip_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase skip success."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-merge").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="skip")

        assert result["status"] == "success"
        assert "Skipped" in result["message"]
        assert "warning" in result
        mock_repo.git.rebase.assert_called_with("--skip")

    @pytest.mark.asyncio
    async def test_github_rebase_unknown_action(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase with unknown action."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="unknown")

        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_git_command_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase handles git command errors."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin
        mock_repo.git.rebase.side_effect = GitCommandError("rebase", "fatal error")

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True)

        assert "error" in result
        assert "Git rebase failed" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase handles generic exceptions."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.side_effect = RuntimeError("Unexpected error")
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True)

        assert "error" in result
        assert "Failed to" in result["error"]  # Can be "Failed to analyze" or "Failed to rebase"

    @pytest.mark.asyncio
    async def test_github_rebase_with_update_remote(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase with update_remote fetches first."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True, update_remote=True)

        assert result["status"] == "success"
        mock_origin.fetch.assert_called()

    @pytest.mark.asyncio
    async def test_github_rebase_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase handles repo name with slash."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="owner/myrepo", action="start", onto="main", confirmed=True)

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_rebase_url_with_existing_auth(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase handles URL with existing authentication token."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        # URL with existing @github.com (e.g., already has auth)
        mock_origin.url = "https://oldtoken@github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_newtoken"

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True, update_remote=True)

        assert result["status"] == "success"
        mock_origin.fetch.assert_called()

    @pytest.mark.asyncio
    async def test_github_rebase_get_authenticated_url_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase handles exception in _get_authenticated_url."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        # Simulate exception when accessing url
        type(mock_origin).url = property(lambda self: (_ for _ in ()).throw(AttributeError("no url")))

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True, update_remote=True)

        # Should still succeed, just without authenticated URL
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_rebase_no_update_remote(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase with update_remote=False skips fetch."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = None

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True, update_remote=False)

        assert result["status"] == "success"
        mock_origin.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_rebase_fetch_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase handles fetch exception gracefully."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.fetch.side_effect = Exception("Network error")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.branches = [mock_main]
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", action="start", onto="main", confirmed=True, update_remote=True)

        # Should still succeed even if fetch fails
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_rebase_continue_reraises_non_conflict_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_rebase continue re-raises non-conflict GitCommandError."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-merge").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        # Error without "conflict" in message
        mock_repo.git.rebase.side_effect = GitCommandError("rebase", "fatal: cannot continue")

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="continue")

        assert "error" in result
        assert "Git rebase failed" in result["error"]

    @pytest.mark.asyncio
    async def test_github_rebase_outer_exception_handler(
        self, mock_deps: ToolDependencies, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test github_rebase outer exception handler with non-GitCommandError."""
        tool = GitHubRebaseTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        git_dir = repo_path / ".git"
        git_dir.mkdir()
        (git_dir / "rebase-merge").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        # Non-GitCommandError exception
        mock_repo.git.rebase.side_effect = RuntimeError("Unexpected runtime error")

        with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_rebase.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="continue")

        assert "error" in result
        assert "Failed to rebase" in result["error"]
        assert "Unexpected runtime error" in result["error"]
