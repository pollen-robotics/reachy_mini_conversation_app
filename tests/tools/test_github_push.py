"""Unit tests for the github_push tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_push import GitHubPushTool


class TestGitHubPushToolAttributes:
    """Tests for GitHubPushTool tool attributes."""

    def test_github_push_has_correct_name(self) -> None:
        """Test GitHubPushTool tool has correct name."""
        tool = GitHubPushTool()
        assert tool.name == "github_push"

    def test_github_push_has_description(self) -> None:
        """Test GitHubPushTool tool has description."""
        tool = GitHubPushTool()
        assert "push" in tool.description.lower()

    def test_github_push_has_parameters_schema(self) -> None:
        """Test GitHubPushTool tool has correct parameters schema."""
        tool = GitHubPushTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_push_spec(self) -> None:
        """Test GitHubPushTool tool spec generation."""
        tool = GitHubPushTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_push"


class TestGitHubPushToolHelpers:
    """Tests for GitHubPushTool helper methods."""

    def test_get_authenticated_url_with_token(self) -> None:
        """Test _get_authenticated_url with token."""
        tool = GitHubPushTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://github.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert "ghp_test123@github.com" in result
        assert "owner/repo" in result

    def test_get_authenticated_url_ssh(self) -> None:
        """Test _get_authenticated_url with SSH URL."""
        tool = GitHubPushTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "git@github.com:owner/repo.git"

        with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result == "https://ghp_test123@github.com/owner/repo.git"

    def test_get_authenticated_url_without_token(self) -> None:
        """Test _get_authenticated_url without token."""
        tool = GitHubPushTool()

        mock_repo = MagicMock()

        with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    def test_get_authenticated_url_replaces_existing_token(self) -> None:
        """Test _get_authenticated_url replaces existing token."""
        tool = GitHubPushTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://oldtoken@github.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_newtoken"
            result = tool._get_authenticated_url(mock_repo)

        assert "ghp_newtoken@github.com" in result
        assert "oldtoken" not in result


class TestGitHubPushToolExecution:
    """Tests for GitHubPushTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_push_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_push returns error when not confirmed."""
        tool = GitHubPushTool()

        result = await tool(mock_deps, repo="myrepo", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_push_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_push returns error when repo is missing."""
        tool = GitHubPushTool()

        result = await tool(mock_deps, repo="", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_push_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push returns error when repo not found."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"
                result = await tool(mock_deps, repo="nonexistent", confirmed=True)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_push_no_token(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push returns error when no token."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = None
                result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push returns error for non-git directory."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo") as mock_repo:
                    mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")
                    result = await tool(mock_deps, repo="notgit", confirmed=True)

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_nothing_to_push(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push when nothing to push."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = []  # No commits ahead
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "nothing_to_push"
        assert "No local commits" in result["message"]

    @pytest.mark.asyncio
    async def test_github_push_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push success."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 0

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]  # 1 commit ahead
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        assert "pushed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_push_success_new_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push success with new branch (no tracking)."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 0

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature"
        mock_repo.active_branch.tracking_branch.return_value = None  # No tracking
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        assert result["upstream_set"] is True
        mock_origin.push.assert_called_with(refspec="feature:feature", set_upstream=True)

    @pytest.mark.asyncio
    async def test_github_push_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push with push error."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 16  # ERROR flag
        mock_push_info.summary = "Push failed"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "Push failed" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_rejected(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push when rejected."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 32  # REJECTED flag

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "rejected" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_push_git_command_error_rejected(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push with git command error (rejected)."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.side_effect = GitCommandError("push", "rejected - non-fast-forward")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "rejected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_push_permission_denied(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push with permission denied."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.side_effect = GitCommandError("push", "permission denied")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push handles generic exceptions."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.side_effect = RuntimeError("Unexpected error")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "Failed to push" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_hides_token_in_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push hides token in error messages."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.side_effect = GitCommandError("push", "https://ghp_secret@github.com failed")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_secret"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "ghp_secret" not in result["error"]
        assert "***" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_restores_original_url(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push restores original URL after push."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        original_url = "https://github.com/owner/repo.git"

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 0

        mock_origin = MagicMock()
        mock_origin.url = original_url
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    await tool(mock_deps, repo="myrepo", confirmed=True)

        # Verify set_url was called twice: once for auth, once to restore
        calls = mock_origin.set_url.call_args_list
        assert len(calls) == 2
        assert calls[1][0][0] == original_url

    @pytest.mark.asyncio
    async def test_github_push_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push handles repo name with slash."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 0

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="owner/myrepo", confirmed=True)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_push_fetch_git_error_sets_needs_upstream(
        self, mock_deps: ToolDependencies, tmp_path: Path
    ) -> None:
        """Test github_push sets needs_upstream when fetch fails."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 0

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        # Fetch fails with GitCommandError
        mock_origin.fetch.side_effect = GitCommandError("fetch", "remote not found")
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        # Should have used set_upstream because fetch failed
        mock_origin.push.assert_called_with(refspec="main:main", set_upstream=True)

    @pytest.mark.asyncio
    async def test_github_push_empty_push_info(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push handles empty push_info."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = []  # Empty list

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_push_error_with_token_in_summary(
        self, mock_deps: ToolDependencies, tmp_path: Path
    ) -> None:
        """Test github_push hides token from error summary."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 16  # ERROR flag
        mock_push_info.summary = "Failed with ghp_secret token"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_secret"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "ghp_secret" not in result["error"]
        assert "***" in result["error"]

    @pytest.mark.asyncio
    async def test_github_push_generic_exception_with_token(
        self, mock_deps: ToolDependencies, tmp_path: Path
    ) -> None:
        """Test github_push hides token in generic exception."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.side_effect = RuntimeError("Error with ghp_mytoken in message")

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_mytoken"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "ghp_mytoken" not in result["error"]
        assert "***" in result["error"]


    @pytest.mark.asyncio
    async def test_github_push_without_auth_url(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_push works without authenticated URL (non-github remote)."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 0

        mock_origin = MagicMock()
        # Use non-github URL so auth_url will be None
        mock_origin.url = "https://gitlab.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert result["status"] == "success"
        # set_url should NOT have been called since auth_url is None
        mock_origin.set_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_push_error_no_summary_attribute(
        self, mock_deps: ToolDependencies, tmp_path: Path
    ) -> None:
        """Test github_push handles error without summary attribute."""
        tool = GitHubPushTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_tracking = MagicMock()
        mock_tracking.name = "origin/main"

        mock_push_info = MagicMock()
        mock_push_info.ERROR = 16
        mock_push_info.REJECTED = 32
        mock_push_info.flags = 16  # ERROR flag
        # Remove summary attribute
        del mock_push_info.summary

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.return_value = [mock_push_info]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.active_branch.tracking_branch.return_value = mock_tracking
        mock_repo.iter_commits.return_value = [MagicMock()]
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.tools.github_push.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
                mock_config.GITHUB_TOKEN = "ghp_token"

                with patch("reachy_mini_conversation_app.tools.github_push.Repo", return_value=mock_repo):
                    result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "Push failed" in result["error"]


class TestGitHubPushToolHelperEdgeCases:
    """Tests for edge cases in helper methods."""

    def test_get_authenticated_url_non_github_url(self) -> None:
        """Test _get_authenticated_url with non-github URL returns None."""
        tool = GitHubPushTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://gitlab.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    def test_get_authenticated_url_exception(self) -> None:
        """Test _get_authenticated_url handles exception."""
        tool = GitHubPushTool()

        class MockOrigin:
            @property
            def url(self) -> str:
                raise RuntimeError("Cannot access URL")

        class MockRemotes:
            origin = MockOrigin()

        class MockRepo:
            remotes = MockRemotes()

        with patch("reachy_mini_conversation_app.tools.github_push.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(MockRepo())

        assert result is None
