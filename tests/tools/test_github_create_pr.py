"""Unit tests for the github_create_pr tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_create_pr import GitHubCreatePRTool, REPOS_DIR
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubCreatePRToolAttributes:
    """Tests for GitHubCreatePRTool tool attributes."""

    def test_github_create_pr_has_correct_name(self) -> None:
        """Test GitHubCreatePRTool tool has correct name."""
        tool = GitHubCreatePRTool()
        assert tool.name == "github_create_pr"

    def test_github_create_pr_has_description(self) -> None:
        """Test GitHubCreatePRTool tool has description."""
        tool = GitHubCreatePRTool()
        assert "pull request" in tool.description.lower()

    def test_github_create_pr_has_parameters_schema(self) -> None:
        """Test GitHubCreatePRTool tool has correct parameters schema."""
        tool = GitHubCreatePRTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "title" in schema["properties"]
        assert "body" in schema["properties"]
        assert "base" in schema["properties"]
        assert "head" in schema["properties"]
        assert "draft" in schema["properties"]
        assert "issue_number" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "title" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_create_pr_spec(self) -> None:
        """Test GitHubCreatePRTool tool spec generation."""
        tool = GitHubCreatePRTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_create_pr"


class TestGitHubCreatePRGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubCreatePRTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubCreatePRTool()
        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubCreatePRTool()
        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubCreatePRGetCurrentBranch:
    """Tests for _get_current_branch helper method."""

    def test_get_current_branch_repo_not_found(self, tmp_path: Path) -> None:
        """Test _get_current_branch when repo not found."""
        tool = GitHubCreatePRTool()
        with patch("reachy_mini_conversation_app.tools.github_create_pr.REPOS_DIR", tmp_path):
            result = tool._get_current_branch("nonexistent")
            assert result is None

    def test_get_current_branch_success(self, tmp_path: Path) -> None:
        """Test _get_current_branch returns branch name."""
        tool = GitHubCreatePRTool()
        repos_dir = tmp_path / "repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "feature-branch"

        with patch("reachy_mini_conversation_app.tools.github_create_pr.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Repo", return_value=mock_repo):
                result = tool._get_current_branch("myrepo")
                assert result == "feature-branch"

    def test_get_current_branch_with_slash(self, tmp_path: Path) -> None:
        """Test _get_current_branch with owner/repo format."""
        tool = GitHubCreatePRTool()
        repos_dir = tmp_path / "repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"

        with patch("reachy_mini_conversation_app.tools.github_create_pr.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Repo", return_value=mock_repo):
                result = tool._get_current_branch("owner/myrepo")
                assert result == "main"


class TestGitHubCreatePRToolExecution:
    """Tests for GitHubCreatePRTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_create_pr_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr requires confirmation."""
        tool = GitHubCreatePRTool()

        result = await tool(mock_deps, repo="owner/repo", title="Test PR", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_create_pr_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr returns error when repo is missing."""
        tool = GitHubCreatePRTool()

        result = await tool(mock_deps, repo="", title="Test PR", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_create_pr_missing_title(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr returns error when title is missing."""
        tool = GitHubCreatePRTool()

        result = await tool(mock_deps, repo="owner/repo", title="", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_create_pr_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr returns error when no token."""
        tool = GitHubCreatePRTool()

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", title="Test PR", confirmed=True)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_create_pr_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr returns error for invalid repo format."""
        tool = GitHubCreatePRTool()

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", title="Test PR", confirmed=True)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_create_pr_no_head_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_create_pr returns error when head branch not determined."""
        tool = GitHubCreatePRTool()

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.REPOS_DIR", tmp_path):
                result = await tool(mock_deps, repo="owner/repo", title="Test PR", confirmed=True)

        assert "error" in result
        assert "current branch" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_create_pr_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr creates PR successfully."""
        tool = GitHubCreatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.title = "Test PR"
        mock_pr.html_url = "https://github.com/owner/repo/pull/42"

        mock_gh_repo = MagicMock()
        mock_gh_repo.default_branch = "main"
        mock_gh_repo.owner.login = "owner"
        mock_gh_repo.get_branch.return_value = MagicMock()
        mock_gh_repo.get_pulls.return_value = []
        mock_gh_repo.create_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Test PR",
                    body="Description",
                    head="feature",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["pr_number"] == 42
        assert result["title"] == "Test PR"
        assert "pull/42" in result["url"]
        mock_gh_repo.create_pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_create_pr_draft(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr creates draft PR."""
        tool = GitHubCreatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Draft PR"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.default_branch = "main"
        mock_gh_repo.owner.login = "owner"
        mock_gh_repo.get_branch.return_value = MagicMock()
        mock_gh_repo.get_pulls.return_value = []
        mock_gh_repo.create_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Draft PR",
                    head="feature",
                    draft=True,
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["draft"] is True
        mock_gh_repo.create_pull.assert_called_once_with(
            title="Draft PR",
            body="",
            base="main",
            head="feature",
            draft=True,
        )

    @pytest.mark.asyncio
    async def test_github_create_pr_with_issue_link(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr links issue."""
        tool = GitHubCreatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Fix issue"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.default_branch = "main"
        mock_gh_repo.owner.login = "owner"
        mock_gh_repo.get_branch.return_value = MagicMock()
        mock_gh_repo.get_pulls.return_value = []
        mock_gh_repo.create_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Fix issue",
                    body="Fixed the bug",
                    head="feature",
                    issue_number=123,
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["linked_issue"] == 123
        # Verify body contains "Closes #123"
        call_args = mock_gh_repo.create_pull.call_args
        assert "Closes #123" in call_args.kwargs["body"]

    @pytest.mark.asyncio
    async def test_github_create_pr_branch_not_pushed(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr handles branch not on remote."""
        tool = GitHubCreatePRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_branch.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Test PR",
                    head="unpushed-branch",
                    confirmed=True,
                )

        assert "error" in result
        assert "not found on remote" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_create_pr_already_exists(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr handles existing PR."""
        tool = GitHubCreatePRTool()

        existing_pr = MagicMock()
        existing_pr.number = 10
        existing_pr.title = "Existing PR"
        existing_pr.html_url = "https://github.com/owner/repo/pull/10"

        mock_gh_repo = MagicMock()
        mock_gh_repo.default_branch = "main"
        mock_gh_repo.owner.login = "owner"
        mock_gh_repo.get_branch.return_value = MagicMock()
        mock_gh_repo.get_pulls.return_value = [existing_pr]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Test PR",
                    head="feature",
                    confirmed=True,
                )

        assert "error" in result
        assert "already exists" in result["error"].lower()
        assert result["existing_pr"]["number"] == 10

    @pytest.mark.asyncio
    async def test_github_create_pr_no_commits_between(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr handles no commits between branches."""
        tool = GitHubCreatePRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.default_branch = "main"
        mock_gh_repo.owner.login = "owner"
        mock_gh_repo.get_branch.return_value = MagicMock()
        mock_gh_repo.get_pulls.return_value = []
        mock_gh_repo.create_pull.side_effect = GithubException(
            422, {"message": "No commits between main and feature"}, None
        )

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Test PR",
                    head="feature",
                    confirmed=True,
                )

        assert "error" in result
        assert "No commits between" in result["error"]

    @pytest.mark.asyncio
    async def test_github_create_pr_repo_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr handles repo not found."""
        tool = GitHubCreatePRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/nonexistent",
                    title="Test PR",
                    head="feature",
                    confirmed=True,
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_create_pr_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_create_pr handles generic exception."""
        tool = GitHubCreatePRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected error")

        with patch("reachy_mini_conversation_app.tools.github_create_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_create_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Test PR",
                    head="feature",
                    confirmed=True,
                )

        assert "error" in result
        assert "Failed to create" in result["error"]
