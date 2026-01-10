"""Unit tests for the github_list_prs tool."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_list_prs import GitHubListPRsTool


class TestGitHubListPRsToolAttributes:
    """Tests for GitHubListPRsTool tool attributes."""

    def test_github_list_prs_has_correct_name(self) -> None:
        """Test GitHubListPRsTool tool has correct name."""
        tool = GitHubListPRsTool()
        assert tool.name == "github_list_prs"

    def test_github_list_prs_has_description(self) -> None:
        """Test GitHubListPRsTool tool has description."""
        tool = GitHubListPRsTool()
        assert "pull request" in tool.description.lower()

    def test_github_list_prs_has_parameters_schema(self) -> None:
        """Test GitHubListPRsTool tool has correct parameters schema."""
        tool = GitHubListPRsTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "state" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_list_prs_spec(self) -> None:
        """Test GitHubListPRsTool tool spec generation."""
        tool = GitHubListPRsTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_list_prs"


class TestGitHubListPRsGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubListPRsTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubListPRsTool()
        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubListPRsTool()
        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubListPRsToolExecution:
    """Tests for GitHubListPRsTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_list_prs_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs returns error when no token."""
        tool = GitHubListPRsTool()

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_prs_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs returns error when repo is missing."""
        tool = GitHubListPRsTool()

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_prs_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs returns error for invalid repo format."""
        tool = GitHubListPRsTool()

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_prs_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs lists PRs successfully."""
        tool = GitHubListPRsTool()

        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.title = "Test PR"
        mock_pr.state = "open"
        mock_pr.user.login = "author"
        mock_pr.base.ref = "main"
        mock_pr.head.ref = "feature"
        mock_pr.mergeable = True
        mock_pr.merged = False
        mock_pr.draft = False
        mock_pr.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_pr.updated_at = datetime(2024, 1, 2, 12, 0, 0)
        mock_pr.comments = 5
        mock_pr.review_comments = 3
        mock_pr.additions = 100
        mock_pr.deletions = 50
        mock_pr.changed_files = 10
        mock_pr.html_url = "https://github.com/owner/repo/pull/42"

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = [mock_pr]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert result["status"] == "success"
        assert result["repo"] == "owner/repo"
        assert result["count"] == 1
        assert len(result["pull_requests"]) == 1
        pr = result["pull_requests"][0]
        assert pr["number"] == 42
        assert pr["title"] == "Test PR"
        assert pr["author"] == "author"
        assert pr["base"] == "main"
        assert pr["head"] == "feature"

    @pytest.mark.asyncio
    async def test_github_list_prs_state_filter(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs with state filter."""
        tool = GitHubListPRsTool()

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", state="closed")

        assert result["status"] == "success"
        assert result["state_filter"] == "closed"
        mock_repo.get_pulls.assert_called_once_with(state="closed", sort="updated", direction="desc")

    @pytest.mark.asyncio
    async def test_github_list_prs_limit(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs respects limit."""
        tool = GitHubListPRsTool()

        # Create 5 mock PRs
        mock_prs = []
        for i in range(5):
            pr = MagicMock()
            pr.number = i + 1
            pr.title = f"PR {i + 1}"
            pr.state = "open"
            pr.user.login = "author"
            pr.base.ref = "main"
            pr.head.ref = f"feature-{i}"
            pr.mergeable = True
            pr.merged = False
            pr.draft = False
            pr.created_at = None
            pr.updated_at = None
            pr.comments = 0
            pr.review_comments = 0
            pr.additions = 0
            pr.deletions = 0
            pr.changed_files = 0
            pr.html_url = f"url/{i}"
            mock_prs.append(pr)

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = mock_prs

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", limit=3)

        assert result["status"] == "success"
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_github_list_prs_limit_max(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs caps limit at 50."""
        tool = GitHubListPRsTool()

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                # Request 100, should be capped at 50
                result = await tool(mock_deps, repo="owner/repo", limit=100)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_list_prs_empty(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs with no PRs."""
        tool = GitHubListPRsTool()

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["pull_requests"] == []

    @pytest.mark.asyncio
    async def test_github_list_prs_auth_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs handles auth error."""
        tool = GitHubListPRsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"}, None)

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "bad-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "authentication" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_prs_permission_denied(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs handles permission denied."""
        tool = GitHubListPRsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(403, {"message": "Forbidden"}, None)

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "permission" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_prs_repo_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs handles repo not found."""
        tool = GitHubListPRsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_prs_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs handles API error."""
        tool = GitHubListPRsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_prs_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_prs handles generic exception."""
        tool = GitHubListPRsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected error")

        with patch("reachy_mini_conversation_app.tools.github_list_prs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_prs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "Failed to list" in result["error"]
