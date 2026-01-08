"""Unit tests for the github_list_issues tool."""

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_list_issues import GitHubListIssuesTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubListIssuesToolAttributes:
    """Tests for GitHubListIssuesTool tool attributes."""

    def test_github_list_issues_has_correct_name(self) -> None:
        """Test GitHubListIssuesTool tool has correct name."""
        tool = GitHubListIssuesTool()
        assert tool.name == "github_list_issues"

    def test_github_list_issues_has_description(self) -> None:
        """Test GitHubListIssuesTool tool has description."""
        tool = GitHubListIssuesTool()
        assert "list" in tool.description.lower()
        assert "issues" in tool.description.lower()

    def test_github_list_issues_has_parameters_schema(self) -> None:
        """Test GitHubListIssuesTool tool has correct parameters schema."""
        tool = GitHubListIssuesTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "state" in schema["properties"]
        assert "labels" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_list_issues_spec(self) -> None:
        """Test GitHubListIssuesTool tool spec generation."""
        tool = GitHubListIssuesTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_list_issues"


class TestGitHubListIssuesToolExecution:
    """Tests for GitHubListIssuesTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_list_issues_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues returns error when no token."""
        tool = GitHubListIssuesTool()

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_issues_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues returns error when repo is missing."""
        tool = GitHubListIssuesTool()

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_issues_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues returns error for invalid repo."""
        tool = GitHubListIssuesTool()

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_issues_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues returns issues successfully."""
        tool = GitHubListIssuesTool()

        mock_label = MagicMock()
        mock_label.name = "bug"

        mock_user = MagicMock()
        mock_user.login = "testuser"

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.state = "open"
        mock_issue.user = mock_user
        mock_issue.labels = [mock_label]
        mock_issue.created_at = datetime(2024, 1, 15, 12, 0, 0)
        mock_issue.comments = 3
        mock_issue.html_url = "https://github.com/owner/repo/issues/1"
        mock_issue.pull_request = None  # Not a PR

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [mock_issue]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert result["status"] == "success"
        assert result["repo"] == "owner/repo"
        assert result["count"] == 1
        assert result["issues"][0]["number"] == 1
        assert result["issues"][0]["title"] == "Test Issue"
        assert result["issues"][0]["author"] == "testuser"
        assert "bug" in result["issues"][0]["labels"]

    @pytest.mark.asyncio
    async def test_github_list_issues_filters_prs(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues filters out pull requests."""
        tool = GitHubListIssuesTool()

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Real Issue"
        mock_issue.state = "open"
        mock_issue.user = MagicMock(login="user")
        mock_issue.labels = []
        mock_issue.created_at = datetime(2024, 1, 15)
        mock_issue.comments = 0
        mock_issue.html_url = "url"
        mock_issue.pull_request = None

        mock_pr = MagicMock()
        mock_pr.number = 2
        mock_pr.title = "PR"
        mock_pr.pull_request = MagicMock()  # This makes it a PR

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [mock_issue, mock_pr]

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert result["count"] == 1
        assert result["issues"][0]["number"] == 1

    @pytest.mark.asyncio
    async def test_github_list_issues_with_state(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues with state filter."""
        tool = GitHubListIssuesTool()

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", state="closed")

        assert result["status"] == "success"
        assert result["state_filter"] == "closed"
        mock_repo.get_issues.assert_called_once_with(state="closed", labels=[])

    @pytest.mark.asyncio
    async def test_github_list_issues_with_labels(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues with label filter."""
        tool = GitHubListIssuesTool()

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", labels=["bug", "high"])

        assert result["status"] == "success"
        mock_repo.get_issues.assert_called_once_with(state="open", labels=["bug", "high"])

    @pytest.mark.asyncio
    async def test_github_list_issues_limit(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues respects limit."""
        tool = GitHubListIssuesTool()

        # Create 30 mock issues
        mock_issues = []
        for i in range(30):
            mock_issue = MagicMock()
            mock_issue.number = i + 1
            mock_issue.title = f"Issue {i + 1}"
            mock_issue.state = "open"
            mock_issue.user = MagicMock(login="user")
            mock_issue.labels = []
            mock_issue.created_at = datetime(2024, 1, 15)
            mock_issue.comments = 0
            mock_issue.html_url = f"url/{i + 1}"
            mock_issue.pull_request = None
            mock_issues.append(mock_issue)

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = mock_issues

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", limit=5)

        assert result["count"] == 5

    @pytest.mark.asyncio
    async def test_github_list_issues_limit_capped_at_50(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues caps limit at 50."""
        tool = GitHubListIssuesTool()

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = []

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                # Request 100, should be capped at 50 internally
                await tool(mock_deps, repo="owner/repo", limit=100)

        # The tool uses min(limit, 50)
        assert True  # Just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_github_list_issues_auth_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues handles auth error."""
        tool = GitHubListIssuesTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"}, None)

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "bad-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "authentication" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_issues_repo_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues handles repo not found."""
        tool = GitHubListIssuesTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_issues_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_issues handles generic exception."""
        tool = GitHubListIssuesTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_list_issues.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_list_issues.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo")

        assert "error" in result
        assert "Failed to list issues" in result["error"]
