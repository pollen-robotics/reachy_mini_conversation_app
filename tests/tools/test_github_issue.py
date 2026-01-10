"""Unit tests for the github_issue tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_issue import GitHubIssueTool


class TestGitHubIssueToolAttributes:
    """Tests for GitHubIssueTool tool attributes."""

    def test_github_issue_has_correct_name(self) -> None:
        """Test GitHubIssueTool tool has correct name."""
        tool = GitHubIssueTool()
        assert tool.name == "github_issue"

    def test_github_issue_has_description(self) -> None:
        """Test GitHubIssueTool tool has description."""
        tool = GitHubIssueTool()
        assert "issue" in tool.description.lower()
        assert "create" in tool.description.lower()

    def test_github_issue_has_parameters_schema(self) -> None:
        """Test GitHubIssueTool tool has correct parameters schema."""
        tool = GitHubIssueTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "title" in schema["properties"]
        assert "body" in schema["properties"]
        assert "labels" in schema["properties"]
        assert "repo" in schema["required"]
        assert "title" in schema["required"]
        assert "body" in schema["required"]

    def test_github_issue_spec(self) -> None:
        """Test GitHubIssueTool tool spec generation."""
        tool = GitHubIssueTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_issue"


class TestGetFullRepoName:
    """Tests for _get_full_repo_name method."""

    def test_full_repo_name_with_slash(self) -> None:
        """Test repo name already has owner."""
        tool = GitHubIssueTool()
        assert tool._get_full_repo_name("owner/repo") == "owner/repo"

    def test_full_repo_name_with_default_owner(self) -> None:
        """Test repo name uses default owner."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("myrepo")

        assert result == "default-owner/myrepo"

    def test_full_repo_name_no_owner_raises(self) -> None:
        """Test repo name without owner raises error."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError) as exc_info:
                tool._get_full_repo_name("myrepo")

        assert "must include owner" in str(exc_info.value)


class TestGitHubIssueToolExecution:
    """Tests for GitHubIssueTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_issue_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue returns error when no token."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", title="Test", body="Body")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_issue_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue returns error when repo is missing."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="", title="Test", body="Body")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_issue_missing_title(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue returns error when title is missing."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="owner/repo", title="", body="Body")

        assert "error" in result
        assert "title" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_issue_missing_body(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue returns error when body is missing."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="owner/repo", title="Test", body="")

        assert "error" in result
        assert "body" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_issue_invalid_repo_name(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue returns error for invalid repo name."""
        tool = GitHubIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", title="Test", body="Body")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_issue_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue creates issue successfully."""
        tool = GitHubIssueTool()

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.html_url = "https://github.com/owner/repo/issues/42"
        mock_issue.title = "Test Issue"

        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Test Issue",
                    body="Issue body",
                )

        assert result["status"] == "success"
        assert result["issue_number"] == 42
        assert result["issue_url"] == "https://github.com/owner/repo/issues/42"
        mock_repo.create_issue.assert_called_once_with(title="Test Issue", body="Issue body", labels=[])

    @pytest.mark.asyncio
    async def test_github_issue_with_labels(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue creates issue with labels."""
        tool = GitHubIssueTool()

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.html_url = "https://github.com/owner/repo/issues/1"
        mock_issue.title = "Bug"

        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    title="Bug",
                    body="Description",
                    labels=["bug", "priority"],
                )

        assert result["status"] == "success"
        mock_repo.create_issue.assert_called_once_with(
            title="Bug",
            body="Description",
            labels=["bug", "priority"],
        )

    @pytest.mark.asyncio
    async def test_github_issue_auth_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue handles authentication error."""
        tool = GitHubIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"}, None)

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "bad-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", title="Test", body="Body")

        assert "error" in result
        assert "authentication" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_issue_permission_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue handles permission error."""
        tool = GitHubIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(403, {"message": "Forbidden"}, None)

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", title="Test", body="Body")

        assert "error" in result
        assert "permission" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_issue_repo_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue handles repo not found."""
        tool = GitHubIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", title="Test", body="Body")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_issue_generic_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue handles generic API error."""
        tool = GitHubIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", title="Test", body="Body")

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_issue_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_issue handles generic exception."""
        tool = GitHubIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected error")

        with patch("reachy_mini_conversation_app.tools.github_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", title="Test", body="Body")

        assert "error" in result
        assert "Failed to create issue" in result["error"]
