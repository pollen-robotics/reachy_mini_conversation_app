"""Unit tests for the github_close_issue tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_close_issue import GitHubCloseIssueTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubCloseIssueToolAttributes:
    """Tests for GitHubCloseIssueTool tool attributes."""

    def test_github_close_issue_has_correct_name(self) -> None:
        """Test GitHubCloseIssueTool tool has correct name."""
        tool = GitHubCloseIssueTool()
        assert tool.name == "github_close_issue"

    def test_github_close_issue_has_description(self) -> None:
        """Test GitHubCloseIssueTool tool has description."""
        tool = GitHubCloseIssueTool()
        assert "close" in tool.description.lower()
        assert "reopen" in tool.description.lower()

    def test_github_close_issue_has_parameters_schema(self) -> None:
        """Test GitHubCloseIssueTool tool has correct parameters schema."""
        tool = GitHubCloseIssueTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "issue_number" in schema["properties"]
        assert "action" in schema["properties"]
        assert "reason" in schema["properties"]
        assert "comment" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "issue_number" in schema["required"]
        assert "action" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_close_issue_spec(self) -> None:
        """Test GitHubCloseIssueTool tool spec generation."""
        tool = GitHubCloseIssueTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_close_issue"


class TestGitHubCloseIssueToolExecution:
    """Tests for GitHubCloseIssueTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_close_issue_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue requires confirmation."""
        tool = GitHubCloseIssueTool()

        result = await tool(mock_deps, repo="owner/repo", issue_number=1, action="close", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_close_issue_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue returns error when repo is missing."""
        tool = GitHubCloseIssueTool()

        result = await tool(mock_deps, repo="", issue_number=1, action="close", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_issue_missing_issue_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue returns error when issue_number is missing."""
        tool = GitHubCloseIssueTool()

        result = await tool(mock_deps, repo="owner/repo", issue_number=None, action="close", confirmed=True)

        assert "error" in result
        assert "issue number" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_issue_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue returns error when no token."""
        tool = GitHubCloseIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", issue_number=1, action="close", confirmed=True)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_issue_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue returns error for invalid repo."""
        tool = GitHubCloseIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", issue_number=1, action="close", confirmed=True)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_issue_close_completed(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue closes issue as completed."""
        tool = GitHubCloseIssueTool()

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.html_url = "https://github.com/owner/repo/issues/1"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="close",
                    reason="completed",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert "closed" in result["message"].lower()
        assert result["reason"] == "completed"
        mock_issue.edit.assert_called_once_with(state="closed", state_reason="completed")

    @pytest.mark.asyncio
    async def test_github_close_issue_close_not_planned(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue closes issue as not_planned."""
        tool = GitHubCloseIssueTool()

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="close",
                    reason="not_planned",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["reason"] == "not_planned"
        mock_issue.edit.assert_called_once_with(state="closed", state_reason="not_planned")

    @pytest.mark.asyncio
    async def test_github_close_issue_reopen(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue reopens issue."""
        tool = GitHubCloseIssueTool()

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="reopen",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert "reopened" in result["message"].lower()
        mock_issue.edit.assert_called_once_with(state="open")

    @pytest.mark.asyncio
    async def test_github_close_issue_with_comment(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue adds comment when closing."""
        tool = GitHubCloseIssueTool()

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="close",
                    comment="Closing this issue as resolved.",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["comment_added"] is True
        mock_issue.create_comment.assert_called_once_with("Closing this issue as resolved.")

    @pytest.mark.asyncio
    async def test_github_close_issue_unknown_action(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue handles unknown action."""
        tool = GitHubCloseIssueTool()

        mock_issue = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="unknown",
                    confirmed=True,
                )

        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_issue_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue handles issue not found."""
        tool = GitHubCloseIssueTool()

        mock_repo = MagicMock()
        mock_repo.get_issue.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=999,
                    action="close",
                    confirmed=True,
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_issue_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue handles API error."""
        tool = GitHubCloseIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="close",
                    confirmed=True,
                )

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_issue_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_issue handles generic exception."""
        tool = GitHubCloseIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_close_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    action="close",
                    confirmed=True,
                )

        assert "error" in result
        assert "Failed to close issue" in result["error"]
