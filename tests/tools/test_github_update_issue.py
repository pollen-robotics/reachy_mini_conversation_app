"""Unit tests for the github_update_issue tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_update_issue import GitHubUpdateIssueTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubUpdateIssueToolAttributes:
    """Tests for GitHubUpdateIssueTool tool attributes."""

    def test_github_update_issue_has_correct_name(self) -> None:
        """Test GitHubUpdateIssueTool tool has correct name."""
        tool = GitHubUpdateIssueTool()
        assert tool.name == "github_update_issue"

    def test_github_update_issue_has_description(self) -> None:
        """Test GitHubUpdateIssueTool tool has description."""
        tool = GitHubUpdateIssueTool()
        assert "update" in tool.description.lower()
        assert "issue" in tool.description.lower()

    def test_github_update_issue_has_parameters_schema(self) -> None:
        """Test GitHubUpdateIssueTool tool has correct parameters schema."""
        tool = GitHubUpdateIssueTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "issue_number" in schema["properties"]
        assert "title" in schema["properties"]
        assert "body" in schema["properties"]
        assert "labels" in schema["properties"]
        assert "assignees" in schema["properties"]
        assert "milestone" in schema["properties"]
        assert "repo" in schema["required"]
        assert "issue_number" in schema["required"]

    def test_github_update_issue_spec(self) -> None:
        """Test GitHubUpdateIssueTool tool spec generation."""
        tool = GitHubUpdateIssueTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_update_issue"


class TestGitHubUpdateIssueToolExecution:
    """Tests for GitHubUpdateIssueTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_update_issue_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue returns error when repo is missing."""
        tool = GitHubUpdateIssueTool()

        result = await tool(mock_deps, repo="", issue_number=1)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_update_issue_missing_issue_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue returns error when issue_number is missing."""
        tool = GitHubUpdateIssueTool()

        result = await tool(mock_deps, repo="owner/repo", issue_number=None)

        assert "error" in result
        assert "issue number" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_update_issue_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue returns error when no token."""
        tool = GitHubUpdateIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", issue_number=1, title="New title")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_issue_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue returns error for invalid repo."""
        tool = GitHubUpdateIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", issue_number=1, title="Title")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_issue_no_changes(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue with no updates provided."""
        tool = GitHubUpdateIssueTool()

        mock_issue = MagicMock()
        mock_issue.html_url = "https://github.com/owner/repo/issues/1"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1)

        assert result["status"] == "no_changes"
        assert "No updates" in result["message"]

    @pytest.mark.asyncio
    async def test_github_update_issue_title(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue updates title."""
        tool = GitHubUpdateIssueTool()

        mock_issue = MagicMock()
        mock_issue.title = "New Title"
        mock_issue.html_url = "https://github.com/owner/repo/issues/1"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, title="New Title")

        assert result["status"] == "success"
        assert "title" in result["updated_fields"]
        mock_issue.edit.assert_called_with(title="New Title")

    @pytest.mark.asyncio
    async def test_github_update_issue_body(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue updates body."""
        tool = GitHubUpdateIssueTool()

        mock_issue = MagicMock()
        mock_issue.title = "Title"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, body="New body")

        assert result["status"] == "success"
        assert "body" in result["updated_fields"]

    @pytest.mark.asyncio
    async def test_github_update_issue_labels(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue updates labels."""
        tool = GitHubUpdateIssueTool()

        mock_issue = MagicMock()
        mock_issue.title = "Title"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, labels=["bug", "high"])

        assert result["status"] == "success"
        assert "labels" in result["updated_fields"]
        mock_issue.set_labels.assert_called_once_with("bug", "high")

    @pytest.mark.asyncio
    async def test_github_update_issue_assignees(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue updates assignees."""
        tool = GitHubUpdateIssueTool()

        mock_assignee = MagicMock()
        mock_issue = MagicMock()
        mock_issue.title = "Title"
        mock_issue.html_url = "url"
        mock_issue.assignees = [mock_assignee]

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, assignees=["newuser"])

        assert result["status"] == "success"
        assert "assignees" in result["updated_fields"]
        mock_issue.remove_from_assignees.assert_called_once()
        mock_issue.add_to_assignees.assert_called_once_with("newuser")

    @pytest.mark.asyncio
    async def test_github_update_issue_milestone(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue updates milestone."""
        tool = GitHubUpdateIssueTool()

        mock_milestone = MagicMock()
        mock_issue = MagicMock()
        mock_issue.title = "Title"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue
        mock_repo.get_milestone.return_value = mock_milestone

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, milestone=5)

        assert result["status"] == "success"
        assert "milestone" in result["updated_fields"]
        mock_repo.get_milestone.assert_called_once_with(5)
        mock_issue.edit.assert_called_with(milestone=mock_milestone)

    @pytest.mark.asyncio
    async def test_github_update_issue_remove_milestone(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue removes milestone with 0."""
        tool = GitHubUpdateIssueTool()

        mock_issue = MagicMock()
        mock_issue.title = "Title"
        mock_issue.html_url = "url"

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, milestone=0)

        assert result["status"] == "success"
        assert "milestone" in result["updated_fields"]
        mock_issue.edit.assert_called_with(milestone=None)

    @pytest.mark.asyncio
    async def test_github_update_issue_multiple_fields(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue updates multiple fields."""
        tool = GitHubUpdateIssueTool()

        mock_issue = MagicMock()
        mock_issue.title = "New Title"
        mock_issue.html_url = "url"
        mock_issue.assignees = []

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    title="New Title",
                    body="New body",
                    labels=["bug"],
                )

        assert result["status"] == "success"
        assert "title" in result["updated_fields"]
        assert "body" in result["updated_fields"]
        assert "labels" in result["updated_fields"]

    @pytest.mark.asyncio
    async def test_github_update_issue_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue handles issue not found."""
        tool = GitHubUpdateIssueTool()

        mock_repo = MagicMock()
        mock_repo.get_issue.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=999, title="New")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_update_issue_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue handles API error."""
        tool = GitHubUpdateIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, title="New")

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_issue_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_issue handles generic exception."""
        tool = GitHubUpdateIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_update_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_issue.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", issue_number=1, title="New")

        assert "error" in result
        assert "Failed to update issue" in result["error"]
