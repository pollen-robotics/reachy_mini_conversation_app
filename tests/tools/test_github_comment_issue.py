"""Unit tests for the github_comment_issue tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_comment_issue import GitHubCommentIssueTool


class TestGitHubCommentIssueToolAttributes:
    """Tests for GitHubCommentIssueTool tool attributes."""

    def test_github_comment_issue_has_correct_name(self) -> None:
        """Test GitHubCommentIssueTool tool has correct name."""
        tool = GitHubCommentIssueTool()
        assert tool.name == "github_comment_issue"

    def test_github_comment_issue_has_description(self) -> None:
        """Test GitHubCommentIssueTool tool has description."""
        tool = GitHubCommentIssueTool()
        assert "comment" in tool.description.lower()
        assert "issue" in tool.description.lower()

    def test_github_comment_issue_has_parameters_schema(self) -> None:
        """Test GitHubCommentIssueTool tool has correct parameters schema."""
        tool = GitHubCommentIssueTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "issue_number" in schema["properties"]
        assert "body" in schema["properties"]
        assert "repo" in schema["required"]
        assert "issue_number" in schema["required"]
        assert "body" in schema["required"]

    def test_github_comment_issue_spec(self) -> None:
        """Test GitHubCommentIssueTool tool spec generation."""
        tool = GitHubCommentIssueTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_comment_issue"


class TestGitHubCommentIssueGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubCommentIssueTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubCommentIssueTool()
        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubCommentIssueTool()
        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubCommentIssueToolExecution:
    """Tests for GitHubCommentIssueTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_comment_issue_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue returns error when repo is missing."""
        tool = GitHubCommentIssueTool()

        result = await tool(mock_deps, repo="", issue_number=1, body="Test comment")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_issue_missing_issue_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue returns error when issue_number is missing."""
        tool = GitHubCommentIssueTool()

        result = await tool(mock_deps, repo="owner/repo", issue_number=None, body="Test comment")

        assert "error" in result
        assert "issue number" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_issue_missing_body(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue returns error when body is missing."""
        tool = GitHubCommentIssueTool()

        result = await tool(mock_deps, repo="owner/repo", issue_number=1, body="")

        assert "error" in result
        assert "body" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_issue_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue returns error when no token."""
        tool = GitHubCommentIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", issue_number=1, body="Test comment")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_issue_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue returns error for invalid repo format."""
        tool = GitHubCommentIssueTool()

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", issue_number=1, body="Test comment")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_issue_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue creates comment successfully."""
        tool = GitHubCommentIssueTool()

        mock_comment = MagicMock()
        mock_comment.id = 12345
        mock_comment.html_url = "https://github.com/owner/repo/issues/1#issuecomment-12345"

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.create_comment.return_value = mock_comment

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    body="This is a test comment",
                )

        assert result["status"] == "success"
        assert result["issue_number"] == 1
        assert result["issue_title"] == "Test Issue"
        assert result["comment_id"] == 12345
        assert "issuecomment-12345" in result["comment_url"]
        mock_issue.create_comment.assert_called_once_with("This is a test comment")

    @pytest.mark.asyncio
    async def test_github_comment_issue_markdown(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue with markdown comment."""
        tool = GitHubCommentIssueTool()

        mock_comment = MagicMock()
        mock_comment.id = 123
        mock_comment.html_url = "url"

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Issue"
        mock_issue.create_comment.return_value = mock_comment

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        markdown_body = "## Summary\n\n- Item 1\n- Item 2\n\n```python\nprint('hello')\n```"

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    body=markdown_body,
                )

        assert result["status"] == "success"
        mock_issue.create_comment.assert_called_once_with(markdown_body)

    @pytest.mark.asyncio
    async def test_github_comment_issue_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue handles issue not found."""
        tool = GitHubCommentIssueTool()

        mock_repo = MagicMock()
        mock_repo.get_issue.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=999,
                    body="Test comment",
                )

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "#999" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_issue_repo_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue handles repo not found."""
        tool = GitHubCommentIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/nonexistent",
                    issue_number=1,
                    body="Test comment",
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_issue_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue handles API error."""
        tool = GitHubCommentIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    body="Test comment",
                )

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_issue_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue handles generic exception."""
        tool = GitHubCommentIssueTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected error")

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    issue_number=1,
                    body="Test comment",
                )

        assert "error" in result
        assert "Failed to comment" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_issue_with_default_owner(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_issue uses default owner."""
        tool = GitHubCommentIssueTool()

        mock_comment = MagicMock()
        mock_comment.id = 123
        mock_comment.html_url = "url"

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.title = "Issue"
        mock_issue.create_comment.return_value = mock_comment

        mock_repo = MagicMock()
        mock_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_issue.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            with patch("reachy_mini_conversation_app.tools.github_comment_issue.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="myrepo",
                    issue_number=1,
                    body="Test comment",
                )

        assert result["status"] == "success"
        mock_github.get_repo.assert_called_once_with("default-owner/myrepo")
