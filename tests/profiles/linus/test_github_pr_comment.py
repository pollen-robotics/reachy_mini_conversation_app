"""Unit tests for the github_pr_comment tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_pr_comment import GitHubPRCommentTool


class TestGitHubPRCommentToolAttributes:
    """Tests for GitHubPRCommentTool tool attributes."""

    def test_github_pr_comment_has_correct_name(self) -> None:
        """Test GitHubPRCommentTool tool has correct name."""
        tool = GitHubPRCommentTool()
        assert tool.name == "github_pr_comment"

    def test_github_pr_comment_has_description(self) -> None:
        """Test GitHubPRCommentTool tool has description."""
        tool = GitHubPRCommentTool()
        assert "comment" in tool.description.lower()
        assert "pull request" in tool.description.lower()

    def test_github_pr_comment_has_parameters_schema(self) -> None:
        """Test GitHubPRCommentTool tool has correct parameters schema."""
        tool = GitHubPRCommentTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "pr_number" in schema["properties"]
        assert "comment" in schema["properties"]
        assert "repo" in schema["required"]
        assert "pr_number" in schema["required"]
        assert "comment" in schema["required"]

    def test_github_pr_comment_spec(self) -> None:
        """Test GitHubPRCommentTool tool spec generation."""
        tool = GitHubPRCommentTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_pr_comment"


class TestGitHubPRCommentGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubPRCommentTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubPRCommentTool()
        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubPRCommentTool()
        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubPRCommentToolExecution:
    """Tests for GitHubPRCommentTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_pr_comment_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment returns error when no token."""
        tool = GitHubPRCommentTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", pr_number=1, comment="Test")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_comment_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment returns error when repo is missing."""
        tool = GitHubPRCommentTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="", pr_number=1, comment="Test")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_comment_missing_pr_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment returns error when pr_number is missing."""
        tool = GitHubPRCommentTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="owner/repo", pr_number=None, comment="Test")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_comment_missing_comment(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment returns error when comment is missing."""
        tool = GitHubPRCommentTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            result = await tool(mock_deps, repo="owner/repo", pr_number=1, comment="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_comment_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment returns error for invalid repo format."""
        tool = GitHubPRCommentTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", pr_number=1, comment="Test")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_comment_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment adds comment successfully."""
        tool = GitHubPRCommentTool()

        mock_comment = MagicMock()
        mock_comment.html_url = "https://github.com/owner/repo/pull/1#issuecomment-123"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.create_issue_comment.return_value = mock_comment

        mock_repo = MagicMock()
        mock_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    comment="This is my comment",
                )

        assert result["status"] == "success"
        assert result["pr_number"] == 1
        assert result["pr_title"] == "Test PR"
        assert "issuecomment-123" in result["comment_url"]
        assert result["repo"] == "owner/repo"
        mock_pr.create_issue_comment.assert_called_once_with("This is my comment")

    @pytest.mark.asyncio
    async def test_github_pr_comment_auth_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment handles auth error."""
        tool = GitHubPRCommentTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(401, {"message": "Bad credentials"}, None)

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "bad-token"
            with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    comment="Test",
                )

        assert "error" in result
        assert "authentication" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_comment_permission_denied(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment handles permission denied."""
        tool = GitHubPRCommentTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(403, {"message": "Forbidden"}, None)

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    comment="Test",
                )

        assert "error" in result
        assert "permission" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_comment_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment handles PR not found."""
        tool = GitHubPRCommentTool()

        mock_repo = MagicMock()
        mock_repo.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_repo

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=999,
                    comment="Test",
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_comment_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment handles API error."""
        tool = GitHubPRCommentTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    comment="Test",
                )

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_comment_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_comment handles generic exception."""
        tool = GitHubPRCommentTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.profiles.linus.github_pr_comment.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    comment="Test",
                )

        assert "error" in result
        assert "Failed to comment" in result["error"]
