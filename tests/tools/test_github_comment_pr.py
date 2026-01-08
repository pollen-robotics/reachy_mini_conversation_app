"""Unit tests for the github_comment_pr tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_comment_pr import GitHubCommentPRTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubCommentPRToolAttributes:
    """Tests for GitHubCommentPRTool tool attributes."""

    def test_github_comment_pr_has_correct_name(self) -> None:
        """Test GitHubCommentPRTool tool has correct name."""
        tool = GitHubCommentPRTool()
        assert tool.name == "github_comment_pr"

    def test_github_comment_pr_has_description(self) -> None:
        """Test GitHubCommentPRTool tool has description."""
        tool = GitHubCommentPRTool()
        assert "comment" in tool.description.lower()
        assert "pull request" in tool.description.lower()

    def test_github_comment_pr_has_parameters_schema(self) -> None:
        """Test GitHubCommentPRTool tool has correct parameters schema."""
        tool = GitHubCommentPRTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "pr_number" in schema["properties"]
        assert "body" in schema["properties"]
        assert "comment_type" in schema["properties"]
        assert "path" in schema["properties"]
        assert "line" in schema["properties"]
        assert "side" in schema["properties"]
        assert "start_line" in schema["properties"]
        assert "repo" in schema["required"]
        assert "pr_number" in schema["required"]
        assert "body" in schema["required"]

    def test_github_comment_pr_spec(self) -> None:
        """Test GitHubCommentPRTool tool spec generation."""
        tool = GitHubCommentPRTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_comment_pr"


class TestGitHubCommentPRGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubCommentPRTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubCommentPRTool()
        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubCommentPRTool()
        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubCommentPRToolExecution:
    """Tests for GitHubCommentPRTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_comment_pr_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error when repo is missing."""
        tool = GitHubCommentPRTool()

        result = await tool(mock_deps, repo="", pr_number=1, body="Comment")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_pr_missing_pr_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error when pr_number is missing."""
        tool = GitHubCommentPRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=None, body="Comment")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_pr_missing_body(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error when body is missing."""
        tool = GitHubCommentPRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=1, body="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_pr_review_missing_path(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error for review comment without path."""
        tool = GitHubCommentPRTool()

        result = await tool(
            mock_deps, repo="owner/repo", pr_number=1, body="Comment", comment_type="review"
        )

        assert "error" in result
        assert "path" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_pr_review_missing_line(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error for review comment without line."""
        tool = GitHubCommentPRTool()

        result = await tool(
            mock_deps,
            repo="owner/repo",
            pr_number=1,
            body="Comment",
            comment_type="review",
            path="file.py",
        )

        assert "error" in result
        assert "line" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_pr_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error when no token."""
        tool = GitHubCommentPRTool()

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", pr_number=1, body="Comment")

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_pr_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr returns error for invalid repo format."""
        tool = GitHubCommentPRTool()

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", pr_number=1, body="Comment")

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_pr_general_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr adds general comment successfully."""
        tool = GitHubCommentPRTool()

        mock_comment = MagicMock()
        mock_comment.id = 12345
        mock_comment.html_url = "https://github.com/owner/repo/pull/1#issuecomment-12345"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.create_issue_comment.return_value = mock_comment

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="This is a test comment",
                )

        assert result["status"] == "success"
        assert result["pr_number"] == 1
        assert result["pr_title"] == "Test PR"
        assert result["comment_id"] == 12345
        mock_pr.create_issue_comment.assert_called_once_with("This is a test comment")

    @pytest.mark.asyncio
    async def test_github_comment_pr_review_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr adds review comment successfully."""
        tool = GitHubCommentPRTool()

        mock_comment = MagicMock()
        mock_comment.id = 67890

        mock_commit = MagicMock()

        mock_commits = MagicMock()
        mock_commits.reversed = [mock_commit]

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.get_commits.return_value = mock_commits
        mock_pr.create_review_comment.return_value = mock_comment

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="Review comment",
                    comment_type="review",
                    path="src/file.py",
                    line=42,
                )

        assert result["status"] == "success"
        assert result["comment_id"] == 67890
        assert result["path"] == "src/file.py"
        assert result["line"] == 42
        mock_pr.create_review_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_comment_pr_review_multiline(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr adds multiline review comment."""
        tool = GitHubCommentPRTool()

        mock_comment = MagicMock()
        mock_comment.id = 123

        mock_commit = MagicMock()

        mock_commits = MagicMock()
        mock_commits.reversed = [mock_commit]

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.get_commits.return_value = mock_commits
        mock_pr.create_review_comment.return_value = mock_comment

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="Multiline comment",
                    comment_type="review",
                    path="file.py",
                    line=50,
                    start_line=45,
                )

        assert result["status"] == "success"
        assert result["start_line"] == 45

    @pytest.mark.asyncio
    async def test_github_comment_pr_unknown_type(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr handles unknown comment type."""
        tool = GitHubCommentPRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="Comment",
                    comment_type="unknown",
                )

        assert "error" in result
        assert "Unknown comment type" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_pr_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr handles PR not found."""
        tool = GitHubCommentPRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=999,
                    body="Comment",
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_comment_pr_invalid_review_comment(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr handles invalid review comment."""
        tool = GitHubCommentPRTool()

        mock_commit = MagicMock()

        mock_commits = MagicMock()
        mock_commits.reversed = [mock_commit]

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.get_commits.return_value = mock_commits
        mock_pr.create_review_comment.side_effect = GithubException(
            422, {"message": "Validation Failed"}, None
        )

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="Comment",
                    comment_type="review",
                    path="nonexistent.py",
                    line=999,
                )

        assert "error" in result
        assert "Invalid review comment" in result["error"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_comment_pr_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr handles API error."""
        tool = GitHubCommentPRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="Comment",
                )

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_comment_pr_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_comment_pr handles generic exception."""
        tool = GitHubCommentPRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_comment_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_comment_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    body="Comment",
                )

        assert "error" in result
        assert "Failed to comment" in result["error"]
