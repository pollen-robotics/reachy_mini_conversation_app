"""Unit tests for the github_close_pr tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_close_pr import GitHubClosePRTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubClosePRToolAttributes:
    """Tests for GitHubClosePRTool tool attributes."""

    def test_github_close_pr_has_correct_name(self) -> None:
        """Test GitHubClosePRTool tool has correct name."""
        tool = GitHubClosePRTool()
        assert tool.name == "github_close_pr"

    def test_github_close_pr_has_description(self) -> None:
        """Test GitHubClosePRTool tool has description."""
        tool = GitHubClosePRTool()
        assert "close" in tool.description.lower()
        assert "merge" in tool.description.lower()

    def test_github_close_pr_has_parameters_schema(self) -> None:
        """Test GitHubClosePRTool tool has correct parameters schema."""
        tool = GitHubClosePRTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "pr_number" in schema["properties"]
        assert "action" in schema["properties"]
        assert "merge_method" in schema["properties"]
        assert "commit_title" in schema["properties"]
        assert "commit_message" in schema["properties"]
        assert "comment" in schema["properties"]
        assert "delete_branch" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "pr_number" in schema["required"]
        assert "action" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_close_pr_spec(self) -> None:
        """Test GitHubClosePRTool tool spec generation."""
        tool = GitHubClosePRTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_close_pr"


class TestGitHubClosePRGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubClosePRTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubClosePRTool()
        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubClosePRTool()
        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubClosePRToolExecution:
    """Tests for GitHubClosePRTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_close_pr_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr requires confirmation."""
        tool = GitHubClosePRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=1, action="close", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_close_pr_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr returns error when repo is missing."""
        tool = GitHubClosePRTool()

        result = await tool(mock_deps, repo="", pr_number=1, action="close", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_pr_missing_pr_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr returns error when pr_number is missing."""
        tool = GitHubClosePRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=None, action="close", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_pr_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr returns error when no token."""
        tool = GitHubClosePRTool()

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", pr_number=1, action="close", confirmed=True)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_pr_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr returns error for invalid repo format."""
        tool = GitHubClosePRTool()

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", pr_number=1, action="close", confirmed=True)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_pr_close_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr closes PR successfully."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "https://github.com/owner/repo/pull/1"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="close",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert "closed" in result["message"].lower()
        assert result["comment_added"] is False
        mock_pr.edit.assert_called_once_with(state="closed")

    @pytest.mark.asyncio
    async def test_github_close_pr_close_with_comment(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr closes PR with comment."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="close",
                    comment="Closing this PR",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["comment_added"] is True
        mock_pr.create_issue_comment.assert_called_once_with("Closing this PR")

    @pytest.mark.asyncio
    async def test_github_close_pr_reopen(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr reopens PR."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="reopen",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert "reopened" in result["message"].lower()
        mock_pr.edit.assert_called_once_with(state="open")

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr merges PR successfully."""
        tool = GitHubClosePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.sha = "abc123"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"
        mock_pr.mergeable = True
        mock_pr.merge.return_value = mock_merge_result

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert "merged" in result["message"].lower()
        assert result["merge_method"] == "merge"
        assert result["sha"] == "abc123"

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_squash(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr merges with squash."""
        tool = GitHubClosePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.sha = "abc123"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"
        mock_pr.mergeable = True
        mock_pr.merge.return_value = mock_merge_result

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    merge_method="squash",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["merge_method"] == "squash"

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_not_mergeable(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles unmergeable PR."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.mergeable = False
        mock_pr.mergeable_state = "dirty"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    confirmed=True,
                )

        assert "error" in result
        assert "cannot be merged" in result["error"].lower()
        assert result["mergeable_state"] == "dirty"

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_with_delete_branch(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr merges and deletes branch."""
        tool = GitHubClosePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.sha = "abc123"

        mock_ref = MagicMock()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"
        mock_pr.mergeable = True
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature-branch"
        mock_pr.head.repo.full_name = "owner/repo"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_git_ref.return_value = mock_ref

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    delete_branch=True,
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["branch_deleted"] == "feature-branch"
        mock_ref.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_close_pr_unknown_action(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles unknown action."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="unknown",
                    confirmed=True,
                )

        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_pr_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles PR not found."""
        tool = GitHubClosePRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=999,
                    action="close",
                    confirmed=True,
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_not_allowed(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles merge not allowed."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()
        mock_pr.mergeable = True
        mock_pr.merge.side_effect = GithubException(405, {"message": "Not Allowed"}, None)

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    confirmed=True,
                )

        assert "error" in result
        assert "not allowed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_conflict(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles merge conflict."""
        tool = GitHubClosePRTool()

        mock_pr = MagicMock()
        mock_pr.mergeable = True
        mock_pr.merge.side_effect = GithubException(409, {"message": "Conflict"}, None)

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    confirmed=True,
                )

        assert "error" in result
        assert "conflict" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_close_pr_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles generic exception."""
        tool = GitHubClosePRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="close",
                    confirmed=True,
                )

        assert "error" in result
        assert "Failed to close" in result["error"]

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_delete_branch_from_fork(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr doesn't delete branch when PR is from fork (branch 170->177)."""
        tool = GitHubClosePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.sha = "abc123"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"
        mock_pr.mergeable = True
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature-branch"
        mock_pr.head.repo.full_name = "forker/repo"  # Different repo (fork)

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    delete_branch=True,
                    confirmed=True,
                )

        assert result["status"] == "success"
        # Branch should NOT be deleted since it's from a fork
        assert "branch_deleted" not in result
        mock_gh_repo.get_git_ref.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_close_pr_merge_delete_branch_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles branch deletion error (lines 174-175)."""
        tool = GitHubClosePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.sha = "abc123"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.html_url = "url"
        mock_pr.mergeable = True
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature-branch"
        mock_pr.head.repo.full_name = "owner/repo"  # Same repo

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_git_ref.side_effect = GithubException(422, {"message": "Reference already gone"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="merge",
                    delete_branch=True,
                    confirmed=True,
                )

        assert result["status"] == "success"
        # Merge succeeded but branch deletion failed
        assert "branch_delete_error" in result
        assert "branch_deleted" not in result

    @pytest.mark.asyncio
    async def test_github_close_pr_other_github_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_close_pr handles other GitHub API errors (lines 207-208)."""
        tool = GitHubClosePRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.side_effect = GithubException(500, {"message": "Internal Server Error"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_close_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_close_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    action="close",
                    confirmed=True,
                )

        assert "error" in result
        assert "GitHub API error" in result["error"]
        assert "Internal Server Error" in result["error"]
