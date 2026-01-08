"""Unit tests for the github_merge_pr tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_merge_pr import GitHubMergePRTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubMergePRToolAttributes:
    """Tests for GitHubMergePRTool tool attributes."""

    def test_github_merge_pr_has_correct_name(self) -> None:
        """Test GitHubMergePRTool tool has correct name."""
        tool = GitHubMergePRTool()
        assert tool.name == "github_merge_pr"

    def test_github_merge_pr_has_description(self) -> None:
        """Test GitHubMergePRTool tool has description."""
        tool = GitHubMergePRTool()
        assert "merge" in tool.description.lower()
        assert "pull request" in tool.description.lower()

    def test_github_merge_pr_has_parameters_schema(self) -> None:
        """Test GitHubMergePRTool tool has correct parameters schema."""
        tool = GitHubMergePRTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "pr_number" in schema["properties"]
        assert "merge_method" in schema["properties"]
        assert "commit_title" in schema["properties"]
        assert "commit_message" in schema["properties"]
        assert "delete_branch" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "pr_number" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_merge_pr_spec(self) -> None:
        """Test GitHubMergePRTool tool spec generation."""
        tool = GitHubMergePRTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_merge_pr"


class TestGitHubMergePRGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubMergePRTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubMergePRTool()
        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubMergePRTool()
        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubMergePRToolExecution:
    """Tests for GitHubMergePRTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_merge_pr_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr requires confirmation."""
        tool = GitHubMergePRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=1, confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_merge_pr_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr returns error when repo is missing."""
        tool = GitHubMergePRTool()

        result = await tool(mock_deps, repo="", pr_number=1, confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_missing_pr_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr returns error when pr_number is missing."""
        tool = GitHubMergePRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=None, confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr returns error when no token."""
        tool = GitHubMergePRTool()

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", pr_number=1, confirmed=True)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_merge_pr_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr returns error for invalid repo format."""
        tool = GitHubMergePRTool()

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", pr_number=1, confirmed=True)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_merge_pr_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr merges successfully."""
        tool = GitHubMergePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.merged = True
        mock_merge_result.sha = "abc123def"
        mock_merge_result.message = "Merged"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.default_branch = "main"

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert "merged" in result["message"].lower()
        assert result["pr_number"] == 1
        assert result["merge_method"] == "merge"
        assert result["sha"] == "abc123def"

    @pytest.mark.asyncio
    async def test_github_merge_pr_squash(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr with squash method."""
        tool = GitHubMergePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.merged = True
        mock_merge_result.sha = "abc123"
        mock_merge_result.message = "Merged"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.default_branch = "main"

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    merge_method="squash",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["merge_method"] == "squash"
        mock_pr.merge.assert_called_once_with(merge_method="squash")

    @pytest.mark.asyncio
    async def test_github_merge_pr_rebase(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr with rebase method."""
        tool = GitHubMergePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.merged = True
        mock_merge_result.sha = "abc123"
        mock_merge_result.message = "Merged"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.default_branch = "main"

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    merge_method="rebase",
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["merge_method"] == "rebase"

    @pytest.mark.asyncio
    async def test_github_merge_pr_with_custom_commit(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr with custom commit title and message."""
        tool = GitHubMergePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.merged = True
        mock_merge_result.sha = "abc123"
        mock_merge_result.message = "Merged"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.default_branch = "main"

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    commit_title="Custom Title",
                    commit_message="Custom message",
                    confirmed=True,
                )

        assert result["status"] == "success"
        mock_pr.merge.assert_called_once_with(
            merge_method="merge",
            commit_title="Custom Title",
            commit_message="Custom message",
        )

    @pytest.mark.asyncio
    async def test_github_merge_pr_delete_branch(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr with branch deletion."""
        tool = GitHubMergePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.merged = True
        mock_merge_result.sha = "abc123"
        mock_merge_result.message = "Merged"

        mock_ref = MagicMock()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.return_value = mock_merge_result
        mock_pr.head.ref = "feature-branch"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.default_branch = "main"
        mock_gh_repo.get_git_ref.return_value = mock_ref

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    delete_branch=True,
                    confirmed=True,
                )

        assert result["status"] == "success"
        assert result["branch_deleted"] == "feature-branch"
        mock_ref.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_merge_pr_pr_not_open(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles closed PR."""
        tool = GitHubMergePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.state = "closed"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert "error" in result
        assert "not open" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_has_conflicts(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles merge conflicts."""
        tool = GitHubMergePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.state = "open"
        mock_pr.mergeable = False
        mock_pr.mergeable_state = "dirty"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert "error" in result
        assert "conflicts" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_blocked(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles blocked PR."""
        tool = GitHubMergePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "blocked"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert "error" in result
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_merge_failed(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles merge failure."""
        tool = GitHubMergePRTool()

        mock_merge_result = MagicMock()
        mock_merge_result.merged = False
        mock_merge_result.message = "Merge failed"

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.return_value = mock_merge_result

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert "error" in result
        assert "Failed to merge" in result["error"]

    @pytest.mark.asyncio
    async def test_github_merge_pr_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles PR not found."""
        tool = GitHubMergePRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=999,
                    confirmed=True,
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_method_not_allowed(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles merge method not allowed."""
        tool = GitHubMergePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.side_effect = GithubException(405, {"message": "Not Allowed"}, None)

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    merge_method="rebase",
                    confirmed=True,
                )

        assert "error" in result
        assert "not allowed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_conflict_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles 409 conflict error."""
        tool = GitHubMergePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.state = "open"
        mock_pr.mergeable = True
        mock_pr.mergeable_state = "clean"
        mock_pr.merge.side_effect = GithubException(409, {"message": "Conflict"}, None)

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert "error" in result
        assert "conflict" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_merge_pr_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_merge_pr handles generic exception."""
        tool = GitHubMergePRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_merge_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_merge_pr.Github", return_value=mock_github):
                result = await tool(
                    mock_deps,
                    repo="owner/repo",
                    pr_number=1,
                    confirmed=True,
                )

        assert "error" in result
        assert "Failed to merge" in result["error"]
