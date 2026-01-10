"""Unit tests for the github_update_pr tool."""

from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_update_pr import GitHubUpdatePRTool


class TestGitHubUpdatePRToolAttributes:
    """Tests for GitHubUpdatePRTool tool attributes."""

    def test_github_update_pr_has_correct_name(self) -> None:
        """Test GitHubUpdatePRTool tool has correct name."""
        tool = GitHubUpdatePRTool()
        assert tool.name == "github_update_pr"

    def test_github_update_pr_has_description(self) -> None:
        """Test GitHubUpdatePRTool tool has description."""
        tool = GitHubUpdatePRTool()
        assert "update" in tool.description.lower()
        assert "pull request" in tool.description.lower()

    def test_github_update_pr_has_parameters_schema(self) -> None:
        """Test GitHubUpdatePRTool tool has correct parameters schema."""
        tool = GitHubUpdatePRTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "pr_number" in schema["properties"]
        assert "title" in schema["properties"]
        assert "body" in schema["properties"]
        assert "labels" in schema["properties"]
        assert "assignees" in schema["properties"]
        assert "reviewers" in schema["properties"]
        assert "milestone" in schema["properties"]
        assert "draft" in schema["properties"]
        assert "repo" in schema["required"]
        assert "pr_number" in schema["required"]

    def test_github_update_pr_spec(self) -> None:
        """Test GitHubUpdatePRTool tool spec generation."""
        tool = GitHubUpdatePRTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_update_pr"


class TestGitHubUpdatePRGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubUpdatePRTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubUpdatePRTool()
        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubUpdatePRTool()
        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubUpdatePRToolExecution:
    """Tests for GitHubUpdatePRTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_update_pr_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr returns error when repo is missing."""
        tool = GitHubUpdatePRTool()

        result = await tool(mock_deps, repo="", pr_number=1)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_update_pr_missing_pr_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr returns error when pr_number is missing."""
        tool = GitHubUpdatePRTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=None)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_update_pr_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr returns error when no token."""
        tool = GitHubUpdatePRTool()

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_pr_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr returns error for invalid repo format."""
        tool = GitHubUpdatePRTool()

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", pr_number=1)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_pr_no_changes(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr with no updates provided."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Original"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "no_changes"
        assert result["pr_number"] == 1

    @pytest.mark.asyncio
    async def test_github_update_pr_title(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr updates title."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "New Title"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, title="New Title")

        assert result["status"] == "success"
        assert "title" in result["updated_fields"]
        mock_pr.edit.assert_called_with(title="New Title")

    @pytest.mark.asyncio
    async def test_github_update_pr_body(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr updates body."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, body="New description")

        assert result["status"] == "success"
        assert "body" in result["updated_fields"]
        mock_pr.edit.assert_called_with(body="New description")

    @pytest.mark.asyncio
    async def test_github_update_pr_labels(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr updates labels."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, labels=["bug", "urgent"])

        assert result["status"] == "success"
        assert "labels" in result["updated_fields"]
        mock_pr.set_labels.assert_called_once_with("bug", "urgent")

    @pytest.mark.asyncio
    async def test_github_update_pr_assignees(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr updates assignees."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        mock_issue = MagicMock()
        mock_issue.assignees = [MagicMock()]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, assignees=["user1", "user2"])

        assert result["status"] == "success"
        assert "assignees" in result["updated_fields"]
        mock_issue.add_to_assignees.assert_called_once_with("user1", "user2")

    @pytest.mark.asyncio
    async def test_github_update_pr_reviewers(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr requests reviewers."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, reviewers=["reviewer1"])

        assert result["status"] == "success"
        assert "reviewers" in result["updated_fields"]
        mock_pr.create_review_request.assert_called_once_with(reviewers=["reviewer1"])

    @pytest.mark.asyncio
    async def test_github_update_pr_milestone(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr updates milestone."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        mock_issue = MagicMock()
        mock_milestone = MagicMock()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_issue.return_value = mock_issue
        mock_gh_repo.get_milestone.return_value = mock_milestone

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, milestone=5)

        assert result["status"] == "success"
        assert "milestone" in result["updated_fields"]
        mock_issue.edit.assert_called_once_with(milestone=mock_milestone)

    @pytest.mark.asyncio
    async def test_github_update_pr_remove_milestone(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr removes milestone with 0."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        mock_issue = MagicMock()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, milestone=0)

        assert result["status"] == "success"
        assert "milestone" in result["updated_fields"]
        mock_issue.edit.assert_called_once_with(milestone=None)

    @pytest.mark.asyncio
    async def test_github_update_pr_convert_to_draft_unsupported(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr converting to draft is unsupported."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.draft = False

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, draft=True)

        assert "error" in result
        assert "GraphQL" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_pr_ready_for_review_unsupported(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr marking ready for review is unsupported."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.draft = True

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, draft=False)

        assert "error" in result
        assert "GraphQL" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_pr_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr handles PR not found."""
        tool = GitHubUpdatePRTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=999, title="New")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_update_pr_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr handles API error."""
        tool = GitHubUpdatePRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, title="New")

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_pr_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr handles generic exception."""
        tool = GitHubUpdatePRTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, title="New")

        assert "error" in result
        assert "Failed to update" in result["error"]

    @pytest.mark.asyncio
    async def test_github_update_pr_empty_assignees_clears_all(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr with empty assignees list clears all assignees."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"

        # Existing assignees that should be removed
        existing_assignee = MagicMock()
        mock_issue = MagicMock()
        mock_issue.assignees = [existing_assignee]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_issue.return_value = mock_issue

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                # Pass empty list - should remove existing but not add any new
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, assignees=[])

        assert result["status"] == "success"
        assert "assignees" in result["updated_fields"]
        # Should remove existing assignee
        mock_issue.remove_from_assignees.assert_called_once_with(existing_assignee)
        # Should NOT call add_to_assignees since list is empty
        mock_issue.add_to_assignees.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_update_pr_draft_false_on_non_draft_pr(self, mock_deps: ToolDependencies) -> None:
        """Test github_update_pr with draft=False on a non-draft PR does nothing."""
        tool = GitHubUpdatePRTool()

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.html_url = "url"
        mock_pr.draft = False  # Already not a draft

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_update_pr.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_update_pr.Github", return_value=mock_github):
                # draft=False on non-draft PR - should be a no-op for draft handling
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, draft=False)

        # Since no other updates, should be no_changes
        assert result["status"] == "no_changes"
