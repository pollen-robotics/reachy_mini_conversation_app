"""Unit tests for the github_pr_checks tool."""

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest
from github import GithubException

from reachy_mini_conversation_app.tools.github_pr_checks import GitHubPRChecksTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubPRChecksToolAttributes:
    """Tests for GitHubPRChecksTool tool attributes."""

    def test_github_pr_checks_has_correct_name(self) -> None:
        """Test GitHubPRChecksTool tool has correct name."""
        tool = GitHubPRChecksTool()
        assert tool.name == "github_pr_checks"

    def test_github_pr_checks_has_description(self) -> None:
        """Test GitHubPRChecksTool tool has description."""
        tool = GitHubPRChecksTool()
        assert "check" in tool.description.lower()
        assert "ci" in tool.description.lower()

    def test_github_pr_checks_has_parameters_schema(self) -> None:
        """Test GitHubPRChecksTool tool has correct parameters schema."""
        tool = GitHubPRChecksTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "pr_number" in schema["properties"]
        assert "filter_status" in schema["properties"]
        assert "include_logs" in schema["properties"]
        assert "repo" in schema["required"]
        assert "pr_number" in schema["required"]

    def test_github_pr_checks_spec(self) -> None:
        """Test GitHubPRChecksTool tool spec generation."""
        tool = GitHubPRChecksTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_pr_checks"


class TestGitHubPRChecksGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubPRChecksTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubPRChecksTool()
        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubPRChecksTool()
        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubPRChecksToolExecution:
    """Tests for GitHubPRChecksTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_pr_checks_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks returns error when repo is missing."""
        tool = GitHubPRChecksTool()

        result = await tool(mock_deps, repo="", pr_number=1)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_checks_missing_pr_number(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks returns error when pr_number is missing."""
        tool = GitHubPRChecksTool()

        result = await tool(mock_deps, repo="owner/repo", pr_number=None)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_checks_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks returns error when no token."""
        tool = GitHubPRChecksTool()

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_checks_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks returns error for invalid repo format."""
        tool = GitHubPRChecksTool()

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", pr_number=1)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_checks_success_all_passing(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with all passing checks."""
        tool = GitHubPRChecksTool()

        mock_check_run = MagicMock()
        mock_check_run.name = "test"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "success"
        mock_check_run.started_at = datetime(2024, 1, 1, 10, 0, 0)
        mock_check_run.completed_at = datetime(2024, 1, 1, 10, 5, 0)
        mock_check_run.details_url = "https://github.com/actions/run/123"
        mock_check_run.html_url = "https://github.com/owner/repo/actions/run/123"
        mock_check_run.output = None

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.head.sha = "abc123def456"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "passing"
        assert result["summary"]["total"] == 1
        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_github_pr_checks_with_failed_check(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with failed checks."""
        tool = GitHubPRChecksTool()

        mock_output = MagicMock()
        mock_output.title = "Test Failed"
        mock_output.summary = "Some tests failed"
        mock_output.text = "Error details"

        mock_check_run = MagicMock()
        mock_check_run.name = "test"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "failure"
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = "url"
        mock_check_run.html_url = "url"
        mock_check_run.output = mock_output

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "Test PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "failing"
        assert result["summary"]["failed"] == 1
        assert "failed_checks" in result

    @pytest.mark.asyncio
    async def test_github_pr_checks_with_pending(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with pending checks."""
        tool = GitHubPRChecksTool()

        mock_check_run = MagicMock()
        mock_check_run.name = "build"
        mock_check_run.status = "in_progress"
        mock_check_run.conclusion = None
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = None
        mock_check_run.html_url = None
        mock_check_run.output = None

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "pending"
        assert result["summary"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_github_pr_checks_filter_failed(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with failed filter."""
        tool = GitHubPRChecksTool()

        mock_failed_check = MagicMock()
        mock_failed_check.name = "test"
        mock_failed_check.status = "completed"
        mock_failed_check.conclusion = "failure"
        mock_failed_check.started_at = None
        mock_failed_check.completed_at = None
        mock_failed_check.details_url = None
        mock_failed_check.html_url = None
        mock_failed_check.output = None

        mock_success_check = MagicMock()
        mock_success_check.name = "build"
        mock_success_check.status = "completed"
        mock_success_check.conclusion = "success"
        mock_success_check.started_at = None
        mock_success_check.completed_at = None
        mock_success_check.details_url = None
        mock_success_check.html_url = None
        mock_success_check.output = None

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_failed_check, mock_success_check]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, filter_status="failed")

        assert result["status"] == "success"
        assert len(result["checks"]) == 1
        assert result["checks"][0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_github_pr_checks_with_external_status(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with external CI status."""
        tool = GitHubPRChecksTool()

        mock_external_status = MagicMock()
        mock_external_status.context = "jenkins/build"
        mock_external_status.state = "failure"
        mock_external_status.description = "Build failed"
        mock_external_status.target_url = "https://jenkins.example.com/job/123"

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = [mock_external_status]

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = []
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "failing"
        assert result["summary"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_github_pr_checks_no_checks(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with no checks."""
        tool = GitHubPRChecksTool()

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = []
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "no_checks"

    @pytest.mark.asyncio
    async def test_github_pr_checks_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks handles PR not found."""
        tool = GitHubPRChecksTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=999)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_pr_checks_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks handles API error."""
        tool = GitHubPRChecksTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_checks_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks handles generic exception."""
        tool = GitHubPRChecksTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert "error" in result
        assert "Failed to get PR checks" in result["error"]

    @pytest.mark.asyncio
    async def test_github_pr_checks_failed_with_partial_output(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with failed check having partial output (only title)."""
        tool = GitHubPRChecksTool()

        mock_output = MagicMock()
        mock_output.title = "Build Error"
        mock_output.summary = None
        mock_output.text = None

        mock_check_run = MagicMock()
        mock_check_run.name = "build"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "failure"
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = None
        mock_check_run.html_url = None
        mock_check_run.output = mock_output

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["summary"]["failed"] == 1
        # Check that only title is captured, not summary/text
        failed_check = result["failed_checks"][0]
        assert "error_title" in failed_check
        assert failed_check["error_title"] == "Build Error"
        assert "error_summary" not in failed_check
        assert "error_text" not in failed_check

    @pytest.mark.asyncio
    async def test_github_pr_checks_failed_with_only_summary(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with failed check having only summary."""
        tool = GitHubPRChecksTool()

        mock_output = MagicMock()
        mock_output.title = None
        mock_output.summary = "Tests failed: 5/10 passed"
        mock_output.text = None

        mock_check_run = MagicMock()
        mock_check_run.name = "tests"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "failure"
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = None
        mock_check_run.html_url = None
        mock_check_run.output = mock_output

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        failed_check = result["failed_checks"][0]
        assert "error_title" not in failed_check
        assert "error_summary" in failed_check
        assert "error_text" not in failed_check

    @pytest.mark.asyncio
    async def test_github_pr_checks_failed_with_only_text(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with failed check having only text."""
        tool = GitHubPRChecksTool()

        mock_output = MagicMock()
        mock_output.title = None
        mock_output.summary = None
        mock_output.text = "Full error log here"

        mock_check_run = MagicMock()
        mock_check_run.name = "lint"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "timed_out"
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = None
        mock_check_run.html_url = None
        mock_check_run.output = mock_output

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        failed_check = result["failed_checks"][0]
        assert "error_title" not in failed_check
        assert "error_summary" not in failed_check
        assert "error_text" in failed_check

    @pytest.mark.asyncio
    async def test_github_pr_checks_external_status_pending(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with external CI status in pending state."""
        tool = GitHubPRChecksTool()

        mock_external_status = MagicMock()
        mock_external_status.context = "jenkins/build"
        mock_external_status.state = "pending"
        mock_external_status.description = "Build in progress"
        mock_external_status.target_url = "https://jenkins.example.com/job/123"

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = [mock_external_status]

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = []
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "pending"
        assert result["summary"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_github_pr_checks_external_status_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with external CI status in success state."""
        tool = GitHubPRChecksTool()

        mock_external_status = MagicMock()
        mock_external_status.context = "circleci/build"
        mock_external_status.state = "success"
        mock_external_status.description = "Build passed"
        mock_external_status.target_url = None  # No URL to test include_logs branch

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = [mock_external_status]

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = []
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "passing"
        assert result["summary"]["successful"] == 1
        # No details_url since target_url is None
        assert "details_url" not in result["checks"][0]

    @pytest.mark.asyncio
    async def test_github_pr_checks_filter_pending(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with pending filter."""
        tool = GitHubPRChecksTool()

        mock_pending_check = MagicMock()
        mock_pending_check.name = "build"
        mock_pending_check.status = "in_progress"
        mock_pending_check.conclusion = None
        mock_pending_check.started_at = None
        mock_pending_check.completed_at = None
        mock_pending_check.details_url = None
        mock_pending_check.html_url = None
        mock_pending_check.output = None

        mock_success_check = MagicMock()
        mock_success_check.name = "lint"
        mock_success_check.status = "completed"
        mock_success_check.conclusion = "success"
        mock_success_check.started_at = None
        mock_success_check.completed_at = None
        mock_success_check.details_url = None
        mock_success_check.html_url = None
        mock_success_check.output = None

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_pending_check, mock_success_check]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, filter_status="pending")

        assert result["status"] == "success"
        assert len(result["checks"]) == 1
        assert result["checks"][0]["name"] == "build"

    @pytest.mark.asyncio
    async def test_github_pr_checks_filter_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with success filter."""
        tool = GitHubPRChecksTool()

        mock_pending_check = MagicMock()
        mock_pending_check.name = "build"
        mock_pending_check.status = "in_progress"
        mock_pending_check.conclusion = None
        mock_pending_check.started_at = None
        mock_pending_check.completed_at = None
        mock_pending_check.details_url = None
        mock_pending_check.html_url = None
        mock_pending_check.output = None

        mock_success_check = MagicMock()
        mock_success_check.name = "lint"
        mock_success_check.status = "completed"
        mock_success_check.conclusion = "success"
        mock_success_check.started_at = None
        mock_success_check.completed_at = None
        mock_success_check.details_url = None
        mock_success_check.html_url = None
        mock_success_check.output = None

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_pending_check, mock_success_check]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, filter_status="success")

        assert result["status"] == "success"
        assert len(result["checks"]) == 1
        assert result["checks"][0]["name"] == "lint"

    @pytest.mark.asyncio
    async def test_github_pr_checks_include_logs_false(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with include_logs=False."""
        tool = GitHubPRChecksTool()

        mock_check_run = MagicMock()
        mock_check_run.name = "test"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "success"
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = "https://github.com/actions/run/123"
        mock_check_run.html_url = "https://github.com/owner/repo/actions/run/123"
        mock_check_run.output = None

        mock_external_status = MagicMock()
        mock_external_status.context = "jenkins"
        mock_external_status.state = "success"
        mock_external_status.description = "Build passed"
        mock_external_status.target_url = "https://jenkins.example.com/job/123"

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = [mock_external_status]

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1, include_logs=False)

        assert result["status"] == "success"
        # details_url should not be present when include_logs=False
        for check in result["checks"]:
            assert "details_url" not in check

    @pytest.mark.asyncio
    async def test_github_pr_checks_action_required_conclusion(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with action_required conclusion (counts as failed)."""
        tool = GitHubPRChecksTool()

        mock_output = MagicMock()
        mock_output.title = "Action Required"
        mock_output.summary = "Manual intervention needed"
        mock_output.text = None

        mock_check_run = MagicMock()
        mock_check_run.name = "approve"
        mock_check_run.status = "completed"
        mock_check_run.conclusion = "action_required"
        mock_check_run.started_at = None
        mock_check_run.completed_at = None
        mock_check_run.details_url = None
        mock_check_run.html_url = None
        mock_check_run.output = mock_output

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = []

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = [mock_check_run]
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "failing"
        assert result["summary"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_github_pr_checks_external_error_state(self, mock_deps: ToolDependencies) -> None:
        """Test github_pr_checks with external CI status in error state."""
        tool = GitHubPRChecksTool()

        mock_external_status = MagicMock()
        mock_external_status.context = "aws-codebuild"
        mock_external_status.state = "error"
        mock_external_status.description = "Internal error occurred"
        mock_external_status.target_url = "https://aws.example.com/build/123"

        mock_combined_status = MagicMock()
        mock_combined_status.statuses = [mock_external_status]

        mock_commit = MagicMock()
        mock_commit.get_check_runs.return_value = []
        mock_commit.get_combined_status.return_value = mock_combined_status

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.title = "PR"
        mock_pr.head.sha = "abc123"

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_pull.return_value = mock_pr
        mock_gh_repo.get_commit.return_value = mock_commit

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_pr_checks.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_pr_checks.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", pr_number=1)

        assert result["status"] == "success"
        assert result["overall_status"] == "failing"
        assert result["summary"]["failed"] == 1
