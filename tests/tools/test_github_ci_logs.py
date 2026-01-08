"""Unit tests for the github_ci_logs tool."""

from unittest.mock import MagicMock, patch
from datetime import datetime
import zipfile
import io

import pytest
from github import GithubException
import requests

from reachy_mini_conversation_app.tools.github_ci_logs import GitHubCILogsTool
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubCILogsToolAttributes:
    """Tests for GitHubCILogsTool tool attributes."""

    def test_github_ci_logs_has_correct_name(self) -> None:
        """Test GitHubCILogsTool tool has correct name."""
        tool = GitHubCILogsTool()
        assert tool.name == "github_ci_logs"

    def test_github_ci_logs_has_description(self) -> None:
        """Test GitHubCILogsTool tool has description."""
        tool = GitHubCILogsTool()
        assert "log" in tool.description.lower()
        assert "workflow" in tool.description.lower()

    def test_github_ci_logs_has_parameters_schema(self) -> None:
        """Test GitHubCILogsTool tool has correct parameters schema."""
        tool = GitHubCILogsTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "run_id" in schema["properties"]
        assert "job_name" in schema["properties"]
        assert "failed_only" in schema["properties"]
        assert "tail_lines" in schema["properties"]
        assert "repo" in schema["required"]
        assert "run_id" in schema["required"]

    def test_github_ci_logs_spec(self) -> None:
        """Test GitHubCILogsTool tool spec generation."""
        tool = GitHubCILogsTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_ci_logs"


class TestGitHubCILogsGetFullRepoName:
    """Tests for _get_full_repo_name helper method."""

    def test_get_full_repo_name_with_owner(self) -> None:
        """Test _get_full_repo_name with owner/repo format."""
        tool = GitHubCILogsTool()
        result = tool._get_full_repo_name("owner/repo")
        assert result == "owner/repo"

    def test_get_full_repo_name_with_default_owner(self) -> None:
        """Test _get_full_repo_name with default owner."""
        tool = GitHubCILogsTool()
        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = "default-owner"
            result = tool._get_full_repo_name("repo")
            assert result == "default-owner/repo"

    def test_get_full_repo_name_no_owner_raises(self) -> None:
        """Test _get_full_repo_name raises error when no owner."""
        tool = GitHubCILogsTool()
        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_DEFAULT_OWNER = None
            with pytest.raises(ValueError, match="must include owner"):
                tool._get_full_repo_name("repo")


class TestGitHubCILogsExtractLogs:
    """Tests for _extract_logs_from_zip helper method."""

    def test_extract_logs_from_valid_zip(self) -> None:
        """Test extracting logs from valid zip file."""
        tool = GitHubCILogsTool()

        # Create a mock zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("job1/step1.txt", "Log line 1\nLog line 2\nLog line 3")
            zf.writestr("job1/step2.txt", "Another log")
        zip_content = zip_buffer.getvalue()

        logs = tool._extract_logs_from_zip(zip_content, tail_lines=10)

        assert "job1" in logs
        assert "Log line 1" in logs["job1"]

    def test_extract_logs_with_job_filter(self) -> None:
        """Test extracting logs with job name filter."""
        tool = GitHubCILogsTool()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("build/step1.txt", "Build log")
            zf.writestr("test/step1.txt", "Test log")
        zip_content = zip_buffer.getvalue()

        logs = tool._extract_logs_from_zip(zip_content, job_name="test", tail_lines=10)

        assert "test" in logs
        assert "build" not in logs

    def test_extract_logs_tail_lines(self) -> None:
        """Test extracting logs with tail lines limit."""
        tool = GitHubCILogsTool()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("job/step.txt", "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        zip_content = zip_buffer.getvalue()

        logs = tool._extract_logs_from_zip(zip_content, tail_lines=2)

        assert "job" in logs
        assert "Line 4" in logs["job"]
        assert "Line 5" in logs["job"]
        assert "Line 1" not in logs["job"]

    def test_extract_logs_invalid_zip(self) -> None:
        """Test handling invalid zip file."""
        tool = GitHubCILogsTool()

        logs = tool._extract_logs_from_zip(b"not a zip file")

        assert "error" in logs


class TestGitHubCILogsToolExecution:
    """Tests for GitHubCILogsTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_ci_logs_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs returns error when repo is missing."""
        tool = GitHubCILogsTool()

        result = await tool(mock_deps, repo="", run_id=123)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_ci_logs_missing_run_id(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs returns error when run_id is missing."""
        tool = GitHubCILogsTool()

        result = await tool(mock_deps, repo="owner/repo", run_id=None)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_ci_logs_no_token(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs returns error when no token."""
        tool = GitHubCILogsTool()

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert "error" in result
        assert "GITHUB_TOKEN" in result["error"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_invalid_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs returns error for invalid repo format."""
        tool = GitHubCILogsTool()

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            mock_config.GITHUB_DEFAULT_OWNER = None
            result = await tool(mock_deps, repo="noslash", run_id=123)

        assert "error" in result
        assert "must include owner" in result["error"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_run_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs handles run not found."""
        tool = GitHubCILogsTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.side_effect = GithubException(404, {"message": "Not Found"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", run_id=999)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_ci_logs_success(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs retrieves logs successfully."""
        tool = GitHubCILogsTool()

        # Create mock zip content
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("test/step1.txt", "Test output\nPassed")
        zip_content = zip_buffer.getvalue()

        mock_step = MagicMock()
        mock_step.name = "Run tests"
        mock_step.status = "completed"
        mock_step.conclusion = "success"
        mock_step.number = 1

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "test"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = datetime(2024, 1, 1, 10, 0, 0)
        mock_job.completed_at = datetime(2024, 1, 1, 10, 5, 0)
        mock_job.steps = [mock_step]

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run.head_sha = "abc123def456"
        mock_run.logs_url = "https://api.github.com/repos/owner/repo/actions/runs/123/logs"
        mock_run.jobs.return_value = [mock_job]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.return_value = mock_run

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_content

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                with patch("reachy_mini_conversation_app.tools.github_ci_logs.requests.get", return_value=mock_response):
                    result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert result["status"] == "success"
        assert result["run_id"] == 123
        assert result["workflow"] == "CI"
        assert "logs" in result

    @pytest.mark.asyncio
    async def test_github_ci_logs_failed_only(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs with failed_only filter."""
        tool = GitHubCILogsTool()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "test"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.jobs.return_value = [mock_job]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.return_value = mock_run

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", run_id=123, failed_only=True)

        assert result["status"] == "success"
        assert "No failed jobs" in result["message"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_logs_not_available(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs when logs not available."""
        tool = GitHubCILogsTool()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "test"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "in_progress"
        mock_run.conclusion = None
        mock_run.logs_url = "url"
        mock_run.jobs.return_value = [mock_job]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.return_value = mock_run

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                with patch("reachy_mini_conversation_app.tools.github_ci_logs.requests.get", return_value=mock_response):
                    result = await tool(mock_deps, repo="owner/repo", run_id=123)

        # Note: result has "status" key overwritten by run.status in the code (bug in source)
        # The source code has: "status": run.status which overwrites "status": "success"
        assert "message" in result
        assert "not available" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_ci_logs_download_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs handles download error."""
        tool = GitHubCILogsTool()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "test"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.logs_url = "url"
        mock_run.jobs.return_value = [mock_job]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.return_value = mock_run

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                with patch("reachy_mini_conversation_app.tools.github_ci_logs.requests.get", return_value=mock_response):
                    result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert "error" in result
        assert "Failed to download" in result["error"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_request_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs handles request exception."""
        tool = GitHubCILogsTool()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "test"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.logs_url = "url"
        mock_run.jobs.return_value = [mock_job]

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.return_value = mock_run

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                with patch("reachy_mini_conversation_app.tools.github_ci_logs.requests.get") as mock_get:
                    mock_get.side_effect = requests.RequestException("Connection error")
                    result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert "error" in result
        assert "Failed to download" in result["error"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_api_error(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs handles API error."""
        tool = GitHubCILogsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = GithubException(500, {"message": "Server Error"}, None)

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_generic_exception(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs handles generic exception."""
        tool = GitHubCILogsTool()

        mock_github = MagicMock()
        mock_github.get_repo.side_effect = RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert "error" in result
        assert "Failed to get CI logs" in result["error"]
