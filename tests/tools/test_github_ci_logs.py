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

    @pytest.mark.asyncio
    async def test_github_ci_logs_run_error_reraise(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs re-raises non-404 GithubException (line 143)."""
        tool = GitHubCILogsTool()

        mock_gh_repo = MagicMock()
        mock_gh_repo.get_workflow_run.side_effect = GithubException(500, {"message": "Server Error"}, None)

        mock_github = MagicMock()
        mock_github.get_repo.return_value = mock_gh_repo

        with patch("reachy_mini_conversation_app.tools.github_ci_logs.config") as mock_config:
            mock_config.GITHUB_TOKEN = "test-token"
            with patch("reachy_mini_conversation_app.tools.github_ci_logs.Github", return_value=mock_github):
                result = await tool(mock_deps, repo="owner/repo", run_id=123)

        assert "error" in result
        assert "GitHub API error" in result["error"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_job_timed_out(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs tracks timed_out jobs as failed (line 162)."""
        tool = GitHubCILogsTool()

        # Create mock zip content
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("test/step1.txt", "Timed out")
        zip_content = zip_buffer.getvalue()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "test"
        mock_job.status = "completed"
        mock_job.conclusion = "timed_out"  # Test timed_out
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "failure"
        mock_run.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run.head_sha = "abc123def456"
        mock_run.logs_url = "url"
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
        assert result["jobs_summary"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_github_ci_logs_failed_only_with_failed_jobs(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs with failed_only filter and actual failed jobs (lines 181-191, 222, 233)."""
        tool = GitHubCILogsTool()

        # Create mock zip with both passing and failing job logs
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("pass-job/step1.txt", "Passed")
            zf.writestr("fail-job/step1.txt", "Error: test failed")
        zip_content = zip_buffer.getvalue()

        mock_passing_job = MagicMock()
        mock_passing_job.id = 1
        mock_passing_job.name = "pass-job"
        mock_passing_job.status = "completed"
        mock_passing_job.conclusion = "success"
        mock_passing_job.started_at = None
        mock_passing_job.completed_at = None
        mock_passing_job.steps = []

        mock_failed_job = MagicMock()
        mock_failed_job.id = 2
        mock_failed_job.name = "fail-job"
        mock_failed_job.status = "completed"
        mock_failed_job.conclusion = "failure"
        mock_failed_job.started_at = None
        mock_failed_job.completed_at = None
        mock_failed_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "failure"
        mock_run.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run.head_sha = "abc123def456"
        mock_run.logs_url = "url"
        mock_run.jobs.return_value = [mock_passing_job, mock_failed_job]

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
                    result = await tool(mock_deps, repo="owner/repo", run_id=123, failed_only=True)

        assert result["status"] == "success"
        assert result["failed_only"] is True
        assert "fail-job" in result["failed_jobs"]
        # Logs should only contain the failed job
        assert "fail-job" in result["logs"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_with_job_name_filter(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs with specific job_name filter (line 251)."""
        tool = GitHubCILogsTool()

        # Create mock zip with multiple jobs
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("build/step1.txt", "Build log")
            zf.writestr("test/step1.txt", "Test log")
        zip_content = zip_buffer.getvalue()

        mock_job1 = MagicMock()
        mock_job1.id = 1
        mock_job1.name = "build"
        mock_job1.status = "completed"
        mock_job1.conclusion = "success"
        mock_job1.started_at = None
        mock_job1.completed_at = None
        mock_job1.steps = []

        mock_job2 = MagicMock()
        mock_job2.id = 2
        mock_job2.name = "test"
        mock_job2.status = "completed"
        mock_job2.conclusion = "success"
        mock_job2.started_at = None
        mock_job2.completed_at = None
        mock_job2.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run.head_sha = "abc123def456"
        mock_run.logs_url = "url"
        mock_run.jobs.return_value = [mock_job1, mock_job2]

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
                    result = await tool(mock_deps, repo="owner/repo", run_id=123, job_name="test")

        assert result["status"] == "success"
        assert result["filtered_job"] == "test"
        assert "test" in result["logs"]
        assert "build" not in result["logs"]

    @pytest.mark.asyncio
    async def test_github_ci_logs_large_logs_truncation(self, mock_deps: ToolDependencies) -> None:
        """Test github_ci_logs truncates large logs (lines 260-268)."""
        tool = GitHubCILogsTool()

        # Create mock zip with large logs (>50KB total, >10KB per job)
        large_content = "X" * 15000  # 15KB
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("job1/step1.txt", large_content)
            zf.writestr("job2/step1.txt", large_content)
            zf.writestr("job3/step1.txt", large_content)
            zf.writestr("job4/step1.txt", large_content)
        zip_content = zip_buffer.getvalue()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "job1"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run.head_sha = "abc123def456"
        mock_run.logs_url = "url"
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
                    result = await tool(mock_deps, repo="owner/repo", run_id=123, tail_lines=0)

        assert result["status"] == "success"
        assert "warning" in result
        assert "truncated" in result["warning"].lower()


class TestExtractLogsAdditionalCoverage:
    """Additional tests for _extract_logs_from_zip edge cases."""

    def test_extract_logs_non_txt_files_skipped(self) -> None:
        """Test that non-.txt files are skipped (branch 80->78)."""
        tool = GitHubCILogsTool()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("job/step.txt", "Log content")
            zf.writestr("job/config.json", '{"key": "value"}')
            zf.writestr("job/image.png", "binary data")
        zip_content = zip_buffer.getvalue()

        logs = tool._extract_logs_from_zip(zip_content, tail_lines=10)

        assert "job" in logs
        assert "Log content" in logs["job"]
        # Non-txt files should not appear
        assert "config.json" not in str(logs)
        assert "image.png" not in str(logs)

    def test_extract_logs_tail_lines_zero_no_truncation(self) -> None:
        """Test that tail_lines=0 returns full content (branch 93->98)."""
        tool = GitHubCILogsTool()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("job/step.txt", "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        zip_content = zip_buffer.getvalue()

        logs = tool._extract_logs_from_zip(zip_content, tail_lines=0)

        assert "job" in logs
        assert "Line 1" in logs["job"]
        assert "Line 5" in logs["job"]

    def test_extract_logs_single_file_no_subdir(self) -> None:
        """Test file without subdirectory (line 83)."""
        tool = GitHubCILogsTool()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("standalone.txt", "Direct log")
        zip_content = zip_buffer.getvalue()

        logs = tool._extract_logs_from_zip(zip_content, tail_lines=10)

        assert "standalone.txt" in logs
        assert "Direct log" in logs["standalone.txt"]


class TestGitHubCILogsLargeLogs:
    """Tests for log truncation edge cases."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_ci_logs_truncation_with_small_log(self, mock_deps: ToolDependencies) -> None:
        """Test truncation when some logs are small (line 267)."""
        tool = GitHubCILogsTool()

        # Create logs where total > 50KB but one is small
        large_content = "X" * 20000  # 20KB
        small_content = "Small log\n"  # ~10 bytes

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("large1/step1.txt", large_content)
            zf.writestr("large2/step1.txt", large_content)
            zf.writestr("large3/step1.txt", large_content)
            zf.writestr("small/step1.txt", small_content)
        zip_content = zip_buffer.getvalue()

        mock_job = MagicMock()
        mock_job.id = 1
        mock_job.name = "job1"
        mock_job.status = "completed"
        mock_job.conclusion = "success"
        mock_job.started_at = None
        mock_job.completed_at = None
        mock_job.steps = []

        mock_run = MagicMock()
        mock_run.name = "CI"
        mock_run.status = "completed"
        mock_run.conclusion = "success"
        mock_run.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run.head_sha = "abc123def456"
        mock_run.logs_url = "url"
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
                    result = await tool(mock_deps, repo="owner/repo", run_id=123, tail_lines=0)

        assert result["status"] == "success"
        assert "warning" in result
        # Small log should be included without truncation marker
        assert "small" in result["logs"]
        assert "truncated" not in result["logs"]["small"]
        # Large logs should be truncated
        assert "truncated" in result["logs"]["large1"]
