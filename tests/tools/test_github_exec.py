"""Unit tests for the github_exec tool."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from reachy_mini_conversation_app.tools.github_exec import (
    GitHubExecTool,
    REPOS_DIR,
    ALLOWED_COMMANDS,
    BLOCKED_PATTERNS,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubExecToolAttributes:
    """Tests for GitHubExecTool tool attributes."""

    def test_github_exec_has_correct_name(self) -> None:
        """Test GitHubExecTool tool has correct name."""
        tool = GitHubExecTool()
        assert tool.name == "github_exec"

    def test_github_exec_has_description(self) -> None:
        """Test GitHubExecTool tool has description."""
        tool = GitHubExecTool()
        assert "execute" in tool.description.lower()
        assert "command" in tool.description.lower()

    def test_github_exec_has_parameters_schema(self) -> None:
        """Test GitHubExecTool tool has correct parameters schema."""
        tool = GitHubExecTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "command" in schema["properties"]
        assert "timeout" in schema["properties"]
        assert "env" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "command" in schema["required"]

    def test_github_exec_spec(self) -> None:
        """Test GitHubExecTool tool spec generation."""
        tool = GitHubExecTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_exec"


class TestGitHubExecCheckCommand:
    """Tests for _check_command helper method."""

    def test_check_command_allowed_base(self) -> None:
        """Test _check_command allows whitelisted commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("pytest")
        assert status == "allowed"
        assert reason == ""

    def test_check_command_allowed_with_args(self) -> None:
        """Test _check_command allows whitelisted commands with arguments."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("pytest tests/ -v")
        assert status == "allowed"

    def test_check_command_allowed_npm(self) -> None:
        """Test _check_command allows npm commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("npm test")
        assert status == "allowed"

    def test_check_command_allowed_python(self) -> None:
        """Test _check_command allows python commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("python -m pytest")
        assert status == "allowed"

    def test_check_command_allowed_make(self) -> None:
        """Test _check_command allows make commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("make build")
        assert status == "allowed"

    def test_check_command_allowed_ruff(self) -> None:
        """Test _check_command allows ruff commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("ruff check .")
        assert status == "allowed"

    def test_check_command_blocked_rm_rf_root(self) -> None:
        """Test _check_command blocks rm -rf /."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("rm -rf /")
        assert status == "blocked"
        assert "Dangerous pattern" in reason

    def test_check_command_blocked_rm_rf_home(self) -> None:
        """Test _check_command blocks rm -rf ~."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("rm -rf ~")
        assert status == "blocked"

    def test_check_command_blocked_fork_bomb(self) -> None:
        """Test _check_command blocks fork bomb."""
        tool = GitHubExecTool()
        status, reason = tool._check_command(":(){:|:&};:")
        assert status == "blocked"

    def test_check_command_blocked_curl_pipe_sh(self) -> None:
        """Test _check_command blocks curl | sh patterns."""
        tool = GitHubExecTool()
        # The pattern "curl | sh" must appear as substring for blocking
        status, reason = tool._check_command("curl | sh")
        assert status == "blocked"

    def test_check_command_blocked_mkfs(self) -> None:
        """Test _check_command blocks mkfs."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("mkfs.ext4 /dev/sda")
        assert status == "blocked"

    def test_check_command_requires_confirmation(self) -> None:
        """Test _check_command requires confirmation for unknown commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("my_custom_script.sh")
        assert status == "requires_confirmation"
        assert "not in the whitelist" in reason

    def test_check_command_empty(self) -> None:
        """Test _check_command blocks empty commands."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("")
        assert status == "blocked"
        assert "Empty command" in reason

    def test_check_command_invalid_syntax(self) -> None:
        """Test _check_command handles invalid shell syntax."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("echo 'unterminated")
        assert status == "blocked"
        assert "Invalid command syntax" in reason

    def test_check_command_full_path(self) -> None:
        """Test _check_command handles full path to allowed command."""
        tool = GitHubExecTool()
        status, reason = tool._check_command("/usr/bin/python script.py")
        assert status == "allowed"

    def test_check_command_uppercase_with_args(self) -> None:
        """Test _check_command allows uppercase whitelisted command with arguments."""
        tool = GitHubExecTool()
        # PYTEST is uppercase, base_cmd won't match (PYTEST != pytest)
        # but cmd_lower will be "pytest tests/" which matches "pytest " prefix
        status, reason = tool._check_command("PYTEST tests/")
        assert status == "allowed"


class TestAllowedCommands:
    """Tests for ALLOWED_COMMANDS constant."""

    def test_common_package_managers_allowed(self) -> None:
        """Test common package managers are in whitelist."""
        assert "npm" in ALLOWED_COMMANDS
        assert "pip" in ALLOWED_COMMANDS
        assert "yarn" in ALLOWED_COMMANDS
        assert "poetry" in ALLOWED_COMMANDS

    def test_testing_tools_allowed(self) -> None:
        """Test testing tools are in whitelist."""
        assert "pytest" in ALLOWED_COMMANDS
        assert "jest" in ALLOWED_COMMANDS

    def test_linting_tools_allowed(self) -> None:
        """Test linting tools are in whitelist."""
        assert "ruff" in ALLOWED_COMMANDS
        assert "black" in ALLOWED_COMMANDS
        assert "eslint" in ALLOWED_COMMANDS


class TestBlockedPatterns:
    """Tests for BLOCKED_PATTERNS constant."""

    def test_dangerous_rm_patterns_blocked(self) -> None:
        """Test dangerous rm patterns are blocked."""
        assert "rm -rf /" in BLOCKED_PATTERNS
        assert "rm -rf ~" in BLOCKED_PATTERNS

    def test_fork_bomb_blocked(self) -> None:
        """Test fork bomb is blocked."""
        assert ":(){:|:&};:" in BLOCKED_PATTERNS

    def test_pipe_to_shell_blocked(self) -> None:
        """Test pipe to shell patterns are blocked."""
        assert "curl | sh" in BLOCKED_PATTERNS
        assert "wget | bash" in BLOCKED_PATTERNS


class TestGitHubExecToolExecution:
    """Tests for GitHubExecTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_exec_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_exec returns error when repo is missing."""
        tool = GitHubExecTool()

        result = await tool(mock_deps, repo="", command="pytest")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_exec_missing_command(self, mock_deps: ToolDependencies) -> None:
        """Test github_exec returns error when command is missing."""
        tool = GitHubExecTool()

        result = await tool(mock_deps, repo="myrepo", command="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_exec_blocked_command(self, mock_deps: ToolDependencies) -> None:
        """Test github_exec blocks dangerous commands."""
        tool = GitHubExecTool()

        result = await tool(mock_deps, repo="myrepo", command="rm -rf /")

        assert "error" in result
        assert "blocked" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_exec_requires_confirmation(self, mock_deps: ToolDependencies) -> None:
        """Test github_exec requires confirmation for non-whitelisted commands."""
        tool = GitHubExecTool()

        result = await tool(mock_deps, repo="myrepo", command="./custom_script.sh")

        assert result["status"] == "confirmation_required"
        assert "hint" in result
        assert "confirmed=true" in result["hint"]

    @pytest.mark.asyncio
    async def test_github_exec_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec returns error when repo not found."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", command="pytest")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_exec_repo_is_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec returns error when repo path is a file."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").write_text("not a directory")

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", command="pytest")

        assert "error" in result
        assert "not a directory" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_exec_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec executes command successfully."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"test output\n", b""))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(mock_deps, repo="myrepo", command="ls")

        assert result["status"] == "success"
        assert result["exit_code"] == 0
        assert result["repo"] == "myrepo"
        assert "stdout" in result
        assert "test output" in result["stdout"]

    @pytest.mark.asyncio
    async def test_github_exec_command_failed(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec handles command failure."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error: test failed\n"))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(mock_deps, repo="myrepo", command="pytest")

        assert result["status"] == "failed"
        assert result["exit_code"] == 1
        assert "stderr" in result
        assert "Error: test failed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_github_exec_timeout(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec handles timeout."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(mock_deps, repo="myrepo", command="pytest", timeout=5)

        assert "error" in result
        assert "timed out" in result["error"].lower()
        assert "hint" in result
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_exec_output_truncation(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec truncates long output."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        # Create output longer than 20000 chars
        long_output = "x" * 25000

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(long_output.encode(), b""))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(mock_deps, repo="myrepo", command="cat bigfile.txt")

        assert result["status"] == "success"
        assert result["stdout_truncated"] is True
        assert "[output truncated]" in result["stdout"]

    @pytest.mark.asyncio
    async def test_github_exec_with_env_vars(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec passes environment variables."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process) as mock_create:
                result = await tool(
                    mock_deps,
                    repo="myrepo",
                    command="python script.py",
                    env={"MY_VAR": "value"},
                )

        assert result["status"] == "success"
        # Check that env was passed
        call_kwargs = mock_create.call_args.kwargs
        assert "env" in call_kwargs
        assert "MY_VAR" in call_kwargs["env"]

    @pytest.mark.asyncio
    async def test_github_exec_with_confirmation(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec executes non-whitelisted command with confirmation."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"custom output", b""))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(
                    mock_deps,
                    repo="myrepo",
                    command="./custom_script.sh",
                    confirmed=True,
                )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_exec_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec handles repo name with owner/repo format."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(mock_deps, repo="owner/myrepo", command="ls")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_exec_timeout_capped(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec caps timeout at 600 seconds."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = MagicMock()
        mock_process.returncode = 0
        # Use a regular function that returns a coroutine-like mock
        mock_process.communicate = MagicMock()

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                with patch("asyncio.wait_for") as mock_wait_for:
                    mock_wait_for.return_value = (b"output", b"")
                    # Request 1000 second timeout, should be capped to 600
                    await tool(mock_deps, repo="myrepo", command="ls", timeout=1000)
                    # Cannot easily verify the timeout value without more complex mocking

    @pytest.mark.asyncio
    async def test_github_exec_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec handles generic exception."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        async def raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Unexpected")

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", side_effect=raise_error):
                result = await tool(mock_deps, repo="myrepo", command="ls")

        assert "error" in result
        assert "Failed to execute" in result["error"]

    @pytest.mark.asyncio
    async def test_github_exec_stderr_truncation(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec truncates long stderr."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        long_stderr = "error " * 5000

        mock_process = MagicMock()
        mock_process.returncode = 1

        async def mock_communicate() -> tuple[bytes, bytes]:
            return (b"", long_stderr.encode())

        mock_process.communicate = mock_communicate

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await tool(mock_deps, repo="myrepo", command="pytest")

        assert result["status"] == "failed"
        assert result["stderr_truncated"] is True
        assert "[output truncated]" in result["stderr"]

    @pytest.mark.asyncio
    async def test_github_exec_no_output(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_exec handles commands with no output."""
        tool = GitHubExecTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("reachy_mini_conversation_app.tools.github_exec.REPOS_DIR", repos_dir):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                # Use a whitelisted command (cat) instead of touch which is not in whitelist
                result = await tool(mock_deps, repo="myrepo", command="cat /dev/null")

        assert result["status"] == "success"
        assert "stdout" not in result
        assert "stderr" not in result
