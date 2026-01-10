"""Unit tests for the execute_code tool."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.execute_code import (
    CODE_OUTPUT_DIR,
    ALLOWED_EXTENSIONS,
    MAX_EXECUTION_TIME,
    ExecuteCodeTool,
)


class TestExecuteCodeToolAttributes:
    """Tests for ExecuteCodeTool tool attributes."""

    def test_execute_code_has_correct_name(self) -> None:
        """Test ExecuteCodeTool tool has correct name."""
        tool = ExecuteCodeTool()
        assert tool.name == "execute_code"

    def test_execute_code_has_description(self) -> None:
        """Test ExecuteCodeTool tool has description."""
        tool = ExecuteCodeTool()
        assert "execute" in tool.description.lower()
        assert "code" in tool.description.lower()

    def test_execute_code_has_parameters_schema(self) -> None:
        """Test ExecuteCodeTool tool has correct parameters schema."""
        tool = ExecuteCodeTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "filepath" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert schema["properties"]["filepath"]["type"] == "string"
        assert schema["properties"]["confirmed"]["type"] == "boolean"
        assert "filepath" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_execute_code_spec(self) -> None:
        """Test ExecuteCodeTool tool spec generation."""
        tool = ExecuteCodeTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "execute_code"


class TestExecuteCodeToolExecution:
    """Tests for ExecuteCodeTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_execute_code_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test execute_code returns error when not confirmed."""
        tool = ExecuteCodeTool()

        result = await tool(mock_deps, filepath="/some/path.py", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_execute_code_no_filepath(self, mock_deps: ToolDependencies) -> None:
        """Test execute_code returns error when no filepath."""
        tool = ExecuteCodeTool()

        result = await tool(mock_deps, filepath="", confirmed=True)

        assert "error" in result
        assert "No filepath" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_code_outside_code_dir(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code returns error for files outside CODE_OUTPUT_DIR."""
        tool = ExecuteCodeTool()

        # File outside CODE_OUTPUT_DIR
        outside_file = tmp_path / "outside.py"
        outside_file.write_text("print('hello')")

        result = await tool(mock_deps, filepath=str(outside_file), confirmed=True)

        assert "error" in result
        assert "Security error" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_code_file_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code returns error when file not found."""
        tool = ExecuteCodeTool()

        nonexistent = tmp_path / "reachy_code" / "nonexistent.py"

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", tmp_path / "reachy_code"):
            # Create the directory so path validation passes
            (tmp_path / "reachy_code").mkdir(exist_ok=True)
            result = await tool(mock_deps, filepath=str(nonexistent), confirmed=True)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_code_disallowed_extension(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code returns error for disallowed extension."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        bad_file = code_dir / "script.exe"
        bad_file.write_text("dangerous content")

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            result = await tool(mock_deps, filepath=str(bad_file), confirmed=True)

        assert "error" in result
        assert "Cannot execute" in result["error"]
        assert ".exe" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_code_python_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code successfully executes Python file."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "test.py"
        py_file.write_text("print('hello world')")

        mock_result = MagicMock()
        mock_result.stdout = "hello world\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert result["status"] == "success"
        assert result["return_code"] == 0
        assert "hello world" in result["output"]
        mock_run.assert_called_once()
        # Verify python3 command was used
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "python3"

    @pytest.mark.asyncio
    async def test_execute_code_shell_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code successfully executes shell file."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        sh_file = code_dir / "test.sh"
        sh_file.write_text("echo 'hello'")

        mock_result = MagicMock()
        mock_result.stdout = "hello\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = await tool(mock_deps, filepath=str(sh_file), confirmed=True)

        assert result["status"] == "success"
        # Verify bash command was used
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "bash"

    @pytest.mark.asyncio
    async def test_execute_code_error_return_code(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code handles non-zero return code."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "failing.py"
        py_file.write_text("import sys; sys.exit(1)")

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_result.returncode = 1

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert result["status"] == "error"
        assert result["return_code"] == 1
        assert "Error occurred" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_code_timeout(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code handles timeout."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "slow.py"
        py_file.write_text("import time; time.sleep(100)")

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python3", timeout=30)):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert "error" in result
        assert "timed out" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_execute_code_permission_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code handles permission error."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "restricted.py"
        py_file.write_text("print('hello')")

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", side_effect=PermissionError("Permission denied")):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert "error" in result
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_code_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code handles generic exceptions."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "crash.py"
        py_file.write_text("print('hello')")

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", side_effect=RuntimeError("Unexpected error")):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert "error" in result
        assert "Execution failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_code_truncates_long_output(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code truncates long output."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "verbose.py"
        py_file.write_text("print('x' * 5000)")

        mock_result = MagicMock()
        mock_result.stdout = "x" * 5000
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert result["status"] == "success"
        assert "truncated" in result["output"]
        assert len(result["output"]) < 5000

    @pytest.mark.asyncio
    async def test_execute_code_truncates_long_stderr(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code truncates long stderr."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "errors.py"
        py_file.write_text("print('hi')")

        mock_result = MagicMock()
        mock_result.stdout = "hi"
        mock_result.stderr = "e" * 5000
        mock_result.returncode = 1

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert "truncated" in result["stderr"]
        assert len(result["stderr"]) < 5000

    @pytest.mark.asyncio
    async def test_execute_code_no_output(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code handles no output."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "silent.py"
        py_file.write_text("x = 1")

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert result["status"] == "success"
        assert "(no output)" in result["output"]
        assert result["stderr"] is None

    @pytest.mark.asyncio
    async def test_execute_code_error_no_stderr(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test execute_code handles error with no stderr."""
        tool = ExecuteCodeTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir(exist_ok=True)
        py_file = code_dir / "fail_silent.py"
        py_file.write_text("import sys; sys.exit(1)")

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("reachy_mini_conversation_app.tools.execute_code.CODE_OUTPUT_DIR", code_dir):
            with patch("subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, filepath=str(py_file), confirmed=True)

        assert result["status"] == "error"
        assert "(no error message)" in result["stderr"]


class TestExecuteCodeConstants:
    """Tests for ExecuteCodeTool constants."""

    def test_allowed_extensions(self) -> None:
        """Test allowed extensions are correct."""
        assert ".py" in ALLOWED_EXTENSIONS
        assert ".sh" in ALLOWED_EXTENSIONS
        assert len(ALLOWED_EXTENSIONS) == 2

    def test_max_execution_time(self) -> None:
        """Test max execution time is reasonable."""
        assert MAX_EXECUTION_TIME == 30

    def test_code_output_dir(self) -> None:
        """Test CODE_OUTPUT_DIR is in home directory."""
        assert "reachy_code" in str(CODE_OUTPUT_DIR)
        assert str(Path.home()) in str(CODE_OUTPUT_DIR)
