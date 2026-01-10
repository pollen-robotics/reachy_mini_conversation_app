"""Unit tests for the code_move_to_repo tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.code_move_to_repo import (
    CodeMoveToRepoTool,
)


class TestCodeMoveToRepoToolAttributes:
    """Tests for CodeMoveToRepoTool tool attributes."""

    def test_code_move_to_repo_has_correct_name(self) -> None:
        """Test CodeMoveToRepoTool tool has correct name."""
        tool = CodeMoveToRepoTool()
        assert tool.name == "code_move_to_repo"

    def test_code_move_to_repo_has_description(self) -> None:
        """Test CodeMoveToRepoTool tool has description."""
        tool = CodeMoveToRepoTool()
        assert "move" in tool.description.lower()
        assert "repository" in tool.description.lower()

    def test_code_move_to_repo_has_parameters_schema(self) -> None:
        """Test CodeMoveToRepoTool tool has correct parameters schema."""
        tool = CodeMoveToRepoTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "filename" in schema["properties"]
        assert "repo" in schema["properties"]
        assert "dest_path" in schema["properties"]
        assert "overwrite" in schema["properties"]
        assert "filename" in schema["required"]
        assert "repo" in schema["required"]
        assert "dest_path" in schema["required"]

    def test_code_move_to_repo_spec(self) -> None:
        """Test CodeMoveToRepoTool tool spec generation."""
        tool = CodeMoveToRepoTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "code_move_to_repo"


class TestCodeMoveToRepoToolExecution:
    """Tests for CodeMoveToRepoTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_code_move_missing_filename(self, mock_deps: ToolDependencies) -> None:
        """Test code_move_to_repo returns error when filename is missing."""
        tool = CodeMoveToRepoTool()

        result = await tool(mock_deps, filename="", repo="myrepo", dest_path="src/file.py")

        assert "error" in result
        assert "Filename" in result["error"]

    @pytest.mark.asyncio
    async def test_code_move_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test code_move_to_repo returns error when repo is missing."""
        tool = CodeMoveToRepoTool()

        result = await tool(mock_deps, filename="test.py", repo="", dest_path="src/file.py")

        assert "error" in result
        assert "Repository" in result["error"]

    @pytest.mark.asyncio
    async def test_code_move_missing_dest_path(self, mock_deps: ToolDependencies) -> None:
        """Test code_move_to_repo returns error when dest_path is missing."""
        tool = CodeMoveToRepoTool()

        result = await tool(mock_deps, filename="test.py", repo="myrepo", dest_path="")

        assert "error" in result
        assert "Destination" in result["error"]

    @pytest.mark.asyncio
    async def test_code_move_file_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo returns error when file not found."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            result = await tool(mock_deps, filename="nonexistent.py", repo="myrepo", dest_path="src/file.py")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_code_move_file_not_found_with_hint(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo returns hint when similar files exist."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "20240101_fibonacci.py").write_text("# fib")

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            result = await tool(mock_deps, filename="fibonacci", repo="myrepo", dest_path="src/file.py")

        assert "error" in result
        assert "hint" in result
        assert "fibonacci" in str(result["hint"])

    @pytest.mark.asyncio
    async def test_code_move_invalid_source_path(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo rejects path traversal in source."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()

        # Create file outside code_dir
        outside_file = tmp_path / "outside.py"
        outside_file.write_text("# outside")

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            result = await tool(mock_deps, filename="../outside.py", repo="myrepo", dest_path="src/file.py")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_code_move_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo returns error when repo not found."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "test.py").write_text("# test")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="test.py", repo="nonexistent", dest_path="src/file.py")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_code_move_dest_outside_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo rejects path traversal in destination."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "test.py").write_text("# test")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="test.py", repo="myrepo", dest_path="../../../etc/passwd")

        assert "error" in result
        assert "outside" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_code_move_dest_exists_no_overwrite(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo returns error when dest exists and no overwrite."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "test.py").write_text("# new content")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_dir = repos_dir / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "existing.py").write_text("# existing")

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="test.py", repo="myrepo", dest_path="existing.py")

        assert "error" in result
        assert "already exists" in result["error"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_code_move_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo successfully moves file."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        source_file = code_dir / "test.py"
        source_file.write_text("# test content")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_dir = repos_dir / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="test.py", repo="myrepo", dest_path="src/utils/test.py")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"
        assert "relative_path" in result
        # Verify file was moved
        assert not source_file.exists()
        assert (repo_dir / "src" / "utils" / "test.py").exists()

    @pytest.mark.asyncio
    async def test_code_move_to_directory(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo moves file to existing directory."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        source_file = code_dir / "mycode.py"
        source_file.write_text("# my code")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_dir = repos_dir / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="mycode.py", repo="myrepo", dest_path="src")

        assert result["status"] == "success"
        # Should keep original filename
        assert (repo_dir / "src" / "mycode.py").exists()

    @pytest.mark.asyncio
    async def test_code_move_to_directory_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo moves file to path ending with /."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        source_file = code_dir / "mycode.py"
        source_file.write_text("# my code")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_dir = repos_dir / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="mycode.py", repo="myrepo", dest_path="lib/utils/")

        assert result["status"] == "success"
        # Should create directory and keep original filename
        assert (repo_dir / "lib" / "utils" / "mycode.py").exists()

    @pytest.mark.asyncio
    async def test_code_move_with_overwrite(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo can overwrite existing file."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "new.py").write_text("# new content")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_dir = repos_dir / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "existing.py").write_text("# old content")

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="new.py", repo="myrepo", dest_path="existing.py", overwrite=True)

        assert result["status"] == "success"
        assert (repo_dir / "existing.py").read_text() == "# new content"

    @pytest.mark.asyncio
    async def test_code_move_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo handles repo name with slash."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "test.py").write_text("# test")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_dir = repos_dir / "myrepo"  # Only the last part is used
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                result = await tool(mock_deps, filename="test.py", repo="owner/myrepo", dest_path="test.py")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_code_move_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code_move_to_repo handles exceptions."""
        tool = CodeMoveToRepoTool()

        code_dir = tmp_path / "reachy_code"
        code_dir.mkdir()
        (code_dir / "test.py").write_text("# test")

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.tools.code_move_to_repo.CODE_OUTPUT_DIR", code_dir):
            with patch("reachy_mini_conversation_app.tools.code_move_to_repo.REPOS_DIR", repos_dir):
                with patch("shutil.move", side_effect=OSError("Permission denied")):
                    result = await tool(mock_deps, filename="test.py", repo="myrepo", dest_path="dest.py")

        assert "error" in result
        assert "Failed to move" in result["error"]
