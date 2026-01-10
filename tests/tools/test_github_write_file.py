"""Unit tests for the github_write_file tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.github_write_file import GitHubWriteFileTool


class TestGitHubWriteFileToolAttributes:
    """Tests for GitHubWriteFileTool tool attributes."""

    def test_github_write_file_has_correct_name(self) -> None:
        """Test GitHubWriteFileTool tool has correct name."""
        tool = GitHubWriteFileTool()
        assert tool.name == "github_write_file"

    def test_github_write_file_has_description(self) -> None:
        """Test GitHubWriteFileTool tool has description."""
        tool = GitHubWriteFileTool()
        assert "write" in tool.description.lower()

    def test_github_write_file_has_parameters_schema(self) -> None:
        """Test GitHubWriteFileTool tool has correct parameters schema."""
        tool = GitHubWriteFileTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "path" in schema["properties"]
        assert "content" in schema["properties"]
        assert "overwrite" in schema["properties"]
        assert "create_dirs" in schema["properties"]
        assert "repo" in schema["required"]
        assert "path" in schema["required"]
        assert "content" in schema["required"]

    def test_github_write_file_spec(self) -> None:
        """Test GitHubWriteFileTool tool spec generation."""
        tool = GitHubWriteFileTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_write_file"


class TestGitHubWriteFileToolExecution:
    """Tests for GitHubWriteFileTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_write_file_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_write_file returns error when repo is missing."""
        tool = GitHubWriteFileTool()

        result = await tool(mock_deps, repo="", path="file.txt", content="test")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_write_file_missing_path(self, mock_deps: ToolDependencies) -> None:
        """Test github_write_file returns error when path is missing."""
        tool = GitHubWriteFileTool()

        result = await tool(mock_deps, repo="myrepo", path="", content="test")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_write_file_missing_content(self, mock_deps: ToolDependencies) -> None:
        """Test github_write_file returns error when content is None."""
        tool = GitHubWriteFileTool()

        result = await tool(mock_deps, repo="myrepo", path="file.txt", content=None)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_write_file_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file returns error when repo not found."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", path="file.txt", content="test")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_write_file_path_outside_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file returns error for path outside repo."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="../../../etc/passwd", content="test")

        assert "error" in result
        assert "outside" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_write_file_create_new(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file creates new file."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="new_file.txt", content="Hello World")

        assert result["status"] == "success"
        assert result["action"] == "created"
        assert result["repo"] == "myrepo"
        assert result["path"] == "new_file.txt"
        assert result["lines"] == 1
        assert "hint" in result

        # Verify file was created
        created_file = repo_path / "new_file.txt"
        assert created_file.exists()
        assert created_file.read_text() == "Hello World"

    @pytest.mark.asyncio
    async def test_github_write_file_modify_existing(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file modifies existing file."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        existing_file = repo_path / "existing.txt"
        existing_file.write_text("Old content")

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="existing.txt", content="New content")

        assert result["status"] == "success"
        assert result["action"] == "modified"
        assert existing_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_github_write_file_overwrite_false(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file fails when overwrite=False and file exists."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        existing_file = repo_path / "existing.txt"
        existing_file.write_text("Old content")

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="existing.txt", content="New", overwrite=False)

        assert "error" in result
        assert "already exists" in result["error"].lower()
        assert "hint" in result
        # Verify original content unchanged
        assert existing_file.read_text() == "Old content"

    @pytest.mark.asyncio
    async def test_github_write_file_create_dirs(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file creates parent directories."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="sub/dir/file.txt", content="content")

        assert result["status"] == "success"
        assert result["action"] == "created"
        assert (repo_path / "sub" / "dir" / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_github_write_file_create_dirs_false(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file fails when create_dirs=False and parent missing."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="sub/dir/file.txt", content="content", create_dirs=False)

        assert "error" in result
        assert "parent directory" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_write_file_empty_content(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file allows empty content."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="empty.txt", content="")

        assert result["status"] == "success"
        assert result["lines"] == 0
        assert (repo_path / "empty.txt").exists()
        assert (repo_path / "empty.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_github_write_file_multiline_content(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file with multiline content."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        content = "line1\nline2\nline3\nline4"

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="multi.txt", content=content)

        assert result["status"] == "success"
        assert result["lines"] == 4

    @pytest.mark.asyncio
    async def test_github_write_file_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file handles repo name with slash."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="owner/myrepo", path="file.txt", content="test")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_write_file_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file handles write exception."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        # Create a directory with the target filename to cause write error
        (repo_path / "file.txt").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="file.txt", content="test")

        assert "error" in result
        assert "Failed to write" in result["error"]

    @pytest.mark.asyncio
    async def test_github_write_file_create_dirs_false_parent_exists(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_write_file with create_dirs=False when parent exists."""
        tool = GitHubWriteFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        # Create the parent directory so create_dirs=False path succeeds
        (repo_path / "existing_dir").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_write_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="existing_dir/file.txt", content="content", create_dirs=False)

        # Should succeed because parent exists
        assert result["status"] == "success"
        assert result["action"] == "created"
        assert (repo_path / "existing_dir" / "file.txt").exists()
