"""Unit tests for the github_list_files tool."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_list_files import GitHubListFilesTool


class TestGitHubListFilesToolAttributes:
    """Tests for GitHubListFilesTool tool attributes."""

    def test_github_list_files_has_correct_name(self) -> None:
        """Test GitHubListFilesTool tool has correct name."""
        tool = GitHubListFilesTool()
        assert tool.name == "github_list_files"

    def test_github_list_files_has_description(self) -> None:
        """Test GitHubListFilesTool tool has description."""
        tool = GitHubListFilesTool()
        assert "list" in tool.description.lower()
        assert "files" in tool.description.lower()

    def test_github_list_files_has_parameters_schema(self) -> None:
        """Test GitHubListFilesTool tool has correct parameters schema."""
        tool = GitHubListFilesTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "path" in schema["properties"]
        assert "recursive" in schema["properties"]
        assert "max_depth" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_list_files_spec(self) -> None:
        """Test GitHubListFilesTool tool spec generation."""
        tool = GitHubListFilesTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_list_files"


class TestGitHubListFilesToolExecution:
    """Tests for GitHubListFilesTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_list_files_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_list_files returns error when repo is missing."""
        tool = GitHubListFilesTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_files_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files returns error when repo not found."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_list_files_path_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files returns error when path not found."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_files_path_is_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files returns error when path is a file."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").write_text("content")

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="file.txt")

        assert "error" in result
        assert "not a directory" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_files_root(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files lists root directory."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file1.txt").write_text("content1")
        (repo_path / "file2.py").write_text("content2")
        (repo_path / "subdir").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"
        assert result["path"] == "/"
        assert result["count"] == 3
        assert result["truncated"] is False

        # Check file types
        paths = {f["path"] for f in result["files"]}
        assert "file1.txt" in paths
        assert "file2.py" in paths
        assert "subdir" in paths

    @pytest.mark.asyncio
    async def test_github_list_files_subdir(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files lists subdirectory."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        subdir = repo_path / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("code")

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="src")

        assert result["status"] == "success"
        assert result["path"] == "src"
        assert result["count"] == 1
        assert result["files"][0]["path"] == "src/main.py"

    @pytest.mark.asyncio
    async def test_github_list_files_hides_hidden_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files hides hidden files."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "visible.txt").write_text("content")
        (repo_path / ".hidden").write_text("hidden")
        (repo_path / ".git").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["count"] == 1
        paths = {f["path"] for f in result["files"]}
        assert "visible.txt" in paths
        assert ".hidden" not in paths
        assert ".git" not in paths

    @pytest.mark.asyncio
    async def test_github_list_files_file_types(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files includes file type info."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").write_text("content")
        (repo_path / "dir").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        files_dict = {f["path"]: f for f in result["files"]}
        assert files_dict["file.txt"]["type"] == "file"
        assert files_dict["dir"]["type"] == "directory"

    @pytest.mark.asyncio
    async def test_github_list_files_includes_size(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files includes file size."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").write_text("12345")
        (repo_path / "dir").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo")

        files_dict = {f["path"]: f for f in result["files"]}
        assert files_dict["file.txt"]["size"] == 5
        assert files_dict["dir"]["size"] is None

    @pytest.mark.asyncio
    async def test_github_list_files_recursive(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files recursive mode."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        subdir = repo_path / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        # Mock subprocess for find command
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = f"{repo_path}\n{subdir}\n{subdir / 'nested.txt'}"

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, repo="myrepo", recursive=True)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_list_files_truncates_results(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files truncates results at 100."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        # Create more than 100 files
        for i in range(150):
            (repo_path / f"file{i:03d}.txt").write_text(f"content{i}")

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["count"] == 100
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_github_list_files_timeout(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files handles timeout."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            with patch(
                "reachy_mini_conversation_app.profiles.linus.github_list_files.subprocess.run",
                side_effect=subprocess.TimeoutExpired("find", 10),
            ):
                result = await tool(mock_deps, repo="myrepo", recursive=True)

        assert "error" in result
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_list_files_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files handles repo name with slash."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "file.txt").write_text("content")

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_list_files_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files handles generic exception."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            with patch.object(Path, "iterdir", side_effect=RuntimeError("Unexpected error")):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to list files" in result["error"]

    @pytest.mark.asyncio
    async def test_github_list_files_sorted_results(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files returns sorted results."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "z_file.txt").write_text("z")
        (repo_path / "a_file.txt").write_text("a")
        (repo_path / "m_file.txt").write_text("m")

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        paths = [f["path"] for f in result["files"]]
        assert paths == sorted(paths)

    @pytest.mark.asyncio
    async def test_github_list_files_recursive_find_fails(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_list_files handles find command failure in recursive mode."""
        tool = GitHubListFilesTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()

        # Mock subprocess with non-zero return code
        mock_result = MagicMock()
        mock_result.returncode = 1  # Non-zero = failure
        mock_result.stdout = ""
        mock_result.stderr = "find: error"

        with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_list_files.subprocess.run", return_value=mock_result):
                result = await tool(mock_deps, repo="myrepo", recursive=True)

        # Should still return success but with empty files list
        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["files"] == []
