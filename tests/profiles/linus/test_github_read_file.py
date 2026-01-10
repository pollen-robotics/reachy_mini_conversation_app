"""Unit tests for the github_read_file tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_read_file import (
    MAX_FILE_SIZE,
    TEXT_EXTENSIONS,
    GitHubReadFileTool,
)


class TestGitHubReadFileToolAttributes:
    """Tests for GitHubReadFileTool tool attributes."""

    def test_github_read_file_has_correct_name(self) -> None:
        """Test GitHubReadFileTool tool has correct name."""
        tool = GitHubReadFileTool()
        assert tool.name == "github_read_file"

    def test_github_read_file_has_description(self) -> None:
        """Test GitHubReadFileTool tool has description."""
        tool = GitHubReadFileTool()
        assert "read" in tool.description.lower()
        assert "file" in tool.description.lower()

    def test_github_read_file_has_parameters_schema(self) -> None:
        """Test GitHubReadFileTool tool has correct parameters schema."""
        tool = GitHubReadFileTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "path" in schema["properties"]
        assert "start_line" in schema["properties"]
        assert "end_line" in schema["properties"]
        assert "analyze" in schema["properties"]
        assert "analyzer" in schema["properties"]
        assert "analysis_prompt" in schema["properties"]
        assert "repo" in schema["required"]
        assert "path" in schema["required"]

    def test_github_read_file_spec(self) -> None:
        """Test GitHubReadFileTool tool spec generation."""
        tool = GitHubReadFileTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_read_file"


class TestGitHubReadFileToolExecution:
    """Tests for GitHubReadFileTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_read_file_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_read_file returns error when repo is missing."""
        tool = GitHubReadFileTool()

        result = await tool(mock_deps, repo="", path="file.txt")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_read_file_missing_path(self, mock_deps: ToolDependencies) -> None:
        """Test github_read_file returns error when path is missing."""
        tool = GitHubReadFileTool()

        result = await tool(mock_deps, repo="myrepo", path="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_read_file_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file returns error when repo not found."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", path="file.txt")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_read_file_file_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file returns error when file not found."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="nonexistent.txt")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_read_file_path_outside_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file returns error for path outside repo."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="../../../etc/passwd")

        assert "error" in result
        assert "outside" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_read_file_is_directory(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file returns error when path is a directory."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "subdir").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="subdir")

        assert "error" in result
        assert "directory" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_read_file_file_too_large(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file returns error for large files."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        large_file = repo_path / "large.txt"
        large_file.write_text("x" * (MAX_FILE_SIZE + 1))

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="large.txt")

        assert "error" in result
        assert "too large" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_read_file_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file reads file successfully."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("line1\nline2\nline3")

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="test.py")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"
        assert result["path"] == "test.py"
        assert result["total_lines"] == 3
        assert result["lines_returned"] == 3
        assert result["truncated"] is False
        assert "line1" in result["content"]

    @pytest.mark.asyncio
    async def test_github_read_file_with_line_range(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file with start_line and end_line."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="test.py", start_line=2, end_line=4)

        assert result["status"] == "success"
        assert result["total_lines"] == 5
        assert result["lines_returned"] == 3  # lines 2, 3, 4 (end_line used as slice upper bound)
        assert "line2" in result["content"]
        assert "line3" in result["content"]
        assert "line4" in result["content"]
        assert "line1" not in result["content"]

    @pytest.mark.asyncio
    async def test_github_read_file_truncates_content(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file truncates very long content."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        # Create file under MAX_FILE_SIZE but content over 50000 chars
        test_file.write_text("x" * 60000)

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="test.py")

        assert result["status"] == "success"
        assert result["truncated"] is True
        assert len(result["content"]) == 50000

    @pytest.mark.asyncio
    async def test_github_read_file_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file handles repo name with slash."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("content")

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="owner/myrepo", path="test.py")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_read_file_unicode_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file handles binary/non-text files."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        binary_file = repo_path / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="test.bin")

        assert "error" in result
        assert "not a text file" in result["error"].lower()


class TestGitHubReadFileAnalysis:
    """Tests for AI analysis functionality."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_read_file_analyze_with_claude(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file with Claude analysis."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("def hello(): pass")

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="Analysis result")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True)

        assert result["status"] == "success"
        assert "analysis" in result
        assert result["analysis"]["analyzer"] == "claude"

    @pytest.mark.asyncio
    async def test_github_read_file_analyze_with_openai(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file with OpenAI analysis."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("def hello(): pass")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Analysis result"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.OPENAI_API_KEY = "test-key"
                mock_config.OPENAI_MODEL = "gpt-4o"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.openai.OpenAI", return_value=mock_client):
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True, analyzer="openai")

        assert result["status"] == "success"
        assert "analysis" in result
        assert result["analysis"]["analyzer"] == "openai"

    @pytest.mark.asyncio
    async def test_github_read_file_analyze_no_api_key(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_read_file analysis with missing API key."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("def hello(): pass")

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = None
                result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True)

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "not configured" in result["analysis_error"].lower()


class TestGitHubReadFileAnalysisErrors:
    """Tests for AI analysis error handling."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.fixture
    def setup_repo(self, tmp_path: Path) -> Path:
        """Set up test repo with a file."""
        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("def hello(): pass")
        return repos_dir

    @pytest.mark.asyncio
    async def test_claude_authentication_error(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test Claude AuthenticationError handling (line 117-118)."""
        import anthropic
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "invalid-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.anthropic.Anthropic") as mock_cls:
                    mock_client = MagicMock()
                    mock_client.messages.create.side_effect = anthropic.AuthenticationError(
                        message="Invalid API key", body=None, response=MagicMock()
                    )
                    mock_cls.return_value = mock_client
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True)

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "Invalid ANTHROPIC_API_KEY" in result["analysis_error"]

    @pytest.mark.asyncio
    async def test_claude_rate_limit_error(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test Claude RateLimitError handling (line 119-120)."""
        import anthropic
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.anthropic.Anthropic") as mock_cls:
                    mock_client = MagicMock()
                    mock_client.messages.create.side_effect = anthropic.RateLimitError(
                        message="Rate limit exceeded", body=None, response=MagicMock()
                    )
                    mock_cls.return_value = mock_client
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True)

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "rate limit" in result["analysis_error"].lower()

    @pytest.mark.asyncio
    async def test_claude_generic_exception(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test Claude generic exception handling (lines 121-123)."""
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.anthropic.Anthropic") as mock_cls:
                    mock_client = MagicMock()
                    mock_client.messages.create.side_effect = RuntimeError("Unexpected error")
                    mock_cls.return_value = mock_client
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True)

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "Claude analysis failed" in result["analysis_error"]

    @pytest.mark.asyncio
    async def test_openai_no_api_key(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test OpenAI no API key error (line 131)."""
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.OPENAI_API_KEY = None
                result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True, analyzer="openai")

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "OPENAI_API_KEY" in result["analysis_error"]

    @pytest.mark.asyncio
    async def test_openai_authentication_error(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test OpenAI AuthenticationError handling (lines 151-152)."""
        import openai
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.OPENAI_API_KEY = "invalid-key"
                mock_config.OPENAI_MODEL = "gpt-4o"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.openai.OpenAI") as mock_cls:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
                        message="Invalid API key", body=None, response=MagicMock()
                    )
                    mock_cls.return_value = mock_client
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True, analyzer="openai")

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "Invalid OPENAI_API_KEY" in result["analysis_error"]

    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test OpenAI RateLimitError handling (lines 153-154)."""
        import openai
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.OPENAI_API_KEY = "test-key"
                mock_config.OPENAI_MODEL = "gpt-4o"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.openai.OpenAI") as mock_cls:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
                        message="Rate limit exceeded", body=None, response=MagicMock()
                    )
                    mock_cls.return_value = mock_client
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True, analyzer="openai")

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "rate limit" in result["analysis_error"].lower()

    @pytest.mark.asyncio
    async def test_openai_generic_exception(self, mock_deps: ToolDependencies, setup_repo: Path) -> None:
        """Test OpenAI generic exception handling (lines 155-157)."""
        tool = GitHubReadFileTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", setup_repo):
            with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.config") as mock_config:
                mock_config.OPENAI_API_KEY = "test-key"
                mock_config.OPENAI_MODEL = "gpt-4o"
                with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.openai.OpenAI") as mock_cls:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.side_effect = RuntimeError("Unexpected error")
                    mock_cls.return_value = mock_client
                    result = await tool(mock_deps, repo="myrepo", path="test.py", analyze=True, analyzer="openai")

        assert result["status"] == "success"
        assert "analysis_error" in result
        assert "OpenAI analysis failed" in result["analysis_error"]


class TestGitHubReadFileExceptions:
    """Tests for file reading exception handling."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_generic_read_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test generic exception handling when reading file (lines 268-270)."""
        tool = GitHubReadFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        test_file = repo_path / "test.py"
        test_file.write_text("content")

        with patch("reachy_mini_conversation_app.profiles.linus.github_read_file.REPOS_DIR", repos_dir):
            # Patch read_text to raise an exception
            with patch.object(Path, "read_text", side_effect=IOError("Disk error")):
                result = await tool(mock_deps, repo="myrepo", path="test.py")

        assert "error" in result
        assert "Failed to read file" in result["error"]


class TestTextExtensions:
    """Tests for text file extensions."""

    def test_common_extensions_included(self) -> None:
        """Test common text extensions are included."""
        assert ".py" in TEXT_EXTENSIONS
        assert ".js" in TEXT_EXTENSIONS
        assert ".ts" in TEXT_EXTENSIONS
        assert ".json" in TEXT_EXTENSIONS
        assert ".md" in TEXT_EXTENSIONS
        assert ".txt" in TEXT_EXTENSIONS
        assert ".yaml" in TEXT_EXTENSIONS
        assert ".yml" in TEXT_EXTENSIONS

    def test_no_binary_extensions(self) -> None:
        """Test binary extensions are not included."""
        assert ".exe" not in TEXT_EXTENSIONS
        assert ".dll" not in TEXT_EXTENSIONS
        assert ".png" not in TEXT_EXTENSIONS
        assert ".jpg" not in TEXT_EXTENSIONS
        assert ".zip" not in TEXT_EXTENSIONS
