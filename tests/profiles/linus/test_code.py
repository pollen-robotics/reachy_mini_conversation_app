"""Unit tests for the code generation tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.code import CodeTool


class TestCodeToolAttributes:
    """Tests for CodeTool tool attributes."""

    def test_code_has_correct_name(self) -> None:
        """Test CodeTool tool has correct name."""
        tool = CodeTool()
        assert tool.name == "code"

    def test_code_has_description(self) -> None:
        """Test CodeTool tool has description."""
        tool = CodeTool()
        assert "code" in tool.description.lower()
        assert "claude" in tool.description.lower()

    def test_code_has_parameters_schema(self) -> None:
        """Test CodeTool tool has correct parameters schema."""
        tool = CodeTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "question" in schema["properties"]
        assert "language" in schema["properties"]
        assert "filename" in schema["properties"]
        assert "repo" in schema["properties"]
        assert "path" in schema["properties"]
        assert "overwrite" in schema["properties"]
        assert "question" in schema["required"]

    def test_code_spec(self) -> None:
        """Test CodeTool tool spec generation."""
        tool = CodeTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "code"


class TestCodeToolExecution:
    """Tests for CodeTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_code_missing_api_key(self, mock_deps: ToolDependencies) -> None:
        """Test code returns error when API key is missing."""
        tool = CodeTool()

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = None

            result = await tool(mock_deps, question="Write hello world")

        assert "error" in result
        assert "ANTHROPIC_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_code_repo_without_path(self, mock_deps: ToolDependencies) -> None:
        """Test code returns error when repo is set but path is missing."""
        tool = CodeTool()

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"

            result = await tool(mock_deps, question="Write hello world", repo="myrepo")

        assert "error" in result
        assert "Path" in result["error"]

    @pytest.mark.asyncio
    async def test_code_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error when repo doesn't exist."""
        tool = CodeTool()

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.code.REPOS_DIR", tmp_path):
                result = await tool(mock_deps, question="Write hello world", repo="nonexistent", path="src/main.py")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_code_file_exists_no_overwrite(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error when file exists and overwrite is False."""
        tool = CodeTool()

        # Create repo directory and file
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "existing.py").write_text("# existing content")

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Write hello world",
                    repo="myrepo",
                    path="existing.py",
                    overwrite=False,
                )

        assert "error" in result
        assert "already exists" in result["error"]

    @pytest.mark.asyncio
    async def test_code_path_traversal_attack(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error for path traversal attempt."""
        tool = CodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Write hello world",
                    repo="myrepo",
                    path="../../../etc/passwd",
                )

        assert "error" in result
        assert "outside" in result["error"].lower() or "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_code_success_to_reachy_code(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code successfully generates and saves to reachy_code."""
        tool = CodeTool()

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="def hello():\n    print('Hello, world!')")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = "claude-test"
            with patch("reachy_mini_conversation_app.profiles.linus.code.CODE_OUTPUT_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(mock_deps, question="Write hello world", language="python")

        assert result["status"] == "success"
        assert "filepath" in result
        assert result["language"] == "python"
        assert "explanation" in result
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_success_to_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code successfully generates and saves to repo."""
        tool = CodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="def hello():\n    print('Hello!')")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None  # Test default model
            with patch("reachy_mini_conversation_app.profiles.linus.code.REPOS_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        question="Write hello world",
                        repo="myrepo",
                        path="src/hello.py",
                    )

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"
        assert result["relative_path"] == "src/hello.py"
        # Check file was created
        assert (repo_dir / "src" / "hello.py").exists()

    @pytest.mark.asyncio
    async def test_code_authentication_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles authentication error."""
        import anthropic

        tool = CodeTool()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "invalid-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.code.CODE_OUTPUT_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(mock_deps, question="Write hello world")

        assert "error" in result
        assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_code_rate_limit_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles rate limit error."""
        import anthropic

        tool = CodeTool()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit"}},
        )

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.code.CODE_OUTPUT_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(mock_deps, question="Write hello world")

        assert "error" in result
        assert "Rate limit" in result["error"]

    @pytest.mark.asyncio
    async def test_code_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles generic exceptions."""
        tool = CodeTool()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Network error")

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.code.CODE_OUTPUT_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(mock_deps, question="Write hello world")

        assert "error" in result
        assert "Failed to generate code" in result["error"]


class TestCodeToolHelpers:
    """Tests for CodeTool helper methods."""

    def test_extract_code_from_markdown_with_code_block(self) -> None:
        """Test extracting code from markdown code blocks."""
        tool = CodeTool()

        content = "```python\ndef hello():\n    print('Hello')\n```"
        result = tool._extract_code_from_markdown(content)

        assert result == "def hello():\n    print('Hello')"

    def test_extract_code_from_markdown_without_code_block(self) -> None:
        """Test extracting code when no markdown blocks present."""
        tool = CodeTool()

        content = "def hello():\n    print('Hello')"
        result = tool._extract_code_from_markdown(content)

        assert result == "def hello():\n    print('Hello')"

    def test_extract_code_from_markdown_no_language(self) -> None:
        """Test extracting code from markdown blocks without language."""
        tool = CodeTool()

        content = "```\ndef hello():\n    print('Hello')\n```"
        result = tool._extract_code_from_markdown(content)

        assert result == "def hello():\n    print('Hello')"

    def test_generate_filename_with_keywords(self) -> None:
        """Test filename generation from question."""
        tool = CodeTool()

        filename = tool._generate_filename("Write a function to calculate fibonacci numbers")

        assert "fibonacci" in filename or "calculate" in filename or "function" not in filename

    def test_generate_filename_empty_question(self) -> None:
        """Test filename generation with minimal keywords."""
        tool = CodeTool()

        filename = tool._generate_filename("a to the for")

        assert filename == "generated_code"

    def test_get_extension_python(self) -> None:
        """Test getting extension for Python."""
        tool = CodeTool()
        assert tool._get_extension("python") == "py"
        assert tool._get_extension("Python") == "py"

    def test_get_extension_javascript(self) -> None:
        """Test getting extension for JavaScript."""
        tool = CodeTool()
        assert tool._get_extension("javascript") == "js"

    def test_get_extension_typescript(self) -> None:
        """Test getting extension for TypeScript."""
        tool = CodeTool()
        assert tool._get_extension("typescript") == "ts"

    def test_get_extension_rust(self) -> None:
        """Test getting extension for Rust."""
        tool = CodeTool()
        assert tool._get_extension("rust") == "rs"

    def test_get_extension_cpp(self) -> None:
        """Test getting extension for C++."""
        tool = CodeTool()
        assert tool._get_extension("cpp") == "cpp"
        assert tool._get_extension("c++") == "cpp"

    def test_get_extension_unknown(self) -> None:
        """Test getting extension for unknown language."""
        tool = CodeTool()
        assert tool._get_extension("unknown_lang") == "txt"

    def test_generate_explanation_python(self) -> None:
        """Test explanation generation for Python code."""
        tool = CodeTool()

        code = """def hello():
    print('Hello')

def world():
    print('World')

class MyClass:
    pass
"""
        explanation = tool._generate_explanation(code, "python")

        assert "8 lines" in explanation
        assert "2 function" in explanation
        assert "1 class" in explanation

    def test_generate_explanation_javascript(self) -> None:
        """Test explanation generation for JavaScript code."""
        tool = CodeTool()

        code = """function hello() {
    console.log('Hello');
}

class MyClass {
}
"""
        explanation = tool._generate_explanation(code, "javascript")

        assert "6 lines" in explanation
        assert "function" in explanation.lower() or "javascript" in explanation.lower()

    def test_generate_explanation_no_functions(self) -> None:
        """Test explanation generation for code without functions."""
        tool = CodeTool()

        code = "print('Hello')\nprint('World')"
        explanation = tool._generate_explanation(code, "python")

        assert "2 lines" in explanation
        assert "function" not in explanation.lower()


class TestCodeToolWithFilename:
    """Tests for CodeTool with custom filename."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_code_with_custom_filename(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code with custom filename."""
        tool = CodeTool()

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="print('hello')")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.code.CODE_OUTPUT_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        question="Write hello world",
                        filename="my_custom_script",
                    )

        assert result["status"] == "success"
        assert "my_custom_script" in result["filename"]

    @pytest.mark.asyncio
    async def test_code_overwrite_existing_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code can overwrite existing file in repo."""
        tool = CodeTool()

        # Create repo directory and file
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "existing.py").write_text("# old content")

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="# new content")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.code.REPOS_DIR", tmp_path):
                with patch("reachy_mini_conversation_app.profiles.linus.code.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        question="Write new content",
                        repo="myrepo",
                        path="existing.py",
                        overwrite=True,
                    )

        assert result["status"] == "success"
        assert (repo_dir / "existing.py").read_text() == "# new content"
