"""Unit tests for the github_edit_file tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.tools.github_edit_file import GitHubEditFileTool, REPOS_DIR
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubEditFileToolAttributes:
    """Tests for GitHubEditFileTool tool attributes."""

    def test_github_edit_file_has_correct_name(self) -> None:
        """Test GitHubEditFileTool tool has correct name."""
        tool = GitHubEditFileTool()
        assert tool.name == "github_edit_file"

    def test_github_edit_file_has_description(self) -> None:
        """Test GitHubEditFileTool tool has description."""
        tool = GitHubEditFileTool()
        assert "edit" in tool.description.lower()
        assert "ai" in tool.description.lower()

    def test_github_edit_file_has_parameters_schema(self) -> None:
        """Test GitHubEditFileTool tool has correct parameters schema."""
        tool = GitHubEditFileTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "path" in schema["properties"]
        assert "edit_prompt" in schema["properties"]
        assert "model_path" in schema["properties"]
        assert "model_repo" in schema["properties"]
        assert "analyzer" in schema["properties"]
        assert "apply" in schema["properties"]
        assert "repo" in schema["required"]
        assert "path" in schema["required"]

    def test_github_edit_file_spec(self) -> None:
        """Test GitHubEditFileTool tool spec generation."""
        tool = GitHubEditFileTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_edit_file"


class TestGitHubEditFileToolExecution:
    """Tests for GitHubEditFileTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_edit_file_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_edit_file returns error when repo is missing."""
        tool = GitHubEditFileTool()

        result = await tool(mock_deps, repo="", path="file.py", edit_prompt="fix bugs")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_missing_path(self, mock_deps: ToolDependencies) -> None:
        """Test github_edit_file returns error when path is missing."""
        tool = GitHubEditFileTool()

        result = await tool(mock_deps, repo="myrepo", path="", edit_prompt="fix bugs")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_missing_prompt_and_model(self, mock_deps: ToolDependencies) -> None:
        """Test github_edit_file requires edit_prompt or model_path."""
        tool = GitHubEditFileTool()

        result = await tool(mock_deps, repo="myrepo", path="file.py")

        assert "error" in result
        assert "edit_prompt" in result["error"].lower() or "model_path" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_edit_file_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file returns error when repo not found."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", path="file.py", edit_prompt="fix")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_target_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file returns error when target file not found."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="myrepo", path="nonexistent.py", edit_prompt="fix")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_model_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file returns error when model repo not found."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("content")

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            result = await tool(
                mock_deps,
                repo="myrepo",
                path="target.py",
                model_path="model.py",
                model_repo="nonexistent",
            )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_model_file_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file returns error when model file not found."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("content")

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            result = await tool(
                mock_deps,
                repo="myrepo",
                path="target.py",
                model_path="nonexistent_model.py",
            )

        assert "error" in result
        assert "model file not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_preview_with_claude(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file preview mode with Claude."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        target_file = repo_path / "target.py"
        target_file.write_text("def hello():\n    pass")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="def hello():\n    print('Hello')")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.tools.github_edit_file.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        path="target.py",
                        edit_prompt="add print statement",
                        apply=False,
                    )

        assert result["status"] == "preview"
        assert result["analyzer"] == "claude"
        assert "new_content" in result
        # File should not be changed
        assert target_file.read_text() == "def hello():\n    pass"

    @pytest.mark.asyncio
    async def test_github_edit_file_apply_with_claude(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file apply mode with Claude."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        target_file = repo_path / "target.py"
        target_file.write_text("def hello():\n    pass")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="def hello():\n    print('Hello')")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.tools.github_edit_file.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        path="target.py",
                        edit_prompt="add print statement",
                        apply=True,
                    )

        assert result["status"] == "applied"
        assert "hint" in result
        # File should be changed
        assert "print" in target_file.read_text()

    @pytest.mark.asyncio
    async def test_github_edit_file_with_openai(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file with OpenAI."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("def hello(): pass")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="def hello(): print('hi')"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.OPENAI_API_KEY = "test-key"
                with patch("reachy_mini_conversation_app.tools.github_edit_file.openai.OpenAI", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        path="target.py",
                        edit_prompt="add print",
                        analyzer="openai",
                    )

        assert result["status"] == "preview"
        assert result["analyzer"] == "openai"

    @pytest.mark.asyncio
    async def test_github_edit_file_with_model_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file with model file reference."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("def hello(): pass")
        (repo_path / "model.py").write_text("def example():\n    '''Docstring'''\n    pass")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="def hello():\n    '''Hello docstring'''\n    pass")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.tools.github_edit_file.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        path="target.py",
                        model_path="model.py",
                    )

        assert result["status"] == "preview"
        assert result["model_file"] == "model.py"

    @pytest.mark.asyncio
    async def test_github_edit_file_strips_markdown_code_blocks(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file strips markdown code blocks from response."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("def hello(): pass")

        # AI returns response wrapped in markdown code block
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="```python\ndef hello():\n    print('hi')\n```")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.tools.github_edit_file.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        path="target.py",
                        edit_prompt="add print",
                    )

        assert result["status"] == "preview"
        # Code blocks should be stripped
        assert not result["new_content"].startswith("```")
        assert "def hello():" in result["new_content"]

    @pytest.mark.asyncio
    async def test_github_edit_file_no_api_key(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file with missing API key."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("def hello(): pass")

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = None
                result = await tool(
                    mock_deps,
                    repo="myrepo",
                    path="target.py",
                    edit_prompt="add print",
                )

        assert "error" in result
        assert "not configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_edit_file_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_edit_file handles repo name with slash."""
        tool = GitHubEditFileTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        repo_path = repos_dir / "myrepo"
        repo_path.mkdir()
        (repo_path / "target.py").write_text("content")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="edited content")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("reachy_mini_conversation_app.tools.github_edit_file.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_edit_file.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
                with patch("reachy_mini_conversation_app.tools.github_edit_file.anthropic.Anthropic", return_value=mock_client):
                    result = await tool(
                        mock_deps,
                        repo="owner/myrepo",
                        path="target.py",
                        edit_prompt="edit",
                    )

        assert result["status"] == "preview"
        assert result["repo"] == "myrepo"


class TestBuildPrompt:
    """Tests for _build_prompt method."""

    def test_build_prompt_with_edit_only(self) -> None:
        """Test building prompt with only edit instructions."""
        tool = GitHubEditFileTool()

        prompt = tool._build_prompt(
            target_content="def hello(): pass",
            target_path="test.py",
            edit_prompt="Add error handling",
        )

        assert "test.py" in prompt
        assert "def hello(): pass" in prompt
        assert "Add error handling" in prompt

    def test_build_prompt_with_model_only(self) -> None:
        """Test building prompt with only model file."""
        tool = GitHubEditFileTool()

        prompt = tool._build_prompt(
            target_content="def hello(): pass",
            target_path="test.py",
            model_content="def example():\n    '''Doc'''\n    pass",
            model_path="model.py",
        )

        assert "test.py" in prompt
        assert "model.py" in prompt
        assert "def example()" in prompt

    def test_build_prompt_with_both(self) -> None:
        """Test building prompt with edit and model."""
        tool = GitHubEditFileTool()

        prompt = tool._build_prompt(
            target_content="content",
            target_path="test.py",
            edit_prompt="Fix bugs",
            model_content="model content",
            model_path="model.py",
        )

        assert "test.py" in prompt
        assert "model.py" in prompt
        assert "Fix bugs" in prompt


class TestReadFile:
    """Tests for _read_file method."""

    def test_read_file_success(self, tmp_path: Path) -> None:
        """Test reading file successfully."""
        tool = GitHubEditFileTool()
        (tmp_path / "test.py").write_text("content")

        result = tool._read_file(tmp_path, "test.py")

        assert result == "content"

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        """Test reading non-existent file."""
        tool = GitHubEditFileTool()

        result = tool._read_file(tmp_path, "nonexistent.py")

        assert result is None

    def test_read_file_outside_repo(self, tmp_path: Path) -> None:
        """Test reading file outside repo returns None."""
        tool = GitHubEditFileTool()

        result = tool._read_file(tmp_path, "../../../etc/passwd")

        assert result is None
