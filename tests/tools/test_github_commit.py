"""Unit tests for the github_commit tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import InvalidGitRepositoryError, GitCommandError

from reachy_mini_conversation_app.tools.github_commit import GitHubCommitTool, REPOS_DIR, COMMIT_TYPES
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubCommitToolAttributes:
    """Tests for GitHubCommitTool tool attributes."""

    def test_github_commit_has_correct_name(self) -> None:
        """Test GitHubCommitTool tool has correct name."""
        tool = GitHubCommitTool()
        assert tool.name == "github_commit"

    def test_github_commit_has_description(self) -> None:
        """Test GitHubCommitTool tool has description."""
        tool = GitHubCommitTool()
        assert "commit" in tool.description.lower()

    def test_github_commit_has_parameters_schema(self) -> None:
        """Test GitHubCommitTool tool has correct parameters schema."""
        tool = GitHubCommitTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "type" in schema["properties"]
        assert "message" in schema["properties"]
        assert "confirmed" in schema["properties"]
        assert "repo" in schema["required"]
        assert "confirmed" in schema["required"]

    def test_github_commit_spec(self) -> None:
        """Test GitHubCommitTool tool spec generation."""
        tool = GitHubCommitTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_commit"


class TestBuildCommitMessage:
    """Tests for _build_commit_message method."""

    def test_build_simple_commit_message(self) -> None:
        """Test building simple commit message."""
        tool = GitHubCommitTool()
        msg = tool._build_commit_message("feat", "add new feature")
        assert msg == "feat: add new feature"

    def test_build_commit_message_with_scope(self) -> None:
        """Test building commit message with scope."""
        tool = GitHubCommitTool()
        msg = tool._build_commit_message("fix", "resolve bug", scope="api")
        assert msg == "fix(api): resolve bug"

    def test_build_commit_message_with_body(self) -> None:
        """Test building commit message with body."""
        tool = GitHubCommitTool()
        msg = tool._build_commit_message("docs", "update readme", body="Added installation instructions")
        assert "docs: update readme" in msg
        assert "Added installation instructions" in msg

    def test_build_commit_message_with_breaking(self) -> None:
        """Test building commit message with breaking change."""
        tool = GitHubCommitTool()
        msg = tool._build_commit_message("feat", "change API", breaking=True)
        assert "feat!: change API" in msg
        assert "BREAKING CHANGE" in msg

    def test_build_commit_message_all_options(self) -> None:
        """Test building commit message with all options."""
        tool = GitHubCommitTool()
        msg = tool._build_commit_message(
            "refactor",
            "restructure module",
            scope="auth",
            body="Complete rewrite of authentication logic",
            breaking=True,
        )
        assert "refactor(auth)!: restructure module" in msg
        assert "Complete rewrite" in msg
        assert "BREAKING CHANGE" in msg


class TestParseResponse:
    """Tests for _parse_response method."""

    def test_parse_simple_json_response(self) -> None:
        """Test parsing simple JSON response."""
        tool = GitHubCommitTool()
        response = '{"type": "feat", "message": "add feature"}'
        result = tool._parse_response(response)
        assert result["type"] == "feat"
        assert result["message"] == "add feature"

    def test_parse_json_with_markdown_block(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        tool = GitHubCommitTool()
        response = '```json\n{"type": "fix", "message": "fix bug"}\n```'
        result = tool._parse_response(response)
        assert result["type"] == "fix"
        assert result["message"] == "fix bug"

    def test_parse_response_with_all_fields(self) -> None:
        """Test parsing response with all fields."""
        tool = GitHubCommitTool()
        response = '{"type": "feat", "scope": "api", "message": "add endpoint", "body": "Details here", "breaking": true}'
        result = tool._parse_response(response)
        assert result["type"] == "feat"
        assert result["scope"] == "api"
        assert result["message"] == "add endpoint"
        assert result["body"] == "Details here"
        assert result["breaking"] is True

    def test_parse_response_defaults(self) -> None:
        """Test parsing response uses defaults for missing fields."""
        tool = GitHubCommitTool()
        response = '{}'
        result = tool._parse_response(response)
        assert result["type"] == "chore"
        assert result["message"] == "update code"
        assert result["breaking"] is False


class TestGitHubCommitToolExecution:
    """Tests for GitHubCommitTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_commit_not_confirmed(self, mock_deps: ToolDependencies) -> None:
        """Test github_commit returns error when not confirmed."""
        tool = GitHubCommitTool()

        result = await tool(mock_deps, repo="myrepo", confirmed=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_commit_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_commit returns error when repo is missing."""
        tool = GitHubCommitTool()

        result = await tool(mock_deps, repo="", confirmed=True)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_commit_missing_type_and_message(self, mock_deps: ToolDependencies) -> None:
        """Test github_commit returns error when type and message missing."""
        tool = GitHubCommitTool()

        result = await tool(mock_deps, repo="myrepo", confirmed=True)

        assert "error" in result
        assert "type" in result["error"].lower() or "message" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_commit_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit returns error when repo not found."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", type="feat", message="test", confirmed=True)

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_commit_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit returns error for non-git directory."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", type="feat", message="test", confirmed=True)

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_commit_invalid_type(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit returns error for invalid commit type."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    result = await tool(mock_deps, repo="myrepo", type="invalid", message="test", confirmed=True)

        assert "error" in result
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_commit_nothing_staged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit returns error when nothing staged."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = ""  # No staged files

        mock_repo = MagicMock()
        mock_repo.git = mock_git

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    result = await tool(mock_deps, repo="myrepo", type="feat", message="test", confirmed=True)

        assert result["status"] == "nothing_to_commit"

    @pytest.mark.asyncio
    async def test_github_commit_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit success."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_index = MagicMock()
        mock_index.commit.return_value = mock_commit

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(mock_deps, repo="myrepo", type="feat", message="add feature", confirmed=True)

        assert result["status"] == "success"
        assert result["commit_hash"] == "abc1234"
        assert result["commit_type"] == "feat"
        mock_index.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_commit_with_scope(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit with scope."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_index = MagicMock()
        mock_index.commit.return_value = mock_commit

        mock_git = MagicMock()
        mock_git.diff.return_value = "api/handler.py"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(
                            mock_deps,
                            repo="myrepo",
                            type="fix",
                            scope="api",
                            message="fix endpoint",
                            confirmed=True,
                        )

        assert result["status"] == "success"
        assert "fix(api)" in result["commit_message"]

    @pytest.mark.asyncio
    async def test_github_commit_amend(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit with amend option."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_head = MagicMock()
        mock_head.commit = mock_commit

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.head = mock_head

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(
                            mock_deps,
                            repo="myrepo",
                            type="fix",
                            message="fix typo",
                            amend=True,
                            confirmed=True,
                        )

        assert result["status"] == "success"
        assert result["amended"] is True
        mock_git.commit.assert_called()

    @pytest.mark.asyncio
    async def test_github_commit_amend_no_commits(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit amend fails when no commits exist."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.diff.return_value = ""  # No staged files

        mock_head = MagicMock()
        type(mock_head).commit = property(lambda self: (_ for _ in ()).throw(ValueError("No commits")))

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.head = mock_head

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        type="fix",
                        message="fix",
                        amend=True,
                        confirmed=True,
                    )

        assert "error" in result
        assert "cannot amend" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_commit_skip_checks(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit with skip_checks option."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_index = MagicMock()
        mock_index.commit.return_value = mock_commit

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    result = await tool(
                        mock_deps,
                        repo="myrepo",
                        type="feat",
                        message="add feature",
                        skip_checks=True,
                        confirmed=True,
                    )

        assert result["status"] == "success"
        assert result["checks_skipped"] is True

    @pytest.mark.asyncio
    async def test_github_commit_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit handles git error."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_index = MagicMock()
        mock_index.commit.side_effect = GitCommandError("commit", "error")

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(mock_deps, repo="myrepo", type="feat", message="test", confirmed=True)

        assert "error" in result
        assert "git command failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_commit_nothing_to_commit_message(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit handles nothing to commit git error."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_index = MagicMock()
        mock_index.commit.side_effect = GitCommandError("commit", "nothing to commit")

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(mock_deps, repo="myrepo", type="feat", message="test", confirmed=True)

        assert result["status"] == "nothing_to_commit"

    @pytest.mark.asyncio
    async def test_github_commit_configures_git_user(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit configures git user from config."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_index = MagicMock()
        mock_index.commit.return_value = mock_commit

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_config_reader = MagicMock()
        mock_config_reader.get_value.side_effect = Exception("not set")

        mock_config_writer = MagicMock()

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index
        mock_repo.config_reader.return_value = mock_config_reader
        mock_repo.config_writer.return_value.__enter__ = MagicMock(return_value=mock_config_writer)
        mock_repo.config_writer.return_value.__exit__ = MagicMock(return_value=False)

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = "testowner"
                    mock_config.GITHUB_OWNER_EMAIL = "test@example.com"
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(mock_deps, repo="myrepo", type="feat", message="test", confirmed=True)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_commit_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_commit handles repo name with slash."""
        tool = GitHubCommitTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_commit = MagicMock()
        mock_commit.hexsha = "abc1234567890"

        mock_index = MagicMock()
        mock_index.commit.return_value = mock_commit

        mock_git = MagicMock()
        mock_git.diff.return_value = "file.txt"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.index = mock_index

        with patch("reachy_mini_conversation_app.tools.github_commit.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_commit.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.tools.github_commit.config") as mock_config:
                    mock_config.GITHUB_DEFAULT_OWNER = None
                    with patch("reachy_mini_conversation_app.tools.github_commit.load_commit_rules", return_value=None):
                        result = await tool(mock_deps, repo="owner/myrepo", type="feat", message="test", confirmed=True)

        assert result["status"] == "success"


class TestCommitTypes:
    """Tests for COMMIT_TYPES constant."""

    def test_commit_types_contains_required_types(self) -> None:
        """Test that COMMIT_TYPES contains all required semantic-release types."""
        required_types = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"]
        for t in required_types:
            assert t in COMMIT_TYPES

    def test_commit_types_have_descriptions(self) -> None:
        """Test that all commit types have descriptions."""
        for commit_type, description in COMMIT_TYPES.items():
            assert isinstance(description, str)
            assert len(description) > 0
