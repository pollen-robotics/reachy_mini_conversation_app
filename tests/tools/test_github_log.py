"""Unit tests for the github_log tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from git import InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.github_log import GitHubLogTool, REPOS_DIR
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


class TestGitHubLogToolAttributes:
    """Tests for GitHubLogTool tool attributes."""

    def test_github_log_has_correct_name(self) -> None:
        """Test GitHubLogTool tool has correct name."""
        tool = GitHubLogTool()
        assert tool.name == "github_log"

    def test_github_log_has_description(self) -> None:
        """Test GitHubLogTool tool has description."""
        tool = GitHubLogTool()
        assert "commit history" in tool.description.lower()

    def test_github_log_has_parameters_schema(self) -> None:
        """Test GitHubLogTool tool has correct parameters schema."""
        tool = GitHubLogTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "count" in schema["properties"]
        assert "branch" in schema["properties"]
        assert "author" in schema["properties"]
        assert "since" in schema["properties"]
        assert "until" in schema["properties"]
        assert "path" in schema["properties"]
        assert "grep" in schema["properties"]
        assert "oneline" in schema["properties"]
        assert "stat" in schema["properties"]
        assert "repo" in schema["required"]

    def test_github_log_spec(self) -> None:
        """Test GitHubLogTool tool spec generation."""
        tool = GitHubLogTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_log"


class TestGitHubLogToolExecution:
    """Tests for GitHubLogTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_log_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_log returns error when repo is missing."""
        tool = GitHubLogTool()

        result = await tool(mock_deps, repo="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_log_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log returns error when repo not found."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_log_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log returns error for non-git directory."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit")

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_log_default(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log shows commits with default format."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        log_output = "abc123|abc123|John Doe|john@example.com|2024-01-15|Initial commit"

        mock_git = MagicMock()
        mock_git.log.return_value = log_output

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["branch"] == "main"
        assert result["commit_count"] == 1
        assert result["commits"][0]["hash"] == "abc123"
        assert result["commits"][0]["author"] == "John Doe"

    @pytest.mark.asyncio
    async def test_github_log_oneline(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with oneline format."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        log_output = "abc123 Initial commit\ndef456 Second commit"

        mock_git = MagicMock()
        mock_git.log.return_value = log_output

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", oneline=True)

        assert result["status"] == "success"
        assert result["commit_count"] == 2
        assert result["commits"][0]["hash"] == "abc123"
        assert result["commits"][0]["message"] == "Initial commit"

    @pytest.mark.asyncio
    async def test_github_log_with_count(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with specific count."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", count=5)

        assert result["status"] == "success"
        # Verify -n5 was used
        call_args = mock_git.log.call_args[0]
        assert "-n5" in call_args

    @pytest.mark.asyncio
    async def test_github_log_count_capped_at_100(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log caps count at 100."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", count=500)

        assert result["status"] == "success"
        # Verify -n100 was used (capped)
        call_args = mock_git.log.call_args[0]
        assert "-n100" in call_args

    @pytest.mark.asyncio
    async def test_github_log_with_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with specific branch."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", branch="develop")

        assert result["status"] == "success"
        assert result["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_github_log_with_author(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log filtered by author."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|John|john@example.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", author="John")

        assert result["status"] == "success"
        assert "author: John" in result["filters"]

    @pytest.mark.asyncio
    async def test_github_log_with_date_range(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with date range."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", since="2024-01-01", until="2024-12-31")

        assert result["status"] == "success"
        assert "since: 2024-01-01" in result["filters"]
        assert "until: 2024-12-31" in result["filters"]

    @pytest.mark.asyncio
    async def test_github_log_with_path(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log filtered by path."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", path="src/main.py")

        assert result["status"] == "success"
        assert "path: src/main.py" in result["filters"]

    @pytest.mark.asyncio
    async def test_github_log_with_grep(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log filtered by message grep."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Fix bug"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", grep="Fix")

        assert result["status"] == "success"
        assert "grep: Fix" in result["filters"]

    @pytest.mark.asyncio
    async def test_github_log_no_commits(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with no matching commits."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = ""

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert result["commit_count"] == 0
        assert "no commits" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_github_log_detached_head(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with detached HEAD."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_head = MagicMock()
        mock_head.commit.hexsha = "abc12345678"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.head = mock_head
        # Simulate detached HEAD by making active_branch raise TypeError
        type(mock_repo).active_branch = property(lambda self: (_ for _ in ()).throw(TypeError()))

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert result["status"] == "success"
        assert "detached" in result["branch"].lower()

    @pytest.mark.asyncio
    async def test_github_log_with_stat(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log with stats."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        # First call for main log
        mock_git.log.side_effect = [
            "abc|abc|Author|a@b.com|2024-01-15|Commit",
            " file.txt | 2 +-",  # stat output
        ]

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", stat=True)

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_github_log_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log handles repo name with slash."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.return_value = "abc|abc|Author|a@b.com|2024-01-15|Commit"

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_log_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_log handles generic exception."""
        tool = GitHubLogTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_git = MagicMock()
        mock_git.log.side_effect = RuntimeError("Unexpected error")

        mock_active_branch = MagicMock()
        mock_active_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.git = mock_git
        mock_repo.active_branch = mock_active_branch

        with patch("reachy_mini_conversation_app.tools.github_log.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.tools.github_log.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo")

        assert "error" in result
        assert "Failed to get commit history" in result["error"]
