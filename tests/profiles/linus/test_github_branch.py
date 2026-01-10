"""Unit tests for the github_branch tool."""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_branch import GitHubBranchTool


class TestGitHubBranchToolAttributes:
    """Tests for GitHubBranchTool tool attributes."""

    def test_github_branch_has_correct_name(self) -> None:
        """Test GitHubBranchTool tool has correct name."""
        tool = GitHubBranchTool()
        assert tool.name == "github_branch"

    def test_github_branch_has_description(self) -> None:
        """Test GitHubBranchTool tool has description."""
        tool = GitHubBranchTool()
        assert "branch" in tool.description.lower()

    def test_github_branch_has_parameters_schema(self) -> None:
        """Test GitHubBranchTool tool has correct parameters schema."""
        tool = GitHubBranchTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "repo" in schema["properties"]
        assert "action" in schema["properties"]
        assert "branch" in schema["properties"]
        assert "repo" in schema["required"]
        assert "action" in schema["required"]

    def test_github_branch_spec(self) -> None:
        """Test GitHubBranchTool tool spec generation."""
        tool = GitHubBranchTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_branch"


class TestGitHubBranchToolHelpers:
    """Tests for GitHubBranchTool helper methods."""

    def test_get_authenticated_url_with_token(self) -> None:
        """Test _get_authenticated_url with token."""
        tool = GitHubBranchTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://github.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result is not None
        assert "ghp_test123@github.com" in result
        assert "owner/repo" in result

    def test_get_authenticated_url_ssh(self) -> None:
        """Test _get_authenticated_url with SSH URL."""
        tool = GitHubBranchTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "git@github.com:owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test123"
            result = tool._get_authenticated_url(mock_repo)

        assert result == "https://ghp_test123@github.com/owner/repo.git"

    def test_get_authenticated_url_without_token(self) -> None:
        """Test _get_authenticated_url without token."""
        tool = GitHubBranchTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://github.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
            mock_config.GITHUB_TOKEN = None
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    def test_get_authenticated_url_existing_token(self) -> None:
        """Test _get_authenticated_url replaces existing token."""
        tool = GitHubBranchTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://oldtoken@github.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_newtoken"
            result = tool._get_authenticated_url(mock_repo)

        assert result is not None
        assert "ghp_newtoken@github.com" in result
        assert "oldtoken" not in result


class TestGitHubBranchToolExecution:
    """Tests for GitHubBranchTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_github_branch_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test github_branch returns error when repo is missing."""
        tool = GitHubBranchTool()

        result = await tool(mock_deps, repo="", action="list")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_branch_missing_action(self, mock_deps: ToolDependencies) -> None:
        """Test github_branch returns error when action is missing."""
        tool = GitHubBranchTool()

        result = await tool(mock_deps, repo="myrepo", action="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_branch_create_missing_branch_name(self, mock_deps: ToolDependencies) -> None:
        """Test github_branch create requires branch name."""
        tool = GitHubBranchTool()

        result = await tool(mock_deps, repo="myrepo", action="create", branch="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_branch_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch returns error when repo not found."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            result = await tool(mock_deps, repo="nonexistent", action="list")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_github_branch_not_git_repo(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch returns error for non-git directory."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "notgit").mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo") as mock_repo:
                mock_repo.side_effect = InvalidGitRepositoryError("not a git repo")

                result = await tool(mock_deps, repo="notgit", action="list")

        assert "error" in result
        assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_list(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch list action."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch1 = MagicMock()
        mock_branch1.name = "main"
        mock_branch2 = MagicMock()
        mock_branch2.name = "develop"

        mock_remote_ref = MagicMock()
        mock_remote_ref.name = "origin/main"

        mock_remote = MagicMock()
        mock_remote.refs = [mock_remote_ref]

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_branch1, mock_branch2]
        mock_repo.remotes = [mock_remote]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="list")

        assert result["status"] == "success"
        assert result["current_branch"] == "main"
        assert "main" in result["local_branches"]
        assert "develop" in result["local_branches"]
        assert "origin/main" in result["remote_branches"]

    @pytest.mark.asyncio
    async def test_github_branch_create_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create action success."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "main"

        mock_new_branch = MagicMock()

        mock_repo = MagicMock()
        mock_repo.active_branch = mock_branch
        mock_repo.branches = [mock_branch]
        mock_repo.create_head.return_value = mock_new_branch

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="create", branch="feature")

        assert result["status"] == "success"
        assert "Created and switched" in result["message"]
        assert result["branch"] == "feature"
        assert result["previous_branch"] == "main"
        mock_new_branch.checkout.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_branch_create_already_exists(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create fails when branch exists."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "feature"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_branch]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="create", branch="feature")

        assert "error" in result
        assert "already exists" in result["error"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_branch_create_from_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create from specific branch."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"
        mock_develop = MagicMock()
        mock_develop.name = "develop"

        mock_new_branch = MagicMock()

        # Create a proper mock for branches that supports both iteration and indexing
        mock_branches = MagicMock()
        mock_branches.__iter__ = MagicMock(side_effect=lambda: iter([mock_main, mock_develop]))
        mock_branches.__getitem__ = MagicMock(side_effect=lambda k: {"main": mock_main, "develop": mock_develop}[k])

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = mock_branches
        mock_repo.create_head.return_value = mock_new_branch

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="create", branch="feature", from_branch="develop")

        assert result["status"] == "success"
        assert result["from_branch"] == "develop"

    @pytest.mark.asyncio
    async def test_github_branch_create_from_nonexistent_branch(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create from nonexistent branch fails."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="create", branch="feature", from_branch="nonexistent")

        assert "error" in result
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_create_with_push(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create with push to remote."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "main"

        mock_new_branch = MagicMock()

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"

        mock_repo = MagicMock()
        mock_repo.active_branch = mock_branch
        mock_repo.branches = [mock_branch]
        mock_repo.create_head.return_value = mock_new_branch
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
                    mock_config.GITHUB_TOKEN = "ghp_token"
                    result = await tool(mock_deps, repo="myrepo", action="create", branch="feature", push=True)

        assert result["status"] == "success"
        assert result["pushed"] is True
        assert result["upstream"] == "origin/feature"

    @pytest.mark.asyncio
    async def test_github_branch_create_push_no_token(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create with push fails without token."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "main"

        mock_new_branch = MagicMock()

        mock_repo = MagicMock()
        mock_repo.active_branch = mock_branch
        mock_repo.branches = [mock_branch]
        mock_repo.create_head.return_value = mock_new_branch

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
                    mock_config.GITHUB_TOKEN = None
                    result = await tool(mock_deps, repo="myrepo", action="create", branch="feature", push=True)

        assert result["status"] == "success"
        assert "push_error" in result
        assert "GITHUB_TOKEN" in result["push_error"]

    @pytest.mark.asyncio
    async def test_github_branch_switch_local(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch switch to local branch."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"
        mock_develop = MagicMock()
        mock_develop.name = "develop"

        # Create a proper mock for branches that supports both iteration and indexing
        mock_branches = MagicMock()
        mock_branches.__iter__ = MagicMock(side_effect=lambda: iter([mock_main, mock_develop]))
        mock_branches.__getitem__ = MagicMock(side_effect=lambda k: {"main": mock_main, "develop": mock_develop}[k])

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = mock_branches

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="switch", branch="develop")

        assert result["status"] == "success"
        assert "Switched to" in result["message"]
        assert result["branch"] == "develop"
        assert result["previous_branch"] == "main"

    @pytest.mark.asyncio
    async def test_github_branch_switch_from_remote(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch switch creates local from remote."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_remote_ref = MagicMock()
        mock_remote_ref.name = "origin/feature"

        mock_remote = MagicMock()
        mock_remote.refs = [mock_remote_ref]

        mock_new_branch = MagicMock()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main]
        mock_repo.remotes = [mock_remote]
        mock_repo.create_head.return_value = mock_new_branch

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="switch", branch="feature")

        assert result["status"] == "success"
        assert "from remote" in result["message"]
        assert result["tracking"] == "origin/feature"

    @pytest.mark.asyncio
    async def test_github_branch_switch_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch switch to nonexistent branch."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_remote = MagicMock()
        mock_remote.refs = []

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main]
        mock_repo.remotes = [mock_remote]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="switch", branch="nonexistent")

        assert "error" in result
        assert "does not exist" in result["error"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_branch_delete_success(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch delete action."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"
        mock_feature = MagicMock()
        mock_feature.name = "feature"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main, mock_feature]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="delete", branch="feature")

        assert result["status"] == "success"
        assert "Deleted" in result["message"]
        assert result["deleted_branch"] == "feature"
        mock_repo.delete_head.assert_called_with("feature")

    @pytest.mark.asyncio
    async def test_github_branch_delete_current(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch cannot delete current branch."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="delete", branch="main")

        assert "error" in result
        assert "Cannot delete the current branch" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_delete_not_merged(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch delete unmerged branch without force."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"
        mock_feature = MagicMock()
        mock_feature.name = "feature"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main, mock_feature]
        mock_repo.delete_head.side_effect = GitCommandError("delete", "not fully merged")

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="delete", branch="feature")

        assert "error" in result
        assert "not fully merged" in result["error"]
        assert "hint" in result

    @pytest.mark.asyncio
    async def test_github_branch_delete_force(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch delete with force."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"
        mock_feature = MagicMock()
        mock_feature.name = "feature"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main, mock_feature]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="delete", branch="feature", force=True)

        assert result["status"] == "success"
        mock_repo.delete_head.assert_called_with("feature", force=True)

    @pytest.mark.asyncio
    async def test_github_branch_delete_nonexistent(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch delete nonexistent branch."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main]

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="delete", branch="nonexistent")

        assert "error" in result
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_unknown_action(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch with unknown action."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="unknown")

        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_git_command_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch handles git command errors."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        type(mock_repo).active_branch = PropertyMock(side_effect=GitCommandError("branch", "error"))

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="list")

        assert "error" in result
        assert "Git command failed" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch handles repo name with slash."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "main"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_branch]
        mock_repo.remotes = []

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="owner/myrepo", action="list")

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_github_branch_create_push_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create with push fails (lines 189-190)."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "main"

        mock_new_branch = MagicMock()

        mock_origin = MagicMock()
        mock_origin.url = "https://github.com/owner/repo.git"
        mock_origin.push.side_effect = GitCommandError("push", "remote rejected")

        mock_repo = MagicMock()
        mock_repo.active_branch = mock_branch
        mock_repo.branches = [mock_branch]
        mock_repo.create_head.return_value = mock_new_branch
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
                    mock_config.GITHUB_TOKEN = "ghp_token"
                    result = await tool(mock_deps, repo="myrepo", action="create", branch="feature", push=True)

        assert result["status"] == "success"
        assert "push_error" in result
        assert "Failed to push" in result["push_error"]

    @pytest.mark.asyncio
    async def test_github_branch_delete_other_git_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch delete with other git error (line 261)."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"
        mock_feature = MagicMock()
        mock_feature.name = "feature"

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main, mock_feature]
        mock_repo.delete_head.side_effect = GitCommandError("delete", "some other error")

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="delete", branch="feature")

        assert "error" in result
        assert "Git command failed" in result["error"]

    @pytest.mark.asyncio
    async def test_github_branch_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch handles generic exception (lines 276-278)."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_repo = MagicMock()
        type(mock_repo).active_branch = PropertyMock(side_effect=RuntimeError("Unexpected error"))

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="list")

        assert "error" in result
        assert "Failed to manage branch" in result["error"]

    def test_get_authenticated_url_exception(self) -> None:
        """Test _get_authenticated_url handles exception (lines 79-81)."""
        tool = GitHubBranchTool()

        mock_repo = MagicMock()
        # Make remotes.origin.url raise an exception
        type(mock_repo.remotes).origin = PropertyMock(side_effect=Exception("No remote"))

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test"
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    def test_get_authenticated_url_non_github_url(self) -> None:
        """Test _get_authenticated_url returns None for non-GitHub URLs (branch 72->81)."""
        tool = GitHubBranchTool()

        mock_repo = MagicMock()
        mock_repo.remotes.origin.url = "https://gitlab.com/owner/repo.git"

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
            mock_config.GITHUB_TOKEN = "ghp_test"
            result = tool._get_authenticated_url(mock_repo)

        assert result is None

    @pytest.mark.asyncio
    async def test_github_branch_create_push_no_auth_url(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch create push without auth URL (branch 176->179, 186->192)."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_branch = MagicMock()
        mock_branch.name = "main"

        mock_new_branch = MagicMock()

        mock_origin = MagicMock()
        mock_origin.url = "https://not-github.com/owner/repo.git"  # Non-github URL

        mock_repo = MagicMock()
        mock_repo.active_branch = mock_branch
        mock_repo.branches = [mock_branch]
        mock_repo.create_head.return_value = mock_new_branch
        mock_repo.remotes.origin = mock_origin

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                with patch("reachy_mini_conversation_app.profiles.linus.github_branch.config") as mock_config:
                    mock_config.GITHUB_TOKEN = "ghp_token"
                    # Explicitly patch _get_authenticated_url to return None
                    with patch.object(tool, "_get_authenticated_url", return_value=None):
                        result = await tool(mock_deps, repo="myrepo", action="create", branch="feature", push=True)

        # Should still try to push even without auth URL
        assert result["status"] == "success"
        mock_origin.push.assert_called_once()
        # set_url should NOT have been called since auth_url is None
        mock_origin.set_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_github_branch_switch_multiple_remotes(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test github_branch switch finds branch in second remote (branch 201->200)."""
        tool = GitHubBranchTool()

        repos_dir = tmp_path / "reachy_repos"
        repos_dir.mkdir()
        (repos_dir / "myrepo").mkdir()

        mock_main = MagicMock()
        mock_main.name = "main"

        # First remote has no matching refs
        mock_remote1_ref = MagicMock()
        mock_remote1_ref.name = "origin/main"
        mock_remote1 = MagicMock()
        mock_remote1.refs = [mock_remote1_ref]

        # Second remote has the feature branch
        mock_remote2_ref = MagicMock()
        mock_remote2_ref.name = "upstream/feature"
        mock_remote2 = MagicMock()
        mock_remote2.refs = [mock_remote2_ref]

        mock_new_branch = MagicMock()

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [mock_main]
        mock_repo.remotes = [mock_remote1, mock_remote2]
        mock_repo.create_head.return_value = mock_new_branch

        with patch("reachy_mini_conversation_app.profiles.linus.github_branch.REPOS_DIR", repos_dir):
            with patch("reachy_mini_conversation_app.profiles.linus.github_branch.Repo", return_value=mock_repo):
                result = await tool(mock_deps, repo="myrepo", action="switch", branch="feature")

        assert result["status"] == "success"
        assert result["tracking"] == "upstream/feature"
