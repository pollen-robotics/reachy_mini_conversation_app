"""Unit tests for the commit_rules module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from reachy_mini_conversation_app.tools.commit_rules import (
    CheckResult,
    CommitRules,
    AutoFixConfig,
    PreCommitCheck,
    run_command,
    run_auto_fix,
    load_commit_rules,
    format_check_results,
    run_pre_commit_checks,
)


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_creation(self) -> None:
        """Test CheckResult can be created."""
        result = CheckResult(
            name="test",
            command="echo test",
            required=True,
            passed=True,
            output="test output",
            return_code=0,
        )
        assert result.name == "test"
        assert result.command == "echo test"
        assert result.required is True
        assert result.passed is True
        assert result.output == "test output"
        assert result.return_code == 0


class TestPreCommitCheck:
    """Tests for PreCommitCheck dataclass."""

    def test_pre_commit_check_defaults(self) -> None:
        """Test PreCommitCheck has correct defaults."""
        check = PreCommitCheck(name="test", command="echo test")
        assert check.required is True
        assert check.timeout == 300

    def test_pre_commit_check_custom_values(self) -> None:
        """Test PreCommitCheck with custom values."""
        check = PreCommitCheck(name="test", command="echo test", required=False, timeout=60)
        assert check.required is False
        assert check.timeout == 60


class TestAutoFixConfig:
    """Tests for AutoFixConfig dataclass."""

    def test_auto_fix_config_defaults(self) -> None:
        """Test AutoFixConfig has correct defaults."""
        config = AutoFixConfig()
        assert config.enabled is False
        assert config.commands == []

    def test_auto_fix_config_with_commands(self) -> None:
        """Test AutoFixConfig with commands."""
        config = AutoFixConfig(enabled=True, commands=["ruff format .", "ruff check --fix ."])
        assert config.enabled is True
        assert len(config.commands) == 2


class TestCommitRules:
    """Tests for CommitRules dataclass."""

    def test_commit_rules_defaults(self) -> None:
        """Test CommitRules has correct defaults."""
        rules = CommitRules()
        assert rules.pre_commit == []
        assert rules.auto_fix.enabled is False

    def test_commit_rules_from_dict(self) -> None:
        """Test CommitRules.from_dict creates correct structure."""
        data = {
            "pre_commit": [
                {"name": "ruff", "command": "ruff check .", "required": True},
                {"name": "test", "command": "pytest", "required": False, "timeout": 600},
            ],
            "auto_fix": {
                "enabled": True,
                "commands": ["ruff format ."],
            },
        }
        rules = CommitRules.from_dict(data)

        assert len(rules.pre_commit) == 2
        assert rules.pre_commit[0].name == "ruff"
        assert rules.pre_commit[0].required is True
        assert rules.pre_commit[1].timeout == 600
        assert rules.auto_fix.enabled is True
        assert len(rules.auto_fix.commands) == 1

    def test_commit_rules_from_dict_empty(self) -> None:
        """Test CommitRules.from_dict with empty data."""
        rules = CommitRules.from_dict({})
        assert rules.pre_commit == []
        assert rules.auto_fix.enabled is False

    def test_commit_rules_from_dict_defaults(self) -> None:
        """Test CommitRules.from_dict uses defaults for missing fields."""
        data = {
            "pre_commit": [{"name": "test"}],  # Missing command, required, timeout
        }
        rules = CommitRules.from_dict(data)
        assert rules.pre_commit[0].name == "test"
        assert rules.pre_commit[0].command == ""
        assert rules.pre_commit[0].required is True
        assert rules.pre_commit[0].timeout == 300


class TestLoadCommitRules:
    """Tests for load_commit_rules function."""

    def test_load_commit_rules_no_file(self, tmp_path: Path) -> None:
        """Test load_commit_rules returns None when no file exists."""
        result = load_commit_rules(tmp_path)
        assert result is None

    def test_load_commit_rules_yaml(self, tmp_path: Path) -> None:
        """Test load_commit_rules loads .yaml file."""
        rules_dir = tmp_path / ".reachy"
        rules_dir.mkdir()
        rules_file = rules_dir / "commit_rules.yaml"
        rules_file.write_text("""
pre_commit:
  - name: lint
    command: ruff check .
""")
        result = load_commit_rules(tmp_path)
        assert result is not None
        assert len(result.pre_commit) == 1
        assert result.pre_commit[0].name == "lint"

    def test_load_commit_rules_yml(self, tmp_path: Path) -> None:
        """Test load_commit_rules loads .yml file."""
        rules_dir = tmp_path / ".reachy"
        rules_dir.mkdir()
        rules_file = rules_dir / "commit_rules.yml"
        rules_file.write_text("""
pre_commit:
  - name: test
    command: pytest
""")
        result = load_commit_rules(tmp_path)
        assert result is not None
        assert result.pre_commit[0].name == "test"

    def test_load_commit_rules_empty_yaml(self, tmp_path: Path) -> None:
        """Test load_commit_rules handles empty YAML."""
        rules_dir = tmp_path / ".reachy"
        rules_dir.mkdir()
        rules_file = rules_dir / "commit_rules.yaml"
        rules_file.write_text("")

        result = load_commit_rules(tmp_path)
        assert result is None

    def test_load_commit_rules_invalid_yaml(self, tmp_path: Path) -> None:
        """Test load_commit_rules handles invalid YAML."""
        rules_dir = tmp_path / ".reachy"
        rules_dir.mkdir()
        rules_file = rules_dir / "commit_rules.yaml"
        rules_file.write_text("invalid: yaml: content: [")

        result = load_commit_rules(tmp_path)
        assert result is None

    def test_load_commit_rules_io_error(self, tmp_path: Path) -> None:
        """Test load_commit_rules handles IO errors."""
        rules_dir = tmp_path / ".reachy"
        rules_dir.mkdir()
        rules_file = rules_dir / "commit_rules.yaml"
        rules_file.write_text("pre_commit: []")

        with patch.object(Path, "read_text", side_effect=IOError("Read error")):
            result = load_commit_rules(tmp_path)
        assert result is None


class TestRunCommand:
    """Tests for run_command function."""

    def test_run_command_success(self, tmp_path: Path) -> None:
        """Test run_command with successful command."""
        mock_result = MagicMock()
        mock_result.stdout = "success output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = run_command("echo test", tmp_path)

        assert result.passed is True
        assert result.output == "success output"
        assert result.return_code == 0

    def test_run_command_with_stderr(self, tmp_path: Path) -> None:
        """Test run_command combines stdout and stderr."""
        mock_result = MagicMock()
        mock_result.stdout = "stdout"
        mock_result.stderr = "stderr"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = run_command("echo test", tmp_path)

        assert "stdout" in result.output
        assert "stderr" in result.output

    def test_run_command_only_stderr(self, tmp_path: Path) -> None:
        """Test run_command with only stderr."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error output"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = run_command("failing command", tmp_path)

        assert result.passed is False
        assert result.output == "error output"
        assert result.return_code == 1

    def test_run_command_timeout(self, tmp_path: Path) -> None:
        """Test run_command handles timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="test", timeout=30)):
            result = run_command("slow command", tmp_path, timeout=30)

        assert result.passed is False
        assert "timed out" in result.output.lower()
        assert result.return_code == -1

    def test_run_command_exception(self, tmp_path: Path) -> None:
        """Test run_command handles exceptions."""
        with patch("subprocess.run", side_effect=OSError("Command not found")):
            result = run_command("nonexistent", tmp_path)

        assert result.passed is False
        assert "Failed to run" in result.output
        assert result.return_code == -1


class TestRunAutoFix:
    """Tests for run_auto_fix function."""

    def test_run_auto_fix_disabled(self, tmp_path: Path) -> None:
        """Test run_auto_fix returns empty list when disabled."""
        config = AutoFixConfig(enabled=False, commands=["echo test"])
        results = run_auto_fix(tmp_path, config)
        assert results == []

    def test_run_auto_fix_no_commands(self, tmp_path: Path) -> None:
        """Test run_auto_fix returns empty list when no commands."""
        config = AutoFixConfig(enabled=True, commands=[])
        results = run_auto_fix(tmp_path, config)
        assert results == []

    def test_run_auto_fix_success(self, tmp_path: Path) -> None:
        """Test run_auto_fix runs commands successfully."""
        config = AutoFixConfig(enabled=True, commands=["echo fix1", "echo fix2"])

        mock_result = MagicMock()
        mock_result.stdout = "fixed"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            results = run_auto_fix(tmp_path, config)

        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_run_auto_fix_failure(self, tmp_path: Path) -> None:
        """Test run_auto_fix handles command failure."""
        config = AutoFixConfig(enabled=True, commands=["failing command"])

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            results = run_auto_fix(tmp_path, config)

        assert len(results) == 1
        assert results[0].passed is False


class TestRunPreCommitChecks:
    """Tests for run_pre_commit_checks function."""

    def test_run_pre_commit_checks_all_pass(self, tmp_path: Path) -> None:
        """Test run_pre_commit_checks with all passing checks."""
        rules = CommitRules(
            pre_commit=[
                PreCommitCheck(name="lint", command="ruff check ."),
                PreCommitCheck(name="test", command="pytest"),
            ],
            auto_fix=AutoFixConfig(),
        )

        mock_result = MagicMock()
        mock_result.stdout = "success"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            results = run_pre_commit_checks(tmp_path, rules)

        assert results["passed"] is True
        assert len(results["checks"]) == 2
        assert "2/2" in results["summary"]

    def test_run_pre_commit_checks_required_fails(self, tmp_path: Path) -> None:
        """Test run_pre_commit_checks with required check failing."""
        rules = CommitRules(
            pre_commit=[PreCommitCheck(name="lint", command="ruff check .", required=True)],
            auto_fix=AutoFixConfig(),
        )

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "lint error"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            results = run_pre_commit_checks(tmp_path, rules)

        assert results["passed"] is False
        assert "Failed (required)" in results["summary"]

    def test_run_pre_commit_checks_optional_fails(self, tmp_path: Path) -> None:
        """Test run_pre_commit_checks with optional check failing."""
        rules = CommitRules(
            pre_commit=[PreCommitCheck(name="optional", command="echo test", required=False)],
            auto_fix=AutoFixConfig(),
        )

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "optional error"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            results = run_pre_commit_checks(tmp_path, rules)

        assert results["passed"] is True  # Optional failures don't block
        assert "Failed (optional)" in results["summary"]

    def test_run_pre_commit_checks_with_auto_fix(self, tmp_path: Path) -> None:
        """Test run_pre_commit_checks runs auto-fix first."""
        rules = CommitRules(
            pre_commit=[PreCommitCheck(name="test", command="pytest")],
            auto_fix=AutoFixConfig(enabled=True, commands=["ruff format ."]),
        )

        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            results = run_pre_commit_checks(tmp_path, rules, run_auto_fix_first=True)

        assert len(results["auto_fix_results"]) == 1
        assert results["passed"] is True

    def test_run_pre_commit_checks_skip_auto_fix(self, tmp_path: Path) -> None:
        """Test run_pre_commit_checks can skip auto-fix."""
        rules = CommitRules(
            pre_commit=[PreCommitCheck(name="test", command="pytest")],
            auto_fix=AutoFixConfig(enabled=True, commands=["ruff format ."]),
        )

        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            results = run_pre_commit_checks(tmp_path, rules, run_auto_fix_first=False)

        assert len(results["auto_fix_results"]) == 0


class TestFormatCheckResults:
    """Tests for format_check_results function."""

    def test_format_check_results_all_pass(self) -> None:
        """Test format_check_results with all passing."""
        results = {
            "passed": True,
            "checks": [{"name": "lint", "passed": True, "required": True}],
            "auto_fix_results": [],
            "summary": "1/1 checks passed",
        }
        output = format_check_results(results)

        assert "✓" in output
        assert "lint" in output
        assert "All required checks passed" in output

    def test_format_check_results_with_failure(self) -> None:
        """Test format_check_results with failure."""
        results = {
            "passed": False,
            "checks": [
                {"name": "lint", "passed": False, "required": True, "output": "error line 1\nerror line 2"}
            ],
            "auto_fix_results": [],
            "summary": "0/1 checks passed. Failed (required): lint",
        }
        output = format_check_results(results)

        assert "✗" in output
        assert "lint" in output
        assert "Commit blocked" in output
        assert "error line" in output

    def test_format_check_results_with_auto_fix(self) -> None:
        """Test format_check_results with auto-fix results."""
        results = {
            "passed": True,
            "checks": [{"name": "test", "passed": True, "required": True}],
            "auto_fix_results": [{"name": "auto_fix: ruff...", "passed": True, "command": "ruff format ."}],
            "summary": "1/1 checks passed",
        }
        output = format_check_results(results)

        assert "Auto-fix Results" in output
        assert "Pre-commit Checks" in output

    def test_format_check_results_truncates_long_output(self) -> None:
        """Test format_check_results truncates long output."""
        long_output = "\n".join([f"error line {i}" for i in range(20)])
        results = {
            "passed": False,
            "checks": [{"name": "lint", "passed": False, "required": True, "output": long_output}],
            "auto_fix_results": [],
            "summary": "0/1",
        }
        output = format_check_results(results)

        # Should show first 5 lines and ...
        assert "..." in output
        assert "error line 0" in output
        assert "error line 4" in output

    def test_format_check_results_optional(self) -> None:
        """Test format_check_results shows optional indicator."""
        results = {
            "passed": True,
            "checks": [{"name": "optional_test", "passed": True, "required": False}],
            "auto_fix_results": [],
            "summary": "1/1",
        }
        output = format_check_results(results)

        assert "(optional)" in output
