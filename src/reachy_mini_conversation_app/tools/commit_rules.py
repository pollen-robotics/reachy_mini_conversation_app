"""Commit rules loader and executor for .reachy/commit_rules.yaml."""

import logging
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import field, dataclass

import yaml


logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single pre-commit check."""

    name: str
    command: str
    required: bool
    passed: bool
    output: str
    return_code: int


@dataclass
class PreCommitCheck:
    """A single pre-commit check configuration."""

    name: str
    command: str
    required: bool = True
    timeout: int = 300  # 5 minutes default


@dataclass
class AutoFixConfig:
    """Auto-fix configuration."""

    enabled: bool = False
    commands: List[str] = field(default_factory=list)


@dataclass
class CommitRules:
    """Full commit rules configuration."""

    pre_commit: List[PreCommitCheck] = field(default_factory=list)
    auto_fix: AutoFixConfig = field(default_factory=AutoFixConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitRules":
        """Create CommitRules from a dictionary (parsed YAML)."""
        pre_commit = []
        for check in data.get("pre_commit", []):
            pre_commit.append(PreCommitCheck(
                name=check.get("name", "unnamed"),
                command=check.get("command", ""),
                required=check.get("required", True),
                timeout=check.get("timeout", 300),
            ))

        auto_fix_data = data.get("auto_fix", {})
        auto_fix = AutoFixConfig(
            enabled=auto_fix_data.get("enabled", False),
            commands=auto_fix_data.get("commands", []),
        )

        return cls(pre_commit=pre_commit, auto_fix=auto_fix)


def load_commit_rules(repo_path: Path) -> Optional[CommitRules]:
    """Load commit rules from .reachy/commit_rules.yaml if it exists."""
    rules_file = repo_path / ".reachy" / "commit_rules.yaml"

    if not rules_file.exists():
        # Also check for .yml extension
        rules_file = repo_path / ".reachy" / "commit_rules.yml"
        if not rules_file.exists():
            logger.debug(f"No commit rules found in {repo_path}")
            return None

    try:
        content = rules_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if data is None:
            return None
        rules = CommitRules.from_dict(data)
        logger.info(f"Loaded {len(rules.pre_commit)} pre-commit checks from {rules_file}")
        return rules
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse commit rules: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load commit rules: {e}")
        return None


def run_command(command: str, cwd: Path, timeout: int = 300) -> CheckResult:
    """Run a single command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n" + result.stderr
            else:
                output = result.stderr

        return CheckResult(
            name="",  # Will be set by caller
            command=command,
            required=True,  # Will be set by caller
            passed=result.returncode == 0,
            output=output.strip() if output else "",
            return_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="",
            command=command,
            required=True,
            passed=False,
            output=f"Command timed out after {timeout} seconds",
            return_code=-1,
        )
    except Exception as e:
        return CheckResult(
            name="",
            command=command,
            required=True,
            passed=False,
            output=f"Failed to run command: {str(e)}",
            return_code=-1,
        )


def run_auto_fix(repo_path: Path, auto_fix: AutoFixConfig) -> List[CheckResult]:
    """Run auto-fix commands."""
    results = []

    if not auto_fix.enabled or not auto_fix.commands:
        return results

    logger.info("Running auto-fix commands...")

    for command in auto_fix.commands:
        result = run_command(command, repo_path)
        result.name = f"auto_fix: {command[:30]}..."
        results.append(result)

        if result.passed:
            logger.info(f"Auto-fix passed: {command}")
        else:
            logger.warning(f"Auto-fix failed: {command}")

    return results


def run_pre_commit_checks(
    repo_path: Path,
    rules: CommitRules,
    run_auto_fix_first: bool = True,
) -> Dict[str, Any]:
    """Run all pre-commit checks and return results.

    Returns:
        Dict with:
        - passed: bool - True if all required checks passed
        - checks: List[Dict] - Individual check results
        - auto_fix_results: List[Dict] - Auto-fix command results
        - summary: str - Human-readable summary

    """
    results: Dict[str, Any] = {
        "passed": True,
        "checks": [],
        "auto_fix_results": [],
        "summary": "",
    }

    # Run auto-fix first if enabled
    if run_auto_fix_first and rules.auto_fix.enabled:
        auto_fix_results = run_auto_fix(repo_path, rules.auto_fix)
        results["auto_fix_results"] = [
            {
                "name": r.name,
                "command": r.command,
                "passed": r.passed,
                "output": r.output[:500] if r.output else "",  # Truncate output
            }
            for r in auto_fix_results
        ]

    # Run pre-commit checks
    passed_count = 0
    failed_required = []
    failed_optional = []

    for check in rules.pre_commit:
        logger.info(f"Running check: {check.name} ({check.command})")

        result = run_command(check.command, repo_path, check.timeout)
        result.name = check.name
        result.required = check.required

        check_result = {
            "name": check.name,
            "command": check.command,
            "required": check.required,
            "passed": result.passed,
            "return_code": result.return_code,
            "output": result.output[:1000] if result.output else "",  # Truncate output
        }
        results["checks"].append(check_result)

        if result.passed:
            passed_count += 1
            logger.info(f"Check passed: {check.name}")
        else:
            if check.required:
                failed_required.append(check.name)
                results["passed"] = False
                logger.error(f"Required check failed: {check.name}")
            else:
                failed_optional.append(check.name)
                logger.warning(f"Optional check failed: {check.name}")

    # Build summary
    total = len(rules.pre_commit)
    summary_parts = [f"{passed_count}/{total} checks passed"]

    if failed_required:
        summary_parts.append(f"Failed (required): {', '.join(failed_required)}")
    if failed_optional:
        summary_parts.append(f"Failed (optional): {', '.join(failed_optional)}")

    results["summary"] = ". ".join(summary_parts)

    return results


def format_check_results(results: Dict[str, Any]) -> str:
    """Format check results for display."""
    lines = []

    # Auto-fix results
    if results.get("auto_fix_results"):
        lines.append("## Auto-fix Results")
        for r in results["auto_fix_results"]:
            status = "✓" if r["passed"] else "✗"
            lines.append(f"  {status} {r['name']}")
        lines.append("")

    # Pre-commit checks
    lines.append("## Pre-commit Checks")
    for check in results.get("checks", []):
        status = "✓" if check["passed"] else "✗"
        required = "(required)" if check["required"] else "(optional)"
        lines.append(f"  {status} {check['name']} {required}")

        # Show output for failed checks
        if not check["passed"] and check.get("output"):
            for line in check["output"].split("\n")[:5]:  # First 5 lines
                lines.append(f"      {line}")
            if len(check["output"].split("\n")) > 5:
                lines.append("      ...")

    lines.append("")
    lines.append(f"**{results['summary']}**")

    if not results["passed"]:
        lines.append("")
        lines.append("❌ Commit blocked: required checks failed.")
    else:
        lines.append("")
        lines.append("✅ All required checks passed.")

    return "\n".join(lines)
