"""Tests for ``robot_comic.stt_service.__main__`` CLI argument parsing."""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path


def _subprocess_env() -> dict[str, str]:
    """Build an env dict that adds this worktree's src/ to PYTHONPATH."""
    # __file__ is at <worktree>/tests/stt_service/test_cli.py
    # parents[2] is <worktree> regardless of how tests are invoked.
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


def test_main_argparse_help_lists_uds_flag() -> None:
    """``python -m robot_comic.stt_service --help`` lists the --uds flag."""
    result = subprocess.run(
        [sys.executable, "-m", "robot_comic.stt_service", "--help"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    # argparse exits 0 on --help.
    assert result.returncode == 0
    assert "--uds" in result.stdout


def test_main_argparse_missing_uds_exits_nonzero() -> None:
    """Omitting --uds causes a non-zero exit (required argument)."""
    result = subprocess.run(
        [sys.executable, "-m", "robot_comic.stt_service"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode != 0


def test_main_argparse_uds_on_windows_exits_with_clear_message() -> None:
    """On Windows, --uds exits with a clear error about UDS being unsupported."""
    if sys.platform != "win32":
        # Only meaningful on Windows; skip on Linux/macOS.
        return

    result = subprocess.run(
        [sys.executable, "-m", "robot_comic.stt_service", "--uds", "/tmp/test.sock"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "Windows" in combined or "Unix-domain" in combined
