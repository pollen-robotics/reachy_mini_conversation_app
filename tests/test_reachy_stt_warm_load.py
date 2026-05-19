"""Tests for scripts/reachy-stt-warm-load.sh.

Approach
--------
Bash scripts are awkward to exercise purely via subprocess because the
thing-under-test (the curl call) requires either a live socket or a mock.
We use a **curl shim** approach: we write a tiny fake ``curl`` script to a
temporary directory and prepend that directory to PATH before invoking the
warm-load script.  The shim records the arguments it received to a file so
the test can assert "curl was / was not called" and "curl was called with
--unix-socket …".

When the shim needs to simulate a curl failure, we make it ``exit 1``.

All tests are skipped on Windows (the script uses bash 4+ features and UDS
paths that don't exist on Windows; CI runs the Linux matrix for coverage).
"""

from __future__ import annotations
import os
import sys
import stat
import textwrap
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Marker applied to every test — skip on Windows
# ---------------------------------------------------------------------------
_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="reachy-stt-warm-load.sh is Linux/POSIX-only",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reachy-stt-warm-load.sh"


def _make_env_file(tmp_path: Path, content: str) -> Path:
    """Write a .env file and return its path."""
    env_file = tmp_path / ".env"
    env_file.write_text(textwrap.dedent(content))
    return env_file


def _make_curl_shim(tmp_path: Path, *, exit_code: int = 0) -> tuple[Path, Path]:
    """
    Write a fake ``curl`` executable to *tmp_path*.

    The shim records its arguments to ``<tmp_path>/curl_args.txt`` then exits
    with *exit_code*.

    Returns (shim_dir, args_file).
    """
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir(exist_ok=True)
    args_file = tmp_path / "curl_args.txt"

    shim = shim_dir / "curl"
    shim.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            echo "$@" >> {args_file}
            exit {exit_code}
            """
        )
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return shim_dir, args_file


# ---------------------------------------------------------------------------
# The script hard-codes ENV_FILE and SOCKET paths; we inject test values by
# prepending variable overrides and piping the combined source to bash -s.
# This avoids disk writes and doesn't require the script to support env-var
# injection natively.
# ---------------------------------------------------------------------------


def _run_patched(
    env_file: Path | None,
    *,
    shim_dir: Path | None = None,
    socket_path: str = "/tmp/reachy-stt-test.sock",
) -> subprocess.CompletedProcess[str]:
    """
    Execute the warm-load script with ENV_FILE and SOCKET overridden.

    We prepend two variable assignments to the script source and pipe the
    combined text to bash, which is the cleanest way to inject test values
    without modifying the script on disk.
    """
    script_source = _SCRIPT.read_text()

    overrides = [
        f'ENV_FILE="{env_file if env_file is not None else "/nonexistent/path/.env"}"',
        f'SOCKET="{socket_path}"',
    ]
    patched = "\n".join(overrides) + "\n" + script_source

    env = os.environ.copy()
    if shim_dir is not None:
        env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"

    return subprocess.run(
        ["bash", "-s"],
        input=patched,
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWarmLoadSkip:
    """Cases where /preload should NOT be called."""

    @_posix_only
    def test_no_env_file_exits_zero(self, tmp_path: Path) -> None:
        """Missing .env → skip, exit 0."""
        result = _run_patched(env_file=None)
        assert result.returncode == 0
        assert "not found" in result.stderr or "skipping" in result.stderr

    @_posix_only
    def test_remote_backend_skips_preload(self, tmp_path: Path) -> None:
        """gemini_live_input backend → skip, exit 0, no curl call."""
        env_file = _make_env_file(
            tmp_path,
            """\
            # My robot config
            REACHY_MINI_AUDIO_OUTPUT_BACKEND=elevenlabs
            REACHY_MINI_AUDIO_INPUT_BACKEND=gemini_live_input
            """,
        )
        shim_dir, args_file = _make_curl_shim(tmp_path)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert "not warm-loading" in result.stderr
        assert "gemini_live_input" in result.stderr
        # curl shim must NOT have been invoked
        assert not args_file.exists(), "curl should not have been called"

    @_posix_only
    def test_absent_key_skips_preload(self, tmp_path: Path) -> None:
        """.env with unrelated keys → no backend configured → skip, exit 0."""
        env_file = _make_env_file(
            tmp_path,
            """\
            REACHY_MINI_AUDIO_OUTPUT_BACKEND=chatterbox
            OPENAI_API_KEY=sk-fake
            """,
        )
        shim_dir, args_file = _make_curl_shim(tmp_path)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert not args_file.exists(), "curl should not have been called"

    @_posix_only
    def test_openai_realtime_skips_preload(self, tmp_path: Path) -> None:
        """openai_realtime_output backend → skip, exit 0."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND=openai_realtime_output\n",
        )
        shim_dir, args_file = _make_curl_shim(tmp_path)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert not args_file.exists(), "curl should not have been called"


class TestWarmLoadAttempted:
    """Cases where /preload SHOULD be attempted."""

    @_posix_only
    def test_faster_whisper_triggers_preload(self, tmp_path: Path) -> None:
        """faster_whisper backend → curl called with /preload."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper\n",
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists(), "curl shim should have been called"
        recorded = args_file.read_text()
        assert "/preload" in recorded
        assert "--unix-socket" in recorded

    @_posix_only
    def test_local_stt_triggers_preload(self, tmp_path: Path) -> None:
        """local_stt backend → curl called with /preload."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND=local_stt\n",
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists()

    @_posix_only
    def test_moonshine_triggers_preload(self, tmp_path: Path) -> None:
        """moonshine backend → curl called with /preload."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND=moonshine\n",
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists()

    @_posix_only
    def test_curl_failure_still_exits_zero(self, tmp_path: Path) -> None:
        """Even when curl exits non-zero, the script must exit 0."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper\n",
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=1)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        # Failure should be logged
        assert "failed" in result.stderr or "exit" in result.stderr

    @_posix_only
    def test_curl_receives_unix_socket_path(self, tmp_path: Path) -> None:
        """curl is called with ``--unix-socket <socket_path>``."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper\n",
        )
        socket_path = str(tmp_path / "test.sock")
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir, socket_path=socket_path)

        assert result.returncode == 0
        assert args_file.exists()
        recorded = args_file.read_text()
        assert socket_path in recorded


class TestEnvFileParsing:
    """Robustness of .env parsing logic."""

    @_posix_only
    def test_comments_and_blank_lines_tolerated(self, tmp_path: Path) -> None:
        """Comments and blank lines in .env don't break parsing."""
        env_file = _make_env_file(
            tmp_path,
            """\

            # This is a comment
            # Another comment

            SOME_OTHER_KEY=value

            # STT backend
            REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper

            TRAILING_KEY=foo
            """,
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists(), "curl should be called for faster_whisper"

    @_posix_only
    def test_double_quoted_value_parsed(self, tmp_path: Path) -> None:
        """Double-quoted values are unquoted before comparison."""
        env_file = _make_env_file(
            tmp_path,
            'REACHY_MINI_AUDIO_INPUT_BACKEND="faster_whisper"\n',
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists()

    @_posix_only
    def test_single_quoted_value_parsed(self, tmp_path: Path) -> None:
        """Single-quoted values are unquoted before comparison."""
        env_file = _make_env_file(
            tmp_path,
            "REACHY_MINI_AUDIO_INPUT_BACKEND='moonshine'\n",
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists()

    @_posix_only
    def test_legacy_pipeline_mode_fallback(self, tmp_path: Path) -> None:
        """REACHY_MINI_PIPELINE_MODE is used when AUDIO_INPUT_BACKEND is absent."""
        env_file = _make_env_file(
            tmp_path,
            """\
            # Legacy config (before composable pipeline)
            REACHY_MINI_PIPELINE_MODE=faster_whisper
            """,
        )
        shim_dir, args_file = _make_curl_shim(tmp_path, exit_code=0)
        result = _run_patched(env_file, shim_dir=shim_dir)

        assert result.returncode == 0
        assert args_file.exists(), "legacy PIPELINE_MODE=faster_whisper should trigger preload"
