"""Verify that importing robot_comic.utils (the early-startup entry point)
does NOT trigger eager loading of heavy ML libraries.

Context (issue #10): onnxruntime, scipy (spatial/signal), and, in isolation,
numpy must not be pulled in by the top-level import chain that runs before
systemd READY=1 is emitted.  Any of these can cost 1.5-4 s on a CM4.

Strategy: run each check in a subprocess so sys.modules is truly clean.
"""

from __future__ import annotations
import os
import sys
import textwrap
import subprocess
from pathlib import Path

import pytest


# Each test spawns a fresh interpreter via subprocess. They are unavoidably
# slow (~20-25 s each on Windows) and they shell out, so they are both
# `slow` AND `integration` per the test-infra-2 marker policy.
pytestmark = [pytest.mark.slow, pytest.mark.integration]


# The worktree's src directory — we prepend it to PYTHONPATH so subprocesses
# import the *worktree* copy of robot_comic rather than the installed package.
_WORKTREE_SRC = str(Path(__file__).parents[1] / "src")


def _run_check(snippet: str) -> tuple[bool, str]:
    """Execute *snippet* in a fresh interpreter; return (passed, output).

    The worktree src is prepended to PYTHONPATH so editable-install resolution
    does not shadow our local changes.
    """
    code = textwrap.dedent(snippet)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _WORKTREE_SRC + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    passed = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    return passed, output


def test_utils_import_does_not_load_onnxruntime() -> None:
    """Importing robot_comic.utils must not load onnxruntime."""
    passed, output = _run_check(
        """
        import sys
        from robot_comic.utils import (
            CameraVisionInitializationError,
            parse_args,
            setup_logger,
            get_requested_head_tracker,
            initialize_camera_and_vision,
            log_connection_troubleshooting,
        )
        loaded = [k for k in sys.modules if "onnxruntime" in k]
        if loaded:
            print(f"FAIL: onnxruntime loaded: {loaded}")
            sys.exit(1)
        print("OK: onnxruntime not loaded")
        """
    )
    assert passed, f"onnxruntime was imported eagerly during utils import:\\n{output}"


def test_utils_import_does_not_load_scipy_signal() -> None:
    """Importing robot_comic.utils must not load scipy.signal."""
    passed, output = _run_check(
        """
        import sys
        from robot_comic.utils import (
            CameraVisionInitializationError,
            parse_args,
            setup_logger,
            get_requested_head_tracker,
            initialize_camera_and_vision,
            log_connection_troubleshooting,
        )
        loaded = [k for k in sys.modules if "scipy.signal" in k]
        if loaded:
            print(f"FAIL: scipy.signal loaded: {loaded}")
            sys.exit(1)
        print("OK: scipy.signal not loaded")
        """
    )
    assert passed, f"scipy.signal was imported eagerly during utils import:\\n{output}"


def test_utils_import_does_not_load_camera_worker() -> None:
    """camera_worker (numpy + scipy) must not be imported by utils at module level."""
    passed, output = _run_check(
        """
        import sys
        from robot_comic.utils import (
            CameraVisionInitializationError,
            parse_args,
            setup_logger,
            get_requested_head_tracker,
            initialize_camera_and_vision,
            log_connection_troubleshooting,
        )
        loaded = [k for k in sys.modules if "camera_worker" in k]
        if loaded:
            print(f"FAIL: camera_worker loaded: {loaded}")
            sys.exit(1)
        print("OK: camera_worker not loaded")
        """
    )
    assert passed, f"camera_worker was imported eagerly:\\n{output}"


def test_motion_safety_import_does_not_load_scipy() -> None:
    """Importing robot_comic.motion_safety alone must not load scipy.spatial.transform."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.motion_safety
        loaded = [k for k in sys.modules if "scipy.spatial.transform" in k]
        if loaded:
            print(f"FAIL: scipy.spatial.transform loaded at import time: {loaded}")
            sys.exit(1)
        print("OK: scipy.spatial.transform not loaded at import time")
        """
    )
    assert passed, f"scipy.spatial.transform was imported eagerly by motion_safety:\\n{output}"


def test_head_tracking_init_does_not_load_numpy() -> None:
    """Importing robot_comic.vision.head_tracking must not load numpy."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.vision.head_tracking
        loaded = [k for k in sys.modules if k == "numpy"]
        if loaded:
            print(f"FAIL: numpy loaded by head_tracking: {loaded}")
            sys.exit(1)
        print("OK: numpy not loaded by head_tracking __init__")
        """
    )
    assert passed, f"numpy was imported eagerly by vision/head_tracking/__init__.py:\\n{output}"


# ---------------------------------------------------------------------------
# Issue #283 — google.genai must NOT be pulled in at module-import time
# ---------------------------------------------------------------------------


def test_elevenlabs_tts_does_not_load_google_genai_at_import() -> None:
    """Importing elevenlabs_tts must not load google.genai (5.5 s on Pi 5)."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.elevenlabs_tts
        loaded = [k for k in sys.modules if k.startswith("google.genai")]
        if loaded:
            print(f"FAIL: google.genai loaded at import time: {loaded}")
            sys.exit(1)
        print("OK: google.genai not loaded at import time")
        """
    )
    assert passed, f"google.genai was imported eagerly by elevenlabs_tts:\\n{output}"


def test_gemini_tts_does_not_load_google_genai_at_import() -> None:
    """Importing gemini_tts must not load google.genai (5.5 s on Pi 5)."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.gemini_tts
        loaded = [k for k in sys.modules if k.startswith("google.genai")]
        if loaded:
            print(f"FAIL: google.genai loaded at import time: {loaded}")
            sys.exit(1)
        print("OK: google.genai not loaded at import time")
        """
    )
    assert passed, f"google.genai was imported eagerly by gemini_tts:\\n{output}"


def test_gemini_llm_does_not_load_google_genai_at_import() -> None:
    """Importing gemini_llm must not load google.genai (5.5 s on Pi 5)."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.gemini_llm
        loaded = [k for k in sys.modules if k.startswith("google.genai")]
        if loaded:
            print(f"FAIL: google.genai loaded at import time: {loaded}")
            sys.exit(1)
        print("OK: google.genai not loaded at import time")
        """
    )
    assert passed, f"google.genai was imported eagerly by gemini_llm:\\n{output}"


def test_gemini_live_does_not_load_google_genai_at_import() -> None:
    """Importing gemini_live must not load google.genai (5.5 s on Pi 5)."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.gemini_live
        loaded = [k for k in sys.modules if k.startswith("google.genai")]
        if loaded:
            print(f"FAIL: google.genai loaded at import time: {loaded}")
            sys.exit(1)
        print("OK: google.genai not loaded at import time")
        """
    )
    assert passed, f"google.genai was imported eagerly by gemini_live:\\n{output}"
