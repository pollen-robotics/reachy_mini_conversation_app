"""Verify that importing tools/camera.py and tools/roast.py does NOT pull in
``av`` at module load time.

Context (issue #472 PR-1): ``av`` (via ``camera_frame_encoding``) was imported
eagerly at the top of both modules, costing ~0.5–0.8 s on the Pi during cold
boot.  After the fix the import is deferred to the first ``__call__`` execution.

Strategy: each test spawns a fresh subprocess so ``sys.modules`` starts clean.
"""

from __future__ import annotations
import os
import sys
import textwrap
import subprocess
from pathlib import Path

import pytest


# Subprocess tests are slow (fresh interpreter spin-up) and shell out.
pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Prepend the worktree src so subprocesses import *our* copy of robot_comic,
# not whatever is editable-installed in the venv.
_WORKTREE_SRC = str(Path(__file__).parents[1] / "src")


def _run_check(snippet: str) -> tuple[bool, str]:
    """Execute *snippet* in a fresh interpreter; return (passed, output)."""
    code = textwrap.dedent(snippet)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _WORKTREE_SRC + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    passed = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    return passed, output


def test_camera_module_does_not_import_av_at_load() -> None:
    """Importing robot_comic.tools.camera must not load ``av``."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.tools.camera
        loaded = [k for k in sys.modules if k == "av" or k.startswith("av.")]
        if loaded:
            print(f"FAIL: av loaded at import time: {loaded[:10]}")
            sys.exit(1)
        print("OK: av not loaded at module import time")
        """
    )
    assert passed, f"av was imported eagerly by tools/camera.py:\n{output}"


def test_roast_module_does_not_import_av_at_load() -> None:
    """Importing robot_comic.tools.roast must not load ``av``."""
    passed, output = _run_check(
        """
        import sys
        import robot_comic.tools.roast
        loaded = [k for k in sys.modules if k == "av" or k.startswith("av.")]
        if loaded:
            print(f"FAIL: av loaded at import time: {loaded[:10]}")
            sys.exit(1)
        print("OK: av not loaded at module import time")
        """
    )
    assert passed, f"av was imported eagerly by tools/roast.py:\n{output}"
