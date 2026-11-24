import asyncio
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import _space_audio
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_wait_for_quiet_needs_an_unbroken_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Talk restarts the pause; without a busy signal nothing waits."""
    monkeypatch.setattr(_space_audio, "QUIET_BEFORE_PLAYING_S", 0.2)
    monkeypatch.setattr(_space_audio, "_QUIET_POLL_S", 0.01)
    deps = _deps()
    await asyncio.wait_for(_space_audio.wait_for_quiet(deps), timeout=0.05)

    loop = asyncio.get_running_loop()
    start = loop.time()
    # Busy, a short gap, busy again, then quiet for good.
    deps.conversation_busy = lambda: (loop.time() - start) < 0.1 or 0.15 < (loop.time() - start) < 0.25

    await _space_audio.wait_for_quiet(deps)

    assert loop.time() - start >= 0.25 + 0.2
