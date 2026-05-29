"""Tests for BackgroundToolManager."""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolProgress,
    BackgroundTool,
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_routine(
    tool_name: str = "test_tool",
    result: dict[str, Any] | None = None,
    error: Exception | None = None,
    delay: float = 0.0,
) -> ToolCallRoutine:
    """Create a mock ToolCallRoutine that returns *result* or raises *error*.

    If *delay* > 0, the routine will sleep for that many seconds before
    returning / raising so we can test cancellation and progress.

    Mirrors the contract of ``_dispatch_tool_call`` in core_tools: exceptions
    (including ``CancelledError``) are caught and returned as
    ``{"error": "..."}`` dicts so that ``_run_tool`` never sees a raw raise.
    """
    routine = MagicMock(spec=ToolCallRoutine)
    routine.tool_name = tool_name
    routine.args_json_str = "{}"

    async def _call(manager: BackgroundToolManager) -> dict[str, Any]:
        try:
            if delay:
                await asyncio.sleep(delay)
            if error is not None:
                raise error
            return result or {"ok": True}
        except asyncio.CancelledError:
            return {"error": "Tool cancelled"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    routine.__call__ = _call  # type: ignore[method-assign]
    routine.side_effect = _call
    return routine


# ---------------------------------------------------------------------------
# Model / data-class sanity checks
# ---------------------------------------------------------------------------


class TestToolProgress:
    """Validate ToolProgress construction and bounds."""

    def test_valid_progress(self) -> None:
        """Accept valid progress values and messages."""
        p = ToolProgress(progress=0.5, message="halfway")
        assert p.progress == 0.5
        assert p.message == "halfway"

    def test_bounds(self) -> None:
        """Allow 0.0 and 1.0 as boundary values."""
        assert ToolProgress(progress=0.0).progress == 0.0
        assert ToolProgress(progress=1.0).progress == 1.0

    def test_out_of_bounds_raises(self) -> None:
        """Reject progress values outside [0, 1]."""
        with pytest.raises(Exception):
            ToolProgress(progress=-0.1)
        with pytest.raises(Exception):
            ToolProgress(progress=1.1)


class TestToolNotification:
    """Validate ToolNotification construction."""

    def test_creation(self) -> None:
        """Create a notification and verify its fields."""
        n = ToolNotification(
            id="abc",
            tool_name="my_tool",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"data": 1},
        )
        assert n.id == "abc"
        assert n.status == ToolState.COMPLETED
        assert n.result == {"data": 1}
        assert n.error is None


class TestBackgroundTool:
    """Validate BackgroundTool helpers."""

    def test_tool_id(self) -> None:
        """Verify the composite tool_id property includes started_at."""
        t = BackgroundTool(
            id="123",
            tool_name="weather",
            is_idle_tool_call=False,
            status=ToolState.RUNNING,
        )
        assert t.tool_id == f"weather-123-{t.started_at}"

    def test_get_notification(self) -> None:
        """Convert a BackgroundTool to a ToolNotification."""
        t = BackgroundTool(
            id="1",
            tool_name="t",
            is_idle_tool_call=True,
            status=ToolState.COMPLETED,
            result={"x": 1},
            error=None,
        )
        n = t.get_notification()
        assert isinstance(n, ToolNotification)
        assert n.id == "1"
        assert n.tool_name == "t"
        assert n.is_idle_tool_call is True
        assert n.status == ToolState.COMPLETED
        assert n.result == {"x": 1}


# ---------------------------------------------------------------------------
# BackgroundToolManager
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> BackgroundToolManager:
    """Return a fresh BackgroundToolManager for each test."""
    return BackgroundToolManager()


class TestSetLoop:
    """Verify event-loop assignment via set_loop."""

    @pytest.mark.asyncio
    async def test_set_loop_uses_running_loop(self, manager: BackgroundToolManager) -> None:
        """Default to the current running loop."""
        manager.set_loop()
        assert manager._loop is asyncio.get_running_loop()

    def test_set_loop_explicit(self, manager: BackgroundToolManager) -> None:
        """Accept an explicitly provided loop."""
        loop = asyncio.new_event_loop()
        try:
            manager.set_loop(loop)
            assert manager._loop is loop
        finally:
            loop.close()

    def test_set_loop_creates_new_when_no_running(self, manager: BackgroundToolManager) -> None:
        """When called outside an async context it falls back to a new loop."""
        manager.set_loop()
        assert manager._loop is not None


class TestStartTool:
    """Verify tool registration via start_tool."""

    @pytest.mark.asyncio
    async def test_start_registers_tool(self, manager: BackgroundToolManager) -> None:
        """Register a tool and verify its initial state."""
        routine = _make_routine("greet")
        bg = await manager.start_tool(
            call_id="c1",
            tool_call_routine=routine,
            is_idle_tool_call=False,
        )
        assert bg.tool_name == "greet"
        assert bg.id == "c1"
        assert bg.status == ToolState.RUNNING
        assert manager.get_tool(bg.tool_id) is bg

        # Let the task finish
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_start_with_progress(self, manager: BackgroundToolManager) -> None:
        """Initialize progress tracking when requested."""
        routine = _make_routine("slow", delay=0.1)
        bg = await manager.start_tool(
            call_id="c2",
            tool_call_routine=routine,
            is_idle_tool_call=True,
            with_progress=True,
        )
        assert bg.progress is not None
        assert bg.progress.progress == 0.0
        await asyncio.sleep(0.15)


class TestRunToolLifecycle:
    """Test _run_tool via start_tool (the public entry point)."""

    @pytest.mark.asyncio
    async def test_successful_completion(self, manager: BackgroundToolManager) -> None:
        """Complete a tool and verify result, status, and notification."""
        routine = _make_routine("ok_tool", result={"answer": 42})
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)

        # Wait for the task to finish
        await asyncio.sleep(0.05)

        assert bg.status == ToolState.COMPLETED
        assert bg.result == {"answer": 42}
        assert bg.completed_at is not None
        assert bg.error is None

        # Notification should be queued
        notification = manager._notification_queue.get_nowait()
        assert notification.status == ToolState.COMPLETED

    @pytest.mark.asyncio
    async def test_tool_failure(self, manager: BackgroundToolManager) -> None:
        """Mark a tool as FAILED when it raises an exception."""
        routine = _make_routine("bad_tool", error=ValueError("boom"))
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)

        await asyncio.sleep(0.05)

        assert bg.status == ToolState.FAILED
        assert "ValueError: boom" in (bg.error or "")
        assert bg.completed_at is not None

        notification = manager._notification_queue.get_nowait()
        assert notification.status == ToolState.FAILED

    @pytest.mark.asyncio
    async def test_tool_cancellation(self, manager: BackgroundToolManager) -> None:
        """Cancel a running tool and verify CANCELLED status."""
        routine = _make_routine("long_tool", delay=10.0)
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)

        # Give the task a moment to start, then cancel
        await asyncio.sleep(0.02)
        cancelled = await manager.cancel_tool(bg.tool_id)
        assert cancelled is True

        # Let cancellation propagate
        await asyncio.sleep(0.05)

        assert bg.status == ToolState.CANCELLED
        assert bg.error == "Tool cancelled"
        assert bg.completed_at is not None


class TestUpdateProgress:
    """Verify progress updates on running tools."""

    @pytest.mark.asyncio
    async def test_update_progress_success(self, manager: BackgroundToolManager) -> None:
        """Update progress value and message on a tracked tool."""
        routine = _make_routine("prog", delay=0.5)
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False, with_progress=True)

        ok = await manager.update_progress(bg.tool_id, 0.5, "half done")
        assert ok is True
        assert bg.progress is not None
        assert bg.progress.progress == 0.5
        assert bg.progress.message == "half done"

        # Cancel to clean up
        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_update_progress_clamps(self, manager: BackgroundToolManager) -> None:
        """Clamp out-of-range progress values to [0, 1]."""
        routine = _make_routine("prog", delay=0.5)
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False, with_progress=True)

        await manager.update_progress(bg.tool_id, 1.5)
        assert bg.progress is not None
        assert bg.progress.progress == 1.0

        await manager.update_progress(bg.tool_id, -0.5)
        assert bg.progress.progress == 0.0

        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_update_progress_unknown_tool(self, manager: BackgroundToolManager) -> None:
        """Return False for an unknown tool_id."""
        ok = await manager.update_progress("nonexistent", 0.5)
        assert ok is False

    @pytest.mark.asyncio
    async def test_update_progress_no_tracking(self, manager: BackgroundToolManager) -> None:
        """Return False when progress tracking is disabled."""
        routine = _make_routine("fast", delay=0.5)
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False, with_progress=False)

        ok = await manager.update_progress(bg.tool_id, 0.5)
        assert ok is False

        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)


class TestCancelTool:
    """Verify tool cancellation behaviour."""

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, manager: BackgroundToolManager) -> None:
        """Return False when the tool_id does not exist."""
        result = await manager.cancel_tool("does-not-exist")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_already_completed(self, manager: BackgroundToolManager) -> None:
        """Return True when cancelling an already-completed tool."""
        routine = _make_routine("done")
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.05)  # let it finish
        assert bg.status == ToolState.COMPLETED

        # Cancelling a completed tool should return True (not running, no-op)
        result = await manager.cancel_tool(bg.tool_id)
        assert result is True


class TestTimeoutTools:
    """Verify automatic timeout of long-running tools."""

    @pytest.mark.asyncio
    async def test_timeout_cancels_old_tools(self, manager: BackgroundToolManager) -> None:
        """Cancel tools exceeding max duration."""
        # Use a very short max duration
        manager._max_tool_duration_seconds = 0.01

        routine = _make_routine("slow", delay=10.0)
        await manager.start_tool("c1", routine, is_idle_tool_call=False)

        # Wait longer than the timeout
        await asyncio.sleep(0.05)

        count = await manager.timeout_tools()
        assert count == 1

        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_timeout_ignores_recent_tools(self, manager: BackgroundToolManager) -> None:
        """Leave recent tools untouched."""
        manager._max_tool_duration_seconds = 9999

        routine = _make_routine("fast", delay=10.0)
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)

        count = await manager.timeout_tools()
        assert count == 0

        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)


class TestCleanupTools:
    """Verify cleanup of completed tools from memory."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_completed(self, manager: BackgroundToolManager) -> None:
        """Remove completed tools past the retention window."""
        manager._max_tool_memory_seconds = 0.01

        routine = _make_routine("old")
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.05)
        assert bg.status == ToolState.COMPLETED

        # Wait for the memory retention to expire
        await asyncio.sleep(0.05)

        removed = await manager.cleanup_tools()
        assert removed == 1
        assert manager.get_tool(bg.tool_id) is None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_completed(self, manager: BackgroundToolManager) -> None:
        """Keep recently completed tools."""
        manager._max_tool_memory_seconds = 9999

        routine = _make_routine("recent")
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.05)

        removed = await manager.cleanup_tools()
        assert removed == 0
        assert manager.get_tool(bg.tool_id) is not None

    @pytest.mark.asyncio
    async def test_cleanup_ignores_running(self, manager: BackgroundToolManager) -> None:
        """Never remove still-running tools."""
        manager._max_tool_memory_seconds = 0.0  # immediate expiry

        routine = _make_routine("still_going", delay=10.0)
        bg = await manager.start_tool("c1", routine, is_idle_tool_call=False)

        removed = await manager.cleanup_tools()
        assert removed == 0

        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)


class TestGetters:
    """Verify tool retrieval helpers."""

    @pytest.mark.asyncio
    async def test_get_tool(self, manager: BackgroundToolManager) -> None:
        """Return None for missing tools and the instance for known ones."""
        assert manager.get_tool("nope") is None

        routine = _make_routine("x")
        bg = await manager.start_tool("1", routine, is_idle_tool_call=False)
        assert manager.get_tool(bg.tool_id) is bg
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_get_running_tools(self, manager: BackgroundToolManager) -> None:
        """Return only tools that are still running."""
        r1 = _make_routine("a", delay=10.0)
        r2 = _make_routine("b", delay=10.0)
        r3 = _make_routine("c")  # finishes immediately

        bg1 = await manager.start_tool("1", r1, is_idle_tool_call=False)
        bg2 = await manager.start_tool("2", r2, is_idle_tool_call=False)
        await manager.start_tool("3", r3, is_idle_tool_call=False)
        await asyncio.sleep(0.05)  # let r3 finish

        running = manager.get_running_tools()
        assert len(running) == 2
        names = {t.tool_name for t in running}
        assert names == {"a", "b"}

        # Clean up
        await manager.cancel_tool(bg1.tool_id)
        await manager.cancel_tool(bg2.tool_id)
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_get_all_tools_sorted(self, manager: BackgroundToolManager) -> None:
        """Tools are returned most-recent-first."""
        r1 = _make_routine("first")
        r2 = _make_routine("second")

        await manager.start_tool("1", r1, is_idle_tool_call=False)
        await asyncio.sleep(0.02)  # ensure different started_at
        await manager.start_tool("2", r2, is_idle_tool_call=False)

        await asyncio.sleep(0.05)

        all_tools = manager.get_all_tools()
        assert len(all_tools) == 2
        assert all_tools[0].tool_name == "second"
        assert all_tools[1].tool_name == "first"

    @pytest.mark.asyncio
    async def test_get_all_tools_limit(self, manager: BackgroundToolManager) -> None:
        """Respect the limit parameter on get_all_tools."""
        for i in range(5):
            r = _make_routine(f"t{i}")
            await manager.start_tool(str(i), r, is_idle_tool_call=False)

        await asyncio.sleep(0.05)

        limited = manager.get_all_tools(limit=3)
        assert len(limited) == 3


class TestStartUp:
    """Verify start_up bootstraps background tasks."""

    @pytest.mark.asyncio
    async def test_startup_creates_tasks(self, manager: BackgroundToolManager) -> None:
        """start_up should create the listener and cleanup background tasks."""
        callback = AsyncMock()
        manager.start_up(tool_callbacks=[callback])

        # Start a tool and let it complete — the listener should invoke the callback
        routine = _make_routine("ping")
        await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.1)

        assert callback.call_count == 1
        notification = callback.call_args[0][0]
        assert isinstance(notification, ToolNotification)
        assert notification.status == ToolState.COMPLETED

    @pytest.mark.asyncio
    async def test_startup_multiple_callbacks(self, manager: BackgroundToolManager) -> None:
        """Invoke all registered callbacks on completion."""
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        manager.start_up(tool_callbacks=[cb1, cb2])

        routine = _make_routine("multi")
        await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.1)

        assert cb1.call_count == 1
        assert cb2.call_count == 1


class TestNotificationQueue:
    """Verify notifications are enqueued on tool completion or failure."""

    @pytest.mark.asyncio
    async def test_notifications_queued_on_completion(self, manager: BackgroundToolManager) -> None:
        """Queue a COMPLETED notification with the tool result."""
        routine = _make_routine("notif", result={"v": 1})
        await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.05)

        n = manager._notification_queue.get_nowait()
        assert n.tool_name == "notif"
        assert n.status == ToolState.COMPLETED
        assert n.result == {"v": 1}

    @pytest.mark.asyncio
    async def test_notifications_queued_on_failure(self, manager: BackgroundToolManager) -> None:
        """Queue a FAILED notification with the error message."""
        routine = _make_routine("fail", error=RuntimeError("oops"))
        await manager.start_tool("c1", routine, is_idle_tool_call=False)
        await asyncio.sleep(0.05)

        n = manager._notification_queue.get_nowait()
        assert n.status == ToolState.FAILED
        assert "RuntimeError: oops" in (n.error or "")
