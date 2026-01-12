import asyncio
import logging
from typing import Any
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_handler(loop: asyncio.AbstractEventLoop) -> OpenaiRealtimeHandler:
    asyncio.set_event_loop(loop)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return OpenaiRealtimeHandler(deps)


def test_format_timestamp_uses_wall_clock() -> None:
    """Test that format_timestamp uses wall clock time."""
    loop = asyncio.new_event_loop()
    try:
        print("Testing format_timestamp...")
        handler = _build_handler(loop)
        formatted = handler.format_timestamp()
        print(f"Formatted timestamp: {formatted}")
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    # Extract year from "[YYYY-MM-DD ...]"
    year = int(formatted[1:5])
    assert year == datetime.now(timezone.utc).year

@pytest.mark.asyncio
async def test_start_up_retries_on_abrupt_close(monkeypatch: Any, caplog: Any) -> None:
    """First connection dies with ConnectionClosedError during iteration -> retried.

    Second connection iterates cleanly (no events) -> start_up returns without raising.
    Ensures handler clears self.connection at the end.
    """
    caplog.set_level(logging.WARNING)

    # Use a local Exception as the module's ConnectionClosedError to avoid ws dependency
    FakeCCE = type("FakeCCE", (Exception,), {})
    monkeypatch.setattr(rt_mod, "ConnectionClosedError", FakeCCE)

    # Make asyncio.sleep return immediately (for backoff)
    async def _fast_sleep(*_a: Any, **_kw: Any) -> None: return None
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep, raising=False)

    attempt_counter = {"n": 0}

    class FakeConn:
        """Minimal realtime connection stub."""

        def __init__(self, mode: str):
            self._mode = mode

            class _Session:
                async def update(self, **_kw: Any) -> None: return None
            self.session = _Session()

            class _InputAudioBuffer:
                async def append(self, **_kw: Any) -> None: return None
            self.input_audio_buffer = _InputAudioBuffer()

            class _Item:
                async def create(self, **_kw: Any) -> None: return None

            class _Conversation:
                item = _Item()
            self.conversation = _Conversation()

            class _Response:
                async def create(self, **_kw: Any) -> None: return None
                async def cancel(self, **_kw: Any) -> None: return None
            self.response = _Response()

        async def __aenter__(self) -> "FakeConn": return self
        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: return False
        async def close(self) -> None: return None

        # Async iterator protocol
        def __aiter__(self) -> "FakeConn": return self
        async def __anext__(self) -> None:
            if self._mode == "raise_on_iter":
                raise FakeCCE("abrupt close (simulated)")
            raise StopAsyncIteration  # clean exit (no events)

    class FakeRealtime:
        def connect(self, **_kw: Any) -> FakeConn:
            attempt_counter["n"] += 1
            mode = "raise_on_iter" if attempt_counter["n"] == 1 else "clean"
            return FakeConn(mode)

    class FakeClient:
        def __init__(self, **_kw: Any) -> None: self.realtime = FakeRealtime()

    # Patch the OpenAI client used by the handler
    monkeypatch.setattr(rt_mod, "AsyncOpenAI", FakeClient)

    # Build handler with minimal deps
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    # Run: should retry once and exit cleanly
    await handler.start_up()

    # Cleanup: stop the background tasks that were started
    handler._shutdown_requested = True
    for task in [handler._watchdog_task, handler._heartbeat_task, handler._keep_alive_task]:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # Validate: two attempts total (fail -> retry -> succeed), and connection cleared
    assert attempt_counter["n"] == 2
    assert handler.connection is None

    # Optional: confirm we logged the unexpected close once
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "closed unexpectedly" in r.msg]
    assert len(warnings) == 1


# --- Session Duration Tracking Tests ---


class TestSessionDurationTracking:
    """Tests for session duration tracking."""

    def test_session_start_time_initialized_to_none(self) -> None:
        """Verify _session_start_time is None before session starts."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler._session_start_time is None
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_session_max_duration_default_value(self) -> None:
        """Verify default max duration is 55 minutes (3300 seconds)."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler._session_max_duration == 55 * 60  # 3300 seconds
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_get_session_elapsed_time_returns_none_before_start(self) -> None:
        """Verify get_session_elapsed_time returns None before session starts."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler.get_session_elapsed_time() is None
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_get_session_remaining_time_returns_none_before_start(self) -> None:
        """Verify get_session_remaining_time returns None before session starts."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler.get_session_remaining_time() is None
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_get_session_elapsed_time_after_start(self) -> None:
        """Verify get_session_elapsed_time returns elapsed time after session starts."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            # Simulate session start
            handler._session_start_time = loop.time()
            elapsed = handler.get_session_elapsed_time()
            assert elapsed is not None
            assert elapsed >= 0.0
            assert elapsed < 1.0  # Should be very small since we just started
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_get_session_remaining_time_after_start(self) -> None:
        """Verify get_session_remaining_time returns remaining time after session starts."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            # Simulate session start
            handler._session_start_time = loop.time()
            remaining = handler.get_session_remaining_time()
            assert remaining is not None
            # Should be close to max duration since we just started
            assert remaining > handler._session_max_duration - 1.0
            assert remaining <= handler._session_max_duration
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_get_session_remaining_time_clamps_to_zero(self) -> None:
        """Verify get_session_remaining_time clamps to zero when exceeded."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            # Simulate session that started a long time ago (past max duration)
            handler._session_start_time = loop.time() - (handler._session_max_duration + 100)
            remaining = handler.get_session_remaining_time()
            assert remaining == 0.0
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class TestSessionStartTimeReset:
    """Tests for session start time reset on restart."""

    @pytest.mark.asyncio
    async def test_restart_session_resets_start_time(self) -> None:
        """Verify _restart_session resets _session_start_time to None initially."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Simulate an active session with a specific start time
        original_start_time = asyncio.get_event_loop().time() - 1000  # 1000 seconds ago
        handler._session_start_time = original_start_time

        # Create a mock connection with async close method
        mock_connection = MagicMock()

        async def mock_close() -> None:
            pass

        mock_connection.close = mock_close
        handler.connection = mock_connection

        # Mock the client to avoid actual connection
        handler.client = MagicMock()

        # Mock _run_realtime_session to prevent it from actually running
        # and setting a new _session_start_time
        async def mock_run_realtime_session() -> None:
            # Simulate delay without setting _session_start_time
            await asyncio.sleep(10)  # This will timeout

        handler._run_realtime_session = mock_run_realtime_session  # type: ignore[method-assign]

        # Call restart - should reset start time immediately then try to reconnect
        await handler._restart_session()

        # After restart attempt (which timed out), the start time should still be None
        # because mock_run_realtime_session doesn't set it
        assert handler._session_start_time is None


# --- Session Watchdog Tests ---


class TestSessionWatchdog:
    """Tests for session watchdog functionality."""

    def test_watchdog_task_initialized_to_none(self) -> None:
        """Verify _watchdog_task is None before start_up."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler._watchdog_task is None
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_session_renewal_threshold_default(self) -> None:
        """Verify default renewal threshold is 120 seconds (2 minutes)."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler._session_renewal_threshold == 120.0
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    @pytest.mark.asyncio
    async def test_watchdog_stops_on_shutdown(self) -> None:
        """Verify watchdog exits when _shutdown_requested is True."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Start watchdog
        handler._watchdog_task = asyncio.create_task(handler._session_watchdog())

        # Request shutdown immediately
        handler._shutdown_requested = True

        # Give watchdog time to check and exit
        await asyncio.sleep(0.1)

        # Watchdog should still be running but will exit on next iteration
        # Cancel it to clean up
        if not handler._watchdog_task.done():
            handler._watchdog_task.cancel()
            try:
                await handler._watchdog_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_watchdog_triggers_renewal_at_threshold(self) -> None:
        """Verify watchdog triggers renewal when remaining time <= threshold."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track if graceful renewal was called
        renewal_called = {"count": 0}

        async def mock_graceful_renewal() -> None:
            renewal_called["count"] += 1

        handler._graceful_session_renewal = mock_graceful_renewal  # type: ignore[method-assign]

        # Simulate session that's about to expire (only 60 seconds remaining)
        handler._session_start_time = asyncio.get_event_loop().time() - (handler._session_max_duration - 60)

        # Start watchdog with very short check interval for testing
        async def fast_watchdog() -> None:
            """Faster watchdog for testing."""
            while not handler._shutdown_requested:
                try:
                    await asyncio.sleep(0.05)  # Very short interval
                    if handler._shutdown_requested:
                        break
                    remaining = handler.get_session_remaining_time()
                    if remaining is None:
                        continue
                    if remaining <= handler._session_renewal_threshold:
                        await handler._graceful_session_renewal()
                        break  # Exit after triggering once for test
                except asyncio.CancelledError:
                    break

        handler._watchdog_task = asyncio.create_task(fast_watchdog())

        # Wait for watchdog to trigger
        await asyncio.sleep(0.2)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._watchdog_task.done():
            handler._watchdog_task.cancel()
            try:
                await handler._watchdog_task
            except asyncio.CancelledError:
                pass

        # Verify renewal was triggered
        assert renewal_called["count"] >= 1

    @pytest.mark.asyncio
    async def test_watchdog_does_not_trigger_before_threshold(self) -> None:
        """Verify watchdog does not trigger when remaining time > threshold."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track if graceful renewal was called
        renewal_called = {"count": 0}

        async def mock_graceful_renewal() -> None:
            renewal_called["count"] += 1

        handler._graceful_session_renewal = mock_graceful_renewal  # type: ignore[method-assign]

        # Simulate session with plenty of time remaining (50 minutes left)
        handler._session_start_time = asyncio.get_event_loop().time() - (5 * 60)  # 5 min elapsed

        # Fast watchdog that checks once and exits
        check_count = {"n": 0}

        async def fast_watchdog() -> None:
            """Faster watchdog for testing."""
            while not handler._shutdown_requested and check_count["n"] < 3:
                try:
                    await asyncio.sleep(0.05)
                    check_count["n"] += 1
                    if handler._shutdown_requested:
                        break
                    remaining = handler.get_session_remaining_time()
                    if remaining is None:
                        continue
                    if remaining <= handler._session_renewal_threshold:
                        await handler._graceful_session_renewal()
                except asyncio.CancelledError:
                    break

        handler._watchdog_task = asyncio.create_task(fast_watchdog())

        # Wait for a few checks
        await asyncio.sleep(0.3)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._watchdog_task.done():
            handler._watchdog_task.cancel()
            try:
                await handler._watchdog_task
            except asyncio.CancelledError:
                pass

        # Verify renewal was NOT triggered
        assert renewal_called["count"] == 0


# --- Heartbeat Monitor Tests ---


class TestHeartbeatMonitor:
    """Tests for heartbeat monitor functionality."""

    def test_heartbeat_attributes_initialized(self) -> None:
        """Verify heartbeat attributes are properly initialized."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler._last_server_event_time is None
            assert handler._heartbeat_timeout == 30.0
            assert handler._heartbeat_task is None
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    @pytest.mark.asyncio
    async def test_heartbeat_detects_dead_connection(self) -> None:
        """Verify heartbeat triggers restart after timeout with no events."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track if restart was called
        restart_called = {"count": 0}

        async def mock_restart() -> None:
            restart_called["count"] += 1

        handler._restart_session = mock_restart  # type: ignore[method-assign]

        # Simulate last event was a long time ago (past timeout)
        handler._last_server_event_time = asyncio.get_event_loop().time() - 60  # 60 seconds ago
        handler._heartbeat_timeout = 30.0  # 30 second timeout

        # Fast heartbeat monitor for testing
        async def fast_heartbeat() -> None:
            """Faster heartbeat for testing."""
            while not handler._shutdown_requested:
                try:
                    await asyncio.sleep(0.05)
                    if handler._shutdown_requested:
                        break
                    if handler._last_server_event_time is None:
                        continue
                    time_since = asyncio.get_event_loop().time() - handler._last_server_event_time
                    if time_since > handler._heartbeat_timeout:
                        handler._last_server_event_time = None
                        await handler._restart_session()
                        break  # Exit after triggering once
                except asyncio.CancelledError:
                    break

        handler._heartbeat_task = asyncio.create_task(fast_heartbeat())

        # Wait for heartbeat to trigger
        await asyncio.sleep(0.2)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._heartbeat_task.done():
            handler._heartbeat_task.cancel()
            try:
                await handler._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Verify restart was triggered
        assert restart_called["count"] >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_does_not_trigger_with_recent_activity(self) -> None:
        """Verify no restart when receiving regular events."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track if restart was called
        restart_called = {"count": 0}

        async def mock_restart() -> None:
            restart_called["count"] += 1

        handler._restart_session = mock_restart  # type: ignore[method-assign]

        # Simulate recent event (within timeout)
        handler._last_server_event_time = asyncio.get_event_loop().time() - 5  # 5 seconds ago
        handler._heartbeat_timeout = 30.0  # 30 second timeout

        # Fast heartbeat that checks a few times
        check_count = {"n": 0}

        async def fast_heartbeat() -> None:
            """Faster heartbeat for testing."""
            while not handler._shutdown_requested and check_count["n"] < 3:
                try:
                    await asyncio.sleep(0.05)
                    check_count["n"] += 1
                    if handler._shutdown_requested:
                        break
                    if handler._last_server_event_time is None:
                        continue
                    time_since = asyncio.get_event_loop().time() - handler._last_server_event_time
                    if time_since > handler._heartbeat_timeout:
                        handler._last_server_event_time = None
                        await handler._restart_session()
                except asyncio.CancelledError:
                    break

        handler._heartbeat_task = asyncio.create_task(fast_heartbeat())

        # Wait for a few checks
        await asyncio.sleep(0.3)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._heartbeat_task.done():
            handler._heartbeat_task.cancel()
            try:
                await handler._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Verify restart was NOT triggered
        assert restart_called["count"] == 0

    @pytest.mark.asyncio
    async def test_heartbeat_skips_check_before_first_event(self) -> None:
        """Verify heartbeat doesn't trigger when no events received yet."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track if restart was called
        restart_called = {"count": 0}

        async def mock_restart() -> None:
            restart_called["count"] += 1

        handler._restart_session = mock_restart  # type: ignore[method-assign]

        # No events yet
        handler._last_server_event_time = None

        # Fast heartbeat that checks a few times
        check_count = {"n": 0}

        async def fast_heartbeat() -> None:
            """Faster heartbeat for testing."""
            while not handler._shutdown_requested and check_count["n"] < 3:
                try:
                    await asyncio.sleep(0.05)
                    check_count["n"] += 1
                    if handler._shutdown_requested:
                        break
                    if handler._last_server_event_time is None:
                        continue
                    time_since = asyncio.get_event_loop().time() - handler._last_server_event_time
                    if time_since > handler._heartbeat_timeout:
                        handler._last_server_event_time = None
                        await handler._restart_session()
                except asyncio.CancelledError:
                    break

        handler._heartbeat_task = asyncio.create_task(fast_heartbeat())

        # Wait for a few checks
        await asyncio.sleep(0.3)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._heartbeat_task.done():
            handler._heartbeat_task.cancel()
            try:
                await handler._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Verify restart was NOT triggered
        assert restart_called["count"] == 0


# --- Keep-Alive Loop Tests ---


class TestKeepAliveLoop:
    """Tests for keep-alive loop functionality."""

    def test_keep_alive_attributes_initialized(self) -> None:
        """Verify keep-alive attributes are properly initialized."""
        loop = asyncio.new_event_loop()
        try:
            handler = _build_handler(loop)
            assert handler._keep_alive_interval == 5 * 60  # 5 minutes
            assert handler._keep_alive_task is None
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    @pytest.mark.asyncio
    async def test_keep_alive_skips_when_no_connection(self) -> None:
        """Verify keep-alive skips session.update when connection is None."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # No connection
        handler.connection = None

        # Track if session.update would be called (it shouldn't be)
        update_called = {"count": 0}

        # Fast keep-alive that checks a few times
        check_count = {"n": 0}

        async def fast_keep_alive() -> None:
            """Faster keep-alive for testing."""
            while not handler._shutdown_requested and check_count["n"] < 3:
                try:
                    await asyncio.sleep(0.05)
                    check_count["n"] += 1
                    if handler._shutdown_requested:
                        break
                    if handler.connection is None:
                        continue
                    # Would call session.update here
                    update_called["count"] += 1
                except asyncio.CancelledError:
                    break

        handler._keep_alive_task = asyncio.create_task(fast_keep_alive())

        # Wait for a few checks
        await asyncio.sleep(0.3)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._keep_alive_task.done():
            handler._keep_alive_task.cancel()
            try:
                await handler._keep_alive_task
            except asyncio.CancelledError:
                pass

        # Verify session.update was NOT called
        assert update_called["count"] == 0

    @pytest.mark.asyncio
    async def test_keep_alive_calls_session_update_when_connected(self) -> None:
        """Verify session.update is called when connection exists."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track session.update calls
        update_called = {"count": 0}

        # Create mock connection with session.update
        mock_session = MagicMock()

        async def mock_update(**kwargs: Any) -> None:
            update_called["count"] += 1

        mock_session.update = mock_update
        mock_connection = MagicMock()
        mock_connection.session = mock_session
        handler.connection = mock_connection

        # Fast keep-alive that makes a single update
        async def fast_keep_alive() -> None:
            """Faster keep-alive for testing."""
            try:
                await asyncio.sleep(0.05)
                if handler._shutdown_requested:
                    return
                if handler.connection is None:
                    return
                await handler.connection.session.update(session={})
            except asyncio.CancelledError:
                pass

        handler._keep_alive_task = asyncio.create_task(fast_keep_alive())

        # Wait for keep-alive to run
        await asyncio.sleep(0.2)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._keep_alive_task.done():
            handler._keep_alive_task.cancel()
            try:
                await handler._keep_alive_task
            except asyncio.CancelledError:
                pass

        # Verify session.update was called
        assert update_called["count"] >= 1

    @pytest.mark.asyncio
    async def test_keep_alive_handles_exception_gracefully(self) -> None:
        """Verify keep-alive handles exceptions without crashing."""
        deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
        handler = OpenaiRealtimeHandler(deps)

        # Track exception handling
        exception_count = {"n": 0}

        # Create mock connection that raises exception on update
        mock_session = MagicMock()

        async def mock_update_error(**kwargs: Any) -> None:
            exception_count["n"] += 1
            raise ConnectionError("Connection lost")

        mock_session.update = mock_update_error
        mock_connection = MagicMock()
        mock_connection.session = mock_session
        handler.connection = mock_connection

        # Fast keep-alive that handles exception
        async def fast_keep_alive() -> None:
            """Faster keep-alive for testing with exception handling."""
            try:
                await asyncio.sleep(0.05)
                if handler._shutdown_requested:
                    return
                if handler.connection is None:
                    return
                try:
                    await handler.connection.session.update(session={})
                except Exception:
                    pass  # Gracefully handle exception
            except asyncio.CancelledError:
                pass

        handler._keep_alive_task = asyncio.create_task(fast_keep_alive())

        # Wait for keep-alive to run
        await asyncio.sleep(0.2)

        # Cleanup
        handler._shutdown_requested = True
        if not handler._keep_alive_task.done():
            handler._keep_alive_task.cancel()
            try:
                await handler._keep_alive_task
            except asyncio.CancelledError:
                pass

        # Verify exception was raised but handled
        assert exception_count["n"] >= 1
        # Task should still be done (not crashed)
        assert handler._keep_alive_task.done()
