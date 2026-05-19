"""Tests for _install_shutdown_signal_handlers in main.py.

Verifies that:
- On Unix, loop.add_signal_handler is called for SIGTERM and SIGINT.
- On Windows (or non-Unix), the function does not crash and uses signal.signal.
- Invoking the installed handler sets the stop_event.
"""

import os
import sys
import signal
import asyncio
import threading
import unittest.mock as mock

import pytest


# Suppress the early welcome WAV before importing main so the boot-time
# subprocess.Popen is never attempted in the test process.
os.environ.setdefault("REACHY_MINI_SKIP_EARLY_WELCOME", "1")

# Import the target at module scope so numpy (transitively imported via
# reachy_mini) is initialised with the REAL sys.platform.  Monkeypatching
# sys.platform during the import would trigger numpy's Linux hugepage check
# on a Windows host and crash with AttributeError: os.uname.
from robot_comic.main import _install_shutdown_signal_handlers  # type: ignore[attr-defined]  # noqa: E402


# ---------------------------------------------------------------------------
# Unix path — loop.add_signal_handler
# ---------------------------------------------------------------------------


class TestInstallHandlersUnix:
    """Exercises the loop.add_signal_handler branch (Unix)."""

    def test_add_signal_handler_called_for_sigterm_and_sigint(self, monkeypatch):
        """On Unix, add_signal_handler must be called for SIGTERM and SIGINT."""
        monkeypatch.setattr(sys, "platform", "linux")

        stop_event = threading.Event()
        loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)

        _install_shutdown_signal_handlers(loop, stop_event)

        called_sigs = {call.args[0] for call in loop.add_signal_handler.call_args_list}
        assert signal.SIGTERM in called_sigs, "SIGTERM handler was not registered"
        assert signal.SIGINT in called_sigs, "SIGINT handler was not registered"

    def test_add_signal_handler_callback_sets_stop_event(self, monkeypatch):
        """The callback passed to add_signal_handler must set stop_event when invoked."""
        monkeypatch.setattr(sys, "platform", "linux")

        stop_event = threading.Event()
        loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)

        _install_shutdown_signal_handlers(loop, stop_event)

        # Find the callback registered for SIGTERM and call it.
        for call in loop.add_signal_handler.call_args_list:
            sig, callback, *cb_args = call.args
            if sig == signal.SIGTERM:
                # The helper is called as: _on_signal(signum)
                callback(*cb_args)
                break
        else:
            pytest.fail("No SIGTERM callback found")

        assert stop_event.is_set(), "stop_event should be set after SIGTERM callback"

    def test_notimplemented_from_add_signal_handler_does_not_raise(self, monkeypatch):
        """If add_signal_handler raises NotImplementedError, the function must not propagate it."""
        monkeypatch.setattr(sys, "platform", "linux")

        stop_event = threading.Event()
        loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)
        loop.add_signal_handler.side_effect = NotImplementedError("not supported")

        # Should complete without raising.
        _install_shutdown_signal_handlers(loop, stop_event)


# ---------------------------------------------------------------------------
# Windows path — signal.signal fallback
# ---------------------------------------------------------------------------


class TestInstallHandlersWindows:
    """Exercises the signal.signal fallback branch (Windows)."""

    def test_does_not_crash_on_windows(self, monkeypatch):
        """On win32 the function must complete without raising."""
        monkeypatch.setattr(sys, "platform", "win32")

        stop_event = threading.Event()
        loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)

        # Stub out signal.signal to avoid actually changing process-level handlers.
        with mock.patch("signal.signal"):
            _install_shutdown_signal_handlers(loop, stop_event)
        # Reaching here means no crash — that is the hard requirement.

    def test_windows_handler_sets_stop_event(self, monkeypatch):
        """On Windows the installed signal.signal callback must set stop_event."""
        monkeypatch.setattr(sys, "platform", "win32")

        stop_event = threading.Event()
        loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)

        captured_handlers: dict = {}

        def _capture_signal(sig, handler):
            captured_handlers[sig] = handler

        with mock.patch("signal.signal", side_effect=_capture_signal):
            _install_shutdown_signal_handlers(loop, stop_event)

        # At least one of SIGTERM / SIGINT should be captured.
        assert len(captured_handlers) >= 1, "No signal handlers were registered"

        # Invoke whichever handler was captured and verify stop_event is set.
        for sig, handler in captured_handlers.items():
            handler(sig, None)  # simulate signal delivery
            break

        assert stop_event.is_set(), "stop_event should be set after Windows signal handler fires"


# ---------------------------------------------------------------------------
# No stop_event — KeyboardInterrupt escalation
# ---------------------------------------------------------------------------


class TestInstallHandlersNoStopEvent:
    """When stop_event is None the handler should raise KeyboardInterrupt."""

    def test_raises_keyboard_interrupt_when_no_stop_event(self, monkeypatch):
        """With stop_event=None, the callback must raise KeyboardInterrupt."""
        monkeypatch.setattr(sys, "platform", "linux")

        loop = mock.MagicMock(spec=asyncio.AbstractEventLoop)

        _install_shutdown_signal_handlers(loop, None)

        for call in loop.add_signal_handler.call_args_list:
            sig, callback, *cb_args = call.args
            if sig == signal.SIGTERM:
                with pytest.raises(KeyboardInterrupt):
                    callback(*cb_args)
                break
        else:
            pytest.fail("No SIGTERM callback found")
