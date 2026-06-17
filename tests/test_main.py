"""Tests for app-level runtime behavior."""

import time
import argparse
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import reachy_mini_conversation_app.main as main_mod


def test_resolve_app_timeout_prefers_cli_over_env(monkeypatch) -> None:
    """CLI timeout should override the environment value."""
    monkeypatch.setenv(main_mod.APP_TIMEOUT_SECONDS_ENV, "30")
    logger = MagicMock()

    timeout = main_mod._resolve_app_timeout_seconds(
        argparse.Namespace(app_timeout_seconds=12.5),
        logger,
    )

    assert timeout == 12.5
    logger.warning.assert_not_called()


def test_resolve_app_timeout_reads_env(monkeypatch) -> None:
    """Environment timeout should be used when no CLI value is provided."""
    monkeypatch.setenv(main_mod.APP_TIMEOUT_SECONDS_ENV, "45")

    timeout = main_mod._resolve_app_timeout_seconds(
        argparse.Namespace(app_timeout_seconds=None),
        MagicMock(),
    )

    assert timeout == 45.0


def test_resolve_app_timeout_disables_non_positive_values(monkeypatch) -> None:
    """Zero and negative timeout values should disable the watchdog."""
    monkeypatch.delenv(main_mod.APP_TIMEOUT_SECONDS_ENV, raising=False)

    assert main_mod._resolve_app_timeout_seconds(argparse.Namespace(app_timeout_seconds=0), MagicMock()) is None
    assert main_mod._resolve_app_timeout_seconds(argparse.Namespace(app_timeout_seconds=-1), MagicMock()) is None


def test_inactivity_timeout_thread_closes_stream_manager() -> None:
    """The inactivity watchdog should close the stream once activity is too old."""
    handler = SimpleNamespace(last_activity_time=time.monotonic() - 10.0)
    stream_manager = SimpleNamespace(handler=handler, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_seconds=0.01,
        stream_manager=stream_manager,
        fallback_handler=handler,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    assert thread is not None
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()


def test_get_last_activity_time_uses_active_stream_handler() -> None:
    """Dynamic stream managers should report the currently installed handler."""
    fallback_handler = SimpleNamespace(last_activity_time=1.0)
    active_handler = SimpleNamespace(last_activity_time=2.0)
    stream_manager = SimpleNamespace(handler=active_handler)

    assert main_mod._get_last_activity_time(stream_manager, fallback_handler) == 2.0
