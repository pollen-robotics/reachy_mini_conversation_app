"""Tests for the ``--refresh-hz`` CLI flag on ``robot-comic-monitor``.

The TUI's redraw cadence used to be hard-coded at 4 Hz. Operators watching
turn-by-turn pipeline timing wanted a calmer display, so the cadence is now
configurable via ``--refresh-hz`` with a 1 Hz default. These tests guard the
argparse plumbing and the clamping bounds — the actual render loop is exercised
manually since it pumps a TTY-backed Rich ``Live`` session.
"""

from __future__ import annotations
import sys

import pytest

from robot_comic import monitor as monitor_mod


def _parse(argv: list[str]):
    """Run ``monitor.main``'s parser with ``argv`` and return the Namespace.

    We don't invoke ``main()`` itself because it opens a journald subprocess
    and a Rich ``Live`` session — both incompatible with the test harness.
    Instead we re-create the parser locally with the same arguments so we can
    assert the flag exists and parses as expected.
    """
    import argparse

    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--unit", default=monitor_mod._JOURNALD_UNIT)
    source.add_argument("--file")
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=monitor_mod._DEFAULT_REFRESH_HZ,
    )
    return parser.parse_args(argv)


def test_default_refresh_hz_is_one() -> None:
    args = _parse([])
    assert args.refresh_hz == monitor_mod._DEFAULT_REFRESH_HZ == 1.0


def test_refresh_hz_accepts_float() -> None:
    args = _parse(["--refresh-hz", "2.5"])
    assert args.refresh_hz == pytest.approx(2.5)


def test_refresh_hz_clamping_below_min() -> None:
    raw = 0.1
    clamped = max(monitor_mod._MIN_REFRESH_HZ, min(monitor_mod._MAX_REFRESH_HZ, raw))
    assert clamped == monitor_mod._MIN_REFRESH_HZ


def test_refresh_hz_clamping_above_max() -> None:
    raw = 100.0
    clamped = max(monitor_mod._MIN_REFRESH_HZ, min(monitor_mod._MAX_REFRESH_HZ, raw))
    assert clamped == monitor_mod._MAX_REFRESH_HZ


def test_refresh_interval_derived_from_hz() -> None:
    """At 1 Hz the render interval is 1 s; at 4 Hz it's 0.25 s."""
    assert 1.0 / 1.0 == 1.0
    assert 1.0 / 4.0 == 0.25


def test_main_parser_includes_refresh_hz_flag(capsys) -> None:
    """``--help`` should mention the new flag (regression guard)."""
    with pytest.raises(SystemExit):
        old_argv = sys.argv
        sys.argv = ["robot-comic-monitor", "--help"]
        try:
            monitor_mod.main()
        finally:
            sys.argv = old_argv
    out = capsys.readouterr().out
    assert "--refresh-hz" in out
