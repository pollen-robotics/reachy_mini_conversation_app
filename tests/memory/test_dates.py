"""Tests for event-date helpers (memory/dates.py)."""

from __future__ import annotations

from reachy_mini_conversation_app.memory.dates import (
    event_date,
    event_dates,
    source_dates,
    parse_log_date,
)


def test_parse_log_date_extracts_date() -> None:
    """A log filename's leading YYYY-MM-DD is parsed; junk returns None."""
    assert parse_log_date("2026-04-17_14-37.log").strftime("%Y-%m-%d") == "2026-04-17"
    assert parse_log_date("2026-04-17_14-37_2.log").strftime("%Y-%m-%d") == "2026-04-17"
    assert parse_log_date("not-a-date.log") is None
    assert parse_log_date(None) is None


def test_event_date_is_latest_conversation_not_created() -> None:
    """event_date uses the most recent source date, ignoring `created`."""
    mem = {
        "created": "2026-06-01T00:00:00Z",
        "sources": ["2026-04-17_14-37.log", "2026-05-05_09-29.log"],
    }
    assert event_date(mem).strftime("%Y-%m-%d") == "2026-05-05"


def test_event_date_falls_back_to_created_without_sources() -> None:
    """With no parseable sources, event_date falls back to created."""
    mem = {"created": "2026-06-01T12:00:00Z", "sources": []}
    assert event_date(mem).strftime("%Y-%m-%d") == "2026-06-01"
    assert event_date({"created": None, "sources": []}) is None


def test_event_dates_are_sorted_unique_strings() -> None:
    """event_dates returns sorted, de-duplicated YYYY-MM-DD strings."""
    mem = {"sources": ["2026-05-05_09-29.log", "2026-04-17_14-37.log", "2026-04-17_18-00.log"]}
    assert event_dates(mem) == ["2026-04-17", "2026-05-05"]


def test_source_dates_skips_unparseable() -> None:
    """source_dates yields only the parseable conversation dates."""
    mem = {"sources": ["2026-04-17_14-37.log", "garbage", "2026-04-18_10-00.log"]}
    assert [d.strftime("%Y-%m-%d") for d in source_dates(mem)] == ["2026-04-17", "2026-04-18"]
