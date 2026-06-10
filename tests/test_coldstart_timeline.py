"""Tests for the cold-start timeline parser/merger (#541)."""

from __future__ import annotations

from robot_comic.observer.coldstart_timeline import (
    rank_gaps,
    build_timeline,
    parse_sink_line,
    render_markdown,
    parse_journal_line,
)


# ---------------------------------------------------------------------------
# Sink (RCSPAN) parsing
# ---------------------------------------------------------------------------


def test_parse_sink_line_extracts_span_event() -> None:
    line = (
        'RCSPAN {"name":"tts.synthesize","trace":"abc","span":"def","parent":null,'
        '"dur_ms":120.5,"status":"UNSET","ts":1781089014917,"attrs":{"x":1}}'
    )
    event = parse_sink_line(line)
    assert event is not None
    assert event["source"] == "span"
    assert event["label"] == "tts.synthesize"
    assert event["epoch_ms"] == 1781089014917
    assert event["detail"]["dur_ms"] == 120.5


def test_parse_sink_line_ignores_non_rcspan_and_garbage() -> None:
    assert parse_sink_line("plain log line") is None
    assert parse_sink_line("RCSPAN {not json}") is None
    assert parse_sink_line("") is None


# ---------------------------------------------------------------------------
# Journal parsing
# ---------------------------------------------------------------------------


def test_parse_journal_line_systemd_started_is_t0_milestone() -> None:
    line = (
        "2026-06-10T11:52:03+0100 ricci systemd[1]: Started reachy-app-autostart.service - Reachy Mini app autostart."
    )
    event = parse_journal_line(line)
    assert event is not None
    assert event["source"] == "journal"
    assert event["label"] == "service_started"
    # 2026-06-10T11:52:03+01:00 == 1781088723000 ms UTC
    assert event["epoch_ms"] == 1781088723000


def test_parse_journal_line_accepts_colon_utc_offset() -> None:
    """ricci's journald emits short-iso with a colon offset (+01:00)."""
    line = (
        "2026-06-10T15:05:25+01:00 reachy-mini systemd[1]: "
        "Started reachy-app-autostart.service - Reachy Mini app autostart (config-driven)."
    )
    event = parse_journal_line(line)
    assert event is not None
    assert event["label"] == "service_started"
    assert event["epoch_ms"] == 1781100325000


def test_parse_journal_line_startup_checkpoint() -> None:
    line = (
        "2026-06-10T11:52:31+0100 ricci python[4242]: "
        "INFO robot_comic.startup_timer Startup: +28.41s first TTS audio frame"
    )
    event = parse_journal_line(line)
    assert event is not None
    assert event["label"] == "first TTS audio frame"
    assert event["detail"]["startup_offset_s"] == 28.41


def test_parse_journal_line_ignores_unrelated() -> None:
    assert parse_journal_line("2026-06-10T11:52:05+0100 ricci python[4242]: chatter") is None
    assert parse_journal_line("not a journal line at all") is None


# ---------------------------------------------------------------------------
# Merge + gap ranking
# ---------------------------------------------------------------------------


def _ev(label: str, epoch_ms: int, source: str = "journal") -> dict:
    return {"source": source, "label": label, "epoch_ms": epoch_ms, "detail": {}}


def test_build_timeline_sorts_and_normalises_to_t0() -> None:
    events = [
        _ev("first sound", 10_500, source="audio"),
        _ev("service_started", 1_000),
        _ev("first TTS audio frame", 9_000),
    ]
    timeline = build_timeline(events, t0_epoch_ms=1_000)
    assert [e["label"] for e in timeline] == [
        "service_started",
        "first TTS audio frame",
        "first sound",
    ]
    assert timeline[0]["t_rel_s"] == 0.0
    assert timeline[1]["t_rel_s"] == 8.0
    assert timeline[2]["t_rel_s"] == 9.5


def test_build_timeline_drops_events_before_t0() -> None:
    events = [_ev("stale", 500), _ev("service_started", 1_000)]
    timeline = build_timeline(events, t0_epoch_ms=1_000)
    assert [e["label"] for e in timeline] == ["service_started"]


def test_rank_gaps_orders_largest_first() -> None:
    timeline = build_timeline(
        [
            _ev("service_started", 0),
            _ev("python_up", 2_000),
            _ev("model_loaded", 22_000),
            _ev("first sound", 23_000),
        ],
        t0_epoch_ms=0,
    )
    gaps = rank_gaps(timeline)
    assert gaps[0]["gap_s"] == 20.0
    assert gaps[0]["from"] == "python_up"
    assert gaps[0]["to"] == "model_loaded"
    assert len(gaps) == 3


def test_render_markdown_contains_timeline_and_gaps() -> None:
    timeline = build_timeline(
        [_ev("service_started", 0), _ev("first sound", 5_000, source="audio")],
        t0_epoch_ms=0,
    )
    md = render_markdown(timeline, rank_gaps(timeline), meta={"host": "ricci"})
    assert "service_started" in md
    assert "first sound" in md
    assert "+5.00s" in md
    assert "ricci" in md
