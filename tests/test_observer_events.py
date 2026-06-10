"""Tests for Tier-0 of the closing-the-loop observer.

Covers the on-disk event log exporter (``telemetry.JsonlEventExporter`` +
the shared ``_span_to_line`` serializer) and the dependency-free reader
(``robot_comic.observer.events``). The MCP server itself is a thin lazy-import
wrapper around the reader and needs the optional ``mcp`` package, so it isn't
exercised here.
"""

from __future__ import annotations
import json
from typing import Any
from pathlib import Path

from robot_comic import telemetry
from robot_comic.observer import events as obs_events


# ---------------------------------------------------------------------------
# Fake span (mirrors the shape OTel hands the exporter; see
# test_telemetry_housekeeping.test_compact_line_exporter_keeps_new_attributes)
# ---------------------------------------------------------------------------
def _make_span(
    name: str = "turn",
    *,
    ts_ms: int = 1_000,
    attrs: dict[str, Any] | None = None,
) -> Any:
    class _SpanCtx:
        trace_id = 0x1
        span_id = 0x2

    class _Status:
        class status_code:
            name = "OK"

    class _FakeSpan:
        pass

    span = _FakeSpan()
    span.name = name
    span.context = _SpanCtx()
    span.parent = None
    span.start_time = 0
    span.end_time = ts_ms * 1_000_000  # ns; exporter divides by 1e6 for ts
    span.status = _Status()
    span.attributes = attrs or {}
    return span


# ---------------------------------------------------------------------------
# JsonlEventExporter
# ---------------------------------------------------------------------------
def test_exporter_appends_one_json_line_per_span(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    exporter = telemetry.JsonlEventExporter(str(log))

    result = exporter.export(
        [
            _make_span("turn", ts_ms=1_000, attrs={"turn.outcome": "ok"}),
            _make_span("tool.call", ts_ms=1_001, attrs={"tool.name": "dance"}),
        ]
    )

    assert result == telemetry.SpanExportResult.SUCCESS
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["name"] == "turn"
    assert first["ts"] == 1_000
    assert first["attrs"]["turn.outcome"] == "ok"
    assert json.loads(lines[1])["attrs"]["tool.name"] == "dance"


def test_exporter_filters_attrs_to_allowlist(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    telemetry.JsonlEventExporter(str(log)).export(
        [_make_span("tts.synthesize", attrs={"tool.name": "x", "secret.random": "drop"})]
    )
    text = log.read_text(encoding="utf-8")
    assert "tool.name" in text
    assert "secret.random" not in text


def test_exporter_appends_across_exports(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    exporter = telemetry.JsonlEventExporter(str(log))
    exporter.export([_make_span("a")])
    exporter.export([_make_span("b")])
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_exporter_rotates_when_over_max_bytes(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    exporter = telemetry.JsonlEventExporter(str(log), max_bytes=10)

    exporter.export([_make_span("first")])  # writes > 10 bytes
    exporter.export([_make_span("second")])  # triggers rotation, then writes

    backup = tmp_path / "events.jsonl.1"
    assert backup.exists(), "oversized log should rotate to a .1 backup"
    assert json.loads(backup.read_text(encoding="utf-8").strip())["name"] == "first"
    assert json.loads(log.read_text(encoding="utf-8").strip())["name"] == "second"


def test_exporter_empty_batch_is_success_and_no_file(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    assert telemetry.JsonlEventExporter(str(log)).export([]) == telemetry.SpanExportResult.SUCCESS
    assert not log.exists()


def test_span_to_line_matches_compact_exporter_payload() -> None:
    """The JSONL log and the RCSPAN stdout stream must serialize identically —
    they share ``_span_to_line``, and this guards against future drift."""
    span = _make_span("turn", ts_ms=1_234, attrs={"turn.outcome": "ok"})
    line = telemetry._span_to_line(span)
    assert line["name"] == "turn"
    assert line["ts"] == 1_234
    assert line["status"] == "OK"
    assert line["parent"] is None
    assert line["attrs"] == {"turn.outcome": "ok"}


# ---------------------------------------------------------------------------
# Reader: events.read_events / recent_events / summarize_events
# ---------------------------------------------------------------------------
def test_read_events_missing_file_returns_empty(tmp_path: Path) -> None:
    assert obs_events.read_events(str(tmp_path / "nope.jsonl")) == []


def test_read_events_skips_blank_and_corrupt_lines(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"name":"a","ts":1}\n'
        "\n"
        "{not valid json\n"  # torn line mid-write
        '{"name":"b","ts":2}\n',
        encoding="utf-8",
    )
    names = [e["name"] for e in obs_events.read_events(str(log))]
    assert names == ["a", "b"]


def test_recent_events_filters_by_window(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    # ts in epoch ms, now=110_000, window=10s -> cutoff=100_000 (inclusive).
    # 99_000 is older than the window; 109_500 is within it.
    log.write_text(
        '{"name":"old","ts":99000}\n{"name":"new","ts":109500}\n',
        encoding="utf-8",
    )
    got = obs_events.recent_events(str(log), window_s=10.0, now_ms=110_000)
    assert [e["name"] for e in got] == ["new"]


def test_summarize_events_reports_speech_and_tools() -> None:
    events = [
        {"name": "turn", "ts": 5, "attrs": {"turn.outcome": "ok"}},
        {"name": "tts.synthesize", "ts": 6, "attrs": {}},
        {"name": "tool.call", "ts": 7, "attrs": {"tool.name": "play_emotion"}},
        {"name": "tool.call", "ts": 8, "attrs": {"tool.name": "dance"}},
    ]
    summary = obs_events.summarize_events(events)
    assert summary["count"] == 4
    assert summary["spoke"] is True
    assert summary["tools_fired"] == ["dance", "play_emotion"]
    assert summary["latest_ts"] == 8


def test_summarize_events_empty() -> None:
    summary = obs_events.summarize_events([])
    assert summary == {"count": 0, "spoke": False, "tools_fired": [], "latest_ts": None}


def test_read_events_tolerates_rcspan_prefixed_lines(tmp_path: Path) -> None:
    """The relay sink (robot-comic-logsink) writes RCSPAN-prefixed lines so the
    monitor can tail them; the reader must accept that file as an event source
    (mixed with raw JSONL lines)."""
    from robot_comic.observer.events import read_events

    log = tmp_path / "relayed_events.log"
    log.write_text(
        'RCSPAN {"name":"turn","ts":1000,"attrs":{}}\n{"name":"tool.execute","ts":2000,"attrs":{}}\n',
        encoding="utf-8",
    )
    events = read_events(str(log))
    assert [e["name"] for e in events] == ["turn", "tool.execute"]
