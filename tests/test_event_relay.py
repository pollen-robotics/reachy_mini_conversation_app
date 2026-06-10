"""Tests for the UDP event relay (robot → workstation log sink).

The robot's journald gets vacuumed on the Pi's small disk, so the app can
fire-and-forget each telemetry span as a UDP datagram to a workstation sink
(``ROBOT_EVENT_RELAY=host:port``). UDP embodies the "graceful attempt": if
the sink is down or unreachable nothing blocks, errors, or accumulates on
the robot. The workstation side (``robot-comic-logsink``) appends each
datagram as an ``RCSPAN``-prefixed line so ``robot-comic-monitor --file``
and the MCP observer can read it.
"""

from __future__ import annotations
import json
import socket
from typing import Any
from pathlib import Path

from robot_comic import telemetry
from robot_comic.log_sink import SinkWriter


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
    span.end_time = ts_ms * 1_000_000
    span.status = _Status()
    span.attributes = attrs or {}
    return span


# ---------------------------------------------------------------------------
# UdpEventExporter
# ---------------------------------------------------------------------------


def test_exporter_sends_one_datagram_per_span() -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]

    exporter = telemetry.UdpEventExporter(f"127.0.0.1:{port}")
    exporter.export(
        [
            _make_span("turn", ts_ms=1_000, attrs={"turn.outcome": "success"}),
            _make_span("tts.synthesize", ts_ms=1_001),
        ]
    )

    first = json.loads(rx.recv(65535).decode("utf-8"))
    second = json.loads(rx.recv(65535).decode("utf-8"))
    rx.close()

    assert first["name"] == "turn"
    assert first["attrs"]["turn.outcome"] == "success"
    assert second["name"] == "tts.synthesize"


def test_exporter_never_raises_when_target_unresolvable() -> None:
    exporter = telemetry.UdpEventExporter("no-such-host.invalid:9477")
    # Must not raise — the relay is strictly best-effort.
    exporter.export([_make_span("turn")])
    exporter.export([_make_span("turn")])


def test_exporter_rejects_bad_spec_without_raising() -> None:
    # A malformed ROBOT_EVENT_RELAY value must not take down telemetry init.
    exporter = telemetry.UdpEventExporter("not-a-host-port")
    exporter.export([_make_span("turn")])


# ---------------------------------------------------------------------------
# SinkWriter (workstation side)
# ---------------------------------------------------------------------------


def test_sink_writes_rcspan_prefixed_lines(tmp_path: Path) -> None:
    out = tmp_path / "remote_events.log"
    writer = SinkWriter(str(out))

    writer.write_datagram(b'{"name":"turn","ts":1000}')
    writer.write_datagram(b'{"name":"tool.execute","ts":1001}')

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("RCSPAN ")
    assert json.loads(lines[0][len("RCSPAN ") :])["name"] == "turn"


def test_sink_ignores_undecodable_garbage(tmp_path: Path) -> None:
    out = tmp_path / "remote_events.log"
    writer = SinkWriter(str(out))

    writer.write_datagram(b"\xff\xfe not json at all")
    writer.write_datagram(b'{"name":"turn","ts":1}')

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0][len("RCSPAN ") :])["name"] == "turn"


def test_sink_rotates_past_max_bytes(tmp_path: Path) -> None:
    out = tmp_path / "remote_events.log"
    writer = SinkWriter(str(out), max_bytes=200)

    payload = json.dumps({"name": "turn", "pad": "x" * 80}).encode()
    for _ in range(5):
        writer.write_datagram(payload)

    assert out.exists()
    assert (tmp_path / "remote_events.log.1").exists()
