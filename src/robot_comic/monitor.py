"""Live TUI monitor for robot_comic turn traces.

Tails the systemd journal (or a log file) for RCSPAN lines emitted by
CompactLineExporter and renders a rolling table of completed turns plus
a live in-flight row that updates as each stage (STT→LLM→TTS) finishes.

Usage:
    robot-comic-monitor                          # tail reachy-app-autostart unit
    robot-comic-monitor --unit my-service        # different systemd unit
    robot-comic-monitor --file /path/to/app.log  # tail a file instead

Keyboard shortcuts (while the monitor is running):
    S     Open the persona-switcher overlay (choose a persona with 1-N,
          confirm with Enter, cancel with Esc).
    Ctrl-C  Quit.
"""

import sys
import json
import time
import argparse
import threading
import subprocess
from typing import Any, Iterator, Optional
from datetime import datetime
from collections import defaultdict
from urllib.error import URLError
from urllib.request import Request, urlopen


try:
    from rich import box
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
except ImportError:
    print("rich is required: uv pip install 'robot_comic[monitor]'", file=sys.stderr)
    raise SystemExit(1)


_JOURNALD_UNIT = "reachy-app-autostart"
_MAX_TURNS = 30

# TUI redraw cadence. 1 Hz is the default — turn events stream in live as RCSPAN
# lines arrive, so the heartbeat redraw only governs spinner / elapsed-time
# animation and benefits from being calm when watching pipeline timing.
_DEFAULT_REFRESH_HZ = 1.0
_MIN_REFRESH_HZ = 0.5
_MAX_REFRESH_HZ = 10.0

# Thresholds (ms) for green / yellow / red colouring per stage
_THRESHOLDS: dict[str, tuple[float, float]] = {
    "stt": (300, 600),
    "llm": (800, 2000),
    "tts": (500, 1500),
    "total": (2000, 4000),
}

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spin() -> str:
    return _SPINNER[int(time.time() * 8) % len(_SPINNER)]


def _service_active(unit: str) -> bool:
    """Return True iff the given systemd unit is currently active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            timeout=1.0,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TurnRecord:
    """Aggregates all spans for one conversational turn."""

    def __init__(self, trace_id: str, root: dict[str, Any], children: list[dict[str, Any]]) -> None:
        """Build a TurnRecord from the root turn span and its buffered children."""
        self.trace_id = trace_id
        self.root = root
        self.children = children
        self.ts = datetime.fromtimestamp(root["ts"] / 1000)

    def _kids(self, name: str) -> list[dict[str, Any]]:
        return [s for s in self.children if s["name"] == name]

    def _sum_ms(self, name: str) -> Optional[float]:
        kids = self._kids(name)
        return sum(s["dur_ms"] for s in kids) if kids else None

    @property
    def stt_ms(self) -> Optional[float]:
        """Total STT inference time."""
        return self._sum_ms("stt.infer")

    @property
    def llm_ms(self) -> Optional[float]:
        """Total LLM time (summed across tool-call chains)."""
        return self._sum_ms("llm.request")

    @property
    def tts_ms(self) -> Optional[float]:
        """Total TTS synthesis time."""
        return self._sum_ms("tts.synthesize")

    @property
    def total_ms(self) -> float:
        """End-to-end turn duration."""
        return float(self.root["dur_ms"])

    @property
    def outcome(self) -> str:
        """Turn outcome attribute."""
        return str(self.root["attrs"].get("turn.outcome", "?"))

    @property
    def mode(self) -> str:
        """Backend mode (openai, chatterbox, gemini, etc.)."""
        return str(self.root["attrs"].get("robot.mode", "?"))

    @property
    def excerpt(self) -> str:
        """Short label: first few words of transcript, or 'greeting' for startup."""
        val = self.root["attrs"].get("turn.excerpt", "")
        if val:
            return str(val)
        return ""

    @property
    def tool_count(self) -> int:
        """Number of tool calls made during this turn."""
        return len(self._kids("tool.execute"))


class SupportingEvent:
    """A synthetic boot-timeline event surfaced as its own monitor row (#301).

    These spans carry ``event.kind=supporting`` and are *not* children of a
    ``turn`` span. They occupy the same chronological lane as full turns but
    render with only ``[ts] [kind] [dur_ms]`` populated.
    """

    def __init__(self, span: dict[str, Any]) -> None:
        """Build from a closed supporting-event span dict (as decoded from RCSPAN)."""
        self.name = str(span.get("name", "event"))
        self.ts_ms: int = int(span.get("ts", 0))
        self.ts = datetime.fromtimestamp(self.ts_ms / 1000)
        # Prefer the externally measured ``event.dur_ms`` (e.g. seconds since
        # process start) over the span's own wall-clock duration, which is
        # effectively zero for synthetic point events.
        attrs = span.get("attrs", {}) or {}
        explicit = attrs.get("event.dur_ms")
        if explicit is None:
            self.dur_ms: float = float(span.get("dur_ms", 0.0))
        else:
            try:
                self.dur_ms = float(explicit)
            except (TypeError, ValueError):
                self.dur_ms = float(span.get("dur_ms", 0.0))


class PendingTurn:
    """Tracks a turn that has started (stt.infer seen) but not yet completed."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.ts = datetime.now()
        self.started_at = time.perf_counter()
        self.stt_ms: Optional[float] = None
        self.llm_ms: float = 0.0
        self.llm_count: int = 0  # completed llm.request spans seen
        self.tts_ms: float = 0.0
        self.tts_count: int = 0  # completed tts.synthesize spans seen
        self.tool_count: int = 0
        self.excerpt: str = ""  # populated from stt.infer turn.excerpt attribute

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000


# ---------------------------------------------------------------------------
# Span buffer
# ---------------------------------------------------------------------------


class SpanBuffer:
    """Accumulates child spans until the root turn span closes the trace."""

    def __init__(self) -> None:
        """Initialise the pending-span store."""
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._inflight: dict[str, PendingTurn] = {}
        # Supporting boot-timeline events (#301) — accumulated in arrival order
        # and rendered on the same chronological lane as full turns.
        self.supporting_events: list[SupportingEvent] = []

    def ingest(self, span: dict[str, Any]) -> Optional[TurnRecord]:
        """Return a completed TurnRecord when the root turn span arrives, else None."""
        trace_id = span["trace"]
        name = span["name"]

        # Supporting events (#301) are top-level (parent is None) but distinguish
        # themselves via the ``event.kind=supporting`` attribute. They never
        # accumulate as children of a turn and are returned via
        # ``supporting_events`` rather than this method's return value.
        attrs = span.get("attrs", {}) or {}
        if attrs.get("event.kind") == "supporting":
            self.supporting_events.append(SupportingEvent(span))
            return None

        if name == "turn" and span["parent"] is None:
            children = self._pending.pop(trace_id, [])
            self._inflight.pop(trace_id, None)
            return TurnRecord(trace_id, span, children)

        self._pending[trace_id].append(span)

        # Update in-flight tracking so the pending row reflects current progress.
        if name == "stt.infer":
            if trace_id not in self._inflight:
                self._inflight[trace_id] = PendingTurn(trace_id)
            self._inflight[trace_id].stt_ms = span.get("dur_ms")
            # Capture transcript excerpt from the completed stt.infer span so the
            # pending row can display it while the outer turn span is still open.
            excerpt = span.get("attrs", {}).get("turn.excerpt", "")
            if excerpt:
                self._inflight[trace_id].excerpt = str(excerpt)
        elif name == "llm.request":
            pt = self._inflight.get(trace_id)
            if pt is not None:
                pt.llm_ms += span.get("dur_ms", 0.0)
                pt.llm_count += 1
        elif name == "tts.synthesize":
            pt = self._inflight.get(trace_id)
            if pt is not None:
                pt.tts_ms += span.get("dur_ms", 0.0)
                pt.tts_count += 1
        elif name == "tool.execute":
            pt = self._inflight.get(trace_id)
            if pt is not None:
                pt.tool_count += 1

        return None

    @property
    def latest_pending(self) -> Optional[PendingTurn]:
        """Return the most-recently-started in-flight turn, if any."""
        if not self._inflight:
            return None
        return max(self._inflight.values(), key=lambda pt: pt.started_at)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(ms: Optional[float], stage: str) -> Text:
    """Format a millisecond value with threshold-based colour."""
    if ms is None:
        return Text("  —  ", style="dim")
    warn, bad = _THRESHOLDS.get(stage, (500, 2000))
    if ms >= bad:
        style = "bold red"
    elif ms >= warn:
        style = "yellow"
    else:
        style = "green"
    label = f"{ms:.0f}ms" if ms < 10_000 else f"{ms / 1000:.1f}s"
    return Text(label.rjust(7), style=style)


def _fmt_spin(ms: Optional[float], active: bool, stage: str) -> Text:
    """Format a stage cell for the in-flight pending row.

    active=True  → stage has started (show spinner while waiting, time when done)
    active=False → stage not yet started (show dim dash)
    """
    if ms is not None and ms > 0:
        return _fmt(ms, stage)
    if active:
        return Text(f"  {_spin()}    ", style="yellow")
    return Text("  —  ", style="dim")


def _fmt_supporting_dur(ms: float) -> Text:
    """Format the duration cell for a supporting-event row in dim cyan."""
    if ms <= 0:
        return Text("    —  ", style="dim cyan")
    if ms < 10_000:
        label = f"{ms:.0f}ms"
    else:
        label = f"{ms / 1000:.1f}s"
    return Text(label.rjust(7), style="dim cyan")


def _add_supporting_row(t: Table, ev: SupportingEvent) -> None:
    """Append a supporting-event row to *t*.

    Only ``Time``, ``What``, and ``Total`` are populated; the other columns
    stay blank so the row reads as a boot-timeline marker rather than a turn.
    The whole row is styled ``dim cyan`` to set it visually apart from real
    conversational turns.
    """
    label = Text(ev.name, style="dim cyan")
    blank = Text("", style="dim cyan")
    t.add_row(
        Text(ev.ts.strftime("%H:%M:%S"), style="dim cyan"),
        label,
        blank,
        blank,
        blank,
        _fmt_supporting_dur(ev.dur_ms),
        blank,
        Text("▪", style="dim cyan"),
    )


def _build_table(
    turns: list[TurnRecord],
    pending: Optional[PendingTurn] = None,
    supporting_events: Optional[list[SupportingEvent]] = None,
) -> Table:
    """Render the most recent turns as a Rich table (newest row first).

    When *supporting_events* is provided, those rows are interleaved with
    *turns* by their ``ts`` so the boot timeline (app.startup, welcome.wav,
    handler.start_up.complete, first_greeting.tts_first_audio …) lines up
    chronologically with subsequent conversational turns (#301).
    """
    t = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style="bold dim",
        row_styles=["", "dim"],
    )
    t.add_column("Time", width=8, no_wrap=True)
    t.add_column("What", width=20, no_wrap=True)
    t.add_column("STT", width=8, justify="right")
    t.add_column("LLM", width=8, justify="right")
    t.add_column("TTS", width=8, justify="right")
    t.add_column("Total", width=8, justify="right")
    t.add_column("Tools", width=5, justify="right")
    t.add_column("", width=1, justify="center")

    if pending is not None:
        sp = _spin()
        stt_done = pending.stt_ms is not None
        llm_started = stt_done  # LLM starts right after STT
        tts_started = pending.llm_count > 0
        # LLM spinner: active while STT is done but no llm.request seen yet,
        # or after the last llm request if tts hasn't started.
        llm_active = llm_started and pending.llm_count == 0
        # Show accumulated LLM time if any requests completed.
        llm_ms = pending.llm_ms if pending.llm_count > 0 else None
        tts_active = tts_started and pending.tts_count == 0
        tts_ms = pending.tts_ms if pending.tts_count > 0 else None

        # Show excerpt from stt.infer span if available, otherwise spinner.
        what_cell: Text
        if pending.excerpt:
            what_cell = Text(pending.excerpt, style="italic yellow")
        else:
            what_cell = Text(f"{sp}", style="bold yellow")
        t.add_row(
            pending.ts.strftime("%H:%M:%S"),
            what_cell,
            _fmt_spin(pending.stt_ms, stt_done, "stt"),
            _fmt_spin(llm_ms, llm_active, "llm"),
            _fmt_spin(tts_ms, tts_active, "tts"),
            _fmt(pending.elapsed_ms, "total"),
            Text(str(pending.tool_count), style="cyan") if pending.tool_count else Text(""),
            Text(sp, style="yellow"),
        )

    # Interleave turns and supporting events by timestamp. Each entry is a
    # (ts_ms, kind, payload) tuple so a stable sort preserves arrival order
    # when timestamps collide.
    merged: list[tuple[int, str, Any]] = []
    for turn in turns:
        merged.append((int(turn.root["ts"]), "turn", turn))
    for ev in supporting_events or []:
        merged.append((ev.ts_ms, "event", ev))
    # Sort ascending then take the last _MAX_TURNS so the most recent rows win
    # when the buffer overflows, and finally render newest-first.
    merged.sort(key=lambda x: x[0])
    merged = merged[-_MAX_TURNS:]

    for _ts, kind, payload in reversed(merged):
        if kind == "event":
            _add_supporting_row(t, payload)
            continue
        turn = payload
        ok_icon = Text("✓", style="green") if turn.outcome == "success" else Text("✗", style="red")
        tools_cell = Text(str(turn.tool_count), style="cyan") if turn.tool_count else Text("")
        excerpt = turn.excerpt or turn.mode
        t.add_row(
            turn.ts.strftime("%H:%M:%S"),
            Text(excerpt, style="italic dim" if excerpt == "greeting" else ""),
            _fmt(turn.stt_ms, "stt"),
            _fmt(turn.llm_ms, "llm"),
            _fmt(turn.tts_ms, "tts"),
            _fmt(turn.total_ms, "total"),
            tools_cell,
            ok_icon,
        )
    return t


# ---------------------------------------------------------------------------
# Log tailing
# ---------------------------------------------------------------------------


def _iter_journald(unit: str) -> Iterator[str]:
    """Stream lines from the systemd journal for a given unit."""
    proc = subprocess.Popen(
        ["journalctl", "-u", unit, "-f", "-o", "cat", "--no-pager"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip()
    finally:
        proc.terminate()


def _iter_file(path: str) -> Iterator[str]:
    """Tail a log file, yielding new lines as they appear."""
    with open(path) as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip()
            else:
                time.sleep(0.05)


def _parse_span(line: str) -> Optional[dict[str, Any]]:
    """Extract a span dict from a line containing an RCSPAN JSON payload."""
    idx = line.find("RCSPAN ")
    if idx == -1:
        return None
    try:
        result: dict[str, Any] = json.loads(line[idx + 7 :])
        return result
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Persona-switcher overlay
# ---------------------------------------------------------------------------

# The app's admin server (LocalStream.init_admin_ui) listens on 7860; the
# reachy_mini daemon listens on 8000 and does not expose /personalities or
# /api/battery. Both endpoints the monitor talks to are app-side.
_ADMIN_BASE_URL = "http://localhost:7860"


def _fetch_personas(base_url: str) -> tuple[list[str], str]:
    """Return ``(choices, current)`` from the admin API.

    Fetches ``GET <base_url>/personalities``.  On error returns empty lists so
    the caller can display a meaningful message rather than crashing.
    """
    try:
        url = f"{base_url}/personalities"
        with urlopen(url, timeout=3.0) as resp:  # noqa: S310
            data: dict[str, Any] = json.loads(resp.read().decode())
        choices: list[str] = list(data.get("choices", []))
        current: str = str(data.get("current", ""))
        return choices, current
    except Exception:
        return [], ""


def _apply_persona(base_url: str, name: str) -> tuple[bool, str]:
    """POST ``{"name": <name>}`` to ``<base_url>/personalities/apply``.

    Returns ``(ok, message)`` where *message* is a short human-readable string.
    """
    try:
        url = f"{base_url}/personalities/apply"
        payload = json.dumps({"name": name}).encode()
        req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")  # noqa: S310
        with urlopen(req, timeout=10.0) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        if data.get("ok"):
            return True, f"Switched to {name!r}"
        return False, str(data.get("error", "unknown error"))
    except Exception as exc:
        return False, str(exc)


def _build_picker_panel(choices: list[str], current: str, selected_idx: Optional[int], status_line: str = "") -> Panel:
    """Build the Rich ``Panel`` for the persona-switcher overlay."""
    t = Table(box=None, show_header=False, show_edge=False, pad_edge=False)
    t.add_column("row", no_wrap=True)

    for i, name in enumerate(choices):
        number = i + 1
        is_active = name == current
        is_selected = selected_idx is not None and i == selected_idx

        if is_selected:
            prefix = Text(f"  {number}. ", style="bold cyan")
        elif is_active:
            prefix = Text(f"▶ {number}. ", style="bold green")
        else:
            prefix = Text(f"  {number}. ", style="dim")

        label = Text(name)
        if is_active:
            label.stylize("bold green")
            label.append("  (active)", style="dim green")
        elif is_selected:
            label.stylize("bold cyan")

        row = Text()
        row.append_text(prefix)
        row.append_text(label)
        t.add_row(row)

    n = len(choices)
    hint_line = Text(f"[1-{n}] choose  ·  [Enter] confirm  ·  [Esc] cancel", style="dim")
    if status_line:
        hint_line = Text(status_line, style="yellow")

    body = Align(t, align="left")
    return Panel(body, title="[bold]Switch persona[/bold]", subtitle=hint_line, border_style="cyan")


class PersonaSwitcher:
    """Overlay state machine for the persona picker.

    Feed key events one at a time via :meth:`handle_key`.  Check
    :attr:`done` to know when to exit overlay mode.  After ``done`` is
    ``True``, read :attr:`apply_name` — it is ``None`` if the user
    cancelled, or the persona name to switch to if the user confirmed.

    Example usage in the render loop::

        switcher = PersonaSwitcher(choices, current)
        while not switcher.done:
            key = inp.poll_key(0.1)
            if key:
                switcher.handle_key(key)
            live.update(switcher.render())
    """

    def __init__(self, choices: list[str], current: str) -> None:
        """Initialise with the available persona names and the active one."""
        self.choices = choices
        self.current = current
        self.selected_idx: Optional[int] = None
        self.done: bool = False
        self.apply_name: Optional[str] = None
        self._status: str = ""

    def handle_key(self, key: str) -> None:
        """Process a single normalised key string from ``MonitorInput.poll_key``."""
        if key == "<esc>":
            self.done = True
            self.apply_name = None
            return
        if key == "<interrupt>":
            self.done = True
            self.apply_name = None
            return
        if key == "<enter>":
            if self.selected_idx is not None and 0 <= self.selected_idx < len(self.choices):
                self.apply_name = self.choices[self.selected_idx]
            self.done = True
            return
        # Digit 1-9 selects a row (0-based index = digit - 1)
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.choices):
                self.selected_idx = idx
            # Out-of-range digits are silently ignored.
            return

    def render(self, status_line: str = "") -> Panel:
        """Render the picker overlay as a Rich ``Panel``."""
        return _build_picker_panel(self.choices, self.current, self.selected_idx, status_line or self._status)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the live TUI monitor."""
    parser = argparse.ArgumentParser(
        description="Live TUI monitor for robot_comic turn traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--unit",
        default=_JOURNALD_UNIT,
        metavar="UNIT",
        help=f"systemd unit to tail (default: {_JOURNALD_UNIT})",
    )
    source.add_argument(
        "--file",
        metavar="PATH",
        help="tail a log file instead of the systemd journal",
    )
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=_DEFAULT_REFRESH_HZ,
        metavar="HZ",
        help=(
            "TUI redraw cadence in Hz (default: %(default)s). Lower values produce "
            "a calmer display when watching turn-by-turn pipeline timing. Key input "
            "remains responsive at ~10 Hz regardless. Clamped to "
            f"[{_MIN_REFRESH_HZ}, {_MAX_REFRESH_HZ}]."
        ),
    )
    args = parser.parse_args()
    refresh_hz = max(_MIN_REFRESH_HZ, min(_MAX_REFRESH_HZ, args.refresh_hz))
    render_interval_s = 1.0 / refresh_hz

    console = Console()
    buffer = SpanBuffer()
    turns: list[TurnRecord] = []

    lines: Iterator[str] = _iter_file(args.file) if args.file else _iter_journald(args.unit)
    source_label = f"file:{args.file}" if args.file else f"journald:{args.unit}"
    # Polling unit name for service status indicator (only relevant for journald mode).
    watch_unit: Optional[str] = args.unit if not args.file else None

    # Service status — polled every ~3 s by the refresh thread.
    _svc_active: list[bool] = [True]  # mutable box so refresh thread can update it
    _svc_last_check: list[float] = [0.0]

    # Battery status — polled every ~30 s from the admin API.
    _battery: list[Optional[dict[str, Any]]] = [None]
    _battery_last_check: list[float] = [0.0]
    _battery_base_url: str = _ADMIN_BASE_URL

    # Current persona — polled every ~30 s from the same admin API used by the
    # S-key overlay. Stored as a single string so the header can render it
    # cheaply alongside battery in the panel subtitle (#303).
    _current_persona: list[Optional[str]] = [None]
    _persona_last_check: list[float] = [0.0]

    def _battery_footer() -> Text:
        """Render a compact battery status line for the panel subtitle."""
        batt = _battery[0]
        if batt is None:
            return Text("")
        source = batt.get("source", "unknown")
        if source == "sim":
            return Text("  battery: sim", style="dim")
        if source == "unknown":
            return Text("")
        percent = batt.get("percent")
        charging = batt.get("charging")
        if percent is None:
            return Text("  battery: --", style="dim")
        if percent < 20:
            style = "bold red"
        elif percent < 50:
            style = "yellow"
        else:
            style = "green"
        charge_mark = " ⚡" if charging else ""
        t = Text(f"  battery: {percent}%{charge_mark}", style=style)
        return t

    def _persona_footer() -> Text:
        """Render a compact current-persona indicator for the panel subtitle."""
        name = _current_persona[0]
        if not name:
            return Text("")
        return Text(f"  persona: {name}", style="cyan")

    def _render() -> Panel:
        pending = buffer.latest_pending

        # Service status indicator
        if watch_unit is not None:
            dot = "● " if _svc_active[0] else "○ "
            dot_style = "green" if _svc_active[0] else "red"
            svc_text = Text(dot, style=dot_style)
            svc_text.append(watch_unit, style="bold" if _svc_active[0] else "dim")
            title_suffix = f"  [dim]{source_label}[/dim]"
        else:
            svc_text = None
            title_suffix = f"  [dim]{source_label}[/dim]"

        body: Text | Table
        supporting = buffer.supporting_events
        if not turns and pending is None and not supporting:
            if watch_unit is not None and not _svc_active[0]:
                body = Text("Service is stopped — start it to see turns.", style="dim italic")
            else:
                body = Text("Waiting for turns… (is ROBOT_INSTRUMENTATION=trace set?)", style="dim italic")
        else:
            body = _build_table(turns, pending, supporting)

        n = len(turns)
        title_parts = "[bold]robot-comic-monitor[/bold]" + title_suffix
        batt_text = _battery_footer()
        persona_text = _persona_footer()
        subtitle_text = Text(f"{n} turn{'s' if n != 1 else ''} recorded", style="dim")
        if persona_text.plain:
            subtitle_text.append_text(persona_text)
        if batt_text.plain:
            subtitle_text.append_text(batt_text)
        subtitle = subtitle_text
        return Panel(
            body,
            title=title_parts,
            subtitle=subtitle,
            border_style="green" if (watch_unit is None or _svc_active[0]) else "red",
        )

    stop_event = threading.Event()

    # Overlay mode flag — set/cleared by _auto_refresh; the main thread only
    # reads it to suppress its own live.update() calls while the overlay is up.
    _overlay_active: list[bool] = [False]

    # Construct MonitorInput on the main thread *before* Live is entered so
    # that tty.cbreak() is called from the main thread.  Rich's Live with
    # screen=True can otherwise race to modify terminal settings, leaving the
    # cbreak setup in an undefined state and causing poll_key() to never
    # deliver keypresses (root cause of #273).
    from robot_comic.monitor_input import MonitorInput

    inp = MonitorInput()

    # Decouple render cadence from key-poll cadence: keys stay responsive at
    # ~10 Hz so the 'S' overlay reacts snappily, but the render heartbeat only
    # fires every ``render_interval_s`` (1 s at the 1 Hz default). Turn events
    # still trigger immediate redraws via the main thread's RCSPAN ingestion.
    _last_render_ts: list[float] = [0.0]

    def _auto_refresh() -> None:
        """Background thread: refresh render + handle keyboard input."""
        try:
            while not stop_event.is_set():
                now = time.time()
                # Re-check service status every ~3 s.
                if watch_unit is not None:
                    if now - _svc_last_check[0] >= 3.0:
                        _svc_active[0] = _service_active(watch_unit)
                        _svc_last_check[0] = now
                # Poll battery status every ~30 s.
                if now - _battery_last_check[0] >= 30.0:
                    try:
                        url = f"{_battery_base_url}/api/battery"
                        with urlopen(url, timeout=2.0) as resp:  # noqa: S310
                            raw_battery = resp.read().decode()
                        _battery[0] = json.loads(raw_battery)
                    except (URLError, OSError, ValueError):
                        pass  # no admin server running — silently skip
                    _battery_last_check[0] = now

                # Poll current persona every ~30 s from the same admin API
                # that powers the S-key overlay (#303). Failures are silent
                # so a stopped app does not spam the header.
                if now - _persona_last_check[0] >= 30.0:
                    try:
                        _choices, _current = _fetch_personas(_battery_base_url)
                        if _current:
                            _current_persona[0] = _current
                    except Exception:
                        pass
                    _persona_last_check[0] = now

                # Poll for a keystroke at ~10 Hz (key responsiveness is independent
                # of redraw cadence).
                key = inp.poll_key(timeout=0.1)

                if key == "s":
                    # Enter persona-switcher overlay.
                    _overlay_active[0] = True
                    _run_persona_overlay(live, inp, stop_event)
                    _overlay_active[0] = False
                elif key == "<interrupt>":
                    stop_event.set()
                    break

                if not _overlay_active[0] and now - _last_render_ts[0] >= render_interval_s:
                    live.update(_render())
                    _last_render_ts[0] = now
        finally:
            pass  # inp is owned by the outer scope; closed there

    def _run_persona_overlay(
        live: Live,
        inp: Any,
        stop_ev: threading.Event,
    ) -> None:
        """Run the persona-switcher overlay until the user confirms or cancels."""
        choices, current = _fetch_personas(_battery_base_url)
        if not choices:
            # Show a brief error and return immediately.
            err_panel = Panel(
                Text("Could not reach admin API — is the app running?", style="red"),
                title="[bold]Switch persona[/bold]",
                border_style="red",
            )
            live.update(err_panel)
            time.sleep(2.0)
            return

        switcher = PersonaSwitcher(choices, current)

        while not switcher.done and not stop_ev.is_set():
            live.update(switcher.render())
            key = inp.poll_key(timeout=0.1)
            if key is not None:
                switcher.handle_key(key)

        if switcher.apply_name is None or stop_ev.is_set():
            # Cancelled — just return.
            return

        # Show "Switching…" while the POST is in-flight.
        live.update(switcher.render(status_line=f"Switching to {switcher.apply_name!r}…"))
        ok, msg = _apply_persona(_battery_base_url, switcher.apply_name)

        result_style = "green" if ok else "red"
        result_panel = Panel(
            Text(msg, style=result_style),
            title="[bold]Switch persona[/bold]",
            border_style=result_style,
        )
        live.update(result_panel)
        time.sleep(1.5)

    try:
        with Live(_render(), console=console, refresh_per_second=refresh_hz, screen=True) as live:
            refresh_thread = threading.Thread(target=_auto_refresh, daemon=True)
            refresh_thread.start()

            for raw in lines:
                span = _parse_span(raw)
                if span is None:
                    continue
                turn = buffer.ingest(span)
                if turn is not None:
                    turns.append(turn)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        inp.close()
