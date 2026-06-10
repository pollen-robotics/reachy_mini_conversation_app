"""Tests for the bonk sweep (#536): daemon-played moves, witnessed one by one.

The daemon trigger (ssh+curl) and mic capture are seams; tests drive
``sweep_move``/``build_plan``/``write_report`` with fakes.
"""

from __future__ import annotations
import json
from typing import Any
from pathlib import Path

from robot_comic.observer import bonk_sweep
from robot_comic.observer.bonk_sweep import DATASETS, build_plan, sweep_move, write_report


class _FakeTrigger:
    def __init__(self, moves: dict[str, list[str]], running_polls: int = 2) -> None:
        self._moves = moves
        self._running_polls = running_polls
        self.played: list[tuple[str, str]] = []

    def motors_mode(self) -> str:
        return "enabled"

    def list_moves(self, dataset: str) -> list[str]:
        return sorted(self._moves[dataset])

    def play(self, dataset: str, move: str) -> str:
        self.played.append((dataset, move))
        return f"uuid-{move}"

    def running(self) -> list[str]:
        if self._running_polls > 0:
            self._running_polls -= 1
            return [f"uuid-{m}" for _, m in self.played[-1:]]
        return []


class _FakeCapture:
    def __init__(self, detections: list[dict[str, Any]] | None = None) -> None:
        self.detections = detections or []
        self.pcm = [b"\x00\x00" * 800]
        self.error: str | None = None

    def __enter__(self) -> "_FakeCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def test_build_plan_enumerates_requested_datasets() -> None:
    trigger = _FakeTrigger({DATASETS["emotions"]: ["sad1", "amazed1"], DATASETS["dances"]: ["twist"]})
    plan = build_plan(trigger, ["emotions", "dances"])
    assert plan == [
        (DATASETS["emotions"], "amazed1"),
        (DATASETS["emotions"], "sad1"),
        (DATASETS["dances"], "twist"),
    ]


def test_build_plan_only_filter() -> None:
    trigger = _FakeTrigger({DATASETS["emotions"]: ["sad1", "amazed1", "rage1"]})
    plan = build_plan(trigger, ["emotions"], only=["rage1"])
    assert plan == [(DATASETS["emotions"], "rage1")]


def test_sweep_move_clean_run_records_entry() -> None:
    trigger = _FakeTrigger({DATASETS["emotions"]: ["sad1"]})
    detection = {"ts_ms": 500, "peak_dbfs": -4.2, "onset_db": 40.0, "hf_ratio": 0.05, "confirm": "decay"}
    entry = sweep_move(
        trigger,
        lambda: _FakeCapture([detection]),  # type: ignore[arg-type]
        DATASETS["emotions"],
        "sad1",
        max_wait_s=2.0,
        tail_s=0.0,
        poll_s=0.0,
    )
    assert trigger.played == [(DATASETS["emotions"], "sad1")]
    assert entry["move"] == "sad1"
    assert entry["detect_count"] == 1
    assert entry["max_peak_dbfs"] == -4.2
    assert entry["duration_s"] >= 0


def test_sweep_move_play_failure_is_recorded_not_raised() -> None:
    class _BrokenTrigger(_FakeTrigger):
        def play(self, dataset: str, move: str) -> str:
            raise RuntimeError("daemon down")

    entry = sweep_move(
        _BrokenTrigger({DATASETS["emotions"]: ["sad1"]}),
        lambda: _FakeCapture(),  # type: ignore[arg-type]
        DATASETS["emotions"],
        "sad1",
        max_wait_s=1.0,
        tail_s=0.0,
    )
    assert "play failed" in entry["error"]
    assert "detect_count" not in entry


def test_write_report_orders_flagged_first(tmp_path: Path) -> None:
    entries = [
        {"dataset": DATASETS["emotions"], "move": "calm1", "duration_s": 3.0, "detect_count": 0},
        {
            "dataset": DATASETS["emotions"],
            "move": "rage1",
            "duration_s": 5.0,
            "detect_count": 2,
            "max_peak_dbfs": -2.0,
        },
    ]
    path = tmp_path / "report.md"
    write_report(entries, str(path))
    text = path.read_text(encoding="utf-8")
    assert text.index("rage1") < text.index("calm1")
    assert "2 moves swept, 1 flagged" in text


def test_main_jsonl_roundtrip_strips_capture(tmp_path: Path, monkeypatch) -> None:
    """The per-move JSONL must serialize (no live capture object in the entry)."""
    detection = {"ts_ms": 1, "peak_dbfs": -3.0, "onset_db": 41.0, "hf_ratio": 0.02, "confirm": "decay"}
    entry = sweep_move(
        _FakeTrigger({DATASETS["emotions"]: ["sad1"]}),
        lambda: _FakeCapture([detection]),  # type: ignore[arg-type]
        DATASETS["emotions"],
        "sad1",
        max_wait_s=1.0,
        tail_s=0.0,
        poll_s=0.0,
    )
    entry.pop("_capture")
    json.dumps(entry)  # must not raise


def test_datasets_registry_names() -> None:
    assert bonk_sweep.DATASETS["emotions"].endswith("emotions-library")
    assert bonk_sweep.DATASETS["dances"].endswith("dances-library")


def test_write_review_page_lists_flagged_with_audio_and_verdicts(tmp_path: Path) -> None:
    entries = [
        {"dataset": DATASETS["emotions"], "move": "calm1", "duration_s": 3.0, "detect_count": 0},
        {
            "dataset": DATASETS["dances"],
            "move": "grid_snap",
            "duration_s": 5.0,
            "detect_count": 2,
            "max_peak_dbfs": -1.0,
            "detections": [
                {"ts_ms": 1000, "peak_dbfs": -1.0, "onset_db": 68.8, "hf_ratio": 0.1, "confirm": "decay"},
                {"ts_ms": 2500, "peak_dbfs": -3.0, "onset_db": 30.0, "hf_ratio": 0.1, "confirm": "decay"},
            ],
        },
    ]
    path = tmp_path / "review.html"
    bonk_sweep.write_review_page(entries, str(path))
    html = path.read_text(encoding="utf-8")
    assert 'src="wav/grid_snap.wav"' in html
    assert 'name="v-grid_snap"' in html  # verdict radios
    assert "Generate feedback JSON" in html
    assert "calm1" in html  # clean move listed in the collapsed section
    assert 'data-move="calm1"' not in html  # ...but gets no review row


def test_load_entries_roundtrip(tmp_path: Path) -> None:
    jsonl = tmp_path / "sweep.jsonl"
    jsonl.write_text('{"move":"a","detect_count":0}\n\n{"move":"b","detect_count":1}\n', encoding="utf-8")
    entries = bonk_sweep._load_entries(str(jsonl))
    assert [e["move"] for e in entries] == ["a", "b"]


def test_parse_running_handles_daemon_object_shape() -> None:
    """The live daemon returns a list of {'uuid': ...} objects, not strings."""
    assert bonk_sweep._parse_running([{"uuid": "abc"}, {"uuid": "def"}]) == ["abc", "def"]
    assert bonk_sweep._parse_running(["abc"]) == ["abc"]
    assert bonk_sweep._parse_running({"uuids": ["abc"]}) == ["abc"]
    assert bonk_sweep._parse_running(None) == []
    assert bonk_sweep._parse_running([]) == []
