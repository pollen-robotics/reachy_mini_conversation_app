"""Bonk sweep (#536): play every preset move via the daemon, witness each one.

Plays the recorded-move datasets (emotions, dances) one move at a time through
the daemon REST API on the robot — the voice app stays STOPPED — while the
workstation witness mic runs the ring-down bonk detector (#522/#534). Output:

- ``<out-dir>/sweep.jsonl`` — one diagnostic entry per move
- ``<out-dir>/report.md``   — human review table, noisiest moves first
- ``<out-dir>/wav/<move>.wav`` — per-move witness audio for auditing
  detections against emotion sound effects
- caution-list entries (``append_caution_entries``) for every flagged move

Caveat for reviewers: daemon playback bypasses the app's ``motion_safety``
clamps and ``MOVEMENT_SPEED_FACTOR``, so the sweep measures the RAW recordings
under daemon kinematics — an upper bound on app-mode behaviour.

CLI (workstation; daemon up on the robot, app stopped)::

    python -m robot_comic.observer.bonk_sweep --host ricci \
        --datasets emotions,dances --out-dir D:/logs/bonk_sweep --device Brio
"""

from __future__ import annotations
import os
import json
import time
import shlex
import argparse
import threading
import subprocess
from typing import Any, Callable

from robot_comic.observer.audio_witness import resolve_input_device
from robot_comic.observer.bonk_detector import DEFAULT_CAUTION_LIST, BonkDetector, append_caution_entries


DATASETS = {
    "emotions": "pollen-robotics/reachy-mini-emotions-library",
    "dances": "pollen-robotics/reachy-mini-dances-library",
}

_SSH_OPTS = ["-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


class DaemonMoveTrigger:
    """Drive the robot daemon's move API over ``ssh <host> curl`` (it binds localhost)."""

    def __init__(self, host: str = "ricci", timeout_s: float = 20.0) -> None:
        """Target ``host`` via ssh; ``timeout_s`` bounds each curl round-trip."""
        self.host = host
        self.timeout_s = timeout_s

    def _curl(self, method: str, path: str) -> Any:
        cmd = ["ssh", *_SSH_OPTS, self.host, f"curl -s -X {method} {shlex.quote('http://localhost:8000' + path)}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        if result.returncode != 0:
            raise RuntimeError(f"ssh/curl failed ({result.returncode}): {result.stderr.strip()[:200]}")
        return json.loads(result.stdout) if result.stdout.strip() else None

    def motors_mode(self) -> str:
        """Return the daemon motor mode (must be ``enabled`` for a sweep)."""
        return str(self._curl("GET", "/api/motors/status")["mode"])

    def list_moves(self, dataset: str) -> list[str]:
        """Enumerate move names in a recorded-move dataset."""
        moves = self._curl("GET", f"/api/move/recorded-move-datasets/list/{dataset}")
        return sorted(str(m) for m in moves)

    def play(self, dataset: str, move: str) -> str:
        """Start a recorded move; return its running-move UUID."""
        response = self._curl("POST", f"/api/move/play/recorded-move-dataset/{dataset}/{move}")
        return str(response["uuid"])

    def running(self) -> list[str]:
        """UUIDs of moves the daemon is currently playing."""
        return _parse_running(self._curl("GET", "/api/move/running"))


def _parse_running(result: Any) -> list[str]:
    """Normalize ``/api/move/running`` output to a list of UUID strings.

    The daemon returns ``[{"uuid": "..."}]`` (observed live 2026-06-10); be
    liberal and also accept plain string lists and ``{"uuids": [...]}``.
    """
    if result is None:
        return []
    if isinstance(result, dict):
        result = result.get("uuids", result.get("moves", []))
    uuids: list[str] = []
    for item in result:
        if isinstance(item, dict):
            value = item.get("uuid")
            if value:
                uuids.append(str(value))
        else:
            uuids.append(str(item))
    return uuids


class WitnessCapture:
    """Background witness-mic capture feeding the bonk detector for one move."""

    def __init__(self, device: int | str | None = None, sample_rate: int = 16_000, block_ms: int = 50) -> None:
        """Resolve the device up front so a misaimed witness fails before the sweep."""
        self.device_index = resolve_input_device(device)
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.detections: list[dict[str, Any]] = []
        self.pcm: list[bytes] = []
        self.error: str | None = None

    def __enter__(self) -> "WitnessCapture":
        """Start the capture thread."""
        self._thread = threading.Thread(target=self._run, name="bonk-sweep-capture", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the capture thread and wait for it to drain."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        import sounddevice as sd

        from robot_comic.observer.bonk_detector import _block_to_pcm

        detector = BonkDetector()
        block_size = int(self.sample_rate * self.block_ms / 1000)
        start = time.time()
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=self.device_index,
            ) as stream:
                while not self._stop.is_set():
                    block, _overflowed = stream.read(block_size)
                    mono = block[:, 0]
                    self.pcm.append(_block_to_pcm(mono))
                    ts_ms = int((time.time() - start) * 1000)
                    hit = detector.update(ts_ms=ts_ms, block=mono, sample_rate=self.sample_rate)
                    if hit is not None:
                        hit["wall_ts"] = int(time.time() * 1000)
                        self.detections.append(hit)
        except Exception as exc:  # capture must never kill the sweep
            self.error = str(exc)

    def save_wav(self, path: str) -> None:
        """Write the captured audio as 16-bit mono WAV."""
        import wave

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.pcm))


def build_plan(
    trigger: DaemonMoveTrigger, dataset_keys: list[str], only: list[str] | None = None
) -> list[tuple[str, str]]:
    """Return ``(dataset, move)`` pairs to sweep, optionally filtered to ``only`` names."""
    plan: list[tuple[str, str]] = []
    for key in dataset_keys:
        dataset = DATASETS[key]
        for move in trigger.list_moves(dataset):
            if only and move not in only:
                continue
            plan.append((dataset, move))
    return plan


def sweep_move(
    trigger: DaemonMoveTrigger,
    capture_factory: Callable[[], WitnessCapture],
    dataset: str,
    move: str,
    *,
    max_wait_s: float = 30.0,
    tail_s: float = 1.5,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    """Play one move while the witness listens; return its diagnostic entry."""
    entry: dict[str, Any] = {"dataset": dataset, "move": move}
    started = time.time()
    with capture_factory() as capture:
        try:
            uuid = trigger.play(dataset, move)
        except Exception as exc:
            entry["error"] = f"play failed: {exc}"
            return entry
        deadline = started + max_wait_s
        while time.time() < deadline:
            try:
                if uuid not in trigger.running():
                    break
            except Exception:
                pass  # transient poll failure: keep listening until the deadline
            time.sleep(poll_s)
        entry["duration_s"] = round(time.time() - started, 1)
        time.sleep(tail_s)
    if capture.error:
        entry["capture_error"] = capture.error
    entry["detections"] = capture.detections
    entry["detect_count"] = len(capture.detections)
    entry["max_peak_dbfs"] = max((d["peak_dbfs"] for d in capture.detections), default=None)
    entry["_capture"] = capture  # for save_wav by the caller; stripped before persisting
    return entry


def write_report(entries: list[dict[str, Any]], path: str) -> None:
    """Write the human review table, noisiest moves first."""
    flagged = [e for e in entries if e.get("detect_count")]
    lines = [
        "# Bonk sweep report",
        "",
        f"{len(entries)} moves swept, {len(flagged)} flagged. Daemon playback — raw recordings,",
        "no app-side motion_safety clamps or speed factor. Audit flagged moves against their",
        "witness WAV before denylisting (emotion sound effects can contain transients).",
        "",
        "| move | dataset | dur (s) | detections | max peak dBFS | notes |",
        "|---|---|---|---|---|---|",
    ]
    order = sorted(entries, key=lambda e: (-(e.get("detect_count") or 0), e["move"]))
    for e in order:
        notes = e.get("error") or e.get("capture_error") or ""
        dataset_short = "emotions" if "emotions" in e["dataset"] else "dances"
        lines.append(
            f"| {e['move']} | {dataset_short} | {e.get('duration_s', '')} | "
            f"{e.get('detect_count', 0)} | {e.get('max_peak_dbfs', '')} | {notes} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Play every preset move via the daemon; witness each for bonks")
    parser.add_argument("--host", default="ricci", help="robot hostname (daemon at localhost:8000 via ssh)")
    parser.add_argument("--datasets", default="emotions,dances", help="comma list of dataset keys: emotions,dances")
    parser.add_argument("--moves", default=None, help="optional comma list of specific move names")
    parser.add_argument("--out-dir", required=True, help="directory for sweep.jsonl / report.md / wav/")
    parser.add_argument("--device", default=None, help="witness input device (default: ROBOT_AUDIO_DEVICE)")
    parser.add_argument("--max-wait", type=float, default=30.0, help="per-move completion cap, seconds")
    parser.add_argument("--tail", type=float, default=1.5, help="listen tail after each move, seconds")
    parser.add_argument("--limit", type=int, default=None, help="sweep only the first N planned moves")
    parser.add_argument("--caution-list", default=DEFAULT_CAUTION_LIST, help="caution list JSONL path")
    args = parser.parse_args()

    trigger = DaemonMoveTrigger(host=args.host)
    mode = trigger.motors_mode()
    if mode != "enabled":
        raise SystemExit(f"daemon motor mode is {mode!r} — enable motors before sweeping")

    only = [m.strip() for m in args.moves.split(",")] if args.moves else None
    plan = build_plan(trigger, [k.strip() for k in args.datasets.split(",")], only)
    if args.limit:
        plan = plan[: args.limit]
    print(f"sweeping {len(plan)} moves on {args.host} (witness device index {resolve_input_device(args.device)})")

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "sweep.jsonl")
    entries: list[dict[str, Any]] = []
    for i, (dataset, move) in enumerate(plan, 1):
        entry = sweep_move(
            trigger,
            lambda: WitnessCapture(device=args.device),
            dataset,
            move,
            max_wait_s=args.max_wait,
            tail_s=args.tail,
        )
        capture = entry.pop("_capture", None)
        if capture is not None and capture.pcm:
            wav_path = os.path.join(args.out_dir, "wav", f"{move}.wav")
            capture.save_wav(wav_path)
            entry["wav"] = wav_path
        entries.append(entry)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        status = f"{entry.get('detect_count', 0)} detection(s)" if not entry.get("error") else entry["error"]
        print(f"[{i}/{len(plan)}] {move}: {status}", flush=True)
        if entry.get("detect_count"):
            append_caution_entries(
                args.caution_list,
                entry["detections"],
                [{"tool": f"sweep:{move}", "offset_s": None} for _ in entry["detections"]],
            )

    write_report(entries, os.path.join(args.out_dir, "report.md"))
    flagged = [e["move"] for e in entries if e.get("detect_count")]
    print(f"done: {len(flagged)} flagged move(s): {', '.join(flagged) if flagged else 'none'}")
    print(f"report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
