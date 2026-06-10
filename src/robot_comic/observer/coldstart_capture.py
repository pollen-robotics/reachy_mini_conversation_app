"""Witnessed cold-start capture (#541).

One command that starts the robot app and records the boot across all three
witness modalities simultaneously, then renders a single merged timeline:

- **journal** (robot clock): ``journalctl -u reachy-app-autostart`` followed
  over ssh — systemd start, ``Startup: +X.XXs`` checkpoints.
- **spans** (robot clock): new ``RCSPAN`` lines appended to the log-sink file.
- **audio** (workstation clock): dBFS onsets from the witness microphone —
  when the robot *actually* made sound.
- **video** (workstation clock): inter-frame motion onsets from the witness
  camera — when the robot *actually* moved.

Workstation clocks are mapped onto the robot clock via a one-shot ssh clock
probe. Run from the workstation that hosts the witness mic/camera::

    python -m robot_comic.observer.coldstart_capture --out D:/logs/coldstart

Requires ``sounddevice``, ``numpy``, and ``opencv-python`` (all lazily
imported). The app is STARTED by this tool; it does not stop it — the caller
owns the live-block lifecycle (see docs/live-block-playbook.md).
"""

from __future__ import annotations
import os
import json
import time
import wave
import argparse
import threading
import subprocess
from typing import Any
from pathlib import Path

from robot_comic.observer.audio_witness import DB_ON, dbfs, resolve_input_device
from robot_comic.observer.coldstart_timeline import (
    TimelineEvent,
    rank_gaps,
    build_timeline,
    parse_sink_line,
    render_markdown,
    parse_journal_line,
)


SERVICE = os.getenv("ROBOT_APP_SERVICE", "reachy-app-autostart")

# Mean absolute inter-frame gray delta above which the robot is "moving".
# The Brio points at a static scene; the only mover in frame is the robot.
MOTION_DELTA_ON = float(os.getenv("ROBOT_MOTION_DELTA_ON", "2.0"))


def _ssh(host: str, command: str, timeout: float = 30.0) -> str:
    """Run a remote command with a non-interactive ssh (-n: no stdin inherit)."""
    result = subprocess.run(
        ["ssh", "-n", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh {host} {command!r} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def estimate_clock_offset_ms(host: str) -> int:
    """Robot epoch-ms minus local epoch-ms, midpoint-corrected for ssh RTT."""
    before = time.time()
    remote_ms = int(_ssh(host, "date +%s%3N"))
    after = time.time()
    local_mid_ms = int((before + after) / 2 * 1000)
    return remote_ms - local_mid_ms


class _AudioWitnessThread(threading.Thread):
    """Record the witness mic; emit onset events and a WAV for the report."""

    def __init__(self, out_dir: Path, offset_ms: int) -> None:
        super().__init__(name="coldstart-audio", daemon=True)
        self.events: list[TimelineEvent] = []
        self._out_dir = out_dir
        self._offset_ms = offset_ms
        # Must not shadow threading.Thread's internal ``_stop`` method —
        # Thread.join() invokes it during teardown.
        self._stop_event = threading.Event()
        self.error: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - hardware I/O
        try:
            import numpy as np
            import sounddevice as sd

            samplerate, block = 16_000, 1_600  # 100 ms blocks
            device = resolve_input_device()
            chunks: list[Any] = []
            in_sound = False
            with sd.InputStream(
                device=device, channels=1, samplerate=samplerate, blocksize=block, dtype="float32"
            ) as stream:
                while not self._stop_event.is_set():
                    frame, _ = stream.read(block)
                    mono = frame[:, 0]
                    chunks.append((mono * 32767).astype(np.int16))
                    level = dbfs(mono)
                    if level >= DB_ON and not in_sound:
                        in_sound = True
                        self.events.append(
                            {
                                "source": "audio",
                                "label": "sound_onset",
                                "epoch_ms": int(time.time() * 1000) + self._offset_ms,
                                "detail": {"dbfs": round(level, 1)},
                            }
                        )
                    elif level < DB_ON - 10 and in_sound:
                        in_sound = False
            wav_path = self._out_dir / "audio_witness.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(samplerate)
                wav.writeframes(np.concatenate(chunks).tobytes())
        except Exception as e:  # noqa: BLE001 - witness failure must not kill capture
            self.error = str(e)


class _VideoWitnessThread(threading.Thread):
    """Sample the witness camera; emit motion-onset events and keyframes."""

    def __init__(self, out_dir: Path, offset_ms: int, camera_index: int) -> None:
        super().__init__(name="coldstart-video", daemon=True)
        self.events: list[TimelineEvent] = []
        self._out_dir = out_dir
        self._offset_ms = offset_ms
        self._camera_index = camera_index
        # Must not shadow threading.Thread's internal ``_stop`` method —
        # Thread.join() invokes it during teardown.
        self._stop_event = threading.Event()
        self.error: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - hardware I/O
        try:
            import cv2

            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"camera index {self._camera_index} did not open")
            previous = None
            in_motion = False
            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.5)
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if previous is not None:
                        delta = float(cv2.absdiff(gray, previous).mean())
                        if delta >= MOTION_DELTA_ON and not in_motion:
                            in_motion = True
                            epoch_ms = int(time.time() * 1000) + self._offset_ms
                            self.events.append(
                                {
                                    "source": "video",
                                    "label": "motion_onset",
                                    "epoch_ms": epoch_ms,
                                    "detail": {"delta": round(delta, 2)},
                                }
                            )
                            cv2.imwrite(str(self._out_dir / f"motion_{epoch_ms}.jpg"), frame)
                        elif delta < MOTION_DELTA_ON / 2 and in_motion:
                            in_motion = False
                    previous = gray
                    time.sleep(0.4)
            finally:
                cap.release()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)


def capture(
    host: str,
    sink_file: Path,
    out_dir: Path,
    duration_s: float,
    camera_index: int,
) -> Path:
    """Run the full capture; return the path of the rendered report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    offset_ms = estimate_clock_offset_ms(host)
    sink_start = sink_file.stat().st_size if sink_file.exists() else 0

    journal_path = out_dir / "journal.log"
    journal_file = journal_path.open("w", encoding="utf-8")
    journal_proc = subprocess.Popen(
        ["ssh", "-n", host, f"journalctl -u {SERVICE} -f -o short-iso --since now"],
        stdout=journal_file,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    audio = _AudioWitnessThread(out_dir, offset_ms)
    video = _VideoWitnessThread(out_dir, offset_ms, camera_index)
    audio.start()
    video.start()

    t_start_local_ms = int(time.time() * 1000)
    _ssh(host, f"sudo -n systemctl start {SERVICE}")
    print(f"[coldstart] {SERVICE} started on {host}; capturing {duration_s:.0f}s …")

    try:
        time.sleep(duration_s)
    finally:
        audio.stop()
        video.stop()
        journal_proc.terminate()
        try:
            journal_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            journal_proc.kill()
        journal_file.close()
        audio.join(timeout=10)
        video.join(timeout=10)

    events: list[TimelineEvent] = []
    for line in journal_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_journal_line(line)
        if parsed is not None:
            events.append(parsed)
    if sink_file.exists():
        with sink_file.open("r", encoding="utf-8", errors="replace") as sink:
            sink.seek(sink_start)
            for line in sink:
                parsed = parse_sink_line(line.strip())
                if parsed is not None:
                    events.append(parsed)
    events.extend(audio.events)
    events.extend(video.events)

    t0_candidates = [e["epoch_ms"] for e in events if e["label"] == "service_started"]
    t0 = t0_candidates[0] if t0_candidates else t_start_local_ms + offset_ms

    timeline = build_timeline(events, t0_epoch_ms=t0)
    meta = {
        "host": host,
        "service": SERVICE,
        "t0_source": "journal service_started" if t0_candidates else "local start command",
        "clock_offset_ms": offset_ms,
        "audio_witness": audio.error or "ok",
        "video_witness": video.error or "ok",
    }
    report = render_markdown(timeline, rank_gaps(timeline), meta=meta)
    report_path = out_dir / "timeline.md"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"[coldstart] report: {report_path}")
    return report_path


def main() -> None:
    """CLI entry point: parse args and run one witnessed cold-start capture."""
    parser = argparse.ArgumentParser(description="Witnessed robot cold-start capture (#541)")
    parser.add_argument("--host", default="ricci")
    parser.add_argument("--sink-file", default=os.getenv("ROBOT_EVENT_LOG", ""), help="log-sink file path")
    parser.add_argument("--out", required=True, help="artifacts directory")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("ROBOT_CAMERA_INDEX", "1")))
    args = parser.parse_args()
    if not args.sink_file:
        parser.error("--sink-file (or ROBOT_EVENT_LOG) is required")
    capture(
        host=args.host,
        sink_file=Path(args.sink_file),
        out_dir=Path(args.out),
        duration_s=args.duration,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    main()
