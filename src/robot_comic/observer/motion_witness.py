"""Witness-side motion smoothness metrics (the burst leg of the visual witness).

``capture_frame``/``describe_scene`` answer *what does the scene look like*
with one frame; this module answers *how smoothly is the robot moving* with a
burst: it records N seconds from the witness camera, reduces each consecutive
frame pair to a scalar motion energy, detects spikes (jerk events), and
correlates each spike with the Tier-0 event log to name the move/emotion that
was rendering at that moment. The spike↔event join is the same backbone the
bonk detector (#522) will use for audio impact signatures.

CLI (run on the workstation, robot in frame):

    python -m robot_comic.observer.motion_witness --seconds 60 \
        --events D:/logs/ricci_events.log --json report.json

Env: ``VISUAL_WITNESS_CAMERA_DEVICE`` — OpenCV device index (default 0),
shared with the still-frame visual witness.
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import Any

import numpy as np
from numpy.typing import NDArray


DEFAULT_FPS = 30.0
# A spike is a sample whose energy exceeds baseline + _SPIKE_MAD_K * MAD,
# with an absolute floor so a perfectly still scene's noise never qualifies.
_SPIKE_MAD_K = 6.0
_SPIKE_MIN_ENERGY = 3.0


# Per-pair energy = the 99.5th percentile of |pixel diff|. A full-frame mean
# dilutes the robot (it covers a few % of the witness FOV) into camera noise;
# a high percentile keys on the most-changed pixels — i.e. the moving robot —
# while staying robust to sensor noise (which a max would amplify).
_ENERGY_PERCENTILE = 99.5


def motion_energy(frames: list[NDArray[Any]]) -> list[float]:
    """Reduce consecutive grayscale frame pairs to localized-motion scalars."""
    energies: list[float] = []
    for prev, cur in zip(frames, frames[1:]):
        diff = np.abs(cur.astype(np.float32) - prev.astype(np.float32))
        energies.append(float(np.percentile(diff, _ENERGY_PERCENTILE)))
    return energies


def analyze_motion(energies: list[float], fps: float, *, start_wall_ts: float | None = None) -> dict[str, Any]:
    """Summarize an energy timeseries: stats, jerk, and spike events.

    ``start_wall_ts`` (epoch seconds) timestamps each spike with wall-clock
    time (``wall_ts``, epoch ms) so it can be joined against the event log.
    """
    if not energies:
        return {"mean_energy": 0.0, "max_energy": 0.0, "max_jerk": 0.0, "spikes": [], "samples": 0}

    arr = np.asarray(energies, dtype=np.float64)
    jerk = np.abs(np.diff(arr)) if len(arr) > 1 else np.zeros(1)

    baseline = float(np.median(arr))
    mad = float(np.median(np.abs(arr - baseline)))
    threshold = max(baseline + _SPIKE_MAD_K * max(mad, 0.1), _SPIKE_MIN_ENERGY)

    spikes: list[dict[str, Any]] = []
    for i, e in enumerate(arr):
        if e <= threshold:
            continue
        # Keep only the local maximum of a contiguous burst.
        if spikes and (i - spikes[-1]["_i"]) <= 3:
            if e > spikes[-1]["energy"]:
                spikes[-1].update(_i=i, t=i / fps, energy=float(e))
            continue
        spikes.append({"_i": i, "t": i / fps, "energy": float(e)})

    for s in spikes:
        i = s.pop("_i")
        if start_wall_ts is not None:
            s["wall_ts"] = int((start_wall_ts + i / fps) * 1000)

    return {
        "mean_energy": float(np.mean(arr)),
        "max_energy": float(np.max(arr)),
        "max_jerk": float(np.max(jerk)),
        "spikes": spikes,
        "samples": len(energies),
    }


def correlate_spikes_with_events(
    spikes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    window_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Attribute each spike to the nearest tool.execute event within the window.

    Spikes need a ``wall_ts`` (epoch ms). Events are Tier-0 span dicts with
    ``ts`` (epoch ms) and ``attrs["tool.name"]``. Unattributed spikes come back
    with ``tool=None`` so coverage gaps are visible rather than dropped.
    """
    tool_events = [e for e in events if e.get("name") == "tool.execute"]
    matches: list[dict[str, Any]] = []
    for spike in spikes:
        wall_ts = spike.get("wall_ts")
        best: dict[str, Any] | None = None
        best_offset = window_s
        if wall_ts is not None:
            for ev in tool_events:
                offset = abs(wall_ts - ev["ts"]) / 1000.0
                if offset <= best_offset:
                    best = ev
                    best_offset = offset
        matches.append(
            {
                "t": spike.get("t"),
                "energy": spike.get("energy"),
                "tool": (best.get("attrs", {}) or {}).get("tool.name") if best else None,
                "offset_s": best_offset if best else None,
            }
        )
    return matches


def record_motion(
    seconds: float,
    *,
    device: int | None = None,
    fps_cap: float = DEFAULT_FPS,
) -> tuple[float, float, list[float]]:
    """Record from the witness camera; return (start_wall_ts, actual_fps, energies).

    Frames are reduced to grayscale energy on the fly — nothing is stored.
    """
    import cv2

    dev = device if device is not None else int(os.getenv("VISUAL_WITNESS_CAMERA_DEVICE", "0"))
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open witness camera device {dev}")
    try:
        start = time.time()
        prev_gray: NDArray[np.float32] | None = None
        energies: list[float] = []
        frame_interval = 1.0 / fps_cap
        next_frame_at = start
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time()
            if now < next_frame_at:
                continue
            next_frame_at = now + frame_interval
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                energies.append(float(np.percentile(np.abs(gray - prev_gray), _ENERGY_PERCENTILE)))
            prev_gray = gray
        elapsed = time.time() - start
        actual_fps = (len(energies) + 1) / elapsed if elapsed > 0 else fps_cap
        return start, actual_fps, energies
    finally:
        cap.release()


def load_events(path: str) -> list[dict[str, Any]]:
    """Load Tier-0 span dicts from an events file (raw JSONL or RCSPAN lines)."""
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("RCSPAN "):
                line = line[len("RCSPAN ") :]
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main() -> None:
    """CLI entry point: record, analyze, correlate, report."""
    parser = argparse.ArgumentParser(description="Measure robot movement smoothness via the witness camera")
    parser.add_argument("--seconds", type=float, default=30.0, help="capture window (default: %(default)s)")
    parser.add_argument("--device", type=int, default=None, help="OpenCV device index")
    parser.add_argument("--events", default=None, help="Tier-0 events file (jsonl or RCSPAN lines) to correlate")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full report to this path")
    args = parser.parse_args()

    print(f"recording {args.seconds:g}s from witness camera...", flush=True)
    start_wall_ts, fps, energies = record_motion(args.seconds, device=args.device)
    report = analyze_motion(energies, fps, start_wall_ts=start_wall_ts)
    report["fps"] = fps
    report["start_wall_ts"] = start_wall_ts
    report["energies"] = [round(e, 2) for e in energies]

    if args.events:
        events = load_events(args.events)
        report["attributions"] = correlate_spikes_with_events(report["spikes"], events)

    print(
        f"samples={report['samples']} fps={fps:.1f} mean_energy={report['mean_energy']:.2f} "
        f"max_energy={report['max_energy']:.2f} max_jerk={report['max_jerk']:.2f} "
        f"spikes={len(report['spikes'])}"
    )
    for att in report.get("attributions", []):
        tool = att["tool"] or "UNATTRIBUTED"
        offset = f" ({att['offset_s']:.1f}s from event)" if att["offset_s"] is not None else ""
        print(f"  spike t={att['t']:.1f}s energy={att['energy']:.1f} -> {tool}{offset}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report written to {args.json_out}")


if __name__ == "__main__":
    main()
