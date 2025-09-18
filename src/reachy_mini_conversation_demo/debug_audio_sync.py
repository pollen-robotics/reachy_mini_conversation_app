"""Thread-safe helpers for correlating playback and wobble timestamps."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List

SYNC_LOG_INTERVAL = 100

_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_state: Dict[int, List[float]] = {}
_order: Deque[int] = deque()
_update_count = 0


def _record(packet_id: int, index: int, timestamp: float | None) -> None:
    """Store a timestamp for a packet. Index 0=playback, 1=wobble."""
    global _update_count
    if packet_id is None:
        return
    ts = time.time() if timestamp is None else timestamp
    with _lock:
        if packet_id not in _state:
            _state[packet_id] = [math.nan, math.nan]
            _order.append(packet_id)
        _state[packet_id][index] = ts
        _update_count += 1
        if _update_count >= SYNC_LOG_INTERVAL:
            _flush_locked(reason="interval")


def record_playback(packet_id: int, timestamp: float | None = None) -> None:
    """Record when a packet is played through the speakers."""
    _record(packet_id, 0, timestamp)


def record_wobble(packet_id: int, timestamp: float | None = None) -> None:
    """Record when a packet is consumed for wobble analysis."""
    _record(packet_id, 1, timestamp)


def force_flush(reason: str = "manual") -> None:
    """Log current state immediately."""
    with _lock:
        _flush_locked(reason=reason)


def reset_state(reason: str = "reset") -> None:
    """Log and reset counters without pruning entries."""
    with _lock:
        _flush_locked(reason=reason)


def _flush_locked(reason: str) -> None:
    """Log snapshot of all tracked packets with relative timings."""
    global _update_count
    if not _state:
        _update_count = 0
        return

    snapshot_order = list(_order)
    snapshot = {pid: values[:] for pid, values in _state.items()}

    all_ts = [ts for values in snapshot.values() for ts in values if not math.isnan(ts)]
    if not all_ts:
        t0 = time.time()
    else:
        t0 = min(all_ts)

    lines = []
    for pid in snapshot_order:
        play_ts, wobble_ts = snapshot[pid]
        adj_play = play_ts - t0 if not math.isnan(play_ts) else math.nan
        adj_wobble = wobble_ts - t0 if not math.isnan(wobble_ts) else math.nan
        delta = (
            adj_wobble - adj_play
            if not math.isnan(adj_play) and not math.isnan(adj_wobble)
            else math.nan
        )
        lines.append(
            "ID %d | play=%.6fs | wobble=%.6fs | delta=%.6fs" % (
                pid,
                adj_play if not math.isnan(adj_play) else float("nan"),
                adj_wobble if not math.isnan(adj_wobble) else float("nan"),
                delta if not math.isnan(delta) else float("nan"),
            )
        )

    _logger.debug(
        "Audio sync snapshot (%d packets, reason=%s, t0=%.6f)\n%s",
        len(lines),
        reason,
        t0,
        "\n".join(lines),
    )

    _update_count = 0
