"""Tests for the motion witness (witness-side movement smoothness metrics).

The visual witness answers "what does the scene look like" with one frame;
the motion witness answers "how smoothly is the robot moving" with a burst:
consecutive-frame motion energy → spike detection → correlation with the
Tier-0 event log (which move/emotion was rendering when the spike happened).
The same spike↔event join is the backbone the bonk detector (#522) will use
for audio impacts.
"""

from __future__ import annotations

import numpy as np

from robot_comic.observer.motion_witness import (
    motion_energy,
    analyze_motion,
    correlate_spikes_with_events,
)


def _frames(*levels: float, shape: tuple[int, int] = (8, 8)) -> list[np.ndarray]:
    """Grayscale frames with uniform intensity per level."""
    return [np.full(shape, lvl, dtype=np.float32) for lvl in levels]


# ---------------------------------------------------------------------------
# motion_energy
# ---------------------------------------------------------------------------


def test_static_scene_has_zero_energy() -> None:
    energies = motion_energy(_frames(10, 10, 10, 10))
    assert len(energies) == 3
    np.testing.assert_allclose(energies, [0.0, 0.0, 0.0])


def test_energy_reflects_frame_change_magnitude() -> None:
    energies = motion_energy(_frames(0, 10, 10, 50))
    assert energies[0] == 10.0  # |10-0|
    assert energies[1] == 0.0
    assert energies[2] == 40.0  # |50-10|


# ---------------------------------------------------------------------------
# analyze_motion
# ---------------------------------------------------------------------------


def test_smooth_motion_reports_no_spikes() -> None:
    # Gentle constant motion: steady energy, zero jerk.
    energies = [2.0] * 60
    report = analyze_motion(energies, fps=30.0)
    assert report["spikes"] == []
    assert report["mean_energy"] == 2.0
    np.testing.assert_allclose(report["max_jerk"], 0.0)


def test_jerky_motion_reports_spike_at_right_time() -> None:
    # Steady low motion with one violent jump at sample 30 (t = 1.0 s @ 30fps).
    energies = [2.0] * 30 + [40.0] + [2.0] * 29
    report = analyze_motion(energies, fps=30.0)
    assert len(report["spikes"]) == 1
    spike = report["spikes"][0]
    np.testing.assert_allclose(spike["t"], 30 / 30.0, atol=0.05)
    assert spike["energy"] == 40.0
    assert report["max_jerk"] > 0


def test_empty_energies_yield_empty_report() -> None:
    report = analyze_motion([], fps=30.0)
    assert report["spikes"] == []
    assert report["mean_energy"] == 0.0


# ---------------------------------------------------------------------------
# correlate_spikes_with_events
# ---------------------------------------------------------------------------


def _event(ts_ms: int, tool: str) -> dict:
    return {"name": "tool.execute", "ts": ts_ms, "attrs": {"tool.name": tool}}


def test_spike_matches_event_within_window() -> None:
    spikes = [{"t": 5.0, "energy": 30.0, "wall_ts": 1_000_000 + 5_000}]
    events = [
        _event(1_000_000 + 4_200, "play_emotion"),
        _event(1_000_000 + 60_000, "move_head"),
    ]
    matches = correlate_spikes_with_events(spikes, events, window_s=2.0)
    assert len(matches) == 1
    assert matches[0]["tool"] == "play_emotion"
    np.testing.assert_allclose(matches[0]["offset_s"], 0.8, atol=1e-6)


def test_spike_with_no_nearby_event_is_unattributed() -> None:
    spikes = [{"t": 5.0, "energy": 30.0, "wall_ts": 1_000_000}]
    events = [_event(2_000_000, "play_emotion")]
    matches = correlate_spikes_with_events(spikes, events, window_s=2.0)
    assert len(matches) == 1
    assert matches[0]["tool"] is None
