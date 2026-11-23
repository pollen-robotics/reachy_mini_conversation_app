import math
import threading
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.speaker_tracking import SpeakerTracking


def test_stable_speech_turns_body_and_head_together() -> None:
    """A stable left bearing rotates body and head while preserving the gesture pose."""
    tracker = SpeakerTracking()
    head = create_head_pose(pitch=10, degrees=True).astype(np.float32)
    for observed_at in (1.0, 1.1, 1.2, 1.3):
        tracker.observe(math.radians(40), True, observed_at, observed_at, enabled=True)
    tracker.apply(head, 0.0, 1.3, allow_turn=True)
    for tick in range(1, 41):
        oriented_head, body_yaw = tracker.apply(head, 0.0, 1.3 + tick * 0.1, allow_turn=True)
    assert math.degrees(body_yaw) == pytest.approx(40, abs=0.1)
    expected = create_head_pose(yaw=body_yaw, degrees=False) @ head
    np.testing.assert_allclose(oriented_head, expected, atol=1e-6)


@pytest.mark.parametrize(
    "yaw,speech,age,enabled",
    [(0.8, False, 0, True), (0.8, True, 1, True), (0.8, True, 0, False), (math.nan, True, 0, True)],
)
def test_unusable_observations_do_not_turn(yaw: float, speech: bool, age: float, enabled: bool) -> None:
    """Silence, stale readings, suppressed speech and invalid bearings leave the pose alone."""
    tracker = SpeakerTracking()
    head = np.eye(4, dtype=np.float32)
    for observed_at in (1.0, 1.1, 1.2, 1.3):
        tracker.observe(yaw, speech, observed_at, observed_at + age, enabled=enabled)
    tracker.apply(head, 0.0, 2.0, allow_turn=True)
    oriented_head, body_yaw = tracker.apply(head, 0.0, 3.0, allow_turn=True)
    assert body_yaw == 0.0
    np.testing.assert_array_equal(oriented_head, head)


def test_noise_and_short_sounds_do_not_acquire_a_target() -> None:
    """A moving bearing or a brief sound cannot establish a speaker target."""
    tracker = SpeakerTracking()
    head = np.eye(4, dtype=np.float32)
    for index, yaw in enumerate((40, -40, 40, -40, 40)):
        observed_at = 1.0 + index * 0.1
        tracker.observe(math.radians(yaw), True, observed_at, observed_at, enabled=True)
    tracker.apply(head, 0.0, 1.4, allow_turn=True)
    _, body_yaw = tracker.apply(head, 0.0, 1.5, allow_turn=True)
    assert body_yaw == 0.0


def test_switches_speakers_and_ignores_small_bearing_changes() -> None:
    """Sustained speech can switch sides without chasing small direction changes."""
    tracker = SpeakerTracking()
    head = np.eye(4, dtype=np.float32)
    tracker.apply(head, 0.0, 0.0, allow_turn=True)
    for start, bearing in ((1.0, 45), (7.0, -45), (13.0, -41)):
        for index in range(4):
            observed_at = start + index * 0.1
            tracker.observe(math.radians(bearing), True, observed_at, observed_at, enabled=True)
        for tick in range(51):
            _, body_yaw = tracker.apply(head, 0.0, start + 0.3 + tick * 0.1, allow_turn=True)
        assert math.degrees(body_yaw) == pytest.approx(45 if start == 1.0 else -45, abs=0.1)


def test_turn_speed_hold_and_limits() -> None:
    """Turns are bounded, pause for playback, and hold their orientation during silence."""
    tracker = SpeakerTracking()
    head = np.eye(4, dtype=np.float32)
    for observed_at in (1.0, 1.1, 1.2, 1.3):
        tracker.observe(math.radians(175), True, observed_at, observed_at, enabled=True)
    tracker.apply(head, 0.0, 1.3, allow_turn=True)
    _, body_yaw = tracker.apply(head, 0.0, 1.4, allow_turn=True)
    assert math.degrees(body_yaw) == pytest.approx(3)
    _, held_yaw = tracker.apply(head, 0.0, 1.5, allow_turn=False)
    assert held_yaw == body_yaw
    for tick in range(1, 100):
        _, body_yaw = tracker.apply(head, 0.0, 1.5 + tick * 0.1, allow_turn=True)
    assert math.degrees(body_yaw) == pytest.approx(150, abs=0.01)
    tracker.observe(0.0, False, 12.0, 12.0, enabled=True)
    _, held_yaw = tracker.apply(head, 0.0, 12.0, allow_turn=True)
    assert held_yaw == pytest.approx(body_yaw, abs=1e-6)
    gesture = create_head_pose(yaw=40, degrees=True).astype(np.float32)
    manual_head, manual_yaw = tracker.apply(gesture, math.radians(20), 12.1, allow_turn=False, manual=True)
    np.testing.assert_array_equal(manual_head, gesture)
    assert math.degrees(manual_yaw) == pytest.approx(20)
    resumed_head, resumed_yaw = tracker.apply(gesture, 0.0, 12.2, allow_turn=True)
    assert math.degrees(resumed_yaw) == pytest.approx(23)
    assert math.degrees(math.atan2(resumed_head[1, 0], resumed_head[0, 0])) == pytest.approx(43)


@pytest.mark.parametrize(
    "angle,head_yaw,expected", [(0, 0, 90), (math.pi, 0, -90), (math.pi / 2, 35, 35), (0, 35, 125)]
)
def test_daemon_readings_use_current_head_orientation(monkeypatch, angle, head_yaw, expected) -> None:
    """Remote board readings become world bearings, including when the head is already turned."""
    robot = MagicMock()
    robot.client.host = "robot.local"
    robot.client.port = 8000
    robot.get_current_head_pose.return_value = create_head_pose(yaw=head_yaw, degrees=True)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, json={"angle": angle, "speech_detected": True})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("reachy_mini_conversation_app.speaker_tracking.httpx.Client", lambda **kwargs: client)
    received = threading.Event()
    bearings = []

    def on_direction(yaw: float, speech: bool, observed_at: float) -> None:
        bearings.append((yaw, speech))
        received.set()

    tracker = SpeakerTracking()
    tracker.start(robot, on_direction)
    try:
        assert received.wait(timeout=1.0)
    finally:
        tracker.stop()
    assert requests[0] == "http://robot.local:8000/api/state/doa"
    assert math.degrees(bearings[0][0]) == pytest.approx(expected)
    assert bearings[0][1] is True


def test_daemon_failure_is_logged_and_retried(monkeypatch, caplog) -> None:
    """An unavailable direction endpoint does not stop later readings."""
    robot = MagicMock()
    robot.client.host = "robot.local"
    robot.client.port = 8000
    robot.get_current_head_pose.return_value = np.eye(4)
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"angle": math.pi / 2, "speech_detected": True})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("reachy_mini_conversation_app.speaker_tracking.httpx.Client", lambda **kwargs: client)
    received = threading.Event()
    tracker = SpeakerTracking()
    tracker.start(robot, lambda yaw, speech, observed_at: received.set())
    try:
        assert received.wait(timeout=1.0)
    finally:
        tracker.stop()
    assert "Speaker direction unavailable" in caplog.text
