"""Unit tests for the gesture library.

All tests are pure-Python: no robot hardware, no SDK I/O, no network.
The MovementManager is mocked so we can assert that queue_move() is called
the expected number of times without running the control loop.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager() -> MagicMock:
    """Return a mock that satisfies the MovementManager interface used by gestures."""
    manager = MagicMock()
    manager.queue_move = MagicMock()
    manager.speed_factor = 1.0  # must be a float so GotoQueueMove clamping works
    return manager


# ---------------------------------------------------------------------------
# GestureRegistry — list_gestures
# ---------------------------------------------------------------------------


class TestListGestures:
    """list_gestures() must return all 7 canonical gestures."""

    EXPECTED = sorted(["shrug", "nod_yes", "nod_no", "point_left", "point_right", "scan", "lean_in"])

    def test_returns_all_canonical_gestures(self) -> None:
        from robot_comic.gestures import registry

        assert registry.list_gestures() == self.EXPECTED

    def test_returns_sorted_list(self) -> None:
        from robot_comic.gestures import registry

        names = registry.list_gestures()
        assert names == sorted(names)

    def test_count_is_seven(self) -> None:
        from robot_comic.gestures import registry

        assert len(registry.list_gestures()) == 7


# ---------------------------------------------------------------------------
# GestureRegistry — register / overwrite
# ---------------------------------------------------------------------------


class TestRegister:
    """register() stores a new gesture and allows overwriting."""

    def test_register_and_list(self) -> None:
        from robot_comic.gestures.registry import GestureRegistry

        reg = GestureRegistry()
        reg.register("wave", lambda m: None)
        assert "wave" in reg.list_gestures()

    def test_overwrite_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from robot_comic.gestures.registry import GestureRegistry

        reg = GestureRegistry()
        reg.register("wave", lambda m: None)
        with caplog.at_level(logging.WARNING, logger="robot_comic.gestures.registry"):
            reg.register("wave", lambda m: None)
        assert any("overwriting" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# GestureRegistry — play
# ---------------------------------------------------------------------------


class TestPlay:
    """play() calls queue_move() at least once and raises KeyError for unknowns."""

    def test_shrug_queues_primary_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("shrug", manager)

        assert manager.queue_move.called
        assert manager.queue_move.call_count >= 1

    def test_nod_yes_queues_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("nod_yes", manager)
        assert manager.queue_move.call_count >= 1

    def test_nod_no_queues_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("nod_no", manager)
        assert manager.queue_move.call_count >= 1

    def test_point_left_queues_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("point_left", manager)
        assert manager.queue_move.call_count >= 1

    def test_point_right_queues_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("point_right", manager)
        assert manager.queue_move.call_count >= 1

    def test_scan_queues_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("scan", manager)
        assert manager.queue_move.call_count >= 1

    def test_lean_in_queues_moves(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play("lean_in", manager)
        assert manager.queue_move.call_count >= 1

    def test_unknown_gesture_raises_key_error(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        with pytest.raises(KeyError, match="Unknown gesture"):
            registry.play("does_not_exist", manager)

    def test_unknown_gesture_error_lists_available(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        with pytest.raises(KeyError) as exc_info:
            registry.play("bogus_name", manager)
        # The error message should hint at what's available
        assert "shrug" in str(exc_info.value)

    def test_unknown_gesture_does_not_call_queue_move(self) -> None:
        from robot_comic.gestures import registry

        manager = _make_manager()
        with pytest.raises(KeyError):
            registry.play("nothing", manager)
        manager.queue_move.assert_not_called()


# ---------------------------------------------------------------------------
# GotoQueueMove payloads — each gesture enqueues GotoQueueMove instances
# ---------------------------------------------------------------------------


class TestGesturePayloads:
    """Verify that queue_move receives GotoQueueMove instances (primary moves)."""

    def test_shrug_enqueues_goto_moves(self) -> None:
        from robot_comic.gestures import registry
        from robot_comic.dance_emotion_moves import GotoQueueMove

        manager = _make_manager()
        registry.play("shrug", manager)

        for c in manager.queue_move.call_args_list:
            move = c.args[0]
            assert isinstance(move, GotoQueueMove), f"Expected GotoQueueMove, got {type(move).__name__}"

    def test_all_gestures_enqueue_goto_moves(self) -> None:
        from robot_comic.gestures import registry
        from robot_comic.dance_emotion_moves import GotoQueueMove

        for gesture_name in registry.list_gestures():
            manager = _make_manager()
            registry.play(gesture_name, manager)
            for c in manager.queue_move.call_args_list:
                move = c.args[0]
                assert isinstance(move, GotoQueueMove), (
                    f"Gesture {gesture_name!r}: expected GotoQueueMove, got {type(move).__name__}"
                )


# ---------------------------------------------------------------------------
# Duration sanity — each gesture stays within the 0.5–2.0 s target window
# ---------------------------------------------------------------------------


class TestGestureDuration:
    """Total queued duration per gesture should be within 0.5–2.0 s."""

    MIN_S = 0.5
    MAX_S = 2.0

    def _total_duration(self, gesture_name: str) -> float:
        from robot_comic.gestures import registry

        manager = _make_manager()
        registry.play(gesture_name, manager)
        total = sum(c.args[0].duration for c in manager.queue_move.call_args_list)
        return total

    def test_all_gestures_within_duration_window(self) -> None:
        from robot_comic.gestures import registry

        for name in registry.list_gestures():
            total = self._total_duration(name)
            assert self.MIN_S <= total <= self.MAX_S, (
                f"Gesture {name!r}: total duration {total:.2f}s is outside [{self.MIN_S}, {self.MAX_S}]"
            )


# ---------------------------------------------------------------------------
# Gesture tool — Gesture(Tool) wrapper
# ---------------------------------------------------------------------------


class TestGestureTool:
    """The Gesture tool should delegate to the registry and return status."""

    @pytest.mark.asyncio
    async def test_known_gesture_returns_queued_status(self) -> None:
        from robot_comic.tools.gesture import Gesture

        tool = Gesture()
        manager = _make_manager()
        deps = MagicMock()
        deps.movement_manager = manager

        result = await tool(deps, name="shrug")

        assert result["status"] == "queued"
        assert result["gesture"] == "shrug"
        manager.queue_move.assert_called()

    @pytest.mark.asyncio
    async def test_unknown_gesture_returns_error(self) -> None:
        from robot_comic.tools.gesture import Gesture

        tool = Gesture()
        deps = MagicMock()
        deps.movement_manager = _make_manager()

        result = await tool(deps, name="does_not_exist")

        assert "error" in result
        assert "does_not_exist" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_name_returns_error(self) -> None:
        from robot_comic.tools.gesture import Gesture

        tool = Gesture()
        deps = MagicMock()
        deps.movement_manager = _make_manager()

        result = await tool(deps)

        assert "error" in result

    def test_tool_name_is_gesture(self) -> None:
        from robot_comic.tools.gesture import Gesture

        assert Gesture.name == "gesture"

    def test_all_canonical_names_in_enum(self) -> None:
        from robot_comic.gestures import registry
        from robot_comic.tools.gesture import Gesture

        tool = Gesture()
        enum_values = tool.parameters_schema["properties"]["name"]["enum"]
        for g in registry.list_gestures():
            assert g in enum_values, f"Gesture {g!r} missing from tool schema enum"
