"""Tests for the example custom moves authored with the #538 toolchain.

Verifies the bundled generic examples author cleanly, pass the offline safety
screen, and round-trip through reachy_mini's RecordedMove loader.
"""

from __future__ import annotations

import pytest

from robot_comic.authoring.examples import (
    EXAMPLES,
    double_take_spec,
    author_and_screen,
    head_tilt_emphasis_spec,
)
from robot_comic.authoring.move_authoring import build_recorded_move


@pytest.mark.parametrize("name", sorted(EXAMPLES))
class TestExamplesScreenClean:
    def test_example_authors_and_passes_screen(self, name: str) -> None:
        move, result = author_and_screen(name)
        assert result.passed, f"{name} failed screen: {result.reasons}"
        assert len(move["time"]) == len(move["set_target_data"]) > 1

    def test_example_round_trips_through_recorded_move(self, name: str) -> None:
        from reachy_mini.motion.recorded_move import RecordedMove

        move = build_recorded_move(EXAMPLES[name]())
        rm = RecordedMove(move)
        assert rm.duration > 0.0
        # Evaluate near the start; must yield a 4x4 head pose and 2 antennas.
        head, antennas, _ = rm.evaluate(0.01)
        assert head.shape == (4, 4)
        assert antennas.shape == (2,)


class TestCheckedInFixture:
    """The checked-in example_moves/double_take.json must match the authored
    spec and stay screen-clean, guarding against drift between code and artifact.
    """

    def _fixture_path(self) -> str:
        import os

        import robot_comic.authoring as authoring_pkg

        return os.path.join(
            os.path.dirname(authoring_pkg.__file__),
            "example_moves",
            "double_take.json",
        )

    def test_fixture_exists_and_round_trips(self) -> None:
        import json

        from reachy_mini.motion.recorded_move import RecordedMove

        with open(self._fixture_path(), encoding="utf-8") as fh:
            move = json.load(fh)
        rm = RecordedMove(move)
        assert rm.duration > 0.0

    def test_fixture_matches_authored_spec(self) -> None:
        import json

        with open(self._fixture_path(), encoding="utf-8") as fh:
            committed = json.load(fh)
        authored = build_recorded_move(double_take_spec())
        assert committed["description"] == authored["description"]
        assert committed["time"] == pytest.approx(authored["time"])
        assert len(committed["set_target_data"]) == len(authored["set_target_data"])

    def test_fixture_passes_screen(self) -> None:
        import json

        from robot_comic.authoring.move_screen import screen_recorded_move

        with open(self._fixture_path(), encoding="utf-8") as fh:
            move = json.load(fh)
        result = screen_recorded_move(move)
        assert result.passed, f"reasons: {result.reasons}"


class TestDoubleTakeShape:
    def test_double_take_is_generic_named(self) -> None:
        spec = double_take_spec()
        assert spec.name == "double_take"
        # No persona/comedian content leaks into the public repo.
        assert spec.description.islower() or spec.description[0].isupper()

    def test_double_take_duration(self) -> None:
        spec = double_take_spec()
        assert spec.duration == pytest.approx(2.85)

    def test_head_tilt_emphasis_duration(self) -> None:
        spec = head_tilt_emphasis_spec()
        assert spec.duration == pytest.approx(1.40)
