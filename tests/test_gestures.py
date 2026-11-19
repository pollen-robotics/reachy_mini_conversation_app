import re
import json
import logging
from types import SimpleNamespace
from typing import Any
from pathlib import Path

import numpy as np
import pytest

import reachy_mini_conversation_app.gestures as gestures_mod
from reachy_mini_conversation_app.gestures import GestureLibrary


_DIMENSIONS = 64
_WORD_IDS = {"accomplished": 10, "died": 11, "hello": 12, "weather": 13}
_LABELS = {"say": ["success1", "downcast1", "welcoming2", "none"], "react": ["success1", "downcast1", "none"]}
# Which word each label's output row responds to; an unknown sentence scores the same for every label.
_LABEL_WORDS = {"success1": "accomplished", "downcast1": "died", "welcoming2": "hello", "none": "weather"}


class _OneHotEncoder:
    """Stand in for SentencePiece and the ONNX session: each known word lights up its own dimension."""

    def encode(self, sentences: list[str]) -> list[list[int]]:
        return [[_WORD_IDS.get(word, 40) for word in re.findall(r"\w+", sentence.lower())] for sentence in sentences]

    def run(self, _outputs: None, inputs: dict[str, Any]) -> list[Any]:
        return [np.eye(_DIMENSIONS, dtype=np.float32)[inputs["input_ids"]]]


@pytest.fixture
def model_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Replace the downloaded model with the fake encoder and output layers that read it."""
    heads = {}
    for kind, labels in _LABELS.items():
        weight = np.zeros((len(labels), _DIMENSIONS), dtype=np.float32)
        for row, label in enumerate(labels):
            weight[row, _WORD_IDS[_LABEL_WORDS[label]] + 1] = 2.0  # +1: the model shifts SentencePiece ids
        heads[f"{kind}_weight"], heads[f"{kind}_bias"] = weight, np.zeros(len(labels), dtype=np.float32)
    np.savez(tmp_path / "gesture_heads.npz", **heads)
    (tmp_path / "gesture_labels.json").write_text(json.dumps(_LABELS))
    encoder = _OneHotEncoder()
    monkeypatch.setattr(gestures_mod, "hf_hub_download", lambda _repo, filename, **_kwargs: str(tmp_path / filename))
    monkeypatch.setattr(gestures_mod, "SentencePieceProcessor", lambda model_file: encoder)
    monkeypatch.setattr(gestures_mod.ort, "InferenceSession", lambda *_args, **_kwargs: encoder)


@pytest.fixture
def recorded_move_names(monkeypatch: pytest.MonkeyPatch, model_files: None) -> list[str]:
    """Replace the emotions library; the returned list is what the library offers."""
    move_names = ["success1", "downcast1", "welcoming2"]
    durations = {"success1": 2.5, "downcast1": 6.0, "welcoming2": 4.3}
    recorded_moves = SimpleNamespace(
        list_moves=lambda: move_names,
        get=lambda name: SimpleNamespace(duration=durations[name], evaluate=lambda t: (t, None, None)),
    )
    monkeypatch.setattr(gestures_mod, "RecordedMoves", lambda _dataset: recorded_moves)
    return move_names


def test_match_picks_the_move_the_model_scores_highest(recorded_move_names: list[str]) -> None:
    """What the robot says and what it hears go through their own output layers."""
    gestures = GestureLibrary()

    said = gestures.match("Mission accomplished!", "say")
    heard = gestures.match("my dog died yesterday", "react")

    assert said is not None and said.emotion_name == "success1"
    assert said.duration == 2.5
    assert heard is not None and heard.emotion_name == "downcast1"


def test_match_returns_none_for_neutral_unsure_or_empty_sentences(recorded_move_names: list[str]) -> None:
    """The "none" label, a probability under the threshold and blank text all mean no gesture."""
    gestures = GestureLibrary()

    assert gestures.match("Nice weather today.", "say") is None
    assert gestures.match("Quantum chromodynamics describes gluons.", "say") is None
    assert gestures.match("   ", "react") is None


def test_match_logs_its_decision_and_duration(
    recorded_move_names: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Every sentence logs the chosen move, its probability and how long the model took."""
    caplog.set_level(logging.DEBUG, logger=gestures_mod.__name__)

    GestureLibrary().match("hello there", "say")

    assert re.search(r"Gesture model: welcoming2 for say sentence \(p=\d\.\d\d, 4 tokens, \d+ ms\)", caplog.text)


def test_long_turns_are_read_from_their_end(recorded_move_names: list[str]) -> None:
    """Past the model's 64 tokens, the start of a long turn is dropped and its ending decides the gesture."""
    rambling = " ".join(["so"] * 80)

    gesture = GestureLibrary().match(f"{rambling} and in the end my dog died", "react")

    assert gesture is not None and gesture.emotion_name == "downcast1"


def test_long_moves_are_cut_to_gesture_length_and_slow_to_a_stop(recorded_move_names: list[str]) -> None:
    """A six-second recording plays for three seconds, easing out over the last half second."""
    gesture = GestureLibrary().match("my dog died yesterday", "react")

    assert gesture is not None and gesture.duration == 3.0
    assert gesture.evaluate(1.0)[0] == 1.0
    assert gesture.evaluate(2.75)[0] == pytest.approx(2.6875)
    assert gesture.evaluate(3.0)[0] == pytest.approx(2.75)


def test_gestures_whose_move_is_unavailable_are_skipped(
    recorded_move_names: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """A move missing from the emotions library is reported and never returned."""
    recorded_move_names.remove("success1")
    caplog.set_level(logging.WARNING, logger=gestures_mod.__name__)

    gestures = GestureLibrary()

    assert gestures.match("Mission accomplished.", "say") is None
    assert "success1" in caplog.text
