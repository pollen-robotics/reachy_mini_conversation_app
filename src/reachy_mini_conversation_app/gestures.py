import json
import time
import logging
from typing import Literal
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from sentencepiece import SentencePieceProcessor
from huggingface_hub import hf_hub_download

from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


logger = logging.getLogger(__name__)
# MiniLM fine-tuned to pick a gesture, with one output layer for what the robot says and one for what it hears.
_MODEL_REPO = "HuggingFaceM4/reachy-gestures"
_MODEL_REVISION = "027bfbeb49656ecb623f89d329c32c190e712c68"
# The fine-tuned model keeps the base model's tokenizer, published there as a SentencePiece file.
_TOKENIZER_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_TOKENIZER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
_MAX_TOKENS = 64  # the length the model was trained on
_EMBEDDING_SCALE = 10.0  # the output layers read the normalized sentence embedding at this scale
_MIN_PROBABILITY = 0.3  # chosen on held-out sentences; below it the model is often wrong
_MAX_GESTURE_SECONDS = 3.0
_SLOWDOWN_SECONDS = 0.5

GestureKind = Literal["say", "react"]


class GestureMove(EmotionQueueMove):
    """A recorded emotion cut to gesture length, slowing to a stop instead of freezing mid-motion."""

    @property
    def duration(self) -> float:
        """Length of the gesture in seconds."""
        return min(super().duration, _MAX_GESTURE_SECONDS)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate the gesture at time t."""
        slowdown_starts_at = self.duration - _SLOWDOWN_SECONDS
        if self.duration < super().duration and t > slowdown_starts_at:
            progress = min(1.0, (t - slowdown_starts_at) / _SLOWDOWN_SECONDS)
            t = slowdown_starts_at + _SLOWDOWN_SECONDS * (progress - progress**2 / 2)
        return super().evaluate(t)


class GestureLibrary:
    """Pick the recorded move that fits something the robot says or hears, using a small trained classifier."""

    def __init__(self) -> None:
        """Download or reuse the cached gesture model and moves."""
        started = time.perf_counter()
        # Creating the session holds Python's interpreter lock for most of a second on the robot, freezing every
        # other Python thread, audio playback included. It goes first so that pause ends before the greeting plays.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            hf_hub_download(_MODEL_REPO, "model_qint8.onnx", revision=_MODEL_REVISION),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = SentencePieceProcessor(
            model_file=hf_hub_download(_TOKENIZER_REPO, "sentencepiece.bpe.model", revision=_TOKENIZER_REVISION)
        )
        with np.load(hf_hub_download(_MODEL_REPO, "gesture_heads.npz", revision=_MODEL_REVISION)) as heads:
            self._heads: dict[GestureKind, tuple[NDArray[np.float32], NDArray[np.float32]]] = {
                kind: (heads[f"{kind}_weight"], heads[f"{kind}_bias"]) for kind in ("say", "react")
            }
        labels_path = Path(hf_hub_download(_MODEL_REPO, "gesture_labels.json", revision=_MODEL_REVISION))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self._recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        available_moves = set(self._recorded_moves.list_moves())
        # "none" and moves missing from the library map to None: the sentence gets no gesture.
        self._labels: dict[GestureKind, list[str | None]] = {}
        for kind in ("say", "react"):
            for move_name in labels[kind]:
                if move_name != "none" and move_name not in available_moves:
                    logger.warning("Gesture move %s is missing from the emotions library", move_name)
            self._labels[kind] = [name if name in available_moves else None for name in labels[kind]]
        logger.info("Gesture model loaded in %.1f s", time.perf_counter() - started)

    def _embed(self, text: str) -> tuple[NDArray[np.float32], int]:
        # XLM-R ids are the SentencePiece ids shifted by one, between <s> (0) and </s> (2); 3 is unknown.
        # A long turn keeps its end: the robot reacts to how the person finished, the feeling they landed on.
        pieces = self._tokenizer.encode([text])[0][-(_MAX_TOKENS - 2) :]
        input_ids = np.array([[0, *(piece + 1 if piece else 3 for piece in pieces), 2]], dtype=np.int64)
        token_embeddings = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids),
                "token_type_ids": np.zeros_like(input_ids),
            },
        )[0][0]
        sentence_embedding = token_embeddings.mean(axis=0)
        normalized: NDArray[np.float32] = sentence_embedding / np.linalg.norm(sentence_embedding)
        return normalized, input_ids.shape[1]

    def match(self, text: str, kind: GestureKind) -> GestureMove | None:
        """Return the move for something the robot says or hears, or None when it calls for no gesture."""
        if not text.strip():
            return None
        started = time.perf_counter()
        embedding, token_count = self._embed(text)
        weight, bias = self._heads[kind]
        logits = weight @ (embedding * _EMBEDDING_SCALE) + bias
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(probabilities.argmax())
        move_name = self._labels[kind][best] if probabilities[best] >= _MIN_PROBABILITY else None
        logger.debug(
            "Gesture model: %s for %s sentence (p=%.2f, %d tokens, %.0f ms): %s",
            move_name or "none",
            kind,
            probabilities[best],
            token_count,
            (time.perf_counter() - started) * 1000,
            text,
        )
        if move_name is None:
            return None
        return GestureMove(move_name, self._recorded_moves)
