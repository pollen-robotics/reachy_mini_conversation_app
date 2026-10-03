from importlib.resources import files

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class SileroVAD:
    """Detect speech in streaming 16 kHz mono PCM for physical listening reactions."""

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512
    SILENCE_SAMPLES = 6400  # 400 ms

    def __init__(self) -> None:
        """Load the bundled Silero model with a single CPU thread."""
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model_path = files("reachy_mini_conversation_app").joinpath("audio/silero_vad_16k_op15.onnx")
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        """Clear speech state and buffered audio between microphone streams."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._speaking = False
        self._silence_samples = 0

    def process(self, audio: NDArray[np.int16], sample_rate: int) -> bool:
        """Return whether speech is active, holding through 400 ms of silence."""
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError("Local VAD requires 16 kHz microphone audio")
        self._pending = np.concatenate((self._pending, audio.astype(np.float32) / 32768.0))
        offset = 0
        while self._pending.size - offset >= self.CHUNK_SAMPLES:
            chunk = self._pending[offset : offset + self.CHUNK_SAMPLES].reshape(1, -1)
            model_input = np.concatenate((self._context, chunk), axis=1)
            probability, state = self._session.run(
                None,
                {"input": model_input, "state": self._state, "sr": np.array(self.SAMPLE_RATE, dtype=np.int64)},
            )
            self._state = np.asarray(state, dtype=np.float32)
            self._context = chunk[:, -64:].copy()
            speech_probability = float(np.asarray(probability).item())
            if speech_probability >= 0.5:
                self._speaking = True
                self._silence_samples = 0
            elif self._speaking and speech_probability < 0.35:
                self._silence_samples += self.CHUNK_SAMPLES
                if self._silence_samples >= self.SILENCE_SAMPLES:
                    self._speaking = False
                    self._silence_samples = 0
            offset += self.CHUNK_SAMPLES
        self._pending = self._pending[offset:].copy()
        return self._speaking
