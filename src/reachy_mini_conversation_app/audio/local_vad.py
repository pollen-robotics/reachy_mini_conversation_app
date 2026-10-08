import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from huggingface_hub import cached_assets_path


logger = logging.getLogger(__name__)
_MODEL_REVISION = "7e30209a3e901f9842f81b225f3e93d8199902b1"
_MODEL_FILENAME = "silero_vad_16k_op15.onnx"
_MODEL_FILES = {
    _MODEL_FILENAME: (
        "src/silero_vad/data/silero_vad_16k_op15.onnx",
        "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49",
    ),
    "LICENSE": ("LICENSE", "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b"),
}


class SileroVAD:
    """Detect speech in streaming 16 kHz mono PCM for listening reactions and confirmed barge-in."""

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512
    SILENCE_SAMPLES = 6400  # 400 ms
    BARGE_IN_SPEECH_SAMPLES = 6144  # 384 ms
    BARGE_IN_SILENCE_SAMPLES = 1024  # 64 ms

    def __init__(self) -> None:
        """Download or reuse the cached Silero model and load it with a single CPU thread."""
        model_directory = cached_assets_path("reachy_mini_conversation_app", "silero-vad", _MODEL_REVISION)
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            for filename, (upstream_path, checksum) in _MODEL_FILES.items():
                cached_file = model_directory / filename
                if cached_file.is_file():
                    if hashlib.sha256(cached_file.read_bytes()).hexdigest() == checksum:
                        continue
                    logger.warning("Cached Silero file failed checksum verification: %s", cached_file)
                logger.info("Downloading Silero VAD file: %s", filename)
                response = client.get(
                    f"https://raw.githubusercontent.com/snakers4/silero-vad/{_MODEL_REVISION}/{upstream_path}"
                )
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != checksum:
                    raise ValueError(f"Downloaded Silero file failed checksum verification: {filename}")
                download = NamedTemporaryFile(dir=model_directory, delete=False)
                download_path = Path(download.name)
                try:
                    with download:
                        download.write(response.content)
                    download_path.replace(cached_file)
                finally:
                    download_path.unlink(missing_ok=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model_path = model_directory / _MODEL_FILENAME
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        """Clear speech state and buffered audio between microphone streams."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._speaking = False
        self._silence_samples = 0
        self.barge_in_candidate = False
        self.barge_in_confirmed = False
        self._barge_in_speech_samples = 0
        self._barge_in_silence_samples = 0

    def process(self, audio: NDArray[np.int16], sample_rate: int, *, detect_barge_in: bool = False) -> bool:
        """Return whether speech is active, holding through 400 ms of silence."""
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError("Local VAD requires 16 kHz microphone audio")
        self.barge_in_confirmed = False
        if not detect_barge_in:
            self.barge_in_candidate = False
            self._barge_in_speech_samples = 0
            self._barge_in_silence_samples = 0
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
            if detect_barge_in:
                if speech_probability >= 0.6 or (self.barge_in_candidate and speech_probability >= 0.45):
                    self.barge_in_candidate = True
                    self._barge_in_speech_samples += self.CHUNK_SAMPLES
                    self._barge_in_silence_samples = 0
                    if self._barge_in_speech_samples >= self.BARGE_IN_SPEECH_SAMPLES:
                        self.barge_in_confirmed = True
                elif self.barge_in_candidate:
                    self._barge_in_silence_samples += self.CHUNK_SAMPLES
                    if self._barge_in_silence_samples >= self.BARGE_IN_SILENCE_SAMPLES:
                        self.barge_in_candidate = False
                        self._barge_in_speech_samples = 0
                        self._barge_in_silence_samples = 0
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
