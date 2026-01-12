"""Tests for SpeakerIDWorker."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from reachy_mini_conversation_app.speaker_identification.worker import (
    TARGET_SAMPLE_RATE,
    SpeakerIDWorker,
)


class TestSpeakerIDWorker:
    """Tests for the SpeakerIDWorker class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test worker initialization."""
        worker = SpeakerIDWorker(
            model_source="speechbrain/spkrec-ecapa-voxceleb",
            threshold=0.3,
            embeddings_path=tmp_path / "embeddings.npz",
            device="cpu",
        )

        assert worker._threshold == 0.3
        assert worker._device == "cpu"
        assert worker._encoder is None  # Lazy loading

    def test_start_stop(self, tmp_path: Path) -> None:
        """Test starting and stopping the worker thread."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()

        worker.stop()
        assert worker._thread is None or not worker._thread.is_alive()

    def test_start_already_running(self, tmp_path: Path) -> None:
        """Test that starting an already running worker logs a warning."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        worker.start()
        worker.start()  # Should not raise, just warn

        worker.stop()

    def test_feed_audio(self, tmp_path: Path) -> None:
        """Test feeding audio to the worker."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        audio = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        worker.feed_audio(audio, 16000)

        assert not worker._audio_queue.empty()

    def test_feed_audio_queue_full(self, tmp_path: Path) -> None:
        """Test that old audio is dropped when queue is full."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        # Fill the queue
        for i in range(15):  # More than maxsize (10)
            audio = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
            worker.feed_audio(audio, 16000)

        # Queue should not exceed maxsize
        assert worker._audio_queue.qsize() <= 10

    def test_get_current_speaker_default(self, tmp_path: Path) -> None:
        """Test that current speaker is None by default."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        speaker, confidence = worker.get_current_speaker()

        assert speaker is None
        assert confidence == 0.0

    def test_resample_audio(self, tmp_path: Path) -> None:
        """Test audio resampling."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        # 1 second of audio at 24kHz
        audio = np.random.rand(24000).astype(np.float32)

        resampled = worker._resample_audio(audio, 24000, 16000)

        # Should be ~1 second at 16kHz
        assert len(resampled) == 16000
        assert resampled.dtype == np.float32

    def test_resample_audio_same_rate(self, tmp_path: Path) -> None:
        """Test that resampling with same rate returns original."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        audio = np.random.rand(16000).astype(np.float32)

        resampled = worker._resample_audio(audio, 16000, 16000)

        np.testing.assert_array_equal(resampled, audio)

    def test_start_registration(self, tmp_path: Path) -> None:
        """Test starting speaker registration."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        worker.start_registration("Alice")

        assert worker.is_registering()
        assert worker._registration_name == "Alice"

    def test_cancel_registration(self, tmp_path: Path) -> None:
        """Test cancelling speaker registration."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        worker.start_registration("Alice")
        worker.cancel_registration()

        assert not worker.is_registering()
        assert worker._registration_name is None

    def test_finish_registration_no_audio(self, tmp_path: Path) -> None:
        """Test finishing registration with no audio collected."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        worker.start_registration("Alice")
        result = worker.finish_registration()

        assert result is False

    def test_finish_registration_no_registration(self, tmp_path: Path) -> None:
        """Test finishing registration when none is in progress."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        result = worker.finish_registration()

        assert result is False

    def test_list_speakers(self, tmp_path: Path) -> None:
        """Test listing registered speakers."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        # Manually register a speaker via the embeddings store
        embedding = np.random.rand(192).astype(np.float32)
        worker._embeddings_store.register("Alice", embedding)

        speakers = worker.list_speakers()

        assert "Alice" in speakers

    def test_remove_speaker(self, tmp_path: Path) -> None:
        """Test removing a registered speaker."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        # Register and then remove
        embedding = np.random.rand(192).astype(np.float32)
        worker._embeddings_store.register("Alice", embedding)

        result = worker.remove_speaker("Alice")

        assert result is True
        assert "Alice" not in worker.list_speakers()


class TestSpeakerIDWorkerWithMockedModel:
    """Tests that mock the SpeechBrain model for CI."""

    @patch("reachy_mini_conversation_app.speaker_identification.worker.SpeakerIDWorker._load_encoder")
    def test_compute_embedding(self, mock_load_encoder: MagicMock, tmp_path: Path) -> None:
        """Test embedding computation with mocked encoder."""
        import torch

        # Mock encoder that returns a fixed embedding
        mock_encoder = MagicMock()
        mock_embedding = torch.randn(1, 192)
        mock_encoder.encode_batch.return_value = mock_embedding
        mock_load_encoder.return_value = mock_encoder

        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        audio = np.random.rand(16000).astype(np.float32)
        embedding = worker._compute_embedding(audio)

        assert embedding is not None
        assert embedding.shape == (192,)
        mock_encoder.encode_batch.assert_called_once()

    @patch("reachy_mini_conversation_app.speaker_identification.worker.SpeakerIDWorker._load_encoder")
    def test_compute_embedding_failure(self, mock_load_encoder: MagicMock, tmp_path: Path) -> None:
        """Test embedding computation failure handling."""
        mock_load_encoder.side_effect = Exception("Model loading failed")

        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        audio = np.random.rand(16000).astype(np.float32)
        embedding = worker._compute_embedding(audio)

        assert embedding is None

    @patch("reachy_mini_conversation_app.speaker_identification.worker.SpeakerIDWorker._compute_embedding")
    def test_working_loop_identifies_speaker(self, mock_compute: MagicMock, tmp_path: Path) -> None:
        """Test that the working loop identifies speakers."""
        # Setup mock embedding
        mock_embedding = np.random.rand(192).astype(np.float32)
        mock_compute.return_value = mock_embedding

        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")

        # Register a speaker
        worker._embeddings_store.register("Alice", mock_embedding)

        # Start worker and feed audio
        worker.start()

        audio = np.random.rand(16000).astype(np.float32)  # 1 second at 16kHz
        worker.feed_audio(audio, TARGET_SAMPLE_RATE)

        # Wait for processing
        time.sleep(0.5)

        speaker, confidence = worker.get_current_speaker()
        worker.stop()

        assert speaker == "Alice"
        assert confidence > 0.9

    @patch("reachy_mini_conversation_app.speaker_identification.worker.SpeakerIDWorker._compute_embedding")
    def test_working_loop_registration_mode(self, mock_compute: MagicMock, tmp_path: Path) -> None:
        """Test that audio is collected during registration mode."""
        mock_embedding = np.random.rand(192).astype(np.float32)
        mock_compute.return_value = mock_embedding

        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")
        worker.start()

        # Start registration
        worker.start_registration("Bob")

        # Feed some audio
        for _ in range(3):
            audio = np.random.rand(8000).astype(np.float32)  # 0.5 seconds
            worker.feed_audio(audio, TARGET_SAMPLE_RATE)
            time.sleep(0.2)

        # Audio should be collected, not processed for identification
        speaker, _ = worker.get_current_speaker()
        assert speaker is None  # Not identified during registration

        # Finish registration
        result = worker.finish_registration()
        worker.stop()

        assert result is True
        assert "Bob" in worker.list_speakers()


class TestSpeakerIDWorkerThreadSafety:
    """Tests for thread safety of the worker."""

    def test_concurrent_access(self, tmp_path: Path) -> None:
        """Test concurrent access to worker state."""
        worker = SpeakerIDWorker(embeddings_path=tmp_path / "embeddings.npz")
        worker.start()

        errors: list[Exception] = []

        def feed_audio_thread() -> None:
            try:
                for _ in range(100):
                    audio = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
                    worker.feed_audio(audio, 16000)
            except Exception as e:
                errors.append(e)

        def get_speaker_thread() -> None:
            try:
                for _ in range(100):
                    worker.get_current_speaker()
            except Exception as e:
                errors.append(e)

        import threading

        threads = [
            threading.Thread(target=feed_audio_thread),
            threading.Thread(target=feed_audio_thread),
            threading.Thread(target=get_speaker_thread),
            threading.Thread(target=get_speaker_thread),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        worker.stop()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
