"""Tests for SpeakerEmbeddingsStore."""

from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app.speaker_identification import SpeakerEmbeddingsStore


class TestSpeakerEmbeddingsStore:
    """Tests for the SpeakerEmbeddingsStore class."""

    def test_register_speaker(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test registering a new speaker."""
        embeddings_store.register("Alice", sample_embedding)

        assert "Alice" in embeddings_store
        assert len(embeddings_store) == 1
        assert "Alice" in embeddings_store.list_speakers()

    def test_register_multiple_speakers(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
        another_embedding: np.ndarray,
    ) -> None:
        """Test registering multiple speakers."""
        embeddings_store.register("Alice", sample_embedding)
        embeddings_store.register("Bob", another_embedding)

        assert len(embeddings_store) == 2
        assert set(embeddings_store.list_speakers()) == {"Alice", "Bob"}

    def test_identify_known_speaker(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test identifying a registered speaker."""
        embeddings_store.register("Alice", sample_embedding)

        # Same embedding should match with high confidence
        speaker, confidence = embeddings_store.identify(sample_embedding)

        assert speaker == "Alice"
        assert confidence > 0.99  # Should be very close to 1.0

    def test_identify_unknown_speaker(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
        another_embedding: np.ndarray,
    ) -> None:
        """Test that unknown speaker returns None when below threshold."""
        embeddings_store.register("Alice", sample_embedding)

        # Different embedding should not match with high threshold
        speaker, confidence = embeddings_store.identify(another_embedding, threshold=0.9)

        assert speaker is None
        assert confidence == 0.0

    def test_identify_empty_store(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test identification with no registered speakers."""
        speaker, confidence = embeddings_store.identify(sample_embedding)

        assert speaker is None
        assert confidence == 0.0

    def test_save_and_load(
        self,
        temp_embeddings_path: Path,
        sample_embedding: np.ndarray,
        another_embedding: np.ndarray,
    ) -> None:
        """Test that embeddings persist across store instances."""
        # Create store and register speakers
        store1 = SpeakerEmbeddingsStore(temp_embeddings_path)
        store1.register("Alice", sample_embedding)
        store1.register("Bob", another_embedding)

        # Create new store instance from same path
        store2 = SpeakerEmbeddingsStore(temp_embeddings_path)

        assert len(store2) == 2
        assert "Alice" in store2
        assert "Bob" in store2

        # Verify embeddings are correct by identification
        speaker, _ = store2.identify(sample_embedding)
        assert speaker == "Alice"

    def test_remove_speaker(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
    ) -> None:
        """Test removing a registered speaker."""
        embeddings_store.register("Alice", sample_embedding)
        assert "Alice" in embeddings_store

        result = embeddings_store.remove("Alice")

        assert result is True
        assert "Alice" not in embeddings_store
        assert len(embeddings_store) == 0

    def test_remove_nonexistent_speaker(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
    ) -> None:
        """Test removing a speaker that doesn't exist."""
        result = embeddings_store.remove("Unknown")

        assert result is False

    def test_list_speakers_empty(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
    ) -> None:
        """Test listing speakers when store is empty."""
        speakers = embeddings_store.list_speakers()

        assert speakers == []

    def test_cosine_similarity_normalized(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
    ) -> None:
        """Test that embeddings are normalized for proper cosine similarity."""
        # Create two embeddings with different magnitudes but same direction
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([10.0, 0.0, 0.0], dtype=np.float32)  # Same direction, different magnitude

        embeddings_store.register("Alice", embedding1)

        # Should match perfectly despite different magnitude
        speaker, confidence = embeddings_store.identify(embedding2)

        assert speaker == "Alice"
        assert confidence > 0.99

    def test_overwrite_speaker(
        self,
        embeddings_store: SpeakerEmbeddingsStore,
        sample_embedding: np.ndarray,
        another_embedding: np.ndarray,
    ) -> None:
        """Test that re-registering a speaker updates their embedding."""
        embeddings_store.register("Alice", sample_embedding)
        embeddings_store.register("Alice", another_embedding)

        assert len(embeddings_store) == 1

        # Should now match the new embedding
        speaker, confidence = embeddings_store.identify(another_embedding)
        assert speaker == "Alice"
        assert confidence > 0.99

    def test_load_nonexistent_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading from a path that doesn't exist yet."""
        path = tmp_path / "nonexistent" / "embeddings.npz"
        store = SpeakerEmbeddingsStore(path)

        assert len(store) == 0
        assert store.list_speakers() == []

    def test_path_expansion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that ~ in path is expanded."""
        # Mock expanduser to return tmp_path
        monkeypatch.setenv("HOME", str(tmp_path))

        store = SpeakerEmbeddingsStore("~/test_embeddings.npz")

        assert store.embeddings_path.is_absolute()
        assert "~" not in str(store.embeddings_path)
