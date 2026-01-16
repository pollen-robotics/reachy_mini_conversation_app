"""Pytest fixtures for speaker identification tests."""

from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app.speaker_identification import SpeakerEmbeddingsStore


@pytest.fixture
def temp_embeddings_path(tmp_path: Path) -> Path:
    """Create a temporary path for embeddings storage."""
    return tmp_path / "test_embeddings.npz"


@pytest.fixture
def embeddings_store(temp_embeddings_path: Path) -> SpeakerEmbeddingsStore:
    """Create a fresh embeddings store for testing."""
    return SpeakerEmbeddingsStore(temp_embeddings_path)


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Create a sample embedding vector."""
    rng = np.random.default_rng(42)
    embedding = rng.random(192).astype(np.float32)  # ECAPA-TDNN produces 192-dim embeddings
    return embedding


@pytest.fixture
def another_embedding() -> np.ndarray:
    """Create another distinct embedding vector."""
    rng = np.random.default_rng(123)
    embedding = rng.random(192).astype(np.float32)
    return embedding
