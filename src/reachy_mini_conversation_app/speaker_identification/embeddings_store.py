"""Store and compare speaker voice embeddings."""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class SpeakerEmbeddingsStore:
    """Store and compare speaker voice embeddings using cosine similarity."""

    def __init__(self, embeddings_path: Path | str) -> None:
        """Initialize the embeddings store.

        Args:
            embeddings_path: Path to the file where embeddings will be stored (.npz format).

        """
        self.embeddings_path = Path(embeddings_path).expanduser()
        self.embeddings: dict[str, NDArray[np.float32]] = {}
        self._load()

    def _load(self) -> None:
        """Load embeddings from disk if the file exists."""
        if self.embeddings_path.exists():
            try:
                data = np.load(self.embeddings_path, allow_pickle=False)
                self.embeddings = {name: data[name] for name in data.files}
                logger.info("Loaded %d speaker embeddings from %s", len(self.embeddings), self.embeddings_path)
            except Exception as e:
                logger.warning("Failed to load embeddings from %s: %s", self.embeddings_path, e)
                self.embeddings = {}
        else:
            logger.debug("No existing embeddings file at %s", self.embeddings_path)

    def save(self) -> None:
        """Save embeddings to disk."""
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(self.embeddings_path), **self.embeddings)  # type: ignore[arg-type]
        logger.debug("Saved %d speaker embeddings to %s", len(self.embeddings), self.embeddings_path)

    def register(self, name: str, embedding: NDArray[np.float32]) -> None:
        """Register a new speaker with their voice embedding.

        Args:
            name: The speaker's name (identifier).
            embedding: The speaker's voice embedding vector.

        """
        # Normalize the embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.embeddings[name] = embedding.astype(np.float32)
        self.save()
        logger.info("Registered speaker: %s", name)

    def identify(self, embedding: NDArray[np.float32], threshold: float = 0.25) -> tuple[str | None, float]:
        """Identify the speaker closest to the given embedding.

        Args:
            embedding: The voice embedding to identify.
            threshold: Minimum cosine similarity to consider a match.

        Returns:
            A tuple of (speaker_name, confidence). If no match is found above
            the threshold, returns (None, 0.0).

        """
        if not self.embeddings:
            return None, 0.0

        # Normalize input embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_match: str | None = None
        best_score: float = 0.0

        for name, stored_embedding in self.embeddings.items():
            # Cosine similarity (embeddings are already normalized)
            score = float(np.dot(embedding, stored_embedding))
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold:
            logger.debug("Identified speaker: %s (score: %.3f)", best_match, best_score)
            return best_match, best_score

        logger.debug("No speaker match above threshold %.2f (best: %.3f)", threshold, best_score)
        return None, 0.0

    def list_speakers(self) -> list[str]:
        """Return a list of all registered speaker names."""
        return list(self.embeddings.keys())

    def remove(self, name: str) -> bool:
        """Remove a speaker from the store.

        Args:
            name: The speaker's name to remove.

        Returns:
            True if the speaker was removed, False if not found.

        """
        if name in self.embeddings:
            del self.embeddings[name]
            self.save()
            logger.info("Removed speaker: %s", name)
            return True
        logger.warning("Speaker not found: %s", name)
        return False

    def __len__(self) -> int:
        """Return the number of registered speakers."""
        return len(self.embeddings)

    def __contains__(self, name: str) -> bool:
        """Check if a speaker is registered."""
        return name in self.embeddings
