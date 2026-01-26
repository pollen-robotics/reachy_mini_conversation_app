"""Speaker identification module for Reachy Mini Conversation App."""

from reachy_mini_conversation_app.speaker_identification.worker import SpeakerIDWorker
from reachy_mini_conversation_app.speaker_identification.embeddings_store import SpeakerEmbeddingsStore


__all__ = ["SpeakerEmbeddingsStore", "SpeakerIDWorker"]
