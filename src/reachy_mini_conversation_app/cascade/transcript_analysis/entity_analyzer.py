"""Named Entity Recognition (NER) based transcript analyzer using GLiNER."""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Callable, Awaitable

from .base import Reaction, TranscriptAnalyzer


logger = logging.getLogger(__name__)


class EntityAnalyzer(TranscriptAnalyzer):
    """Analyzes transcripts for named entities using GLiNER.

    Features:
    - Uses GLiNER for zero-shot NER
    - Configurable model (small, medium, large)
    - Runs in executor to avoid blocking event loop
    - State tracking to prevent duplicate triggers

    """

    def __init__(
        self,
        entity_callbacks: Dict[str, Callable[..., Awaitable[None]]],
        model_name: str = "urchade/gliner_small-v2.1",
    ):
        """Initialize entity analyzer.

        Args:
            entity_callbacks: Dict mapping entity labels to async callbacks
                            Callback signature: async def(deps, entity_text, entity_label, confidence) -> None
            model_name: GLiNER model name (default: gliner_small-v2.1)

        Raises:
            ImportError: If gliner is not installed

        """
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError(
                "GLiNER not installed. Install with: pip install 'reachy_mini_conversation_app[cascade_gliner]'"
            )

        self.entity_callbacks = entity_callbacks
        self.model_name = model_name
        self.triggered_entities: set[tuple[str, str]] = set()  # (entity_text, label) tuples

        # Load GLiNER model
        logger.info(f"Loading GLiNER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)
        logger.info(f"EntityAnalyzer initialized with {len(entity_callbacks)} entity types")

    async def analyze(self, text: str, is_final: bool) -> List[Reaction]:
        """Analyze text for named entities.

        Args:
            text: Transcript text to analyze
            is_final: Whether this is the final transcript

        Returns:
            List of triggered Reaction objects

        """
        # GLiNER is CPU-bound, run in executor to not block event loop
        import time

        start_time = time.time()
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            None, lambda: self.model.predict_entities(text, list(self.entity_callbacks.keys()))
        )
        elapsed = time.time() - start_time

        logger.info(
            f"🔍 GLiNER analyzed '{text[:50]}...' in {elapsed*1000:.0f}ms, "
            f"found {len(entities)} entities"
        )

        reactions = []

        for entity in entities:
            entity_text: str = entity["text"]
            entity_label: str = entity["label"]
            confidence: float = entity["score"]

            entity_key = (entity_text.lower(), entity_label)

            # Check if this entity hasn't triggered yet
            if entity_key not in self.triggered_entities:
                callback = self.entity_callbacks.get(entity_label)
                if callback:
                    logger.debug(
                        f"Entity match: '{entity_text}' ({entity_label}, " f"confidence={confidence:.2f})"
                    )

                    reactions.append(
                        Reaction(
                            trigger=entity_text,
                            trigger_type=f"entity:{entity_label}",
                            callback=callback,
                            metadata={
                                "entity_text": entity_text,
                                "entity_label": entity_label,
                                "confidence": confidence,
                            },
                        )
                    )

                    # Mark as triggered to prevent duplicates
                    self.triggered_entities.add(entity_key)

        return reactions

    def reset(self) -> None:
        """Reset triggered entities for next conversation."""
        logger.debug(f"Resetting EntityAnalyzer ({len(self.triggered_entities)} entities triggered)")
        self.triggered_entities.clear()
