"""Keyword-based transcript analyzer."""

from __future__ import annotations
import logging
from typing import Dict, List, Callable, Awaitable

from .base import Reaction, TranscriptAnalyzer


logger = logging.getLogger(__name__)


class KeywordAnalyzer(TranscriptAnalyzer):
    """Analyzes transcripts for keyword matches.

    Features:
    - Case-insensitive matching
    - State tracking to prevent duplicate triggers as transcript grows
    - Simple and fast

    """

    def __init__(self, keyword_callbacks: Dict[str, Callable[..., Awaitable[None]]]):
        """Initialize keyword analyzer.

        Args:
            keyword_callbacks: Dict mapping keywords (case-insensitive) to async callbacks
                              Callback signature: async def(deps: ToolDependencies) -> None

        """
        self.keyword_callbacks = keyword_callbacks
        self.triggered_keywords: set[str] = set()  # Track what already triggered

        logger.info(f"KeywordAnalyzer initialized with {len(keyword_callbacks)} keywords")

    async def analyze(self, text: str, is_final: bool) -> List[Reaction]:
        """Analyze text for keyword matches.

        Args:
            text: Transcript text to analyze
            is_final: Whether this is the final transcript

        Returns:
            List of triggered Reaction objects

        """
        reactions = []
        text_lower = text.lower()

        for keyword, callback in self.keyword_callbacks.items():
            keyword_lower = keyword.lower()

            # Check if keyword is in text and hasn't triggered yet
            if keyword_lower in text_lower and keyword_lower not in self.triggered_keywords:
                logger.debug(f"Keyword match: '{keyword}' in '{text[:50]}...'")

                reactions.append(
                    Reaction(
                        trigger=keyword,
                        trigger_type="keyword",
                        callback=callback,
                        metadata={},
                    )
                )

                # Mark as triggered to prevent duplicates
                self.triggered_keywords.add(keyword_lower)

        return reactions

    def reset(self) -> None:
        """Reset triggered keywords for next conversation."""
        logger.debug(f"Resetting KeywordAnalyzer ({len(self.triggered_keywords)} keywords triggered)")
        self.triggered_keywords.clear()
