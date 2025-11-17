"""Transcript analysis for real-time reaction to user speech.

This module provides tools for analyzing transcripts (both partial and final)
and triggering demo-configured reactions based on keywords and named entities.

Example usage in a demo:

    # demos/food/__init__.py
    import logging
    from reachy_mini_conversation_app.tools import ToolDependencies

    logger = logging.getLogger(__name__)

    async def excited_about_pizza(deps: ToolDependencies) -> None:
        logger.info("🍕 User mentioned pizza!")
        # Queue robot gesture, animation, etc.

    async def react_to_food(deps, entity_text, entity_label, confidence):
        logger.info(f"Detected food: {entity_text}")

    reactions = {
        "keywords": {
            "pizza": excited_about_pizza,
        },
        "entities": {
            "food": react_to_food,
        },
        "gliner_model": "urchade/gliner_small-v2.1"  # Optional
    }

"""

from .base import Reaction, TranscriptAnalyzer
from .loader import get_demo_reactions
from .manager import TranscriptAnalysisManager
from .keyword_analyzer import KeywordAnalyzer


# EntityAnalyzer is optional (requires gliner extra)
try:
    from .entity_analyzer import EntityAnalyzer

    __all__ = [
        "Reaction",
        "TranscriptAnalyzer",
        "KeywordAnalyzer",
        "EntityAnalyzer",
        "TranscriptAnalysisManager",
        "get_demo_reactions",
    ]
except ImportError:
    __all__ = [
        "Reaction",
        "TranscriptAnalyzer",
        "KeywordAnalyzer",
        "TranscriptAnalysisManager",
        "get_demo_reactions",
    ]
