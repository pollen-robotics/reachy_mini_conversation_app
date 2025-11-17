"""Base abstractions for transcript analysis."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Callable, Awaitable
from dataclasses import field, dataclass


@dataclass
class Reaction:
    """Represents a triggered reaction from transcript analysis.

    Attributes:
        trigger: The text that triggered this reaction (keyword or entity text)
        trigger_type: Type of trigger ("keyword" or "entity:{label}")
        callback: Async callback function to execute
        metadata: Additional data (entity label, confidence, etc.)

    """

    trigger: str
    trigger_type: str
    callback: Callable[..., Awaitable[None]]
    metadata: dict[str, Any] = field(default_factory=dict)


class TranscriptAnalyzer(ABC):
    """Abstract base class for transcript analyzers."""

    @abstractmethod
    async def analyze(self, text: str, is_final: bool) -> List[Reaction]:
        """Analyze transcript text and return triggered reactions.

        Args:
            text: The transcript text to analyze
            is_final: Whether this is the final transcript (vs partial/interim)

        Returns:
            List of Reaction objects that were triggered

        """
        pass

    def reset(self) -> None:
        """Reset analyzer state between conversations.

        Called after each conversation finishes to clear any state tracking
        (e.g., which keywords/entities have already triggered).

        """
        pass
