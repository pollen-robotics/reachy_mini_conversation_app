"""LLM abstraction for cascade pipeline."""

from __future__ import annotations
import abc
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class LLMChunk:
    """Represents a chunk from LLM streaming response."""

    type: str  # "text_delta", "tool_call", "done"
    content: Optional[str] = None  # Text content for text_delta
    tool_call: Optional[Dict[str, Any]] = None  # Tool call data for tool_call type


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from LLM.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of available tools
            temperature: Sampling temperature

        Yields:
            LLMChunk objects with text deltas, tool calls, or completion signal

        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_tool_call(self, tool_call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """Parse a tool call into its components.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Tuple of (call_id, tool_name, arguments_dict)

        """
        raise NotImplementedError

