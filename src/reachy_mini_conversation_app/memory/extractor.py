"""Memory extractor for automatic fact extraction from conversations."""

import logging
from typing import Any

from openai import AsyncOpenAI

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory.memory_manager import get_memory_manager


logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extract facts from conversation sessions using GPT-4o-mini."""

    EXTRACTION_PROMPT = """Review this conversation and extract key facts worth remembering.
Focus on: user's name/details, preferences, important facts discussed, corrections to previous knowledge.
Output as a bulleted list. If no significant new facts, respond with "No new facts."

Conversation:
{conversation}"""

    def __init__(self) -> None:
        """Initialize the memory extractor."""
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def extract_and_store(self, conversation: list[dict[str, Any]]) -> None:
        """Extract facts from conversation and merge into facts block.

        Args:
            conversation: List of conversation turns with 'role' and 'content' keys
        """
        if not conversation:
            logger.debug("No conversation to extract from")
            return

        try:
            # Format conversation for extraction
            formatted_conversation = self._format_conversation(conversation)

            # Call GPT-4o-mini for extraction
            logger.info("Extracting facts from conversation (%d turns)", len(conversation))

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": self.EXTRACTION_PROMPT.format(conversation=formatted_conversation),
                    }
                ],
                temperature=0.3,
                max_tokens=500,
            )

            extracted_facts = response.choices[0].message.content
            if not extracted_facts:
                logger.info("No facts extracted from conversation")
                return

            # Check if there are actually new facts
            if "no new facts" in extracted_facts.lower():
                logger.info("No significant new facts found in conversation")
                return

            # Get or create facts block
            manager = get_memory_manager()
            await manager.initialize()

            existing_facts_block = await manager.get_block("facts")

            if existing_facts_block:
                # Merge new facts with existing ones
                merged_content = self._merge_facts(existing_facts_block.content, extracted_facts)
            else:
                merged_content = extracted_facts

            # Update facts block
            await manager.update_block("facts", merged_content)
            logger.info("Facts extracted and stored successfully")

        except Exception as e:
            logger.error("Failed to extract and store facts: %s", e)

    def _format_conversation(self, conversation: list[dict[str, Any]]) -> str:
        """Format conversation turns into readable text.

        Args:
            conversation: List of conversation turns

        Returns:
            Formatted conversation string
        """
        lines = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            # Skip empty content
            if not content or not isinstance(content, str):
                continue

            # Format as "Role: content"
            role_label = "User" if role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    def _merge_facts(self, existing_facts: str, new_facts: str) -> str:
        """Merge new facts with existing facts, avoiding duplicates.

        Args:
            existing_facts: Current facts content
            new_facts: Newly extracted facts

        Returns:
            Merged facts content
        """
        # Simple merge: append new facts with a separator
        # Could be enhanced with deduplication logic
        separator = "\n\n--- New facts extracted ---\n"
        return f"{existing_facts}{separator}{new_facts}"
