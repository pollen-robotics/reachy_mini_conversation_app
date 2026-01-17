"""Conversation synthesis for long-term memory using LLM."""

import logging

from openai import OpenAI

from .store import load_memories, save_memories, estimate_tokens
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


def synthesize_conversation(conversation_transcript: str) -> str:
    """Synthesize conversation into memory, merging with existing memories.

    Uses gpt-4o-mini to extract important facts from the conversation and
    integrate them with existing memories into a coherent narrative summary.

    Args:
        conversation_transcript: The conversation text to analyze.

    Returns:
        The updated memory summary.

    """
    if not conversation_transcript.strip():
        logger.debug("Empty conversation transcript, skipping synthesis")
        return load_memories()

    client = OpenAI()
    existing = load_memories()

    prompt = f"""You are a memory synthesis assistant. Analyze this conversation and update the user memory.

Existing memory about the user:
{existing if existing else "(No existing memories)"}

Recent conversation:
{conversation_transcript}

Write an updated memory summary that:
1. Extracts important facts about the user (name, preferences, interests, context)
2. Integrates new information with existing memories
3. Stays concise (max 500 words)
4. Uses narrative style
5. Keeps the same language as the conversation
6. Only keeps information that would be useful in future conversations

If the conversation contains no useful information to remember, return the existing memory unchanged.

Updated memory summary:"""

    logger.info("Synthesizing conversation into memory...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
    )

    updated = (response.choices[0].message.content or "").strip()

    if not updated:
        logger.warning("Empty response from LLM, keeping existing memory")
        return existing

    # Check token limit and condense if needed
    estimated = estimate_tokens(updated)
    if estimated > config.MEMORY_MAX_TOKENS:
        logger.warning(
            f"Memory exceeds token limit ({estimated} > {config.MEMORY_MAX_TOKENS}), "
            "requesting condensed version"
        )
        updated = _condense_memory(client, updated)

    save_memories(updated)
    logger.info(f"Memory synthesized successfully ({estimate_tokens(updated)} tokens)")
    return updated


def _condense_memory(client: OpenAI, memory: str) -> str:
    """Condense memory to fit within token limit.

    Args:
        client: OpenAI client instance.
        memory: The memory text to condense.

    Returns:
        Condensed memory text.

    """
    prompt = f"""Condense this memory summary to be more concise while keeping the most important information.
Target length: approximately {config.MEMORY_MAX_TOKENS * 4} characters.

Memory to condense:
{memory}

Condensed memory:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,
    )

    return (response.choices[0].message.content or memory).strip()
