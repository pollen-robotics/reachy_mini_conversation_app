"""Manager for orchestrating transcript analyzers and executing reactions."""

from __future__ import annotations
import time
import asyncio
import inspect
import logging
from typing import Any, List, cast

from .base import Reaction, TranscriptAnalyzer
from reachy_mini_conversation_app.tools import ToolDependencies


logger = logging.getLogger(__name__)


class TranscriptAnalysisManager:
    """Orchestrates transcript analyzers and executes reactions.

    Features:
    - Runs multiple analyzers in parallel
    - Debounces partial transcript analysis (500ms minimum interval)
    - Fire-and-forget execution (never blocks main pipeline)
    - Handles both keyword and entity callback signatures

    """

    DEBOUNCE_INTERVAL = 0.4  # minimum between partial analyses

    def __init__(self, analyzers: List[TranscriptAnalyzer], deps: ToolDependencies):
        """Initialize transcript analysis manager.

        Args:
            analyzers: List of TranscriptAnalyzer instances to run
            deps: Tool dependencies for executing reactions

        """
        self.analyzers = analyzers
        self.deps = deps
        self._pending_tasks: List[asyncio.Task[Any]] = []
        self._last_analysis_time = 0.0

        logger.info(f"TranscriptAnalysisManager initialized with {len(analyzers)} analyzers")

    async def analyze_partial(self, text: str) -> None:
        """Analyze partial transcript (streaming ASR) with debouncing.

        Fire-and-forget - doesn't block caller.

        Args:
            text: Partial transcript text to analyze

        """
        # Debounce: only analyze if at least 500ms since last analysis
        current_time = time.time()
        if current_time - self._last_analysis_time < self.DEBOUNCE_INTERVAL:
            #logger.debug("Debouncing partial analysis (too soon)")
            return

        self._last_analysis_time = current_time

        # Log what we're analyzing
        logger.info(f"📝 Analyzing PARTIAL transcript: '{text[:100]}...' ({len(text)} chars)")

        # Fire and forget
        task = asyncio.create_task(self._analyze_and_execute(text, is_final=False))
        self._pending_tasks.append(task)

    async def analyze_final(self, text: str) -> None:
        """Analyze final transcript (batch ASR or end of streaming).

        Blocks until analysis completes (but reactions are still fire-and-forget).

        Args:
            text: Final transcript text to analyze

        """
        logger.info(f"📝 Analyzing FINAL transcript: '{text[:100]}...' ({len(text)} chars)")
        await self._analyze_and_execute(text, is_final=True)

    async def _analyze_and_execute(self, text: str, is_final: bool) -> None:
        """Run all analyzers and execute triggered reactions.

        Args:
            text: Transcript text to analyze
            is_final: Whether this is the final transcript

        """
        try:
            # Run all analyzers in parallel
            results = await asyncio.gather(
                *[analyzer.analyze(text, is_final) for analyzer in self.analyzers],
                return_exceptions=True,
            )

            # Execute triggered reactions
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Analyzer error: {result}")
                    continue

                # result is List[Reaction] at this point (not an exception)
                reactions_list = cast(List[Reaction], result)
                for reaction in reactions_list:
                    logger.info(f"🎯 Reaction triggered: {reaction.trigger} ({reaction.trigger_type})")

                    # Fire and forget - don't wait for reaction to complete
                    asyncio.create_task(self._execute_reaction(reaction))

        except Exception as e:
            logger.exception(f"Error in transcript analysis: {e}")

    async def _execute_reaction(self, reaction: Reaction) -> None:
        """Execute a reaction callback with appropriate arguments.

        Inspects callback signature to determine how to call it:
        - 1 param: keyword callback (deps only)
        - 4 params: entity callback (deps, entity_text, entity_label, confidence)

        Args:
            reaction: The Reaction to execute

        """
        try:
            # Inspect callback signature
            sig = inspect.signature(reaction.callback)
            params = list(sig.parameters.keys())

            if len(params) == 1:
                # Keyword callback: async def callback(deps) -> None
                await reaction.callback(self.deps)

            elif len(params) >= 4:
                # Entity callback: async def callback(deps, entity_text, entity_label, confidence) -> None
                await reaction.callback(
                    self.deps,
                    reaction.metadata["entity_text"],
                    reaction.metadata["entity_label"],
                    reaction.metadata["confidence"],
                )

            else:
                logger.warning(
                    f"Reaction callback has unexpected signature: {params}. "
                    f"Expected 1 param (keyword) or 4+ params (entity)"
                )

        except Exception as e:
            logger.exception(f"Error executing reaction '{reaction.trigger}': {e}")

    def reset(self) -> None:
        """Reset all analyzers for next conversation.

        Should be called after each conversation completes.

        """
        logger.debug("Resetting transcript analysis manager")
        for analyzer in self.analyzers:
            analyzer.reset()
        self._pending_tasks.clear()
        self._last_analysis_time = 0.0
