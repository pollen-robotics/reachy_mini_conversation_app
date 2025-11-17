"""Latency tracking for cascade pipeline."""

from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional, Generator
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class LatencyTracker:
    """Centralized latency tracking for the conversation pipeline."""

    def __init__(self) -> None:
        """Initialize latency tracker."""
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.reference_name: str = "pipeline_start"

    def reset(self, reference_name: str = "pipeline_start") -> None:
        """Reset tracker for new conversation turn."""
        self.events = []
        self.start_time = time.perf_counter()
        self.reference_name = reference_name
        logger.info(f"⏱️  LATENCY TRACKING STARTED: {reference_name}")

    def mark(self, event_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a timing event."""
        if self.start_time is None:
            self.reset()

        timestamp = time.perf_counter()
        # After reset(), self.start_time is guaranteed to be set
        assert self.start_time is not None
        elapsed_ms = (timestamp - self.start_time) * 1000

        event = {
            "name": event_name,
            "timestamp": timestamp,
            "elapsed_ms": elapsed_ms,
            "metadata": metadata or {}
        }
        self.events.append(event)

        # Log immediately for real-time tracking
        metadata_str = ""
        if metadata:
            # Format metadata compactly
            parts = [f"{k}={v}" for k, v in metadata.items()]
            metadata_str = f" ({', '.join(parts)})"

        logger.info(f"⏱️  [{elapsed_ms:7.1f}ms] {event_name}{metadata_str}")

    @contextmanager
    def measure(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        """Context manager for measuring operation duration."""
        start = time.perf_counter()
        self.mark(f"{operation_name}_start", metadata)
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            final_metadata = {**(metadata or {}), "duration_ms": round(duration_ms, 1)}
            self.mark(f"{operation_name}_end", final_metadata)
            logger.info(f"⏱️  └─ {operation_name}: {duration_ms:.1f}ms")

    def get_duration(self, start_event: str, end_event: str, use_first: bool = True) -> Optional[float]:
        """Get duration between two events in milliseconds.

        Args:
            start_event: Name of the starting event
            end_event: Name of the ending event
            use_first: If True, use first occurrence of each event (default).
                      If False, use last occurrence.

        Returns:
            Duration in milliseconds, or None if events not found

        """
        start_time = None
        end_time = None

        for event in self.events:
            if event["name"] == start_event:
                if use_first and start_time is None:
                    start_time = event["timestamp"]
                elif not use_first:
                    start_time = event["timestamp"]
            elif event["name"] == end_event:
                if use_first and end_time is None:
                    end_time = event["timestamp"]
                elif not use_first:
                    end_time = event["timestamp"]

        if start_time is not None and end_time is not None:
            return float((end_time - start_time) * 1000)
        return None


    def print_summary(self) -> None:
        """Print detailed latency summary."""
        if not self.events:
            logger.warning("No timing events recorded")
            return

        logger.info("=" * 80)
        logger.info("LATENCY SUMMARY")
        logger.info("=" * 80)

        # Main sequential stages (these should add up to total perceived latency)
        recording_capture = self.get_duration("user_stop_click", "recording_ready")
        asr_processing = self.get_duration("recording_ready", "asr_complete")
        llm_generation = self.get_duration("asr_complete", "llm_complete")
        tts_generation = self.get_duration("tts_start", "tts_first_chunk_ready")
        audio_system = self.get_duration("tts_first_chunk_ready", "audio_playback_started")

        # Gaps (overhead between stages)
        asr_to_llm_gap = self.get_duration("asr_complete", "llm_start")
        llm_to_tts_gap = self.get_duration("llm_complete", "tts_start")

        # Total perceived latency
        total_latency = self.get_duration("user_stop_click", "audio_playback_started")

        # Display main stages
        logger.info("")
        if recording_capture is not None:
            logger.info(f"  1. Recording Capture.......................... {recording_capture:>8.1f}ms")
        if asr_processing is not None:
            logger.info(f"  2. ASR Processing............................. {asr_processing:>8.1f}ms")
        if asr_to_llm_gap is not None and asr_to_llm_gap > 1.0:
            logger.info(f"     ↳ Gap (ASR → LLM start).................... {asr_to_llm_gap:>8.1f}ms")
        if llm_generation is not None:
            logger.info(f"  3. LLM Generation............................. {llm_generation:>8.1f}ms")
        if llm_to_tts_gap is not None and llm_to_tts_gap > 1.0:
            logger.info(f"     ↳ Gap (LLM → TTS start).................... {llm_to_tts_gap:>8.1f}ms")
        if tts_generation is not None:
            logger.info(f"  4. TTS time to first audio.................... {tts_generation:>8.1f}ms")
        if audio_system is not None and audio_system > 0:
            logger.info(f"     ↳ Audio system delay....................... {audio_system:>8.1f}ms")

        logger.info("")
        logger.info("TOTAL PERCEIVED LATENCY:")
        if total_latency is not None:
            logger.info(f"     Click → First Audio........................ {total_latency:>8.1f}ms")

        logger.info("=" * 80)

# Global tracker instance
tracker = LatencyTracker()
