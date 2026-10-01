import json
import time
import wave
import logging
import tempfile
from typing import TextIO, Literal
from pathlib import Path


logger = logging.getLogger(__name__)

AudioStream = Literal["received", "sent"]
CAPTURE_SECONDS = 120


class AudioDiagnostics:
    """Save microphone PCM, submitted PCM, and a timeline for a short diagnostic session."""

    def __init__(self, directory: Path) -> None:
        """Create a unique capture directory with a two-minute recording limit."""
        directory = directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="session-", dir=directory))
        self._started_at = time.monotonic()
        self.closed = False
        self._writers: dict[AudioStream, wave.Wave_write] = {}
        self._sample_counts: dict[AudioStream, int] = {"received": 0, "sent": 0}
        self._events: TextIO | None = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        logger.info("Audio diagnostics recording for up to %s seconds in %s", CAPTURE_SECONDS, self.directory)

    def record_event(self, event_type: str, fields: dict[str, object] | None = None) -> None:
        """Append an event with its local elapsed time and audio sample positions."""
        if self._events is None:
            return
        elapsed = time.monotonic() - self._started_at
        if elapsed >= CAPTURE_SECONDS:
            self.close()
            return
        try:
            self._events.write(
                json.dumps(
                    {
                        "elapsed_s": elapsed,
                        "type": event_type,
                        "received_samples": self._sample_counts["received"],
                        "sent_samples": self._sample_counts["sent"],
                        **(fields or {}),
                    }
                )
                + "\n"
            )
            self._events.flush()
        except OSError:
            logger.warning("Audio diagnostic event write failed; stopping capture", exc_info=True)
            self.close()

    def record_audio(self, stream: AudioStream, sample_rate: int, pcm: bytes, channels: int = 1) -> None:
        """Append PCM16 samples to a WAV and record their position in the timeline."""
        self.record_event(
            "audio." + stream,
            {"sample_rate": sample_rate, "channels": channels, "samples": len(pcm) // (2 * channels)},
        )
        if self._events is None:
            return
        try:
            writer = self._writers.get(stream)
            if writer is None:
                writer = wave.open(str(self.directory / (stream + ".wav")), "wb")
                self._writers[stream] = writer
                writer.setnchannels(channels)
                writer.setsampwidth(2)
                writer.setframerate(sample_rate)
            elif writer.getframerate() != sample_rate or writer.getnchannels() != channels:
                raise ValueError("Microphone format changed during audio capture")
            writer.writeframes(pcm)
            self._sample_counts[stream] += len(pcm) // (2 * channels)
        except (OSError, ValueError, wave.Error):
            logger.warning("Audio diagnostic WAV write failed; stopping capture", exc_info=True)
            self.close()

    def close(self) -> None:
        """Finalize capture files without interrupting the conversation on disk errors."""
        self.closed = True
        for writer in self._writers.values():
            try:
                writer.close()
            except (OSError, wave.Error):
                logger.warning("Could not finalize audio diagnostic WAV", exc_info=True)
        self._writers.clear()
        if self._events is not None:
            try:
                self._events.close()
            except OSError:
                logger.warning("Could not close audio diagnostic timeline", exc_info=True)
            self._events = None
            logger.info("Audio diagnostics saved in %s", self.directory)
