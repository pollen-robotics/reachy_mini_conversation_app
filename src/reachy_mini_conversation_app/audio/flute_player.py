"""Emotion flute audio helper.

TODO: Replace this helper with reachy_mini's native `mini.play_sound(...)` once the
      daemon exposes that API. At that point this module (and its dependencies)
      can be deleted entirely.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmotionFlutePlayer:
    """Plays the WAV companion track that ships with recorded emotions."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._backend_warning_logged = False

    def play(self, emotion_name: str, recorded_moves: Any) -> None:
        """Start playback for the given emotion, interrupting any current clip."""
        local_path = getattr(recorded_moves, "local_path", None)
        if not local_path:
            logger.debug("Recorded moves missing local_path; skipping audio for %s", emotion_name)
            return

        audio_path = Path(local_path) / f"{emotion_name}.wav"
        if not audio_path.exists():
            logger.debug("No audio file found for %s at %s", emotion_name, audio_path)
            return

        with self._lock:
            self._stop_locked()
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._thread = threading.Thread(
                target=self._play_thread,
                args=(audio_path, stop_event),
                name=f"emotion-flute-{emotion_name}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop any ongoing playback."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def _play_thread(self, audio_path: Path, stop_event: threading.Event) -> None:
        try:
            import sounddevice as sd  # type: ignore
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            if not self._backend_warning_logged:
                logger.warning(
                    "Install the emotion_reader_profile extra to enable flute audio (missing %s)",
                    exc.name if hasattr(exc, "name") else "sounddevice/soundfile",
                )
                self._backend_warning_logged = True
            return

        try:
            data, sample_rate = sf.read(str(audio_path), dtype="float32")
        except Exception as read_exc:
            logger.warning("Failed to load flute audio %s: %s", audio_path, read_exc)
            return

        logger.debug("Starting flute audio playback from %s", audio_path)
        sd.stop()
        sd.play(data, samplerate=sample_rate)

        duration = len(data) / sample_rate
        start_time = time.time()
        while not stop_event.wait(0.05):
            if (time.time() - start_time) >= duration:
                break

        sd.stop()
        logger.debug("Flute audio playback finished for %s", audio_path.name)


def build_flute_player() -> EmotionFlutePlayer:
    """Factory kept for dependency injection and future replacements."""
    return EmotionFlutePlayer()
