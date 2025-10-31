"""Emotion flute player built on simpleaudio."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class EmotionFlutePlayer:
    """Play companion WAV tracks for recorded emotions using simpleaudio."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._play_obj: Any | None = None
        self._simpleaudio_missing_logged = False

    def play(self, emotion_name: str, recorded_moves: Any) -> None:
        """Start playback of emotion audio, stopping any previous clip."""
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
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._play_thread,
                args=(audio_path, self._stop_event),
                name=f"emotion-audio-{emotion_name}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop any ongoing playback."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop_event.set()
        if self._play_obj is not None:
            try:
                self._play_obj.stop()
            except Exception:
                pass
            self._play_obj = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def _play_thread(self, audio_path: Path, stop_event: threading.Event) -> None:
        try:
            import simpleaudio  # type: ignore[import]
        except ImportError:
            if not self._simpleaudio_missing_logged:
                logger.warning("simpleaudio not installed; cannot play %s", audio_path)
                self._simpleaudio_missing_logged = True
            return

        try:
            wave_obj = simpleaudio.WaveObject.from_wave_file(str(audio_path))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", audio_path, exc)
            return

        play_obj = wave_obj.play()
        self._play_obj = play_obj

        while not stop_event.wait(0.05):
            if not play_obj.is_playing():
                break

        try:
            play_obj.stop()
        except Exception:
            pass
        finally:
            self._play_obj = None


def build_flute_player() -> EmotionFlutePlayer:
    """Create a new flute player instance."""
    return EmotionFlutePlayer()
