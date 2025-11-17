"""Sound player utility for ReachyMiniScript."""

import logging
import threading
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move


logger = logging.getLogger(__name__)


class SoundQueueMove(Move):  # type: ignore
    """Wrapper for sound playback to work with the movement queue system."""

    def __init__(self, sound_file_path: str, duration: float = 0.0, blocking: bool = False):
        """Initialize a SoundQueueMove.

        Args:
            sound_file_path: Absolute path to the sound file
            duration: Duration of the sound in seconds (for blocking mode)
            blocking: If True, move lasts for sound duration. If False, move is instant.

        """
        from pathlib import Path
        self.sound_file_path = str(Path(sound_file_path))
        self._duration = duration if blocking else 0.01  # Instant for async, full duration for blocking
        self.blocking = blocking
        self._played = False

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate sound move - plays sound on first call, maintains current pose."""
        # Play sound only once at the start
        if not self._played and t <= 0.01:
            self._played = True
            try:
                if self.blocking:
                    # Blocking mode: play in foreground thread (blocks the evaluate call)
                    from pathlib import Path

                    from reachy_mini_conversation_app.rmscript.sound_player import play_sound_blocking
                    play_sound_blocking(Path(self.sound_file_path))
                else:
                    # Async mode: play in background thread
                    from pathlib import Path

                    from reachy_mini_conversation_app.rmscript.sound_player import play_sound_async
                    play_sound_async(Path(self.sound_file_path))
            except Exception as e:
                logger.error(f"Error playing sound {self.sound_file_path}: {e}")

        # Return None for all joints - maintains current pose
        return (None, None, None)


def find_sound_file(sound_name: str, search_paths: list[Path]) -> Optional[Path]:
    """Find a sound file by name in the given search paths.

    Searches for .wav and .mp3 files with the given name.

    Args:
        sound_name: Name of the sound (without extension)
        search_paths: List of directories to search in

    Returns:
        Path to the sound file, or None if not found

    """
    extensions = [".wav", ".mp3", ".ogg", ".flac"]

    for search_path in search_paths:
        for ext in extensions:
            file_path = search_path / f"{sound_name}{ext}"
            if file_path.exists():
                logger.info(f"Found sound file: {file_path}")
                return file_path

    logger.warning(f"Sound file not found: {sound_name} inside {search_paths}")
    return None


def get_sound_duration(file_path: Path) -> float:
    """Get the duration of a sound file in seconds.

    Args:
        file_path: Path to the sound file

    Returns:
        Duration in seconds, or 0.0 if unable to determine

    """
    try:
        # Try using soundfile first (fastest and most reliable)
        try:
            import soundfile as sf
            with sf.SoundFile(str(file_path)) as f:
                duration: float = len(f) / f.samplerate
                return duration
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"soundfile failed: {e}")

        # Try using pydub as fallback
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) / 1000.0  # pydub returns milliseconds
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"pydub failed: {e}")

        # Fallback: estimate 0 (couldn't determine duration)
        logger.warning(f"Could not determine duration for {file_path}")
        return 0.0

    except Exception as e:
        logger.error(f"Error getting sound duration: {e}")
        return 0.0


def play_sound_blocking(file_path: Path) -> Tuple[bool, float]:
    """Play a sound file and wait for it to finish.

    Args:
        file_path: Path to the sound file

    Returns:
        Tuple of (success: bool, duration: float)

    """
    try:
        # Get duration first
        duration = get_sound_duration(file_path)

        # Try using pydub for playback (most compatible)
        try:
            from pydub import AudioSegment
            from pydub.playback import play

            audio = AudioSegment.from_file(str(file_path))
            play(audio)
            return True, duration
        except ImportError:
            logger.debug("pydub not available for playback")
        except Exception as e:
            logger.debug(f"pydub playback failed: {e}")

        # Fallback: use subprocess with system audio player
        import platform
        import subprocess

        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", str(file_path)], check=True)
        elif system == "Linux":
            # Try different players in order of preference
            players = ["aplay", "paplay", "ffplay", "mpg123"]
            for player in players:
                try:
                    subprocess.run([player, str(file_path)], check=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            import winsound  # type: ignore[import-not-found,unused-ignore]
            if file_path.suffix == ".wav":
                winsound.PlaySound(str(file_path), winsound.SND_FILENAME)  # type: ignore[attr-defined,unused-ignore]
            else:
                # Use PowerShell for non-WAV files
                subprocess.run(
                    ["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync()"],
                    check=True
                )

        return True, duration

    except Exception as e:
        logger.error(f"Error playing sound: {e}")
        return False, 0.0


def play_sound_async(file_path: Path) -> bool:
    """Play a sound file in the background (non-blocking).

    Args:
        file_path: Path to the sound file

    Returns:
        True if playback started successfully

    """
    def _play() -> None:
        play_sound_blocking(file_path)

    try:
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
        return True
    except Exception as e:
        logger.error(f"Error starting async playback: {e}")
        return False
