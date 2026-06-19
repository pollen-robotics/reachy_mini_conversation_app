"""Packaged sound playback helpers."""

from typing import Protocol
from importlib.resources import files


class _SoundPlayer(Protocol):
    def play_sound(self, sound_file: str) -> None: ...


def play(media: _SoundPlayer, filename: str) -> None:
    """Play one packaged conversation-app sound through Reachy Mini media."""
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Sound filename must be a plain filename: {filename!r}")

    sound_path = files(__package__).joinpath(filename)
    if not sound_path.is_file():
        raise FileNotFoundError(f"Sound file not found: {filename}")

    media.play_sound(str(sound_path))
