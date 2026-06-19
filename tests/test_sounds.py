from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import sounds


def test_play_resolves_packaged_sound() -> None:
    """Packaged sounds should be played through the provided media object."""
    media = MagicMock()

    sounds.play(media, "dream_start.wav")

    media.play_sound.assert_called_once()
    sound_path = Path(media.play_sound.call_args.args[0])
    assert sound_path.name == "dream_start.wav"
    assert sound_path.is_file()


@pytest.mark.parametrize("filename", ["", "../dream_start.wav", "nested/dream_start.wav", r"nested\dream_start.wav"])
def test_play_rejects_non_plain_filenames(filename: str) -> None:
    """Sound playback only accepts filenames packaged in the sounds directory."""
    media = MagicMock()

    with pytest.raises(ValueError, match="plain filename"):
        sounds.play(media, filename)

    media.play_sound.assert_not_called()
