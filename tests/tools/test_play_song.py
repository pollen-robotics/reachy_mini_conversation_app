from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from reachy_mini_conversation_app.tools import play_song, compose_song
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


WAV_BYTES = b"RIFF0000WAVEfmt "


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_play_song_plays_the_latest_song_and_dances_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The song file is played and enough dance moves are queued to cover it; without a song, nothing plays."""
    deps = _deps()
    monkeypatch.setattr(compose_song, "latest_song", None)
    assert "error" in await play_song.PlaySong()(deps)
    deps.reachy_mini.media.play_sound.assert_not_called()

    path = tmp_path / "wiggle.wav"
    path.write_bytes(WAV_BYTES)
    monkeypatch.setattr(compose_song, "latest_song", compose_song.Song(title="Wiggle", path=path, duration_s=20.0))

    result = await play_song.PlaySong()(deps)

    assert result["status"] == "playing"
    deps.reachy_mini.media.play_sound.assert_called_once_with(str(path))
    queued = [call.args[0] for call in deps.movement_manager.queue_move.call_args_list]
    assert result["dance_moves"] == len(queued)
    assert sum(move.duration for move in queued) >= 20.0


@pytest.mark.parametrize(
    ("style", "bpm", "expected"),
    [
        ("lullaby, soft female vocals, 70 bpm", 70, play_song.CALM),
        ("acoustic ballad, warm male vocals", None, play_song.CALM),
        ("pop, upbeat, female vocals, 120 bpm, bright", 120, play_song.GROOVE),
        ("chiptune, male vocals, 140 bpm, upbeat", 140, play_song.GROOVE),
        ("hard rock, distorted guitars, 110 bpm", 110, play_song.ENERGETIC),
    ],
)
def test_dance_style_follows_the_song(style: str, bpm: int | None, expected: play_song.DanceStyle) -> None:
    """Calm songs get small smooth moves, loud or fast ones the whole library."""
    song = compose_song.Song(title="t", path=Path("t.wav"), duration_s=20.0, style=style, bpm=bpm)
    assert play_song.dance_style(song) == expected


def test_song_moves_are_scaled_and_timed_without_touching_the_library() -> None:
    """A song move is scaled and set to the song's tempo on its own copy; the regular dance tool is unaffected."""
    library_params = {name: dict(params) for name, (_, params, _) in AVAILABLE_MOVES.items()}

    move = play_song._song_move("simple_nod", play_song.CALM, bpm=140)

    scaled = move.dance_move.move_params
    assert scaled["amplitude_rad"] == pytest.approx(library_params["simple_nod"]["amplitude_rad"] * 0.5)
    assert scaled["antenna_move_name"] == library_params["simple_nod"]["antenna_move_name"]
    assert move.dance_move.default_bpm == 140
    assert {name: dict(params) for name, (_, params, _) in AVAILABLE_MOVES.items()} == library_params
    assert type(move.dance_move).default_bpm == 114


def test_every_library_move_has_a_character() -> None:
    """A move added to the library must be classified before songs can dance to it."""
    groups = play_song.SMOOTH_MOVES + play_song.RHYTHMIC_MOVES + play_song.INTENSE_MOVES
    assert sorted(groups) == sorted(AVAILABLE_MOVES)
