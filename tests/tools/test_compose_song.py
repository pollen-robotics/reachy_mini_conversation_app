import json
import base64
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import _space_audio, compose_song
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


WAV_BYTES = b"RIFF0000WAVEfmt "


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.fixture
def space(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Fake the song Space's MCP client; tests set ``space.result`` or ``space.error``."""
    space = MagicMock(result=None, error=None)

    class _FakeClient:
        def __init__(self, server: Any) -> None:
            space.server = server

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            space.arguments = arguments
            if space.error is not None:
                raise space.error
            result: dict[str, Any] = space.result
            return result

    monkeypatch.setattr(_space_audio, "RemoteMcpToolClient", _FakeClient)
    monkeypatch.setattr(_space_audio, "AUDIO_DIRECTORY", tmp_path)
    monkeypatch.setattr(_space_audio, "current_hf_token", lambda: "hf_user")
    monkeypatch.setattr(compose_song, "latest_song", None)
    return space


def _song_result(**fields: Any) -> dict[str, Any]:
    payload = {"title": "Wiggle", "tags": "pop, 120 bpm", "duration": 20.0, **fields}
    return {"status": "ok", "content_blocks": [{"type": "text", "text": json.dumps(payload)}]}


@pytest.mark.asyncio
async def test_compose_song_saves_the_audio_and_keeps_it_ready(space: MagicMock) -> None:
    """The sung audio is saved and remembered, signed in, with a clamped duration."""
    space.result = _song_result(audio="data:audio/wav;base64," + base64.b64encode(WAV_BYTES).decode())

    result = await compose_song.ComposeSong()(_deps(), description="a song about antennas", duration_s=500)

    assert (result["status"], result["title"], result["duration_s"]) == ("ready", "Wiggle", 20.0)
    assert space.arguments == {"description": "a song about antennas", "audio_duration": 60}
    assert space.server.headers == {"Authorization": "Bearer hf_user"}
    # The Space sends the WAV inline in one event, far above the SDK's 1 MiB default.
    assert space.server.max_event_bytes >= 16 * 1024 * 1024
    song = compose_song.latest_song
    assert song is not None
    assert (song.path.suffix, song.path.read_bytes()) == (".wav", WAV_BYTES)
    assert (song.style, song.bpm) == ("pop, 120 bpm", 120)

    # A new song replaces the old one on disk too, even with the same title in the same second.
    await compose_song.ComposeSong()(_deps(), description="a song about antennas")
    assert compose_song.latest_song.path.exists() and not song.path.exists()


@pytest.mark.asyncio
async def test_compose_song_reports_why_the_space_failed(space: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Space-side failure is explained and no song is kept."""
    quota = "You have exceeded your free ZeroGPU quota (290s requested vs. 10s left)."
    space.result = {"status": "error", "text": quota, "content_blocks": []}
    monkeypatch.setattr(
        _space_audio,
        "explain_space_failure",
        lambda slug, detail: {"error": "GPU time used up", "reason": "gpu_quota", "detail": detail},
    )

    result = await compose_song.ComposeSong()(_deps(), description="a lullaby")

    assert (result["reason"], result["detail"]) == ("gpu_quota", quota)
    assert compose_song.latest_song is None
