import wave
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import speak_in_voice
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_speak_in_voice_plays_the_line_and_returns_once_heard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The designed voice line is downloaded from the Space's file URL, played, and waited for."""
    clip = tmp_path / "line.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 800)  # 0.05 s
    url = "https://hugging-apps-firered-tts3.hf.space/gradio_api/file=/tmp/gradio/abc/audio.wav"
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_call(slug: str, mcp_url: str, tool: str, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        calls.append((tool, arguments))
        return {"status": "ok", "text": f"[{{'path': '/tmp/gradio/abc/audio.wav', 'url': '{url}'}}]"}

    downloaded: list[str] = []

    def _fake_save(audio: str, name: str) -> Path:
        downloaded.append(audio)
        return clip

    monkeypatch.setattr(speak_in_voice, "call_space_tool", _fake_call)
    monkeypatch.setattr(speak_in_voice, "save_audio", _fake_save)
    deps = _deps()

    result = await speak_in_voice.SpeakInVoice()(
        deps, text="I find your lack of snacks disturbing.", voice="deep villain", language="EN"
    )

    assert calls == [
        (
            "firered_tts3_voice_design",
            {"instruction": "deep villain", "text": "I find your lack of snacks disturbing."},
        )
    ]
    assert downloaded == [url]
    deps.reachy_mini.media.play_sound.assert_called_once_with(str(clip))
    assert (result["status"], result["duration_s"]) == ("played", 0.1)


@pytest.mark.asyncio
async def test_speak_in_voice_passes_space_failures_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explained Space failure reaches the model unchanged and nothing plays."""

    async def _failing_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": "The Space is paused by its owner.", "reason": "space_paused"}

    monkeypatch.setattr(speak_in_voice, "call_space_tool", _failing_call)
    deps = _deps()

    result = await speak_in_voice.SpeakInVoice()(deps, text="Hello.", voice="a squeaky mouse", language="en")

    assert result["reason"] == "space_paused"
    deps.reachy_mini.media.play_sound.assert_not_called()


@pytest.mark.asyncio
async def test_speak_in_voice_refuses_other_languages_before_any_space_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A French line is turned down at once, with a hint to ask the user for English, and costs no GPU time."""
    space_call = MagicMock()
    monkeypatch.setattr(speak_in_voice, "call_space_tool", space_call)

    result = await speak_in_voice.SpeakInVoice()(_deps(), text="Bonjour !", voice="a villain", language="fr")

    assert result["reason"] == "unsupported_language"
    assert "English" in result["next"]
    space_call.assert_not_called()
