from pathlib import Path
from collections import deque
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools import speak_in_voice, repeat_voice_line
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


WAV_BYTES = b"RIFF0000WAVEfmt "


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_repeat_voice_line_replays_the_latest_or_a_matching_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without words the latest line plays again; words pick an earlier one; nothing is generated."""
    snacks, droid = tmp_path / "snacks.wav", tmp_path / "droid.wav"
    snacks.write_bytes(WAV_BYTES)
    droid.write_bytes(WAV_BYTES)
    lines: deque[speak_in_voice.VoiceLine] = deque(maxlen=5)
    monkeypatch.setattr(speak_in_voice, "recent_lines", lines)
    repeat = repeat_voice_line.RepeatVoiceLine()
    deps = _deps()

    assert "error" in await repeat(deps)

    lines.append(speak_in_voice.VoiceLine("I find your lack of snacks disturbing.", "villain", snacks, 3.0))
    lines.append(speak_in_voice.VoiceLine("To get to the Dark Side!", "villain", droid, 4.0))

    latest = await repeat(deps)
    earlier = await repeat(deps, words="SNACKS")
    missing = await repeat(deps, words="banana")

    assert (latest["text"], earlier["text"]) == ("To get to the Dark Side!", "I find your lack of snacks disturbing.")
    assert [call.args[0] for call in deps.reachy_mini.media.play_sound.call_args_list] == [str(droid), str(snacks)]
    assert missing["recent_lines"] == ["I find your lack of snacks disturbing.", "To get to the Dark Side!"]
