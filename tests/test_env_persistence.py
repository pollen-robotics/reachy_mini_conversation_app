"""Values the settings UI persists must not corrupt the instance `.env`."""

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import dotenv
import pytest

from reachy_mini_conversation_app.console import LocalStream


def _stream(instance_path: Path) -> LocalStream:
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace()))
    stream = LocalStream(MagicMock(), robot)
    stream._instance_path = str(instance_path)
    return stream


def test_persist_env_values_rejects_a_line_break_before_writing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A line break ends the record, so the remainder is parsed as its own assignment."""
    monkeypatch.setenv("SOME_TOKEN", "")
    stream = _stream(tmp_path)

    with pytest.raises(ValueError, match="line breaks"):
        stream._persist_env_values({"SOME_TOKEN": "value\nOPENAI_API_KEY=injected"})

    # Nothing was written.
    assert not (tmp_path / ".env").exists()


def test_persist_env_values_rejects_interpolation_syntax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """python-dotenv expands `${VAR}` and offers no way to escape it."""
    monkeypatch.setenv("SOME_TOKEN", "")
    stream = _stream(tmp_path)

    with pytest.raises(ValueError, match=r"line breaks"):
        stream._persist_env_values({"SOME_TOKEN": "literal-${HOME}-please"})

    assert not (tmp_path / ".env").exists()


def test_persist_env_values_still_writes_an_ordinary_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal token still persists."""
    monkeypatch.setenv("SOME_TOKEN", "")
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=keep-me\nSOME_TOKEN=old\n", encoding="utf-8")

    _stream(tmp_path)._persist_env_values({"SOME_TOKEN": "hf_abc123"})

    values = dotenv.dotenv_values(env_path)
    assert values["OTHER"] == "keep-me"
    assert values["SOME_TOKEN"] == "hf_abc123"
