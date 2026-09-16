"""Tests for the instance settings store."""

import json
from pathlib import Path

import pytest

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.settings_store import (
    AppSettings,
    read_settings,
    update_settings,
    load_settings_into_runtime,
)


def test_absent_file_reads_as_unset(tmp_path: Path) -> None:
    """A fresh instance has no settings and keeps the environment defaults."""
    assert read_settings(tmp_path) == AppSettings()


def test_update_merges_and_round_trips(tmp_path: Path) -> None:
    """Each write changes one field and leaves the others alone."""
    update_settings(tmp_path, AppSettings(language="fr"))
    merged = update_settings(tmp_path, AppSettings(memory_enabled=False))

    assert merged == AppSettings(language="fr", memory_enabled=False)
    assert read_settings(tmp_path) == merged

    payload = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["memory_enabled"] is False


def test_booleans_survive_as_booleans(tmp_path: Path) -> None:
    """The reason for leaving .env: a switch round-trips as a real boolean."""
    update_settings(tmp_path, AppSettings(camera_enabled=False))

    assert read_settings(tmp_path).camera_enabled is False


def test_awkward_value_cannot_write_another_key(tmp_path: Path) -> None:
    """A newline in a value stays inside that value (#532 is impossible here)."""
    update_settings(tmp_path, AppSettings(hf_ws_url="ws://host\nHF_TOKEN=leaked"))

    assert read_settings(tmp_path).hf_ws_url == "ws://host\nHF_TOKEN=leaked"
    payload = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert set(payload) == {"version", "hf_ws_url"}


@pytest.mark.parametrize("content", ["{ not json", '"a string"', '{"memory_enabled": "false"}'])
def test_unusable_file_is_ignored(tmp_path: Path, content: str) -> None:
    """A corrupt or wrongly-typed file must not stop the app from starting."""
    (tmp_path / "settings.json").write_text(content, encoding="utf-8")

    assert read_settings(tmp_path) == AppSettings()


def test_load_applies_over_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored settings win over the env-derived defaults."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "en")
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    update_settings(tmp_path, AppSettings(language="fr", memory_enabled=False))

    load_settings_into_runtime(tmp_path)

    assert config.REALTIME_TRANSCRIPTION_LANGUAGE == "fr"
    assert config.MEMORY_ENABLED is False


def test_load_leaves_unset_fields_to_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A setting the user never changed keeps its environment value."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "de")
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    update_settings(tmp_path, AppSettings(memory_enabled=False))

    load_settings_into_runtime(tmp_path)

    assert config.REALTIME_TRANSCRIPTION_LANGUAGE == "de"
