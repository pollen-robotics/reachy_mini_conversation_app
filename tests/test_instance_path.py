"""Tests for the instance directory resolution and its one-time migration."""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.config import INSTANCE_PATH_ENV, resolve_instance_path


@pytest.fixture(autouse=True)
def _config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the default instance directory at a temporary config home."""
    config_home = tmp_path / "config"
    monkeypatch.setattr(config_mod, "user_config_dir", lambda name: str(config_home / name))
    monkeypatch.delenv(INSTANCE_PATH_ENV, raising=False)
    return config_home


def test_defaults_beside_the_daemon_config(tmp_path: Path, _config_home: Path) -> None:
    """Without an override the data lives next to the daemon's own config."""
    resolved = resolve_instance_path(tmp_path / "site-packages")

    assert resolved == _config_home / "reachy_mini" / "conversation_app"
    assert resolved.is_dir()


def test_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REACHY_MINI_INSTANCE_PATH overrides the default and is created."""
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(tmp_path / "mine"))

    assert resolve_instance_path(tmp_path / "site-packages") == tmp_path / "mine"
    assert (tmp_path / "mine").is_dir()


def test_migrates_legacy_data_once(tmp_path: Path) -> None:
    """Data written in the package directory moves over on the first run."""
    legacy = tmp_path / "site-packages"
    (legacy / "user_personalities" / "guide").mkdir(parents=True)
    (legacy / "user_personalities" / "guide" / "profile.md").write_text("+++\n+++\nBe a guide.")
    (legacy / "memory.v1.json").write_text('{"version": 1, "facts": []}')

    resolved = resolve_instance_path(legacy)

    assert (resolved / "memory.v1.json").read_text() == '{"version": 1, "facts": []}'
    assert (resolved / "user_personalities" / "guide" / "profile.md").exists()

    # A second run must not overwrite live data with the stale legacy copy.
    (resolved / "memory.v1.json").write_text('{"version": 1, "facts": ["kept"]}')
    resolve_instance_path(legacy)
    assert "kept" in (resolved / "memory.v1.json").read_text()


def test_unusable_path_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad override warns and keeps the previous directory, never crashes."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(blocker / "nope"))

    legacy = tmp_path / "site-packages"
    assert resolve_instance_path(legacy) == legacy
