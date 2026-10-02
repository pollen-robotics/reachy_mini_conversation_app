"""Tests for the instance directory resolution and its one-time migration."""

import shutil
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


def _legacy(tmp_path: Path, memory: bool = False) -> Path:
    """Build a package directory holding data the old layout wrote there."""
    legacy = tmp_path / "site-packages"
    (legacy / "user_personalities" / "guide").mkdir(parents=True)
    (legacy / "user_personalities" / "guide" / "profile.md").write_text("+++\n+++\nBe a guide.")
    if memory:
        (legacy / "memory.v1.json").write_text('{"version": 1, "facts": []}')
    return legacy


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
    legacy = _legacy(tmp_path, memory=True)

    resolved = resolve_instance_path(legacy)

    assert (resolved / "memory.v1.json").read_text() == '{"version": 1, "facts": []}'
    assert (resolved / "user_personalities" / "guide" / "profile.md").exists()

    assert not list(resolved.glob(".*.migrating"))

    # A second run must not overwrite live data with the stale legacy copy.
    (resolved / "memory.v1.json").write_text('{"version": 1, "facts": ["kept"]}')
    resolve_instance_path(legacy)
    assert "kept" in (resolved / "memory.v1.json").read_text()


def test_an_interrupted_migration_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy that dies halfway must not pass for a finished migration."""
    legacy = _legacy(tmp_path)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    real_copytree = shutil.copytree

    def _dies_halfway(src: Path, dst: Path, **kwargs: object) -> None:
        Path(dst).mkdir(parents=True)
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copytree", _dies_halfway)
    resolve_instance_path(legacy)
    assert not (target / "user_personalities").exists()

    monkeypatch.setattr(shutil, "copytree", real_copytree)
    resolve_instance_path(legacy)
    assert (target / "user_personalities" / "guide" / "profile.md").exists()


def test_a_migration_that_fails_between_items_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory landing must not make the next start skip the personalities."""
    legacy = _legacy(tmp_path, memory=True)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    real_copytree = shutil.copytree

    def _fails(src: Path, dst: Path, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copytree", _fails)
    resolve_instance_path(legacy)
    assert (target / "memory.v1.json").exists()
    assert not (target / ".migrated").exists()

    monkeypatch.setattr(shutil, "copytree", real_copytree)
    resolve_instance_path(legacy)
    assert (target / "user_personalities" / "guide" / "profile.md").exists()
    assert (target / ".migrated").exists()


def test_data_deleted_after_migration_stays_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once migrated, the legacy copy never comes back."""
    legacy = _legacy(tmp_path)
    target = tmp_path / "instance"
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(target))

    resolve_instance_path(legacy)
    shutil.rmtree(target / "user_personalities")
    resolve_instance_path(legacy)

    assert not (target / "user_personalities").exists()


def test_unusable_path_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad override warns and keeps the previous directory, never crashes."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv(INSTANCE_PATH_ENV, str(blocker / "nope"))

    legacy = tmp_path / "site-packages"
    assert resolve_instance_path(legacy) == legacy
