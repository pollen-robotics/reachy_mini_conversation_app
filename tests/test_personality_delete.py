"""Regression coverage for deleting custom personalities."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.personality as personality_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.personality import delete_personality
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_override,
    write_profile_tool_override,
)
from reachy_mini_conversation_app.personality_routes import (
    RouteError,
    PersonalityOps,
    build_personality_ops,
)


def _make_user_profile(name: str) -> None:
    personality_mod.save_user_personality(name, "Be brief.")


def _ops(persisted: str | None = None) -> PersonalityOps:
    return build_personality_ops(
        MagicMock(),
        lambda: None,
        get_persisted_personality=lambda: persisted,
    )


def test_delete_removes_user_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a user profile also removes its tool override."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    _make_user_profile("doomed")
    profile_dir = tmp_path / "user_personalities" / "doomed"
    write_profile_tool_override("user_personalities/doomed", ["dance"], tmp_path)
    assert profile_dir.is_dir()

    assert delete_personality("user_personalities/doomed") is True
    assert not profile_dir.exists()
    assert read_profile_tool_override("user_personalities/doomed", tmp_path) is None


def test_delete_refuses_builtin_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in profiles cannot be deleted."""
    builtin_dir = config.resolve_profile_dir("mad_scientist_assistant")
    assert builtin_dir.is_dir()

    assert delete_personality("mad_scientist_assistant") is False
    assert builtin_dir.is_dir()


def test_delete_refuses_path_outside_user_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile deletion cannot escape the user profile root."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    victim = tmp_path / "user_personalities" / "outside_target"
    victim.mkdir(parents=True)

    assert delete_personality("user_personalities/../outside_target") is False
    assert victim.is_dir()


def test_ops_refuses_deleting_current_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The active profile cannot be deleted."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/live")
    _make_user_profile("live")

    with pytest.raises(RouteError) as error:
        _ops().delete("user_personalities/live")

    assert error.value.reason == "profile_in_use"
    assert (tmp_path / "user_personalities" / "live").is_dir()


def test_ops_refuses_deleting_startup_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The persisted startup profile cannot be deleted."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    _make_user_profile("boots")

    with pytest.raises(RouteError) as error:
        _ops(persisted="user_personalities/boots").delete("user_personalities/boots")

    assert error.value.reason == "profile_in_use"
    assert (tmp_path / "user_personalities" / "boots").is_dir()


def test_ops_deletes_inactive_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inactive user profile can be deleted."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/live")
    _make_user_profile("live")
    _make_user_profile("spare")

    result = _ops().delete("user_personalities/spare")

    assert result["ok"] is True
    assert not (tmp_path / "user_personalities" / "spare").exists()


def test_ops_refuses_non_deletable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A built-in deletion reports the stable not-deletable reason."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    with pytest.raises(RouteError) as error:
        _ops().delete("mad_scientist_assistant")

    assert error.value.reason == "not_deletable"


def test_locked_mode_rejects_profile_creation_and_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locked mode prevents profile creation and deletion."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr("reachy_mini_conversation_app.personality_routes.LOCKED_PROFILE", "default")
    ops = _ops()

    with pytest.raises(RouteError) as save_error:
        ops.save({"name": "new_profile", "instructions": "Hello."})
    with pytest.raises(RouteError) as delete_error:
        ops.delete("user_personalities/old")

    assert save_error.value.reason == "profile_locked"
    assert delete_error.value.reason == "profile_locked"
    assert not (tmp_path / "user_personalities" / "new_profile").exists()
