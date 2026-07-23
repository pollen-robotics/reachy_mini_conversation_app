"""Regression coverage for deleting custom personalities.

Delete touches persisted user data, so we pin down both the storage-level
contract (`delete_personality`) and the route-level guard that protects the
active/startup profile from being removed underneath a running app.
"""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.personality as personality_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.personality import delete_personality
from reachy_mini_conversation_app.personality_routes import (
    RouteError,
    PersonalityOps,
    build_personality_ops,
)


def _make_user_profile(name: str) -> None:
    """Create a minimal UI-style profile under the writable user root."""
    personality_mod._write_profile(name, "Be brief.", "")


def test_delete_removes_user_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A UI-created profile under the user root can be deleted."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    _make_user_profile("doomed")
    profile_dir = tmp_path / "user_personalities" / "doomed"
    assert profile_dir.is_dir()

    assert delete_personality("user_personalities/doomed") is True
    assert not profile_dir.exists()


def test_delete_refuses_builtin_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in profiles live outside the user root and must never be deletable."""
    builtin_dir = personality_mod.resolve_profile_dir("mad_scientist_assistant")
    assert builtin_dir.is_dir()

    assert delete_personality("mad_scientist_assistant") is False
    assert builtin_dir.is_dir()


def test_delete_refuses_path_outside_user_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A traversal selection escaping the user root is refused, not followed."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    victim = tmp_path / "user_personalities" / "outside_target"
    victim.mkdir(parents=True)

    assert delete_personality("user_personalities/../outside_target") is False
    assert victim.is_dir()


def _ops(persisted: str | None = None) -> PersonalityOps:
    """Build personality ops with stub callbacks for delete-guard tests."""
    return build_personality_ops(
        object(),  # type: ignore[arg-type]  # delete does not touch the handler
        lambda: None,
        get_persisted_personality=(lambda: persisted),
    )


def test_ops_refuses_deleting_current_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting the live profile would break get_session_instructions() next startup."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/live")
    _make_user_profile("live")

    with pytest.raises(RouteError) as ei:
        _ops().delete("user_personalities/live")

    assert ei.value.reason == "profile_in_use"
    assert (tmp_path / "user_personalities" / "live").is_dir()


def test_ops_refuses_deleting_startup_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The persisted startup profile is guarded even when it is not the live one."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    _make_user_profile("boots")

    with pytest.raises(RouteError) as ei:
        _ops(persisted="user_personalities/boots").delete("user_personalities/boots")

    assert ei.value.reason == "profile_in_use"
    assert (tmp_path / "user_personalities" / "boots").is_dir()


def test_ops_deletes_inactive_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile that is neither live nor startup is deleted and returns ok."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/live")
    _make_user_profile("live")
    _make_user_profile("spare")

    result = _ops().delete("user_personalities/spare")

    assert result["ok"] is True
    assert not (tmp_path / "user_personalities" / "spare").exists()


def test_ops_refuses_non_deletable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in/missing deletes raise not_deletable so the UI keeps the card."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    with pytest.raises(RouteError) as ei:
        _ops().delete("mad_scientist_assistant")

    assert ei.value.reason == "not_deletable"
