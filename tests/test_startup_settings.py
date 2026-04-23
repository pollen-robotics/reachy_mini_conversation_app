"""Tests for persisted instance-local startup settings."""

from reachy_mini_conversation_app.startup_settings import (
    StartupSettings,
    read_startup_settings,
    write_startup_settings,
    load_startup_settings_into_runtime,
)


def test_write_and_read_startup_settings(tmp_path) -> None:
    """Startup settings should round-trip through startup_settings.json."""
    write_startup_settings(tmp_path, profile="sorry_bro", voice="shimmer")

    assert read_startup_settings(tmp_path) == StartupSettings(profile="sorry_bro", voice="shimmer")


def test_load_startup_settings_into_runtime_applies_profile_when_no_env(monkeypatch, tmp_path) -> None:
    """Startup settings should seed the runtime profile when no explicit env override exists."""
    write_startup_settings(tmp_path, profile="sorry_bro", voice="shimmer")
    applied_profiles: list[str | None] = []
    monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(profile="sorry_bro", voice="shimmer")
    assert applied_profiles == ["sorry_bro"]


def test_load_startup_settings_into_runtime_saved_settings_override_instance_env(monkeypatch, tmp_path) -> None:
    """Saved startup settings should override an instance-local profile env value."""
    write_startup_settings(tmp_path, profile="sorry_bro", voice="shimmer")
    applied_profiles: list[str | None] = []
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "env_profile")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(profile="sorry_bro", voice="shimmer")
    assert applied_profiles == ["sorry_bro"]


def test_load_startup_settings_into_runtime_saved_settings_override_inherited_env(monkeypatch, tmp_path) -> None:
    """Saved startup settings should override a profile inherited from another `.env`."""
    write_startup_settings(tmp_path, profile="nature_documentarian", voice="cedar")
    applied_profiles: list[str | None] = []
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "example")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(profile="nature_documentarian", voice="cedar")
    assert applied_profiles == ["nature_documentarian"]


def test_load_startup_settings_into_runtime_preserves_inherited_env_without_saved_settings(
    monkeypatch, tmp_path
) -> None:
    """Inherited env config should still apply when no startup settings have been saved."""
    applied_profiles: list[str | None] = []
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "example")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings()
    assert applied_profiles == []
