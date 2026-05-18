"""Tests for persisted instance-local startup settings."""

import json

from robot_comic.startup_settings import (
    STARTUP_SETTINGS_FILENAME,
    StartupSettings,
    read_startup_settings,
    write_startup_settings,
    load_startup_settings_into_runtime,
)


def test_write_and_read_startup_settings(tmp_path) -> None:
    """Startup settings should round-trip through startup_settings.json."""
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer")

    assert read_startup_settings(tmp_path) == StartupSettings(profile="house_comedian", voice="shimmer")


def test_write_and_read_movement_speed_round_trip(tmp_path) -> None:
    """movement_speed should round-trip through startup_settings.json (#309)."""
    write_startup_settings(tmp_path, movement_speed=0.45)

    assert read_startup_settings(tmp_path) == StartupSettings(movement_speed=0.45)


def test_write_movement_speed_clamps_out_of_range(tmp_path) -> None:
    """Out-of-range speeds are clamped to [0.1, 2.0] before persisting."""
    write_startup_settings(tmp_path, movement_speed=5.0)
    assert read_startup_settings(tmp_path).movement_speed == 2.0

    write_startup_settings(tmp_path, movement_speed=0.0)
    assert read_startup_settings(tmp_path).movement_speed == 0.1


def test_write_movement_speed_ignores_non_numeric(tmp_path) -> None:
    """A non-numeric speed clears the field rather than persisting garbage."""
    write_startup_settings(tmp_path, movement_speed=0.45)
    write_startup_settings(tmp_path, movement_speed="fast")  # type: ignore[arg-type]
    # "fast" normalises to None, which under partial-update means "clear it".
    assert read_startup_settings(tmp_path).movement_speed is None


def test_partial_update_preserves_other_fields(tmp_path) -> None:
    """Passing only movement_speed must not clobber profile/voice."""
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer")
    write_startup_settings(tmp_path, movement_speed=0.45)

    assert read_startup_settings(tmp_path) == StartupSettings(
        profile="house_comedian", voice="shimmer", movement_speed=0.45
    )


def test_partial_update_movement_speed_then_profile_preserves_speed(tmp_path) -> None:
    """Conversely, updating profile must not clear a previously-saved movement_speed."""
    write_startup_settings(tmp_path, movement_speed=0.45)
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer")

    assert read_startup_settings(tmp_path) == StartupSettings(
        profile="house_comedian", voice="shimmer", movement_speed=0.45
    )


def test_explicit_none_clears_field(tmp_path) -> None:
    """Passing an explicit ``None`` clears the field instead of preserving it."""
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer", movement_speed=0.45)
    write_startup_settings(tmp_path, movement_speed=None)

    assert read_startup_settings(tmp_path) == StartupSettings(profile="house_comedian", voice="shimmer")


def test_file_removed_when_all_fields_cleared(tmp_path) -> None:
    """Settings file should be removed when every field ends up None."""
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer", movement_speed=0.45)
    write_startup_settings(tmp_path, profile=None, voice=None, movement_speed=None)

    assert not (tmp_path / STARTUP_SETTINGS_FILENAME).exists()


def test_read_ignores_invalid_movement_speed_on_disk(tmp_path) -> None:
    """A malformed movement_speed value on disk reads back as None (no crash)."""
    (tmp_path / STARTUP_SETTINGS_FILENAME).write_text(
        json.dumps({"profile": "p", "movement_speed": "fast"}),
        encoding="utf-8",
    )
    assert read_startup_settings(tmp_path) == StartupSettings(profile="p", movement_speed=None)


def test_load_startup_settings_into_runtime_applies_profile_when_no_env(monkeypatch, tmp_path) -> None:
    """Startup settings should seed the runtime profile when no explicit env override exists."""
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer")
    applied_profiles: list[str | None] = []
    monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    monkeypatch.setattr(
        "robot_comic.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(profile="house_comedian", voice="shimmer")
    assert applied_profiles == ["house_comedian"]


def test_load_startup_settings_into_runtime_saved_settings_override_instance_env(monkeypatch, tmp_path) -> None:
    """Saved startup settings should override an instance-local profile env value."""
    write_startup_settings(tmp_path, profile="house_comedian", voice="shimmer")
    applied_profiles: list[str | None] = []
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "env_profile")
    monkeypatch.setattr(
        "robot_comic.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(profile="house_comedian", voice="shimmer")
    assert applied_profiles == ["house_comedian"]


def test_load_startup_settings_into_runtime_saved_settings_override_inherited_env(monkeypatch, tmp_path) -> None:
    """Saved startup settings should override a profile inherited from another `.env`."""
    write_startup_settings(tmp_path, profile="default", voice="cedar")
    applied_profiles: list[str | None] = []
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "house_comedian")
    monkeypatch.setattr(
        "robot_comic.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(profile="default", voice="cedar")
    assert applied_profiles == ["default"]


def test_load_startup_settings_into_runtime_preserves_inherited_env_without_saved_settings(
    monkeypatch, tmp_path
) -> None:
    """Inherited env config should still apply when no startup settings have been saved."""
    applied_profiles: list[str | None] = []
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "house_comedian")
    monkeypatch.setattr(
        "robot_comic.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings()
    assert applied_profiles == []
