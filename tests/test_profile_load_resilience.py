"""A broken selected profile must not stop the app from starting.

The selected profile is persisted, so it can become unreadable long after it
was chosen: the app is upgraded past the profile format it was authored in, or
its directory is partly removed. That used to exit the app with code 1, which
is unrecoverable in practice because the profile can only be changed from a UI
that never comes up.

`core_tools.initialize_tools` still raises, so library callers keep a catchable
failure (see test_external_loading.py). The recovery policy lives in the app
layer, which degrades to the packaged default profile instead of exiting.
"""

import sys
import importlib
from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.profile_store import write_profile


def _reset_core_tools() -> None:
    """Drop the cached tool registry so each case reloads from its profile."""
    for module_name in list(sys.modules):
        if module_name.startswith(
            ("reachy_mini_conversation_app.tools.", "reachy_mini_conversation_app._external_tools.")
        ):
            sys.modules.pop(module_name, None)
    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    importlib.reload(app_lifecycle)


def _use_profile(monkeypatch: pytest.MonkeyPatch, profiles_root: Path, profile: str) -> None:
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", profile)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    monkeypatch.setattr(config_mod, "LOCKED_PROFILE", None, raising=False)


@pytest.fixture(autouse=True)
def _isolate_tool_registry() -> None:
    _reset_core_tools()
    yield
    _reset_core_tools()


def test_legacy_profile_keeps_its_own_persona_rather_than_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-profile.md profile is read, so the fallback must not engage.

    The reported crash was a profile in this shape. It is handled by reading the
    legacy layout (test_profile_legacy_layout.py); the fallback below exists for
    profiles that are genuinely unreadable, and must not fire here or the user
    would lose their persona to the default one.
    """
    import logging

    profiles_root = tmp_path / "profiles"
    legacy_profile = profiles_root / "legacy_profile"
    legacy_profile.mkdir(parents=True)
    (legacy_profile / "instructions.txt").write_text("tu es un robot", encoding="utf-8")
    (legacy_profile / "greeting.txt").write_text("bonjour", encoding="utf-8")

    _use_profile(monkeypatch, profiles_root, "legacy_profile")

    abandoned = app_lifecycle.initialize_tools_with_default_fallback(None, logging.getLogger(__name__))

    assert abandoned is None, "a legacy profile is readable and must be kept"
    assert config_mod.config.REACHY_MINI_CUSTOM_PROFILE == "legacy_profile"

    from reachy_mini_conversation_app.tools import core_tools

    assert core_tools.ALL_TOOLS


def test_malformed_profile_document_falls_back_to_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile.md that fails to parse must not be fatal either."""
    import logging

    profiles_root = tmp_path / "profiles"
    broken_profile = profiles_root / "broken_profile"
    broken_profile.mkdir(parents=True)
    (broken_profile / "profile.md").write_text("no front matter here", encoding="utf-8")

    _use_profile(monkeypatch, profiles_root, "broken_profile")

    abandoned = app_lifecycle.initialize_tools_with_default_fallback(None, logging.getLogger(__name__))

    assert abandoned == "broken_profile"
    from reachy_mini_conversation_app.tools import core_tools

    assert core_tools.ALL_TOOLS


def test_readable_profile_is_left_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must not disturb a profile that reads correctly."""
    import logging

    profiles_root = tmp_path / "profiles"
    good_profile = profiles_root / "good_profile"
    write_profile("good_profile", good_profile, "sois utile", ["dance"])

    _use_profile(monkeypatch, profiles_root, "good_profile")

    abandoned = app_lifecycle.initialize_tools_with_default_fallback(None, logging.getLogger(__name__))

    assert abandoned is None
    assert config_mod.config.REACHY_MINI_CUSTOM_PROFILE == "good_profile"
    from reachy_mini_conversation_app.tools import core_tools

    assert "dance" in core_tools.ALL_TOOLS


def test_failure_propagates_when_the_fallback_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the default profile cannot load either, the caller still gets the error.

    main() turns that into exit 1, which is the right outcome: without a usable
    tool registry there is no conversation to run.
    """
    import logging

    profiles_root = tmp_path / "profiles"
    broken_profile = profiles_root / "broken_profile"
    broken_profile.mkdir(parents=True)
    (broken_profile / "profile.md").write_text("no front matter here", encoding="utf-8")

    _use_profile(monkeypatch, profiles_root, "broken_profile")

    def _always_fails(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tool registry is unusable")

    monkeypatch.setattr(app_lifecycle, "initialize_tools", _always_fails)

    with pytest.raises(RuntimeError, match="tool registry is unusable"):
        app_lifecycle.initialize_tools_with_default_fallback(None, logging.getLogger(__name__))


def test_locked_profile_build_does_not_claim_a_fallback_it_cannot_make(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With LOCKED_PROFILE pinned, set_custom_profile is a no-op.

    Falling through would retry the same broken profile and log that we started
    on the default, which is not what happened. Report the real situation and
    let the error propagate.
    """
    import logging

    profiles_root = tmp_path / "profiles"
    broken_profile = profiles_root / "locked_profile"
    broken_profile.mkdir(parents=True)
    (broken_profile / "profile.md").write_text("no front matter here", encoding="utf-8")

    _use_profile(monkeypatch, profiles_root, "locked_profile")
    monkeypatch.setattr(config_mod, "LOCKED_PROFILE", "locked_profile", raising=False)

    calls: list[object] = []
    real_initialize_tools = app_lifecycle.initialize_tools

    def _counting(*args: object, **kwargs: object) -> None:
        calls.append(object())
        return real_initialize_tools(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_lifecycle, "initialize_tools", _counting)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(Exception):
            app_lifecycle.initialize_tools_with_default_fallback(None, logging.getLogger(__name__))

    assert len(calls) == 1, "the locked profile must not be retried against itself"
    assert not any("starting on the 'default' profile instead" in r.message for r in caplog.records), (
        "must not claim a fallback that the lock prevents"
    )


def test_default_profile_selection_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already on the default profile, a failure must propagate without a retry."""
    import logging

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod, "LOCKED_PROFILE", None, raising=False)

    calls: list[object] = []

    def _always_fails(*_args: object, **_kwargs: object) -> None:
        calls.append(object())
        raise RuntimeError("tool registry is unusable")

    monkeypatch.setattr(app_lifecycle, "initialize_tools", _always_fails)

    with pytest.raises(RuntimeError, match="tool registry is unusable"):
        app_lifecycle.initialize_tools_with_default_fallback(None, logging.getLogger(__name__))

    assert len(calls) == 1, "the default profile should not be retried against itself"
