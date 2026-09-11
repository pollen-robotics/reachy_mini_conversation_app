from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.profile_store import write_profile


def test_config_raises_on_external_profile_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail fast when external/built-in profile names collide."""
    external_profiles = tmp_path / "external_profiles"
    write_profile("default", external_profiles / "default", "External default.", [])

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Ambiguous profile names"):
        config_mod.Config()


def test_config_raises_on_external_profile_name_collision_with_builtin_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should treat compact built-in profile names as reserved."""
    external_profiles = tmp_path / "external_profiles"
    write_profile(
        "mad_scientist_assistant",
        external_profiles / "mad_scientist_assistant",
        "External scientist.",
        [],
    )

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Ambiguous profile names"):
        config_mod.Config()


def test_config_raises_on_external_tool_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail fast when external/built-in tool names collide."""
    external_tools = tmp_path / "external_tools"
    external_tools.mkdir(parents=True)
    (external_tools / "dance.py").write_text("# collision with built-in dance tool\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", external_tools)

    with pytest.raises(RuntimeError, match="Ambiguous tool names"):
        config_mod.Config()


def test_config_raises_when_selected_external_profile_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should fail fast when selected profile is absent from external root."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod.Config, "REACHY_MINI_CUSTOM_PROFILE", "missing_profile")
    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Selected profile 'missing_profile' was not found"):
        config_mod.Config()


def test_config_allows_packaged_default_with_external_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical default should not require an external profile copy."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir()

    monkeypatch.setattr(config_mod.Config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    configured = config_mod.Config()

    assert configured.REACHY_MINI_CUSTOM_PROFILE == "default"
    assert not (external_profiles / "default").exists()


def test_obsolete_backend_env_is_ignored_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Stale multi-backend selectors should be ignored with a warning, not change behaviour."""
    monkeypatch.setenv("BACKEND_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_NAME", "gpt-realtime-2")

    with caplog.at_level("WARNING"):
        config_mod.refresh_runtime_config_from_env()

    assert "BACKEND_PROVIDER" in caplog.text
    assert "MODEL_NAME" in caplog.text
    assert "Hugging Face backend only" in caplog.text


def test_hf_default_session_url_uses_stable_space_proxy() -> None:
    """The app should not embed the raw, replaceable Inference Endpoint allocator URL."""
    assert config_mod.HF_DEFAULTS.session_url == "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session"
    assert ".aws.endpoints.huggingface.cloud" not in config_mod.HF_DEFAULTS.session_url


def test_refresh_runtime_config_reloads_hf_runtime_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instance-local .env reloads should update every env-backed Hugging Face runtime field."""
    monkeypatch.setenv("HF_TOKEN", "hf-runtime-token")

    monkeypatch.setattr(config_mod.config, "HF_TOKEN", None)

    config_mod.refresh_runtime_config_from_env()

    assert config_mod.config.HF_TOKEN == "hf-runtime-token"


@pytest.mark.parametrize(
    ("configured_mode", "session_url", "direct_ws_url", "expected_mode", "expected_has_target"),
    [
        ("local", "https://hf.example.test/session", None, "local", False),
        ("deployed", "https://hf.example.test/session", "ws://127.0.0.1:8765/v1/realtime", "deployed", True),
        ("local", None, "ws://127.0.0.1:8765/v1/realtime", "local", True),
        ("deployed", None, "ws://127.0.0.1:8765/v1/realtime", "deployed", False),
    ],
)
def test_hf_connection_selection_uses_explicit_mode_for_target(
    monkeypatch: pytest.MonkeyPatch,
    configured_mode: str | None,
    session_url: str | None,
    direct_ws_url: str | None,
    expected_mode: str,
    expected_has_target: bool,
) -> None:
    """Hugging Face selection should use the configured mode without inferring from URLs."""
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_CONNECTION_MODE", configured_mode)
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_SESSION_URL", session_url)
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_WS_URL", direct_ws_url)

    selection = config_mod.get_hf_connection_selection()

    assert selection.mode == expected_mode
    assert selection.has_target is expected_has_target
    assert selection.session_url == session_url
    assert selection.direct_ws_url == direct_ws_url


def test_hf_connection_selection_requires_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hugging Face selection should fail instead of inferring a missing mode."""
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_CONNECTION_MODE", None)
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_SESSION_URL", "https://hf.example.test/session")
    monkeypatch.setattr(config_mod.config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    with pytest.raises(RuntimeError, match="HF_REALTIME_CONNECTION_MODE must be set"):
        config_mod.get_hf_connection_selection()
