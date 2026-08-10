"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config
from reachy_mini_conversation_app.companion.settings import CompanionSettings


SAVED_API_URL = "https://alice-smolagents-assistant-reachy-mini.hf.space"
SAVED_API_TOKEN = "a" * 32


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


def test_companion_connection_uses_coherent_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment values never mix with a saved connection."""
    settings = CompanionSettings(api_url=SAVED_API_URL, api_token=SAVED_API_TOKEN)
    monkeypatch.delenv(config.SMOL_ASSISTANT_API_URL_ENV, raising=False)
    monkeypatch.delenv(config.SMOL_ASSISTANT_API_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config.config, "HF_TOKEN", None)
    monkeypatch.setattr(config, "get_token", lambda: None)

    assert config.get_companion_connection(settings) is None

    monkeypatch.setattr(config, "get_token", lambda: "hf_saved_token")

    saved_connection = config.get_companion_connection(settings)

    assert saved_connection is not None
    assert saved_connection.api_url == SAVED_API_URL
    assert saved_connection.api_token == SAVED_API_TOKEN
    assert saved_connection.hf_token == "hf_saved_token"

    monkeypatch.setenv(config.SMOL_ASSISTANT_API_URL_ENV, "http://127.0.0.1:9000/")
    assert config.get_companion_connection(settings) is None
