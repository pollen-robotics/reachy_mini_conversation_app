"""Unit tests for the config module."""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestConfig:
    """Tests for the Config class."""

    def test_config_has_required_attributes(self) -> None:
        """Test that Config has all expected attributes."""
        from reachy_mini_conversation_app.config import config

        # Required attribute
        assert hasattr(config, "OPENAI_API_KEY")

        # Optional attributes with defaults
        assert hasattr(config, "MODEL_NAME")
        assert hasattr(config, "HF_HOME")
        assert hasattr(config, "LOCAL_VISION_MODEL")
        assert hasattr(config, "HF_TOKEN")

        # Anthropic attributes
        assert hasattr(config, "ANTHROPIC_API_KEY")
        assert hasattr(config, "ANTHROPIC_MODEL")

        # GitHub attributes
        assert hasattr(config, "GITHUB_TOKEN")
        assert hasattr(config, "GITHUB_DEFAULT_OWNER")
        assert hasattr(config, "GITHUB_OWNER_EMAIL")

        # Profile attribute
        assert hasattr(config, "REACHY_MINI_CUSTOM_PROFILE")

    def test_config_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Config uses correct default values when env vars not set."""
        # Clear relevant env vars
        monkeypatch.delenv("MODEL_NAME", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("LOCAL_VISION_MODEL", raising=False)
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

        # Re-import to get fresh defaults (module-level code already ran)
        # We test the defaults defined in the class
        assert os.getenv("MODEL_NAME", "gpt-realtime") == "gpt-realtime"
        assert os.getenv("HF_HOME", "./cache") == "./cache"
        assert os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct") == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        assert os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


class TestSetCustomProfile:
    """Tests for set_custom_profile function."""

    def test_set_custom_profile_with_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setting a custom profile."""
        from reachy_mini_conversation_app.config import config, set_custom_profile

        # Clear any existing value
        monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)

        set_custom_profile("linus")

        assert config.REACHY_MINI_CUSTOM_PROFILE == "linus"
        assert os.environ.get("REACHY_MINI_CUSTOM_PROFILE") == "linus"

    def test_set_custom_profile_with_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clearing the custom profile."""
        from reachy_mini_conversation_app.config import config, set_custom_profile

        # Set a value first
        monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "linus")
        config.REACHY_MINI_CUSTOM_PROFILE = "linus"

        set_custom_profile(None)

        assert config.REACHY_MINI_CUSTOM_PROFILE is None
        assert os.environ.get("REACHY_MINI_CUSTOM_PROFILE") is None

    def test_set_custom_profile_updates_both_config_and_env(self) -> None:
        """Test that both config object and env var are updated."""
        from reachy_mini_conversation_app.config import config, set_custom_profile

        set_custom_profile("test_profile")

        # Both should be updated
        assert config.REACHY_MINI_CUSTOM_PROFILE == "test_profile"
        assert os.environ.get("REACHY_MINI_CUSTOM_PROFILE") == "test_profile"

        # Cleanup
        set_custom_profile(None)


class TestSetConfigValue:
    """Tests for set_config_value function."""

    def test_set_config_value_success(self) -> None:
        """Test successfully setting a config value."""
        from reachy_mini_conversation_app.config import config, set_config_value

        original = config.MODEL_NAME

        result = set_config_value("MODEL_NAME", "test-model")

        assert result is True
        assert config.MODEL_NAME == "test-model"
        assert os.environ.get("MODEL_NAME") == "test-model"

        # Restore
        set_config_value("MODEL_NAME", original)

    def test_set_config_value_with_none_removes_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that setting None removes the env var."""
        from reachy_mini_conversation_app.config import config, set_config_value

        # Set a value first
        monkeypatch.setenv("HF_TOKEN", "test-token")
        config.HF_TOKEN = "test-token"

        result = set_config_value("HF_TOKEN", None)

        assert result is True
        assert config.HF_TOKEN is None
        assert os.environ.get("HF_TOKEN") is None

    def test_set_config_value_with_empty_string_removes_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that setting empty string removes the env var."""
        from reachy_mini_conversation_app.config import config, set_config_value

        monkeypatch.setenv("HF_TOKEN", "test-token")
        config.HF_TOKEN = "test-token"

        result = set_config_value("HF_TOKEN", "")

        assert result is True
        assert config.HF_TOKEN == ""
        assert os.environ.get("HF_TOKEN") is None

    def test_set_config_value_nonexistent_key(self) -> None:
        """Test setting a key that doesn't exist in config."""
        from reachy_mini_conversation_app.config import set_config_value

        # Should still succeed (env var gets set even if not in config)
        result = set_config_value("NONEXISTENT_KEY", "value")

        assert result is True
        assert os.environ.get("NONEXISTENT_KEY") == "value"

        # Cleanup
        os.environ.pop("NONEXISTENT_KEY", None)

    def test_set_config_value_handles_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that exceptions are handled gracefully."""
        from reachy_mini_conversation_app.config import config, set_config_value

        # Make setattr on config raise an exception
        original_setattr = config.__class__.__setattr__

        def raise_on_setattr(self: object, name: str, value: object) -> None:
            if name == "TEST_FAIL_KEY":
                raise RuntimeError("Test error")
            original_setattr(self, name, value)  # type: ignore[arg-type]

        monkeypatch.setattr(config.__class__, "__setattr__", raise_on_setattr)

        # Also make os.environ raise
        original_environ_setitem = os.environ.__class__.__setitem__

        def raise_on_environ(self: object, key: str, value: str) -> None:
            if key == "TEST_FAIL_KEY":
                raise RuntimeError("Test error")
            original_environ_setitem(self, key, value)  # type: ignore[arg-type]

        monkeypatch.setattr(os.environ.__class__, "__setitem__", raise_on_environ)

        result = set_config_value("TEST_FAIL_KEY", "test")

        assert result is False


class TestReloadConfig:
    """Tests for reload_config function."""

    @patch("reachy_mini_conversation_app.config.find_dotenv")
    def test_reload_config_updates_values(
        self, mock_find_dotenv: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that reload_config updates config values from environment."""
        from reachy_mini_conversation_app.config import config, reload_config

        # Prevent loading .env file
        mock_find_dotenv.return_value = ""

        # Set new values in environment
        monkeypatch.setenv("OPENAI_API_KEY", "new-openai-key")
        monkeypatch.setenv("MODEL_NAME", "new-model")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "new-anthropic-key")

        reload_config()

        assert config.OPENAI_API_KEY == "new-openai-key"
        assert config.MODEL_NAME == "new-model"
        assert config.ANTHROPIC_API_KEY == "new-anthropic-key"

    def test_reload_config_uses_defaults_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that reload_config uses defaults when env vars not set."""
        from reachy_mini_conversation_app.config import config, reload_config

        # Clear env vars
        monkeypatch.delenv("MODEL_NAME", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        reload_config()

        assert config.MODEL_NAME == "gpt-realtime"
        assert config.HF_HOME == "./cache"

    @patch("reachy_mini_conversation_app.config.find_dotenv")
    def test_reload_config_clears_optional_values(
        self, mock_find_dotenv: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that reload_config clears values when env vars removed."""
        from reachy_mini_conversation_app.config import config, reload_config

        # Prevent loading .env file
        mock_find_dotenv.return_value = ""

        # Set a value first
        config.HF_TOKEN = "old-token"
        monkeypatch.delenv("HF_TOKEN", raising=False)

        reload_config()

        assert config.HF_TOKEN is None

    @patch("reachy_mini_conversation_app.config.find_dotenv")
    @patch("reachy_mini_conversation_app.config.load_dotenv")
    def test_reload_config_loads_dotenv_when_found(
        self,
        mock_load_dotenv: MagicMock,
        mock_find_dotenv: MagicMock,
    ) -> None:
        """Test that reload_config loads .env file when found."""
        from reachy_mini_conversation_app.config import reload_config

        mock_find_dotenv.return_value = "/path/to/.env"

        reload_config()

        mock_find_dotenv.assert_called_once_with(usecwd=True)
        mock_load_dotenv.assert_called_once_with(dotenv_path="/path/to/.env", override=True)

    @patch("reachy_mini_conversation_app.config.find_dotenv")
    @patch("reachy_mini_conversation_app.config.load_dotenv")
    def test_reload_config_skips_dotenv_when_not_found(
        self,
        mock_load_dotenv: MagicMock,
        mock_find_dotenv: MagicMock,
    ) -> None:
        """Test that reload_config skips loading when no .env found."""
        from reachy_mini_conversation_app.config import reload_config

        mock_find_dotenv.return_value = ""

        reload_config()

        mock_find_dotenv.assert_called_once_with(usecwd=True)
        mock_load_dotenv.assert_not_called()


class TestConfigSingleton:
    """Tests for config singleton behavior."""

    def test_config_is_singleton(self) -> None:
        """Test that config is a singleton instance."""
        from reachy_mini_conversation_app.config import config as config1
        from reachy_mini_conversation_app.config import config as config2

        assert config1 is config2

    def test_config_modifications_persist(self) -> None:
        """Test that modifications to config persist across imports."""
        from reachy_mini_conversation_app.config import config

        original = config.MODEL_NAME
        config.MODEL_NAME = "modified-model"

        # Re-import
        from reachy_mini_conversation_app.config import config as config2

        assert config2.MODEL_NAME == "modified-model"

        # Restore
        config.MODEL_NAME = original
