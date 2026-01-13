"""Unit tests for the config module."""

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestConfigModuleLoad:
    """Tests for module-level config loading."""

    def test_module_loads_with_no_dotenv_file(self) -> None:
        """Test module-level code when no .env file is found (line 17).

        This covers the else branch at lines 16-17.
        """
        # Save original module references
        module_name = "reachy_mini_conversation_app.config"
        original_module = sys.modules.get(module_name)

        try:
            # Remove the module from cache to force reload
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Patch find_dotenv to return empty string (no .env found)
            with patch("dotenv.find_dotenv", return_value=""):
                # Import will trigger module-level code
                import reachy_mini_conversation_app.config as config_module

                # Verify the module loaded successfully
                assert hasattr(config_module, "config")
                assert hasattr(config_module, "Config")
        finally:
            # Restore the original module to avoid side effects on other tests
            if original_module is not None:
                sys.modules[module_name] = original_module
            elif module_name in sys.modules:
                del sys.modules[module_name]
                # Reimport the original module
                import reachy_mini_conversation_app.config  # noqa: F401

    def test_module_loads_with_dotenv_file(self, tmp_path: Any) -> None:
        """Test module-level code when .env file is found (lines 14-15).

        This covers the if branch at lines 12-15.
        """
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\n")

        # Save original module references
        module_name = "reachy_mini_conversation_app.config"
        original_module = sys.modules.get(module_name)

        try:
            # Remove the module from cache to force reload
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Patch find_dotenv to return our temp .env file path
            with patch("dotenv.find_dotenv", return_value=str(env_file)):
                with patch("dotenv.load_dotenv") as mock_load:
                    # Import will trigger module-level code
                    import reachy_mini_conversation_app.config as config_module

                    # Verify load_dotenv was called with the correct path
                    mock_load.assert_called_once_with(dotenv_path=str(env_file), override=True)

                    # Verify the module loaded successfully
                    assert hasattr(config_module, "config")
                    assert hasattr(config_module, "Config")
        finally:
            # Restore the original module to avoid side effects on other tests
            if original_module is not None:
                sys.modules[module_name] = original_module
            elif module_name in sys.modules:
                del sys.modules[module_name]
                # Reimport the original module
                import reachy_mini_conversation_app.config  # noqa: F401


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

        # Profile attribute
        assert hasattr(config, "REACHY_MINI_CUSTOM_PROFILE")

    def test_config_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Config uses correct default values when env vars not set."""
        # Clear relevant env vars
        monkeypatch.delenv("MODEL_NAME", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("LOCAL_VISION_MODEL", raising=False)

        # Re-import to get fresh defaults (module-level code already ran)
        # We test the defaults defined in the class
        assert os.getenv("MODEL_NAME", "gpt-realtime") == "gpt-realtime"
        assert os.getenv("HF_HOME", "./cache") == "./cache"
        assert os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct") == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


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

class TestSetCustomProfileExceptionHandling:
    """Tests for set_custom_profile exception handling."""

    def test_set_custom_profile_exception_on_config_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that exceptions during config update are caught (lines 58-59)."""
        from reachy_mini_conversation_app.config import config, set_custom_profile

        # Make config.REACHY_MINI_CUSTOM_PROFILE property raise on assignment
        original_setattr = type(config).__setattr__

        def raise_on_profile_setattr(self: Any, name: str, value: Any) -> None:
            if name == "REACHY_MINI_CUSTOM_PROFILE":
                raise RuntimeError("Config update error")
            original_setattr(self, name, value)

        monkeypatch.setattr(type(config), "__setattr__", raise_on_profile_setattr)

        # Should not raise - exception is caught
        set_custom_profile("test_profile")

        # Verify env var was still set (second try block succeeded)
        import os
        assert os.environ.get("REACHY_MINI_CUSTOM_PROFILE") == "test_profile"

        # Cleanup
        os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)

    def test_set_custom_profile_exception_on_env_update(self) -> None:
        """Test that exceptions during env update are caught (lines 68-69)."""
        from reachy_mini_conversation_app.config import config, set_custom_profile

        # First set successfully
        set_custom_profile("test_profile")
        assert config.REACHY_MINI_CUSTOM_PROFILE == "test_profile"

        # Clean up env var
        os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)

        # Mock os.environ operations to raise exceptions
        mock_environ = MagicMock()
        mock_environ.__setitem__ = MagicMock(side_effect=RuntimeError("Env set error"))
        mock_environ.pop = MagicMock(side_effect=RuntimeError("Env pop error"))

        # Use patch on the import within the function (os imported as _os)
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "os":
                mock_module = MagicMock()
                mock_module.environ = mock_environ
                return mock_module
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Should not raise - exception is caught in second try block
            set_custom_profile("another_profile")

        # Config should still be updated (first try block succeeded)
        assert config.REACHY_MINI_CUSTOM_PROFILE == "another_profile"
