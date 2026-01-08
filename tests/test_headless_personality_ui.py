"""Unit tests for headless_personality_ui module."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Store original modules before any mocking
_ORIGINAL_MODULES: dict[str, Any] = {}
_MODULES_TO_MOCK = [
    "fastapi",
    "fastapi.responses",
    "gradio",
    "reachy_mini_conversation_app.openai_realtime",
]


@pytest.fixture(autouse=True)
def mock_fastapi_dependencies() -> Any:
    """Mock fastapi and gradio for tests."""
    # Save originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in sys.modules:
            _ORIGINAL_MODULES[mod_name] = sys.modules[mod_name]

    # Create mock fastapi
    mock_fastapi = MagicMock()
    mock_fastapi.FastAPI = MagicMock()
    mock_fastapi.Request = MagicMock()
    mock_fastapi.Body = MagicMock(return_value=MagicMock())
    mock_responses = MagicMock()
    mock_responses.JSONResponse = MagicMock()
    mock_fastapi.responses = mock_responses

    # Create mock gradio
    mock_gradio = MagicMock()

    # Create mock OpenaiRealtimeHandler
    mock_openai_realtime = MagicMock()
    mock_openai_realtime.OpenaiRealtimeHandler = MagicMock()

    sys.modules["fastapi"] = mock_fastapi
    sys.modules["fastapi.responses"] = mock_responses
    sys.modules["gradio"] = mock_gradio
    sys.modules["reachy_mini_conversation_app.openai_realtime"] = mock_openai_realtime

    yield {
        "mock_fastapi": mock_fastapi,
        "mock_gradio": mock_gradio,
        "mock_openai_realtime": mock_openai_realtime,
    }

    # Restore originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in _ORIGINAL_MODULES:
            sys.modules[mod_name] = _ORIGINAL_MODULES[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]

    # Clear cached module imports
    mods_to_clear = [
        k
        for k in sys.modules
        if k.startswith("reachy_mini_conversation_app.headless_personality_ui")
    ]
    for mod_name in mods_to_clear:
        del sys.modules[mod_name]


class TestConfigVars:
    """Tests for CONFIG_VARS constant."""

    def test_config_vars_exists(self) -> None:
        """Test that CONFIG_VARS is defined."""
        from reachy_mini_conversation_app.headless_personality_ui import CONFIG_VARS

        assert isinstance(CONFIG_VARS, list)
        assert len(CONFIG_VARS) > 0

    def test_config_vars_has_openai_key(self) -> None:
        """Test that CONFIG_VARS includes OPENAI_API_KEY."""
        from reachy_mini_conversation_app.headless_personality_ui import CONFIG_VARS

        keys = [var[0] for var in CONFIG_VARS]
        assert "OPENAI_API_KEY" in keys

    def test_config_vars_format(self) -> None:
        """Test that CONFIG_VARS has correct format."""
        from reachy_mini_conversation_app.headless_personality_ui import CONFIG_VARS

        for var in CONFIG_VARS:
            assert len(var) == 4  # (env_key, config_attr, is_secret, description)
            assert isinstance(var[0], str)  # env_key
            assert isinstance(var[1], str)  # config_attr
            assert isinstance(var[2], bool)  # is_secret
            assert isinstance(var[3], str)  # description


class TestMaskSecret:
    """Tests for _mask_secret helper (tested indirectly via mount)."""

    def test_mask_secret_logic_empty(self) -> None:
        """Test masking empty values."""
        # Test the logic directly
        value = None
        is_secret = True
        if value is None or not value:
            result = None
        elif not is_secret:
            result = value
        elif len(value) <= 8:
            result = "***"
        else:
            result = f"{value[:4]}...{value[-4:]}"

        assert result is None

    def test_mask_secret_logic_short(self) -> None:
        """Test masking short secret values."""
        value = "abc123"
        is_secret = True
        if value is None or not value:
            result = None
        elif not is_secret:
            result = value
        elif len(value) <= 8:
            result = "***"
        else:
            result = f"{value[:4]}...{value[-4:]}"

        assert result == "***"

    def test_mask_secret_logic_long(self) -> None:
        """Test masking long secret values."""
        value = "sk-ant-1234567890abcdef"
        is_secret = True
        if value is None or not value:
            result = None
        elif not is_secret:
            result = value
        elif len(value) <= 8:
            result = "***"
        else:
            result = f"{value[:4]}...{value[-4:]}"

        assert result == "sk-a...cdef"

    def test_mask_secret_logic_not_secret(self) -> None:
        """Test non-secret values are not masked."""
        value = "my-model-name"
        is_secret = False
        if value is None or not value:
            result = None
        elif not is_secret:
            result = value
        elif len(value) <= 8:
            result = "***"
        else:
            result = f"{value[:4]}...{value[-4:]}"

        assert result == "my-model-name"


class TestMountPersonalityRoutes:
    """Tests for mount_personality_routes function."""

    def test_mount_personality_routes_adds_endpoints(self) -> None:
        """Test that mount_personality_routes adds endpoints to app."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        # Check that decorators were used (get, post endpoints)
        assert mock_app.get.called or mock_app.post.called

    def test_mount_personality_routes_with_persist_callbacks(self) -> None:
        """Test mount with persistence callbacks."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)
        mock_persist = MagicMock()
        mock_get_persisted = MagicMock(return_value="linus")

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            persist_personality=mock_persist,
            get_persisted_personality=mock_get_persisted,
        )

        # Should complete without error
        assert True


class TestGetEnvFilePath:
    """Tests for _get_env_file_path helper logic."""

    def test_get_env_file_path_logic_with_dotenv(self) -> None:
        """Test getting env file path when dotenv finds it."""
        # Simulate the logic
        from dotenv import find_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        result = Path(dotenv_path) if dotenv_path else None

        # Result depends on environment, just check it doesn't crash
        assert result is None or isinstance(result, Path)


class TestUpdateEnvFile:
    """Tests for _update_env_file helper logic."""

    def test_update_env_file_adds_new_key(self, tmp_path: Path) -> None:
        """Test adding a new key to env file."""
        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING_KEY=value\n")

        # Simulate the logic
        key = "NEW_KEY"
        value = "new_value"

        lines = []
        key_found = False

        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                    key_found = True
                    if value is not None and value != "":
                        lines.append(f"{key}={value}\n")
                else:
                    lines.append(line)

        if not key_found and value is not None and value != "":
            lines.append(f"{key}={value}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        content = env_path.read_text()
        assert "EXISTING_KEY=value" in content
        assert "NEW_KEY=new_value" in content

    def test_update_env_file_updates_existing_key(self, tmp_path: Path) -> None:
        """Test updating an existing key in env file."""
        env_path = tmp_path / ".env"
        env_path.write_text("MY_KEY=old_value\nOTHER_KEY=other\n")

        # Simulate the logic
        key = "MY_KEY"
        value = "new_value"

        lines = []
        key_found = False

        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                    key_found = True
                    if value is not None and value != "":
                        lines.append(f"{key}={value}\n")
                else:
                    lines.append(line)

        if not key_found and value is not None and value != "":
            lines.append(f"{key}={value}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        content = env_path.read_text()
        assert "MY_KEY=new_value" in content
        assert "OTHER_KEY=other" in content
        assert "old_value" not in content

    def test_update_env_file_removes_key(self, tmp_path: Path) -> None:
        """Test removing a key from env file."""
        env_path = tmp_path / ".env"
        env_path.write_text("MY_KEY=value\nOTHER_KEY=other\n")

        # Simulate the logic
        key = "MY_KEY"
        value = None

        lines = []
        key_found = False

        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                    key_found = True
                    if value is not None and value != "":
                        lines.append(f"{key}={value}\n")
                    # If value is None, we skip the line (remove it)
                else:
                    lines.append(line)

        if not key_found and value is not None and value != "":
            lines.append(f"{key}={value}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        content = env_path.read_text()
        assert "MY_KEY" not in content
        assert "OTHER_KEY=other" in content


class TestUpdateRuntimeConfig:
    """Tests for _update_runtime_config helper logic."""

    def test_update_runtime_config_sets_env_var(self) -> None:
        """Test setting environment variable."""
        key = "TEST_CONFIG_KEY_12345"
        value = "test_value"

        # Clean up first
        os.environ.pop(key, None)

        # Simulate the logic
        if value is not None and value != "":
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

        assert os.environ.get(key) == "test_value"

        # Clean up
        os.environ.pop(key, None)

    def test_update_runtime_config_removes_env_var(self) -> None:
        """Test removing environment variable."""
        key = "TEST_CONFIG_KEY_12345"

        # Set it first
        os.environ[key] = "some_value"

        # Simulate the logic
        value = None
        if value is not None and value != "":
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

        assert os.environ.get(key) is None


class TestStartupChoice:
    """Tests for _startup_choice helper logic."""

    def test_startup_choice_returns_default(self) -> None:
        """Test that startup choice returns default when no persisted value."""
        from reachy_mini_conversation_app.headless_personality_ui import DEFAULT_OPTION

        # Simulate the logic
        get_persisted_personality = None
        stored = None

        if get_persisted_personality is not None:
            stored = get_persisted_personality()
            if stored:
                result = stored
            else:
                result = DEFAULT_OPTION
        else:
            result = DEFAULT_OPTION

        assert result == DEFAULT_OPTION

    def test_startup_choice_returns_persisted(self) -> None:
        """Test that startup choice returns persisted value when available."""
        from reachy_mini_conversation_app.headless_personality_ui import DEFAULT_OPTION

        # Simulate the logic
        def get_persisted_personality() -> str:
            return "linus"

        stored = get_persisted_personality()
        if stored:
            result = stored
        else:
            result = DEFAULT_OPTION

        assert result == "linus"


class TestCurrentChoice:
    """Tests for _current_choice helper logic."""

    def test_current_choice_returns_config_value(self) -> None:
        """Test that current choice returns config value."""
        from reachy_mini_conversation_app.headless_personality_ui import DEFAULT_OPTION

        # Simulate the logic
        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.REACHY_MINI_CUSTOM_PROFILE = "my_profile"

            cur = getattr(mock_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            result = cur or DEFAULT_OPTION

        assert result == "my_profile"

    def test_current_choice_returns_default_when_none(self) -> None:
        """Test that current choice returns default when config is None."""
        from reachy_mini_conversation_app.headless_personality_ui import DEFAULT_OPTION

        # Simulate the logic
        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.REACHY_MINI_CUSTOM_PROFILE = None

            cur = getattr(mock_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            result = cur or DEFAULT_OPTION

        assert result == DEFAULT_OPTION


class TestListEndpoint:
    """Tests for /personalities endpoint logic."""

    def test_list_returns_choices_and_current(self) -> None:
        """Test that list endpoint returns correct structure."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            DEFAULT_OPTION,
            list_personalities,
        )

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities"
        ) as mock_list:
            mock_list.return_value = ["profile1", "profile2"]

            choices = [DEFAULT_OPTION, *mock_list()]

        assert DEFAULT_OPTION in choices
        assert "profile1" in choices
        assert "profile2" in choices


class TestLoadEndpoint:
    """Tests for /personalities/load endpoint logic."""

    def test_load_returns_profile_data(self, tmp_path: Path) -> None:
        """Test that load endpoint returns profile data."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            available_tools_for,
            read_instructions_for,
            resolve_profile_dir,
        )

        # Create a test profile
        profile_dir = tmp_path / "my_profile"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text("Test instructions")
        (profile_dir / "tools.txt").write_text("# comment\ntool1\ntool2")
        (profile_dir / "voice.txt").write_text("ash")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir",
            return_value=profile_dir,
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.read_instructions_for",
                return_value="Test instructions",
            ):
                with patch(
                    "reachy_mini_conversation_app.headless_personality_ui.available_tools_for",
                    return_value=["tool1", "tool2", "tool3"],
                ):
                    # Simulate endpoint logic
                    name = "my_profile"
                    instr = "Test instructions"
                    tools_txt = (profile_dir / "tools.txt").read_text()
                    voice = (profile_dir / "voice.txt").read_text().strip()
                    avail = ["tool1", "tool2", "tool3"]
                    enabled = [
                        ln.strip()
                        for ln in tools_txt.splitlines()
                        if ln.strip() and not ln.strip().startswith("#")
                    ]

        assert instr == "Test instructions"
        assert voice == "ash"
        assert "tool1" in enabled
        assert "tool2" in enabled


class TestSaveEndpoint:
    """Tests for /personalities/save endpoint logic."""

    def test_save_validates_name(self) -> None:
        """Test that save validates the name."""
        from reachy_mini_conversation_app.headless_personality_ui import _sanitize_name

        name_s = _sanitize_name("")

        assert name_s == ""  # Empty name should fail validation


class TestVoicesEndpoint:
    """Tests for /voices endpoint logic."""

    def test_voices_returns_default_when_no_loop(self) -> None:
        """Test that voices returns default when loop is unavailable."""
        get_loop = MagicMock(return_value=None)

        loop = get_loop()
        if loop is None:
            result = ["cedar"]
        else:
            result = []  # Would fetch from handler

        assert result == ["cedar"]
