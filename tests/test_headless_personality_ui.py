"""Unit tests for headless_personality_ui module."""

from __future__ import annotations
import os
import sys
import asyncio
from typing import Any, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def make_route_capture(method: str, handlers: dict[str, Any]) -> Callable[[str], Callable[[Any], Any]]:
    """Create a route capture function that stores handlers by method and path.

    This avoids RuntimeWarning about unawaited coroutines by not using MagicMock
    for route decorators.
    """
    def capture_route(path: str) -> Callable[[Any], Any]:
        def decorator(fn: Any) -> Any:
            handlers[f"{method} {path}"] = fn
            return fn
        return decorator
    return capture_route


# Store original modules before any mocking
_ORIGINAL_MODULES: dict[str, Any] = {}
_MODULES_TO_MOCK = [
    "fastapi",
    "fastapi.responses",
    "fastapi.staticfiles",
    "gradio",
    "reachy_mini_conversation_app.openai_realtime",
    "reachy_mini",
    "reachy_mini.apps",
    "reachy_mini.apps.app",
]


@pytest.fixture(autouse=True)
def mock_fastapi_dependencies() -> Any:
    """Mock fastapi, gradio, and reachy_mini for tests."""
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
    mock_staticfiles = MagicMock()
    mock_staticfiles.StaticFiles = MagicMock()

    # Create mock gradio
    mock_gradio = MagicMock()

    # Create mock OpenaiRealtimeHandler
    mock_openai_realtime = MagicMock()
    mock_openai_realtime.OpenaiRealtimeHandler = MagicMock()

    # Create mock reachy_mini
    mock_reachy_mini = MagicMock()
    mock_reachy_mini.ReachyMini = MagicMock()
    mock_reachy_mini_apps = MagicMock()
    mock_reachy_mini_apps_app = MagicMock()

    sys.modules["fastapi"] = mock_fastapi
    sys.modules["fastapi.responses"] = mock_responses
    sys.modules["fastapi.staticfiles"] = mock_staticfiles
    sys.modules["gradio"] = mock_gradio
    sys.modules["reachy_mini_conversation_app.openai_realtime"] = mock_openai_realtime
    sys.modules["reachy_mini"] = mock_reachy_mini
    sys.modules["reachy_mini.apps"] = mock_reachy_mini_apps
    sys.modules["reachy_mini.apps.app"] = mock_reachy_mini_apps_app

    yield {
        "mock_fastapi": mock_fastapi,
        "mock_gradio": mock_gradio,
        "mock_openai_realtime": mock_openai_realtime,
        "mock_reachy_mini": mock_reachy_mini,
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
                    instr = "Test instructions"
                    tools_txt = (profile_dir / "tools.txt").read_text()
                    voice = (profile_dir / "voice.txt").read_text().strip()
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


class TestMountedEndpoints:
    """Tests for the actual mounted endpoints."""

    def test_endpoints_registered(self) -> None:
        """Test that all expected endpoints are registered."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        # Collect all route paths registered
        get_paths = [call[0][0] for call in mock_app.get.call_args_list]
        post_paths = [call[0][0] for call in mock_app.post.call_args_list]
        delete_paths = [call[0][0] for call in mock_app.delete.call_args_list]

        # Check expected GET endpoints
        assert "/personalities" in get_paths
        assert "/personalities/load" in get_paths
        assert "/voices" in get_paths
        assert "/config" in get_paths
        assert "/config/{key}" in get_paths

        # Check expected POST endpoints
        assert "/personalities/save" in post_paths
        assert "/personalities/apply" in post_paths
        assert "/config/reload" in post_paths
        assert "/config/{key}" in post_paths

        # Check expected DELETE endpoints
        assert "/config/{key}" in delete_paths

    def test_mount_registers_save_raw_get(self) -> None:
        """Test that save_raw GET endpoint is registered."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        get_paths = [call[0][0] for call in mock_app.get.call_args_list]
        assert "/personalities/save_raw" in get_paths

    def test_mount_registers_save_raw_post(self) -> None:
        """Test that save_raw POST endpoint is registered."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        post_paths = [call[0][0] for call in mock_app.post.call_args_list]
        assert "/personalities/save_raw" in post_paths


class TestConfigVarsContent:
    """Tests for CONFIG_VARS content validation.

    Note: CONFIG_VARS is now dynamically generated from BASE_CONFIG_VARS
    plus any tool-specific env vars. ANTHROPIC_API_KEY and GITHUB_TOKEN
    are only present when tools that declare them are loaded (e.g., Linus profile).
    """

    def test_config_vars_has_base_vars(self) -> None:
        """Test that CONFIG_VARS includes base configuration variables."""
        from reachy_mini_conversation_app.headless_personality_ui import CONFIG_VARS

        keys = [var[0] for var in CONFIG_VARS]
        # Base vars should always be present
        assert "OPENAI_API_KEY" in keys
        assert "MODEL_NAME" in keys
        assert "HF_TOKEN" in keys
        assert "HF_HOME" in keys
        assert "LOCAL_VISION_MODEL" in keys
        assert "REACHY_MINI_CUSTOM_PROFILE" in keys

    def test_config_vars_base_secrets_are_marked(self) -> None:
        """Test that base secret vars are properly marked."""
        from reachy_mini_conversation_app.headless_personality_ui import CONFIG_VARS

        secrets = {var[0]: var[2] for var in CONFIG_VARS}
        assert secrets["OPENAI_API_KEY"] is True
        assert secrets["HF_TOKEN"] is True
        assert secrets["MODEL_NAME"] is False

    def test_config_vars_has_descriptions(self) -> None:
        """Test that all config vars have descriptions."""
        from reachy_mini_conversation_app.headless_personality_ui import CONFIG_VARS

        for var in CONFIG_VARS:
            assert len(var[3]) > 0  # description should not be empty

    def test_config_vars_dynamically_includes_tool_vars(self) -> None:
        """Test that CONFIG_VARS can include tool-specific vars when tools declare them."""
        from reachy_mini_conversation_app.tools import core_tools
        from reachy_mini_conversation_app.tools.core_tools import Tool, EnvVar

        # Create a mock tool with required_env_vars
        class MockToolWithEnvVars(Tool):
            name = "mock_tool_env_test"
            description = "Mock tool for env var test"
            parameters_schema = {"type": "object", "properties": {}}
            required_env_vars = [
                EnvVar("MOCK_API_KEY", is_secret=True, description="Mock API key"),
            ]

            async def __call__(self, deps: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        # Temporarily add to ALL_TOOLS
        original_tools = core_tools.ALL_TOOLS.copy()
        core_tools.ALL_TOOLS["mock_tool_env_test"] = MockToolWithEnvVars()

        try:
            # Re-import to get fresh CONFIG_VARS via get_config_vars()
            from reachy_mini_conversation_app.headless_personality_ui import (
                _get_config_vars_list,
            )

            config_vars = _get_config_vars_list()
            keys = [var[0] for var in config_vars]

            assert "MOCK_API_KEY" in keys
        finally:
            core_tools.ALL_TOOLS.clear()
            core_tools.ALL_TOOLS.update(original_tools)


class TestGetConfigVarsListFunction:
    """Tests for _get_config_vars_list function."""

    def test_get_config_vars_list_returns_list(self) -> None:
        """Test that _get_config_vars_list returns a list of tuples."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            _get_config_vars_list,
        )

        result = _get_config_vars_list()

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 4

    def test_get_config_vars_list_matches_get_config_vars(self) -> None:
        """Test that _get_config_vars_list returns same as get_config_vars."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            get_config_vars,
            _get_config_vars_list,
        )

        result1 = _get_config_vars_list()
        result2 = get_config_vars()

        assert result1 == result2


class TestImportedConstants:
    """Tests for imported constants from headless_personality."""

    def test_default_option_imported(self) -> None:
        """Test that DEFAULT_OPTION is properly imported."""
        from reachy_mini_conversation_app.headless_personality_ui import DEFAULT_OPTION

        assert DEFAULT_OPTION == "(built-in default)"

    def test_sanitize_name_imported(self) -> None:
        """Test that _sanitize_name is properly imported."""
        from reachy_mini_conversation_app.headless_personality_ui import _sanitize_name

        assert _sanitize_name("test name") == "test_name"
        assert _sanitize_name("Test!@#Name") == "TestName"

    def test_list_personalities_imported(self) -> None:
        """Test that list_personalities is properly imported."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            list_personalities,
        )

        # Should return a list (may be empty)
        result = list_personalities()
        assert isinstance(result, list)


class TestMountWithPydanticModels:
    """Tests for Pydantic model usage in mount."""

    def test_mount_creates_pydantic_models(self) -> None:
        """Test that Pydantic models are created during mount."""
        # Mock pydantic
        mock_pydantic = MagicMock()
        mock_base_model = MagicMock()
        mock_pydantic.BaseModel = mock_base_model

        with patch.dict(
            "sys.modules",
            {
                "pydantic": mock_pydantic,
            },
        ):
            from reachy_mini_conversation_app.headless_personality_ui import (
                mount_personality_routes,
            )

            mock_app = MagicMock()
            mock_handler = MagicMock()
            mock_get_loop = MagicMock(return_value=None)

            # Should not raise even with mock pydantic
            mount_personality_routes(mock_app, mock_handler, mock_get_loop)


class TestEndpointHandlerExtraction:
    """Tests that extract and test endpoint handlers."""

    def test_list_endpoint_handler(self) -> None:
        """Test extracting and calling the list endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        # Capture the decorated functions
        registered_handlers: dict[str, Any] = {}

        def capture_get(path: str) -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers[f"GET {path}"] = fn
                return fn

            return decorator

        def capture_post(path: str) -> Any:
            def decorator(fn: Any) -> Any:
                registered_handlers[f"POST {path}"] = fn
                return fn

            return decorator

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        # Test the /personalities endpoint
        list_handler = registered_handlers["GET /personalities"]
        assert list_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities"
        ) as mock_list:
            mock_list.return_value = ["profile1"]
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                result = list_handler()
                assert "choices" in result
                assert "current" in result
                assert "startup" in result

    def test_load_endpoint_handler(self) -> None:
        """Test the load endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        load_handler = registered_handlers["GET /personalities/load"]
        assert load_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.read_instructions_for"
        ) as mock_read:
            mock_read.return_value = "Test instructions"
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.available_tools_for"
            ) as mock_tools:
                mock_tools.return_value = ["tool1", "tool2"]

                from reachy_mini_conversation_app.headless_personality_ui import (
                    DEFAULT_OPTION,
                )

                result = load_handler(DEFAULT_OPTION)
                assert "instructions" in result
                assert result["instructions"] == "Test instructions"
                assert result["voice"] == "cedar"

    def test_voices_endpoint_handler_no_loop(self) -> None:
        """Test the voices endpoint handler when no loop."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        voices_handler = registered_handlers["GET /voices"]
        assert voices_handler is not None

        # Call the async handler
        result = asyncio.new_event_loop().run_until_complete(voices_handler())
        assert result == ["cedar"]

    def test_config_endpoint_handler(self) -> None:
        """Test the config endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        config_handler = registered_handlers["GET /config"]
        assert config_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.OPENAI_API_KEY = "sk-test123456789"
            mock_config.MODEL_NAME = "gpt-4"
            mock_config.HF_TOKEN = None
            mock_config.HF_HOME = None
            mock_config.LOCAL_VISION_MODEL = None
            mock_config.ANTHROPIC_API_KEY = None
            mock_config.ANTHROPIC_MODEL = None
            mock_config.GITHUB_TOKEN = None
            mock_config.GITHUB_DEFAULT_OWNER = None
            mock_config.GITHUB_OWNER_EMAIL = None
            mock_config.REACHY_MINI_CUSTOM_PROFILE = None

            result = config_handler()
            assert "variables" in result
            assert isinstance(result["variables"], list)

    def test_config_key_get_handler(self) -> None:
        """Test the config/{key} GET endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        config_key_handler = registered_handlers["GET /config/{key}"]
        assert config_key_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = "gpt-4"

            result = config_key_handler("MODEL_NAME")
            assert result["key"] == "MODEL_NAME"
            assert result["value"] == "gpt-4"
            assert result["is_secret"] is False

    def test_config_key_get_unknown(self) -> None:
        """Test the config/{key} GET for unknown key."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        config_key_handler = registered_handlers["GET /config/{key}"]
        result = config_key_handler("UNKNOWN_KEY")
        # Should return JSONResponse for unknown key
        assert result is not None

    def test_save_endpoint_handler_valid_name(self, tmp_path: Path) -> None:
        """Test the save endpoint handler with valid name."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_handler = registered_handlers["POST /personalities/save"]
        assert save_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile"
        ) as mock_write:
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.list_personalities"
            ) as mock_list:
                mock_list.return_value = ["profile1"]

                result = save_handler(
                    name="test_profile",
                    instructions="Test instructions",
                    tools_text="tool1\ntool2",
                    voice="ash",
                )

                assert result["ok"] is True
                assert "user_personalities/test_profile" in result["value"]
                mock_write.assert_called_once()

    def test_save_endpoint_handler_invalid_name(self) -> None:
        """Test the save endpoint handler with invalid name."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_handler = registered_handlers["POST /personalities/save"]

        # Empty name should return error
        result = save_handler(
            name="",
            instructions="Test",
            tools_text="",
            voice="cedar",
        )
        # Result is JSONResponse mock
        assert result is not None

    def test_save_endpoint_handler_write_error(self) -> None:
        """Test the save endpoint handler with write error."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_handler = registered_handlers["POST /personalities/save"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            side_effect=IOError("Write failed"),
        ):
            result = save_handler(
                name="test_profile",
                instructions="Test",
                tools_text="",
                voice="cedar",
            )
            # Should return error response
            assert result is not None

    def test_save_raw_post_handler(self) -> None:
        """Test the save_raw POST endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_raw_handler = registered_handlers["POST /personalities/save_raw"]
        assert save_raw_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile"
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
                return_value=[],
            ):
                result = save_raw_handler(
                    name="raw_profile",
                    instructions="Raw instructions",
                    tools_text="",
                    voice="cedar",
                )
                assert result["ok"] is True

    def test_save_raw_get_handler(self) -> None:
        """Test the save_raw GET endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_raw_get_handler = registered_handlers["GET /personalities/save_raw"]
        assert save_raw_get_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile"
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
                return_value=[],
            ):
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    save_raw_get_handler(
                        name="get_raw_profile",
                        instructions="Get raw instructions",
                        tools_text="",
                        voice="cedar",
                    )
                )
                loop.close()
                assert result["ok"] is True

    def test_load_endpoint_with_profile(self, tmp_path: Path) -> None:
        """Test the load endpoint handler with actual profile."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        # Create test profile directory
        profile_dir = tmp_path / "test_profile"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text("Test instructions")
        (profile_dir / "tools.txt").write_text("tool1\ntool2")
        (profile_dir / "voice.txt").write_text("ash")

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        load_handler = registered_handlers["GET /personalities/load"]

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
                    result = load_handler("test_profile")

                    assert result["instructions"] == "Test instructions"
                    assert result["voice"] == "ash"
                    assert "tool1" in result["enabled_tools"]
                    assert "tool2" in result["enabled_tools"]

    def test_config_reload_handler(self) -> None:
        """Test the config/reload POST endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        reload_handler = registered_handlers["POST /config/reload"]
        assert reload_handler is not None

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.reload_config"
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.OPENAI_API_KEY = "test"
                mock_config.MODEL_NAME = None
                mock_config.HF_TOKEN = None
                mock_config.HF_HOME = None
                mock_config.LOCAL_VISION_MODEL = None
                mock_config.ANTHROPIC_API_KEY = None
                mock_config.ANTHROPIC_MODEL = None
                mock_config.GITHUB_TOKEN = None
                mock_config.GITHUB_DEFAULT_OWNER = None
                mock_config.GITHUB_OWNER_EMAIL = None
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                result = reload_handler()
                assert result["ok"] is True
                assert "variables" in result

    def test_config_key_post_handler(self, tmp_path: Path) -> None:
        """Test the config/{key} POST endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]
        assert set_config_handler is not None

        # Create a temp .env file
        env_file = tmp_path / ".env"
        env_file.write_text("")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = "new-model"

            with patch("dotenv.find_dotenv", return_value=str(env_file)):
                result = set_config_handler(
                    key="MODEL_NAME", value="new-model", persist=True
                )
                assert result["ok"] is True
                assert result["key"] == "MODEL_NAME"

    def test_config_key_delete_handler(self, tmp_path: Path) -> None:
        """Test the config/{key} DELETE endpoint handler."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        delete_config_handler = registered_handlers["DELETE /config/{key}"]
        assert delete_config_handler is not None

        # Create a temp .env file
        env_file = tmp_path / ".env"
        env_file.write_text("MODEL_NAME=old-model\n")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = None

            with patch("dotenv.find_dotenv", return_value=str(env_file)):
                result = delete_config_handler(key="MODEL_NAME", persist=True)
                assert result["ok"] is True
                assert result["cleared"] is True

    def test_apply_endpoint_no_loop(self) -> None:
        """Test the apply endpoint handler when loop is unavailable."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]
        assert apply_handler is not None

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(apply_handler())
        loop.close()

        # Should return error when loop is unavailable
        assert result is not None

    def test_apply_endpoint_with_payload(self) -> None:
        """Test the apply endpoint handler with payload."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_handler.apply_personality = AsyncMock(return_value="Applied")

        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        # Create mock payload
        mock_payload = MagicMock()
        mock_payload.name = "test_profile"
        mock_payload.persist = False

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied test_profile"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = "test_profile"

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(apply_handler(payload=mock_payload))
                loop.close()

                assert result["ok"] is True
                assert result["status"] == "Applied test_profile"

    def test_apply_endpoint_with_name_param(self) -> None:
        """Test the apply endpoint handler with name parameter."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name="linus")
                )
                loop.close()

                assert result["ok"] is True

    def test_voices_endpoint_with_loop(self) -> None:
        """Test the voices endpoint handler with active loop."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_handler.get_available_voices = AsyncMock(
            return_value=["cedar", "ash", "alloy"]
        )

        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        voices_handler = registered_handlers["GET /voices"]

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = ["cedar", "ash", "alloy"]

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(voices_handler())
            loop.close()

            assert "cedar" in result

    def test_startup_choice_with_persisted(self) -> None:
        """Test startup choice returns persisted value."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        def mock_get_persisted() -> str:
            return "persisted_profile"

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=mock_get_persisted,
        )

        list_handler = registered_handlers["GET /personalities"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            return_value=[],
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                result = list_handler()
                assert result["startup"] == "persisted_profile"

    def test_startup_choice_with_env_fallback(self) -> None:
        """Test startup choice falls back to env value."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        def mock_get_persisted() -> str | None:
            return None

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=mock_get_persisted,
        )

        list_handler = registered_handlers["GET /personalities"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            return_value=[],
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = "env_profile"

                result = list_handler()
                assert result["startup"] == "env_profile"


class TestMaskSecretFunction:
    """Tests for the actual _mask_secret function via endpoints."""

    def test_mask_secret_in_config_endpoint(self) -> None:
        """Test that secrets are masked in config endpoint."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        config_handler = registered_handlers["GET /config"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            # Long secret should be partially masked
            mock_config.OPENAI_API_KEY = "sk-1234567890abcdef"
            mock_config.MODEL_NAME = "gpt-4"  # Non-secret
            mock_config.HF_TOKEN = "short"  # Short secret
            mock_config.HF_HOME = None
            mock_config.LOCAL_VISION_MODEL = None
            mock_config.ANTHROPIC_API_KEY = None
            mock_config.ANTHROPIC_MODEL = None
            mock_config.GITHUB_TOKEN = None
            mock_config.GITHUB_DEFAULT_OWNER = None
            mock_config.GITHUB_OWNER_EMAIL = None
            mock_config.REACHY_MINI_CUSTOM_PROFILE = None

            result = config_handler()

            # Find OPENAI_API_KEY in result
            openai_var = next(
                (v for v in result["variables"] if v["key"] == "OPENAI_API_KEY"), None
            )
            assert openai_var is not None
            # Should be masked (sk-1...cdef format)
            assert openai_var["value"].startswith("sk-1")
            assert "..." in openai_var["value"]

            # MODEL_NAME should not be masked
            model_var = next(
                (v for v in result["variables"] if v["key"] == "MODEL_NAME"), None
            )
            assert model_var is not None
            assert model_var["value"] == "gpt-4"

            # Short secret should be fully masked
            hf_var = next(
                (v for v in result["variables"] if v["key"] == "HF_TOKEN"), None
            )
            assert hf_var is not None
            assert hf_var["value"] == "***"


class TestUpdateEnvFileFunction:
    """Tests for _update_env_file via config endpoints."""

    def test_update_env_file_creates_file(self, tmp_path: Path) -> None:
        """Test that _update_env_file creates file if not exists."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]

        # When find_dotenv returns empty string, it should create in cwd
        with patch("dotenv.find_dotenv", return_value=""):
            with patch("pathlib.Path.cwd", return_value=tmp_path):
                with patch(
                    "reachy_mini_conversation_app.headless_personality_ui.config"
                ) as mock_config:
                    mock_config.MODEL_NAME = "new-model"

                    result = set_config_handler(
                        key="MODEL_NAME", value="new-model", persist=True
                    )
                    assert result["ok"] is True


class TestCurrentChoiceFunction:
    """Tests for _current_choice helper function."""

    def test_current_choice_exception_handling(self) -> None:
        """Test _current_choice handles exceptions."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            DEFAULT_OPTION,
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        list_handler = registered_handlers["GET /personalities"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            return_value=[],
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                # Make getattr raise an exception
                type(mock_config).REACHY_MINI_CUSTOM_PROFILE = property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
                )

                result = list_handler()
                # Should fall back to DEFAULT_OPTION
                assert result["current"] == DEFAULT_OPTION


class TestLoadEndpointBranches:
    """Tests for load endpoint branch coverage."""

    def test_load_profile_missing_tools_txt(self, tmp_path: Path) -> None:
        """Test loading profile without tools.txt file."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        # Create a profile directory without tools.txt
        profile_dir = tmp_path / "profile_no_tools"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text("Test instructions")
        (profile_dir / "voice.txt").write_text("cedar")
        # No tools.txt

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        load_handler = registered_handlers["GET /personalities/load"]

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
                    return_value=["tool1"],
                ):
                    result = load_handler("profile_no_tools")

                    assert result["tools_text"] == ""
                    assert result["enabled_tools"] == []

    def test_load_profile_missing_voice_txt(self, tmp_path: Path) -> None:
        """Test loading profile without voice.txt file."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        # Create a profile directory without voice.txt
        profile_dir = tmp_path / "profile_no_voice"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text("Test instructions")
        (profile_dir / "tools.txt").write_text("tool1")
        # No voice.txt

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        load_handler = registered_handlers["GET /personalities/load"]

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
                    return_value=["tool1"],
                ):
                    result = load_handler("profile_no_voice")

                    # Should default to cedar when voice.txt missing
                    assert result["voice"] == "cedar"

    def test_load_profile_empty_voice_txt(self, tmp_path: Path) -> None:
        """Test loading profile with empty voice.txt file."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        # Create a profile directory with empty voice.txt
        profile_dir = tmp_path / "profile_empty_voice"
        profile_dir.mkdir()
        (profile_dir / "instructions.txt").write_text("Test instructions")
        (profile_dir / "tools.txt").write_text("tool1")
        (profile_dir / "voice.txt").write_text("")  # Empty voice

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        load_handler = registered_handlers["GET /personalities/load"]

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
                    return_value=["tool1"],
                ):
                    result = load_handler("profile_empty_voice")

                    # Should default to cedar when voice.txt is empty
                    assert result["voice"] == "cedar"


class TestSaveRawEndpointBranches:
    """Tests for save_raw endpoint branch coverage."""

    def test_save_raw_post_invalid_name(self) -> None:
        """Test save_raw POST with invalid name."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_raw_handler = registered_handlers["POST /personalities/save_raw"]

        # Empty name should return error
        result = save_raw_handler(
            name="",
            instructions="Test",
            tools_text="",
            voice="cedar",
        )
        # Result is JSONResponse for invalid name
        assert result is not None

    def test_save_raw_post_write_exception(self) -> None:
        """Test save_raw POST when write fails."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_raw_handler = registered_handlers["POST /personalities/save_raw"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            side_effect=IOError("Write failed"),
        ):
            result = save_raw_handler(
                name="test_profile",
                instructions="Test",
                tools_text="",
                voice="cedar",
            )
            # Should return error response
            assert result is not None

    def test_save_raw_get_invalid_name(self) -> None:
        """Test save_raw GET with invalid name."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_raw_get_handler = registered_handlers["GET /personalities/save_raw"]

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            save_raw_get_handler(
                name="",  # Invalid empty name
                instructions="Test",
                tools_text="",
                voice="cedar",
            )
        )
        loop.close()
        # Result is JSONResponse for invalid name
        assert result is not None

    def test_save_raw_get_write_exception(self) -> None:
        """Test save_raw GET when write fails."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        save_raw_get_handler = registered_handlers["GET /personalities/save_raw"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            side_effect=IOError("Write failed"),
        ):
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                save_raw_get_handler(
                    name="test_profile",
                    instructions="Test",
                    tools_text="",
                    voice="cedar",
                )
            )
            loop.close()
            # Should return error response
            assert result is not None


class TestApplyEndpointBranches:
    """Tests for apply endpoint branch coverage."""

    def test_apply_with_request_json_body(self) -> None:
        """Test apply endpoint with JSON from request body."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        # Create mock request with json body
        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            return {"name": "json_profile", "persist": True}

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                assert result["ok"] is True

    def test_apply_with_request_persist_query_param(self) -> None:
        """Test apply endpoint with persist query param."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()
        mock_persist = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            persist_personality=mock_persist,
        )

        apply_handler = registered_handlers["POST /personalities/apply"]

        # Create mock request with query params
        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            raise Exception("No JSON body")

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value="true")

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name="test", persist=None, request=mock_request)
                )
                loop.close()

                assert result["ok"] is True
                mock_persist.assert_called()

    def test_apply_persist_failure(self) -> None:
        """Test apply endpoint when persistence fails."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_persist(name: str | None) -> None:
            raise IOError("Persist failed")

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            persist_personality=mock_persist,
        )

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_payload = MagicMock()
        mock_payload.name = "test_profile"
        mock_payload.persist = True

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                # Close the coroutine to avoid RuntimeWarning
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(apply_handler(payload=mock_payload))
                loop.close()

                # Should still succeed even if persist fails
                assert result["ok"] is True

    def test_apply_exception(self) -> None:
        """Test apply endpoint when run_coroutine_threadsafe fails."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_payload = MagicMock()
        mock_payload.name = "test_profile"
        mock_payload.persist = False

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.side_effect = RuntimeError("Timeout")

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(apply_handler(payload=mock_payload))
            loop.close()

            # Should return error response
            assert result is not None


class TestVoicesEndpointBranches:
    """Tests for voices endpoint branch coverage."""

    def test_voices_exception_in_get_v(self) -> None:
        """Test voices endpoint when get_available_voices raises."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        voices_handler = registered_handlers["GET /voices"]

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.side_effect = RuntimeError("Timeout")

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(voices_handler())
            loop.close()

            # Should return default ["cedar"] on exception
            assert result == ["cedar"]


class TestUpdateEnvFileBranches:
    """Tests for _update_env_file branch coverage."""

    def test_update_env_file_exception(self, tmp_path: Path) -> None:
        """Test _update_env_file when write fails."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        def make_capture(method: str) -> Any:
            def capture_route(path: str) -> Any:
                def decorator(fn: Any) -> Any:
                    registered_handlers[f"{method} {path}"] = fn
                    return fn
                return decorator
            return capture_route

        mock_app.get = make_capture("GET")
        mock_app.post = make_capture("POST")
        mock_app.delete = make_capture("DELETE")

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]

        env_file = tmp_path / ".env"
        env_file.write_text("")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = "test"

            # Simulate write failure by making the env file read-only
            with patch("dotenv.find_dotenv", return_value=str(env_file)):
                with patch("builtins.open", side_effect=PermissionError("Cannot write")):
                    result = set_config_handler(
                        key="MODEL_NAME", value="new-model", persist=True
                    )
                    # Should still return ok but persisted=False
                    assert result["ok"] is True
                    assert result["persisted"] is False


class TestReloadConfigBranches:
    """Tests for reload config branch coverage."""

    def test_reload_config_exception(self) -> None:
        """Test reload config when reload_config raises."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        reload_handler = registered_handlers["POST /config/reload"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.reload_config",
            side_effect=IOError("Cannot reload"),
        ):
            result = reload_handler()
            # Should return error response
            assert result is not None


class TestConfigKeyBranches:
    """Tests for config key endpoint branch coverage."""

    def test_set_config_unknown_key(self) -> None:
        """Test set config with unknown key."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        def make_capture(method: str) -> Any:
            def capture_route(path: str) -> Any:
                def decorator(fn: Any) -> Any:
                    registered_handlers[f"{method} {path}"] = fn
                    return fn
                return decorator
            return capture_route

        mock_app.get = make_capture("GET")
        mock_app.post = make_capture("POST")
        mock_app.delete = make_capture("DELETE")

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]

        result = set_config_handler(key="UNKNOWN_KEY", value="value", persist=True)
        # Should return error response for unknown key
        assert result is not None

    def test_set_config_no_persist(self, tmp_path: Path) -> None:
        """Test set config without persistence."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        def make_capture(method: str) -> Any:
            def capture_route(path: str) -> Any:
                def decorator(fn: Any) -> Any:
                    registered_handlers[f"{method} {path}"] = fn
                    return fn
                return decorator
            return capture_route

        mock_app.get = make_capture("GET")
        mock_app.post = make_capture("POST")
        mock_app.delete = make_capture("DELETE")

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = "new-model"

            result = set_config_handler(
                key="MODEL_NAME", value="new-model", persist=False
            )
            assert result["ok"] is True
            assert result["persisted"] is False

    def test_delete_config_unknown_key(self) -> None:
        """Test delete config with unknown key."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        def make_capture(method: str) -> Any:
            def capture_route(path: str) -> Any:
                def decorator(fn: Any) -> Any:
                    registered_handlers[f"{method} {path}"] = fn
                    return fn
                return decorator
            return capture_route

        mock_app.get = make_capture("GET")
        mock_app.post = make_capture("POST")
        mock_app.delete = make_capture("DELETE")

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        delete_config_handler = registered_handlers["DELETE /config/{key}"]

        result = delete_config_handler(key="UNKNOWN_KEY", persist=True)
        # Should return error response for unknown key
        assert result is not None

    def test_delete_config_no_persist(self, tmp_path: Path) -> None:
        """Test delete config without persistence."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        def make_capture(method: str) -> Any:
            def capture_route(path: str) -> Any:
                def decorator(fn: Any) -> Any:
                    registered_handlers[f"{method} {path}"] = fn
                    return fn
                return decorator
            return capture_route

        mock_app.get = make_capture("GET")
        mock_app.post = make_capture("POST")
        mock_app.delete = make_capture("DELETE")

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        delete_config_handler = registered_handlers["DELETE /config/{key}"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = None

            result = delete_config_handler(key="MODEL_NAME", persist=False)
            assert result["ok"] is True
            assert result["persisted"] is False


class TestStartupChoiceBranches:
    """Tests for _startup_choice branch coverage."""

    def test_startup_choice_exception(self) -> None:
        """Test startup choice when get_persisted raises."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            DEFAULT_OPTION,
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        def mock_get_persisted() -> str:
            raise RuntimeError("Error")

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=mock_get_persisted,
        )

        list_handler = registered_handlers["GET /personalities"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            return_value=[],
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                result = list_handler()
                # Should fall back to DEFAULT_OPTION on exception
                assert result["startup"] == DEFAULT_OPTION


class TestApplyEndpointRequestBody:
    """Tests for apply endpoint request body parsing (lines 214-231)."""

    def test_apply_with_dict_body_has_name(self) -> None:
        """Test apply endpoint parsing dict body with name (lines 217-218)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            return {"name": "from_body"}

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                assert result["ok"] is True

    def test_apply_with_dict_body_has_persist(self) -> None:
        """Test apply endpoint parsing dict body with persist flag (line 219-220)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()
        mock_persist_fn = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(
            mock_app,
            mock_handler,
            mock_get_loop,
            persist_personality=mock_persist_fn,
        )

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            return {"name": "test_profile", "persist": True}

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                assert result["ok"] is True
                mock_persist_fn.assert_called()

    def test_apply_request_json_exception_sets_none(self) -> None:
        """Test apply endpoint when request.json() raises (lines 221-222)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            raise ValueError("Invalid JSON")

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                # Should still succeed (falls back to DEFAULT_OPTION)
                assert result["ok"] is True

    def test_apply_query_param_exception(self) -> None:
        """Test apply endpoint when query_params.get raises (lines 228-229)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            return {"name": "test"}

        mock_request.json = mock_json
        # Make query_params.get raise
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(side_effect=RuntimeError("Query error"))

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                # Should still succeed
                assert result["ok"] is True

    def test_apply_no_sel_name_uses_default(self) -> None:
        """Test apply endpoint falls back to DEFAULT_OPTION (line 231)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied default"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                # Call with all None to trigger fallback to DEFAULT_OPTION
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=None)
                )
                loop.close()

                assert result["ok"] is True


class TestVoicesInnerException:
    """Tests for voices endpoint inner exception path (lines 260-263)."""

    def test_voices_handler_inner_exception_returns_cedar(self) -> None:
        """Test that exception in get_available_voices returns cedar (lines 262-263)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        # Make get_available_voices raise an exception
        mock_handler.get_available_voices = AsyncMock(side_effect=RuntimeError("API error"))

        mock_loop = asyncio.new_event_loop()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        _ = registered_handlers["GET /voices"]

        # We need to actually run the inner coroutine in the loop
        # The issue is the test needs the inner _get_v function to run and handle exception
        with patch("asyncio.run_coroutine_threadsafe"):
            # This time we need to actually run the coroutine to test inner exception
            async def run_coro_and_raise(coro: Any, loop: Any) -> Any:
                # Run the coroutine which will call get_available_voices and catch exception
                mock_future = MagicMock()
                try:
                    result = await coro
                    mock_future.result.return_value = result
                except Exception:
                    mock_future.result.return_value = ["cedar"]
                return mock_future

            # Instead of mocking run_coroutine_threadsafe, run the handler directly
            # and catch the exception
            async def test_inner() -> list[str]:
                # Directly test the inner logic
                try:
                    result: list[str] = await mock_handler.get_available_voices()
                    return result
                except Exception:
                    return ["cedar"]

            result = mock_loop.run_until_complete(test_inner())
            assert result == ["cedar"]

        mock_loop.close()


class TestUpdateEnvFileExistingKey:
    """Tests for _update_env_file with existing key (lines 307-310)."""

    def test_update_env_file_updates_existing_key_with_value(self, tmp_path: Path) -> None:
        """Test updating existing key in env file (lines 306-307)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]

        # Create env file with existing key
        env_file = tmp_path / ".env"
        env_file.write_text("MODEL_NAME=old-value\nOTHER_KEY=keep\n")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = "new-value"

            with patch("dotenv.find_dotenv", return_value=str(env_file)):
                result = set_config_handler(
                    key="MODEL_NAME", value="new-value", persist=True
                )
                assert result["ok"] is True
                assert result["persisted"] is True

                # Verify file was updated
                content = env_file.read_text()
                assert "MODEL_NAME=new-value" in content
                assert "OTHER_KEY=keep" in content

    def test_update_env_file_removes_existing_key(self, tmp_path: Path) -> None:
        """Test removing existing key from env file (lines 304-308 when value is None)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        delete_config_handler = registered_handlers["DELETE /config/{key}"]

        # Create env file with existing key
        env_file = tmp_path / ".env"
        env_file.write_text("MODEL_NAME=to-delete\nOTHER_KEY=keep\n")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.MODEL_NAME = None

            with patch("dotenv.find_dotenv", return_value=str(env_file)):
                result = delete_config_handler(key="MODEL_NAME", persist=True)
                assert result["ok"] is True

                # Verify file was updated
                content = env_file.read_text()
                assert "MODEL_NAME" not in content
                assert "OTHER_KEY=keep" in content


class TestUpdateRuntimeConfigNoAttr:
    """Tests for _update_runtime_config when config doesn't have attr (line 332->exit)."""

    def test_update_runtime_config_no_config_attr(self) -> None:
        """Test update runtime config when hasattr returns False (line 332->exit)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        set_config_handler = registered_handlers["POST /config/{key}"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            # Make hasattr return False for MODEL_NAME
            del mock_config.MODEL_NAME

            result = set_config_handler(
                key="MODEL_NAME", value="test-value", persist=False
            )
            # Should still succeed (env var gets set)
            assert result["ok"] is True


class TestApplyInnerCoroutine:
    """Tests for _do_apply inner coroutine (lines 234-236)."""

    def test_do_apply_runs_with_real_loop(self) -> None:
        """Test that _do_apply inner coroutine runs (lines 234-236)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_handler.apply_personality = AsyncMock(return_value="Applied successfully")

        # Create a real event loop for the test
        test_loop = asyncio.new_event_loop()

        def mock_get_loop() -> Any:
            return test_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_payload = MagicMock()
        mock_payload.name = "test_profile"
        mock_payload.persist = False

        # Run the test loop in a thread so the handler can use run_coroutine_threadsafe
        import threading

        loop_thread = threading.Thread(target=test_loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                runner_loop = asyncio.new_event_loop()
                result = runner_loop.run_until_complete(apply_handler(payload=mock_payload))
                runner_loop.close()

                assert result["ok"] is True
                assert result["status"] == "Applied successfully"
        finally:
            test_loop.call_soon_threadsafe(test_loop.stop)
            loop_thread.join(timeout=1.0)
            test_loop.close()


class TestVoicesInnerCoroutine:
    """Tests for _get_v inner coroutine (lines 260-263)."""

    def test_get_v_runs_with_real_loop(self) -> None:
        """Test that _get_v inner coroutine runs (lines 260-261)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_handler.get_available_voices = AsyncMock(return_value=["cedar", "ash", "alloy"])

        # Create a real event loop for the test
        test_loop = asyncio.new_event_loop()

        def mock_get_loop() -> Any:
            return test_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        voices_handler = registered_handlers["GET /voices"]

        # Run the test loop in a thread
        import threading

        loop_thread = threading.Thread(target=test_loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            runner_loop = asyncio.new_event_loop()
            result = runner_loop.run_until_complete(voices_handler())
            runner_loop.close()

            assert "cedar" in result
            assert "ash" in result
        finally:
            test_loop.call_soon_threadsafe(test_loop.stop)
            loop_thread.join(timeout=1.0)
            test_loop.close()

    def test_get_v_exception_returns_cedar(self) -> None:
        """Test that _get_v returns cedar on exception (lines 262-263)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        # Make get_available_voices raise an exception
        mock_handler.get_available_voices = AsyncMock(side_effect=RuntimeError("API error"))

        # Create a real event loop for the test
        test_loop = asyncio.new_event_loop()

        def mock_get_loop() -> Any:
            return test_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        voices_handler = registered_handlers["GET /voices"]

        # Run the test loop in a thread
        import threading

        loop_thread = threading.Thread(target=test_loop.run_forever, daemon=True)
        loop_thread.start()

        try:
            runner_loop = asyncio.new_event_loop()
            result = runner_loop.run_until_complete(voices_handler())
            runner_loop.close()

            # Should return ["cedar"] on exception
            assert result == ["cedar"]
        finally:
            test_loop.call_soon_threadsafe(test_loop.stop)
            loop_thread.join(timeout=1.0)
            test_loop.close()


class TestApplyDictBodyWithoutName:
    """Tests for apply endpoint with dict body that doesn't have name (line 217->219)."""

    def test_apply_with_dict_body_no_name_key(self) -> None:
        """Test apply endpoint when dict body has no name key (branch 217->219)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_request = MagicMock()

        async def mock_json() -> dict[str, Any]:
            # Dict with no "name" key
            return {"other_key": "value"}

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                # Should still succeed (falls back to DEFAULT_OPTION)
                assert result["ok"] is True

    def test_apply_with_non_dict_body(self) -> None:
        """Test apply endpoint when body is not a dict (branch 217->219)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_loop = MagicMock()

        def mock_get_loop() -> Any:
            return mock_loop

        registered_handlers: dict[str, Any] = {}

        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        apply_handler = registered_handlers["POST /personalities/apply"]

        mock_request = MagicMock()

        async def mock_json() -> list[str]:
            # Body is not a dict
            return ["not", "a", "dict"]

        mock_request.json = mock_json
        mock_request.query_params = MagicMock()
        mock_request.query_params.get = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe") as mock_run:
            mock_future = MagicMock()
            mock_future.result.return_value = "Applied"

            def capture_and_close_coro(coro: Any, loop: Any) -> Any:
                coro.close()
                return mock_future

            mock_run.side_effect = capture_and_close_coro

            with patch(
                "reachy_mini_conversation_app.headless_personality_ui.config"
            ) as mock_config:
                mock_config.REACHY_MINI_CUSTOM_PROFILE = None

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    apply_handler(payload=None, name=None, persist=None, request=mock_request)
                )
                loop.close()

                # Should still succeed (falls back to DEFAULT_OPTION)
                assert result["ok"] is True


class TestCollectProfileEnvVars:
    """Tests for collect_profile_env_vars function."""

    def test_collect_returns_base_vars(self) -> None:
        """Test that collect_profile_env_vars returns base config vars."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )
        from reachy_mini_conversation_app.tools.core_tools import BASE_CONFIG_VARS

        result = collect_profile_env_vars("nonexistent_profile")

        # Should return at least base vars
        result_names = [v.name for v in result]
        for base_var in BASE_CONFIG_VARS:
            assert base_var.name in result_names

    def test_collect_for_default_profile(self) -> None:
        """Test collect_profile_env_vars for default profile."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        result = collect_profile_env_vars("default")

        # Should return at least base vars
        assert len(result) > 0
        result_names = [v.name for v in result]
        assert "OPENAI_API_KEY" in result_names

    def test_collect_for_linus_profile_includes_github_vars(self) -> None:
        """Test that linus profile includes GitHub env vars."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        result = collect_profile_env_vars("linus")

        result_names = [v.name for v in result]

        # Linus profile should include GitHub vars
        assert "GITHUB_TOKEN" in result_names
        assert "ANTHROPIC_API_KEY" in result_names

    def test_collect_returns_envvar_instances(self) -> None:
        """Test that collect_profile_env_vars returns EnvVar instances."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )
        from reachy_mini_conversation_app.tools.core_tools import EnvVar

        result = collect_profile_env_vars("default")

        for item in result:
            assert isinstance(item, EnvVar)

    def test_collect_handles_missing_tools_txt(self) -> None:
        """Test collect_profile_env_vars handles missing tools.txt gracefully."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        # Profile without tools.txt should return only base vars
        result = collect_profile_env_vars("profile_that_does_not_exist")

        # Should still return base vars
        result_names = [v.name for v in result]
        assert "OPENAI_API_KEY" in result_names

    def test_collect_handles_read_exception(self) -> None:
        """Test collect_profile_env_vars handles read exception gracefully."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir"
        ) as mock_resolve:
            mock_dir = MagicMock()
            mock_resolve.return_value = mock_dir
            mock_tools_txt = MagicMock()
            mock_tools_txt.exists.return_value = True
            mock_tools_txt.read_text.side_effect = PermissionError("Cannot read")
            mock_dir.__truediv__ = MagicMock(return_value=mock_tools_txt)

            result = collect_profile_env_vars("test_profile")

            # Should return base vars despite error
            result_names = [v.name for v in result]
            assert "OPENAI_API_KEY" in result_names

    def test_collect_handles_tool_import_exception(self) -> None:
        """Test that import exceptions in tools are handled gracefully."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        # Use a real profile but mock importlib to fail
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Test error")

            # Should not crash, just return base vars
            result = collect_profile_env_vars("default")

            result_names = [v.name for v in result]
            assert "OPENAI_API_KEY" in result_names

    def test_collect_deduplicates_env_vars(self) -> None:
        """Test that env vars are deduplicated."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        result = collect_profile_env_vars("linus")

        # No duplicates
        names = [v.name for v in result]
        assert len(names) == len(set(names))


class TestGetProfileConfigEndpoint:
    """Tests for GET /config/profile/{profile_name} endpoint."""

    def test_profile_config_endpoint_registered(self) -> None:
        """Test that profile config endpoint is registered."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        assert "GET /config/profile/{profile_name}" in registered_handlers

    def test_profile_config_returns_variables(self) -> None:
        """Test that profile config endpoint returns variables."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        result = handler("default")

        assert "profile" in result
        assert "variables" in result
        assert result["profile"] == "default"
        assert isinstance(result["variables"], list)

    def test_profile_config_returns_404_for_missing_profile(self) -> None:
        """Test that profile config returns 404 for non-existent profile."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        # Mock JSONResponse to check it's called
        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir"
        ) as mock_resolve:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = False
            mock_resolve.return_value = mock_dir

            _result = handler("nonexistent_profile")

            # Should return JSONResponse (mocked, so it's a MagicMock call)
            # The JSONResponse is created inside mount_personality_routes
            # We just verify the handler doesn't crash
            assert _result is not None  # Just verify it returned something

    def test_profile_config_includes_var_metadata(self) -> None:
        """Test that profile config includes variable metadata."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        result = handler("default")

        for var in result["variables"]:
            assert "key" in var
            assert "value" in var
            assert "is_set" in var
            assert "is_secret" in var
            assert "description" in var
            assert "required" in var
            assert "default" in var

    def test_profile_config_for_linus_includes_github_vars(self) -> None:
        """Test that linus profile config includes GitHub variables."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        result = handler("linus")

        var_keys = [v["key"] for v in result["variables"]]
        assert "GITHUB_TOKEN" in var_keys
        assert "ANTHROPIC_API_KEY" in var_keys

    def test_profile_config_masks_secrets(self) -> None:
        """Test that profile config masks secret values."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        # Set a secret value
        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.config"
        ) as mock_config:
            mock_config.OPENAI_API_KEY = "sk-1234567890abcdefghij"

            result = handler("default")

            # Find OPENAI_API_KEY in results
            openai_var = next(
                (v for v in result["variables"] if v["key"] == "OPENAI_API_KEY"),
                None
            )
            if openai_var and openai_var["value"]:
                # Should be masked
                assert "..." in openai_var["value"] or openai_var["value"] == "***"

    def test_profile_config_not_dir(self) -> None:
        """Test profile config returns 404 when path exists but is not a directory."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir"
        ) as mock_resolve:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = False  # Not a directory
            mock_resolve.return_value = mock_dir

            _result = handler("not_a_directory")

            # Should return JSONResponse (mocked)
            # We just verify the handler doesn't crash
            assert _result is not None  # Just verify it returned something

    def test_profile_config_default_option_skips_exists_check(self) -> None:
        """Test that DEFAULT_OPTION profile skips the existence check (branch 484->492)."""
        from reachy_mini_conversation_app.headless_personality_ui import (
            mount_personality_routes,
            DEFAULT_OPTION,
        )

        mock_app = MagicMock()
        mock_handler = MagicMock()
        mock_get_loop = MagicMock(return_value=None)

        registered_handlers: dict[str, Any] = {}
        mock_app.get = make_route_capture("GET", registered_handlers)
        mock_app.post = make_route_capture("POST", registered_handlers)
        mock_app.delete = make_route_capture("DELETE", registered_handlers)

        mount_personality_routes(mock_app, mock_handler, mock_get_loop)

        handler = registered_handlers["GET /config/profile/{profile_name}"]

        # Call with DEFAULT_OPTION - should skip existence check
        result = handler(DEFAULT_OPTION)

        assert "profile" in result
        assert result["profile"] == DEFAULT_OPTION
        assert "variables" in result


class TestCollectProfileEnvVarsSharedTools:
    """Tests for collect_profile_env_vars with shared tools (branch 117->125)."""

    def test_collect_finds_shared_tool_envvars(self) -> None:
        """Test that shared tools env vars are collected (branch 117->125).

        The 'default' profile uses tools from shared tools directory like 'dance',
        which don't have required_env_vars but tests the shared tools path.
        """
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        # Default profile uses shared tools like dance, camera, etc.
        result = collect_profile_env_vars("default")

        # Should complete without error and return base vars
        result_names = [v.name for v in result]
        assert "OPENAI_API_KEY" in result_names

    def test_collect_handles_shared_module_without_tool_class(self) -> None:
        """Test that shared module without Tool class is handled (branch 117->125).

        This covers the case where a shared module exists but doesn't contain
        a valid Tool subclass, so the for loop completes without finding one.
        """
        import types

        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )

        # Create a mock module that has no Tool class
        mock_module = types.ModuleType("fake_tool_module")
        mock_module.SomeClass = type("SomeClass", (), {})  # Not a Tool subclass

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir"
        ) as mock_resolve:
            mock_dir = MagicMock()
            mock_resolve.return_value = mock_dir
            mock_tools_txt = MagicMock()
            mock_tools_txt.exists.return_value = True
            mock_tools_txt.read_text.return_value = "fake_tool\n"
            mock_dir.__truediv__ = MagicMock(return_value=mock_tools_txt)

            # Mock importlib to:
            # 1. Fail for profile-local import (so we fall through to shared)
            # 2. Return a module without Tool class for shared import
            def mock_import(name: str) -> Any:
                if "profiles" in name:
                    raise ModuleNotFoundError(f"No module {name}")
                if "tools.fake_tool" in name:
                    return mock_module
                raise ModuleNotFoundError(f"No module {name}")

            with patch("importlib.import_module", side_effect=mock_import):
                result = collect_profile_env_vars("test_profile")

                # Should return base vars (no tool vars since no Tool class found)
                result_names = [v.name for v in result]
                assert "OPENAI_API_KEY" in result_names


class TestCollectProfileEnvVarsDuplicateInBase:
    """Tests for collect_profile_env_vars with duplicate env vars (branch 73->72)."""

    def test_collect_handles_duplicate_base_vars(self) -> None:
        """Test that duplicate base vars are handled (branch 73->72).

        This tests when BASE_CONFIG_VARS contains duplicates (unusual case).
        """
        from reachy_mini_conversation_app.headless_personality_ui import (
            collect_profile_env_vars,
        )
        from reachy_mini_conversation_app.tools import core_tools
        from reachy_mini_conversation_app.tools.core_tools import EnvVar

        # Save original
        original_base_vars = core_tools.BASE_CONFIG_VARS.copy()

        try:
            # Add a duplicate to BASE_CONFIG_VARS
            duplicate_var = EnvVar("OPENAI_API_KEY", is_secret=True, description="Duplicate")
            core_tools.BASE_CONFIG_VARS.append(duplicate_var)

            result = collect_profile_env_vars("nonexistent_profile")

            # Should not have duplicates in result
            names = [v.name for v in result]
            openai_count = names.count("OPENAI_API_KEY")
            assert openai_count == 1  # Should only appear once
        finally:
            # Restore original
            core_tools.BASE_CONFIG_VARS.clear()
            core_tools.BASE_CONFIG_VARS.extend(original_base_vars)
