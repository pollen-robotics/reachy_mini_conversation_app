import sys
import importlib
from types import ModuleType
from pathlib import Path

import pytest

import robot_comic.config as config_mod


def _reload_core_tools() -> ModuleType:
    """Reload core_tools after config object has been patched."""
    for module_name in list(sys.modules):
        if module_name.startswith("robot_comic.tools."):
            sys.modules.pop(module_name, None)
    # External file-loaded modules are registered by bare tool name.
    sys.modules.pop("ext_ping", None)
    sys.modules.pop("sweep_look", None)

    sys.modules.pop("robot_comic.tools.core_tools", None)
    core_tools_mod = importlib.import_module("robot_comic.tools.core_tools")
    return core_tools_mod


def test_external_profile_can_use_builtin_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """External profile tools.txt can reference built-in src tools."""
    profile_name = "ext_profile_test"
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / profile_name
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text("dance\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", profile_name)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.get_tool_specs()  # trigger lazy init

    assert "dance" in core_tools_mod.ALL_TOOLS
    assert "dance" not in sys.modules


def test_external_tools_can_be_loaded_without_external_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """External tools can be loaded with built-in profile via autoload mode."""
    external_tools_root = tmp_path / "external_tools"
    external_tools_root.mkdir(parents=True)

    (external_tools_root / "ext_ping.py").write_text(
        "\n".join(
            [
                "from typing import Any, Dict",
                "from robot_comic.tools.core_tools import Tool, ToolDependencies",
                "",
                "class ExtPingTool(Tool):",
                '    name = "ext_ping"',
                '    description = "External ping tool"',
                '    parameters_schema = {"type": "object", "properties": {}, "required": []}',
                "",
                "    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:",
                '        return {"status": "ok"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", external_tools_root)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", True)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.get_tool_specs()  # trigger lazy init

    assert "ext_ping" in core_tools_mod.ALL_TOOLS


def test_builtin_profile_can_load_profile_local_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-local .py tools should be discovered and loaded alongside built-in tools."""
    profile_name = "local_tool_profile"
    profiles_root = tmp_path / "profiles"
    profile_dir = profiles_root / profile_name
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("test\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text("dance\nlocal_ping\n", encoding="utf-8")
    (profile_dir / "local_ping.py").write_text(
        "\n".join(
            [
                "from typing import Any, Dict",
                "from robot_comic.tools.core_tools import Tool, ToolDependencies",
                "",
                "class LocalPingTool(Tool):",
                '    name = "local_ping"',
                '    description = "Profile-local ping tool"',
                '    parameters_schema = {"type": "object", "properties": {}, "required": []}',
                "",
                "    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:",
                '        return {"status": "ok"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", profile_name)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    # Clear any cached sweep_look module from previous test runs
    sys.modules.pop("local_ping", None)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.get_tool_specs()  # trigger lazy init

    assert "local_ping" in core_tools_mod.ALL_TOOLS
    assert "dance" in core_tools_mod.ALL_TOOLS


def test_refresh_picks_up_external_profiles_directory_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """refresh_runtime_config_from_env must update PROFILES_DIRECTORY.

    Class-body reads of REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY fire once at
    module import — before main.py's load_dotenv() runs. Without an explicit
    refresh, values in the instance .env are silently ignored at runtime,
    even though refresh_runtime_config_from_env() updates ~70 other attrs.
    """
    external_root = tmp_path / "external_profiles"
    external_root.mkdir()
    external_tools = tmp_path / "external_tools"
    external_tools.mkdir()

    original_profiles = config_mod.config.PROFILES_DIRECTORY
    original_tools = config_mod.config.TOOLS_DIRECTORY

    monkeypatch.setenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", str(external_root))
    monkeypatch.setenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", str(external_tools))
    monkeypatch.setenv("AUTOLOAD_EXTERNAL_TOOLS", "1")

    try:
        config_mod.refresh_runtime_config_from_env()
        assert config_mod.config.PROFILES_DIRECTORY == external_root
        assert config_mod.config.TOOLS_DIRECTORY == external_tools
        assert config_mod.config.AUTOLOAD_EXTERNAL_TOOLS is True
    finally:
        # Restore — refresh() will reset using the monkeypatched env which
        # gets unwound after the test, but be explicit so failures don't
        # leak into the rest of the session.
        config_mod.config.PROFILES_DIRECTORY = original_profiles
        config_mod.config.TOOLS_DIRECTORY = original_tools


def test_refresh_falls_back_to_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsetting the env var must restore PROFILES_DIRECTORY to the default."""
    original_profiles = config_mod.config.PROFILES_DIRECTORY
    original_tools = config_mod.config.TOOLS_DIRECTORY

    monkeypatch.delenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", raising=False)
    monkeypatch.delenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", raising=False)

    try:
        config_mod.refresh_runtime_config_from_env()
        assert config_mod.config.PROFILES_DIRECTORY == config_mod.DEFAULT_PROFILES_DIRECTORY
        assert config_mod.config.TOOLS_DIRECTORY is None
    finally:
        config_mod.config.PROFILES_DIRECTORY = original_profiles
        config_mod.config.TOOLS_DIRECTORY = original_tools
