import sys
import importlib
from types import ModuleType
from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod


def _reload_core_tools() -> ModuleType:
    """Reload core_tools after config object has been patched."""
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)
    # External file-loaded modules are registered by bare tool name.
    sys.modules.pop("ext_ping", None)

    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    core_tools_mod = importlib.import_module("reachy_mini_conversation_app.tools.core_tools")
    return core_tools_mod


def test_external_profile_can_use_builtin_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    assert "dance" in core_tools_mod.ALL_TOOLS


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
                "from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies",
                "",
                "class ExtPingTool(Tool):",
                "    name = \"ext_ping\"",
                "    description = \"External ping tool\"",
                "    parameters_schema = {\"type\": \"object\", \"properties\": {}, \"required\": []}",
                "",
                "    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:",
                "        return {\"status\": \"ok\"}",
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

    assert "ext_ping" in core_tools_mod.ALL_TOOLS
