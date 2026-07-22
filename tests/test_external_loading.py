import sys
import json
import importlib
import threading
from types import ModuleType
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.profile_store import write_profile


def _reload_core_tools() -> ModuleType:
    """Reload core_tools after config object has been patched."""
    for module_name in list(sys.modules):
        if module_name.startswith(
            ("reachy_mini_conversation_app.tools.", "reachy_mini_conversation_app._external_tools.")
        ):
            sys.modules.pop(module_name, None)

    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    core_tools_mod = importlib.import_module("reachy_mini_conversation_app.tools.core_tools")
    core_tools_mod.initialize_tools()
    return core_tools_mod


def test_external_profile_can_use_builtin_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """External profile defaults can reference shared application tools."""
    profile_name = "ext_profile_test"
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / profile_name
    write_profile(profile_name, profile_dir, "hello", ["dance"])

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", profile_name)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()

    assert "dance" in core_tools_mod.ALL_TOOLS
    assert "dance" not in sys.modules


def test_packaged_default_tools_load_with_external_profiles_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No active external profile should retain the packaged default toolset."""
    external_profiles_root = tmp_path / "external_profiles"
    write_profile("guide", external_profiles_root / "guide", "Guide.", ["dance"])

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()

    assert not (external_profiles_root / "default").exists()
    assert "sweep_look" in core_tools_mod.ALL_TOOLS


def test_missing_profile_raises_runtime_error_instead_of_exiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library callers should receive a catchable initialization failure."""
    external_profiles_root = tmp_path / "external_profiles"
    external_profiles_root.mkdir()

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "missing")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    with pytest.raises(RuntimeError, match="Failed to read tools for profile 'missing'"):
        _reload_core_tools()


def test_external_tools_can_be_loaded_without_external_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """External tools can be loaded with built-in profile via autoload mode."""
    external_tools_root = tmp_path / "external_tools"
    external_tools_root.mkdir(parents=True)

    (external_tools_root / "json.py").write_text(
        "\n".join(
            [
                "from typing import Any, Dict",
                "from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies",
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
    monkeypatch.setitem(sys.modules, "json", json)

    core_tools_mod = _reload_core_tools()

    assert sys.modules["json"] is json
    assert "ext_ping" in core_tools_mod.ALL_TOOLS
    assert "reachy_mini_conversation_app._external_tools.json" in sys.modules


def test_external_tools_fail_on_duplicate_tool_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading must fail if multiple tools declare the same Tool.name."""
    external_tools_root = tmp_path / "external_tools"
    external_tools_root.mkdir(parents=True)

    duplicate_tool_source = "\n".join(
        [
            "from typing import Any, Dict",
            "from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies",
            "",
            "class DupTool(Tool):",
            '    name = "dup_tool"',
            '    description = "Duplicate tool name"',
            '    parameters_schema = {"type": "object", "properties": {}, "required": []}',
            "",
            "    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:",
            '        return {"status": "ok"}',
            "",
        ]
    )
    (external_tools_root / "ext_dup_a.py").write_text(duplicate_tool_source, encoding="utf-8")
    (external_tools_root / "ext_dup_b.py").write_text(duplicate_tool_source, encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", external_tools_root)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", True)

    with pytest.raises(RuntimeError, match="Duplicate Tool.name values detected"):
        _reload_core_tools()


def test_builtin_profile_can_load_shared_sweep_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A built-in profile can enable the shared sweep tool from its profile document."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()

    assert "sweep_look" in core_tools_mod.ALL_TOOLS


def test_tool_registry_reloads_when_profile_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime profile changes should refresh enabled tools without restarting Python."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()

    initial_tool_names = {spec["name"] for spec in core_tools_mod.get_tool_specs()}
    assert "sweep_look" in initial_tool_names
    assert "camera" in initial_tool_names

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mars_rover")

    reloaded_tool_names = {spec["name"] for spec in core_tools_mod.get_tool_specs()}
    assert "camera" in reloaded_tool_names
    assert "move_head" in reloaded_tool_names
    assert "sweep_look" not in reloaded_tool_names
    assert "camera" in core_tools_mod.ALL_TOOLS
    assert "sweep_look" not in core_tools_mod.ALL_TOOLS


def test_forced_tool_registry_reload_does_not_duplicate_shared_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading the registry should not duplicate an already imported shared tool."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools(force=True)

    assert "sweep_look" in core_tools_mod.ALL_TOOLS


def test_tool_registry_reads_wait_for_forced_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec readers should observe one complete registry generation during reload."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    core_tools_mod = _reload_core_tools()

    reload_entered = threading.Event()
    release_reload = threading.Event()
    read_started = threading.Event()
    read_tool_names = core_tools_mod._read_profile_tool_names

    def pause_profile_read(instance_path: str | Path | None) -> list[str]:
        if not reload_entered.is_set():
            reload_entered.set()
            assert release_reload.wait(timeout=1.0)
        return read_tool_names(instance_path)

    def read_specs() -> list[dict[str, object]]:
        read_started.set()
        return core_tools_mod.get_tool_specs()

    monkeypatch.setattr(core_tools_mod, "_read_profile_tool_names", pause_profile_read)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mars_rover")

    with ThreadPoolExecutor(max_workers=2) as executor:
        reload_future = executor.submit(core_tools_mod.initialize_tools, force=True)
        try:
            assert reload_entered.wait(timeout=1.0)
            read_future = executor.submit(read_specs)
            assert read_started.wait(timeout=1.0)
            with pytest.raises(TimeoutError):
                read_future.result(timeout=0.1)
        finally:
            release_reload.set()
        reload_future.result(timeout=1.0)
        spec_names = {spec["name"] for spec in read_future.result(timeout=1.0)}

    assert "move_head" in spec_names
    assert "sweep_look" not in spec_names
    assert spec_names == set(core_tools_mod.get_tools())
