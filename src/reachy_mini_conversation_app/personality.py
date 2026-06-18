"""Personality (profile) data layer.

Provides functions to list, read, and write personality profiles stored
on disk. No HTTP or framework dependencies — importable anywhere.
"""

from __future__ import annotations
from typing import List
from pathlib import Path

from .config import USER_PERSONALITIES_DIRNAME, config, get_default_voice_for_backend
from .tools.tool_constants import SystemTool


DEFAULT_OPTION = "(built-in default)"

# Dev-only profiles, hidden from the UI, but still loadable via REACHY_MINI_CUSTOM_PROFILE
UNLISTED_PROFILES = {"tedai"}


def _prompts_dir() -> Path:
    return Path(__file__).parent / "prompts"


def _tools_dir() -> Path:
    return Path(__file__).parent / "tools"


def _sanitize_name(name: str) -> str:
    import re

    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
    return s


def list_personalities() -> List[str]:
    """List available personality profile names."""
    names: List[str] = []
    try:
        builtin_root = config.PROFILES_DIRECTORY
        if builtin_root.exists():
            for p in sorted(builtin_root.iterdir()):
                if p.name == USER_PERSONALITIES_DIRNAME or p.name in UNLISTED_PROFILES:
                    continue
                if p.is_dir() and (p / "instructions.txt").exists():
                    names.append(p.name)
        udir = config.user_personalities_root()
        if udir.exists():
            for p in sorted(udir.iterdir()):
                if p.is_dir() and (p / "instructions.txt").exists():
                    names.append(f"{USER_PERSONALITIES_DIRNAME}/{p.name}")
    except Exception:
        pass
    return names


def resolve_profile_dir(selection: str) -> Path:
    """Resolve the directory path for the given profile selection."""
    return config.resolve_profile_dir(selection)


def read_instructions_for(name: str) -> str:
    """Read the instructions.txt content for the given profile name."""
    try:
        if name == DEFAULT_OPTION:
            df = _prompts_dir() / "default_prompt.txt"
            return df.read_text(encoding="utf-8").strip() if df.exists() else ""
        target = resolve_profile_dir(name) / "instructions.txt"
        return target.read_text(encoding="utf-8").strip() if target.exists() else ""
    except Exception as e:
        return f"Could not load instructions: {e}"


def read_tools_for(name: str) -> str:
    """Read the tools.txt content for the given profile name."""
    try:
        profile_name = "default" if name == DEFAULT_OPTION else name
        target = resolve_profile_dir(profile_name) / "tools.txt"
        return target.read_text(encoding="utf-8") if target.exists() else ""
    except Exception:
        return ""


def available_tools_for(selected: str) -> List[str]:
    """List available tool modules for the given profile selection."""
    shared: List[str] = []
    try:
        for py in _tools_dir().glob("*.py"):
            if py.stem in {"__init__", "core_tools", "background_tool_manager", "tool_constants"} or py.stem in {
                t.value for t in SystemTool
            }:
                continue
            shared.append(py.stem)
    except Exception:
        pass
    local: List[str] = []
    try:
        if selected != DEFAULT_OPTION:
            for py in resolve_profile_dir(selected).glob("*.py"):
                local.append(py.stem)
    except Exception:
        pass
    return sorted(set(shared + local))


def _write_profile(sanitized_name: str, instructions: str, tools_text: str, voice: str | None = None) -> None:
    default_voice = get_default_voice_for_backend()
    target_dir = config.user_personalities_root() / sanitized_name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
    (target_dir / "tools.txt").write_text((tools_text or "").strip() + "\n", encoding="utf-8")
    (target_dir / "voice.txt").write_text((voice or default_voice).strip() + "\n", encoding="utf-8")
