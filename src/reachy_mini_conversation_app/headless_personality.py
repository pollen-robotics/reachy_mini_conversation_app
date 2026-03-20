"""Headless personality management (console-based).

Provides an interactive CLI to browse, preview, apply, create and edit
"personalities" (profiles) when running without Gradio.

This module is intentionally not shared with the Gradio implementation to
avoid coupling and keep responsibilities clear for headless mode.
"""

from __future__ import annotations
from typing import List
from pathlib import Path


DEFAULT_OPTION = "(built-in default)"


def _profiles_root() -> Path:
    return Path(__file__).parent / "profiles"


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
    root = _profiles_root()
    try:
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.name == "user_personalities":
                    continue
                if p.is_dir() and (p / "instructions.txt").exists():
                    names.append(p.name)
        udir = root / "user_personalities"
        if udir.exists():
            for p in sorted(udir.iterdir()):
                if p.is_dir() and (p / "instructions.txt").exists():
                    names.append(f"user_personalities/{p.name}")
    except Exception:
        pass
    return names


def resolve_profile_dir(selection: str) -> Path:
    """Resolve the directory path for the given profile selection."""
    return _profiles_root() / selection


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


def available_tools_for(selected: str) -> List[str]:
    """List available tool modules for the given profile selection."""
    shared: List[str] = []
    try:
        for py in _tools_dir().glob("*.py"):
            if py.stem in {"__init__", "core_tools"}:
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


def _write_profile(name_s: str, instructions: str, tools_text: str, voice: str = "cedar") -> None:
    target_dir = _profiles_root() / "user_personalities" / name_s
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
    (target_dir / "tools.txt").write_text((tools_text or "").strip() + "\n", encoding="utf-8")
    (target_dir / "voice.txt").write_text((voice or "cedar").strip() + "\n", encoding="utf-8")
