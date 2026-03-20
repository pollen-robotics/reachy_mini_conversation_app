from pathlib import Path

import pytest

import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.headless_personality import (
    resolve_profile_dir,
    read_instructions_for,
)


def test_profile_name_resolves_directly_to_storage_dir() -> None:
    """Built-in profile names should map directly to their on-disk directory."""
    profile_dir = resolve_profile_dir("mad_scientist_assistant")

    assert profile_dir.name == "mad_scientist_assistant"
    assert (profile_dir / "instructions.txt").is_file()


def test_prompts_load_from_compact_builtin_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt loading should read compact built-in profile instructions directly."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "mad_scientist_assistant")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)

    expected = (
        DEFAULT_PROFILES_DIRECTORY / "mad_scientist_assistant" / "instructions.txt"
    ).read_text(encoding="utf-8").strip()

    assert prompts_mod.get_session_instructions() == expected
    assert read_instructions_for("mad_scientist_assistant") == expected


def test_project_file_paths_stay_within_windows_budget() -> None:
    """Project file paths should stay below the agreed in-repo budget."""
    project_root = Path(__file__).resolve().parents[1]
    ignored_parts = {".git", ".venv", "__pycache__", "build", "dist"}

    project_files = [
        path
        for path in project_root.rglob("*")
        if path.is_file() and not any(part in ignored_parts for part in path.relative_to(project_root).parts)
    ]

    longest_path = max(project_files, key=lambda path: len(str(path.relative_to(project_root))))
    longest_length = len(str(longest_path.relative_to(project_root)))

    assert longest_length <= 140, (
        "Project path budget exceeded: "
        f"{longest_path.relative_to(project_root)} is {longest_length} characters long"
    )
