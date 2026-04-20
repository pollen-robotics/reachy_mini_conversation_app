import shutil
import zipfile
import subprocess
from pathlib import Path, PurePosixPath

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.headless_personality import (
    resolve_profile_dir,
    read_instructions_for,
)


# Path characters budget computation
# ─────────────────
# Windows MAX_PATH limit: 259 usable characters (failures start at 260)
#
# Project files (WINDOWS_PATH_BUDGET = 130):
#   C:\Users\<username(20)>
#     \.cache\huggingface\hub
#     \spaces--pollen-robotics--reachy_mini_conversation_app
#     \snapshots\<commit_hash(40)>\
#   = 158 characters  =>  101 remaining to 259.
#   The project root folder is not cloned in the snapshot, so we add it
#   back to the budget: 101 + len("reachy_mini_conversation_app\") (29) = 130.
#
# Wheel files (WINDOWS_WHEEL_PATH_BUDGET = 71):
#   C:\Users\<username(20)>
#     \.cache\huggingface\hub
#     \spaces--pollen-robotics--reachy_mini_conversation_app
#     \snapshots\<commit_hash(40)>
#     \build\bdist.win-amd64\wheel\
#   = 186 characters  =>  73 remaining to 259.
#   In practice the copy fails at 257 because of an intermediate \.\
#   folder, bringing the real budget down to 71.

WINDOWS_PATH_BUDGET = 130
WINDOWS_WHEEL_PATH_BUDGET = 71


def _git_tracked_files(project_root: Path) -> list[Path]:
    """Return git-tracked files that still exist in the working tree."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git-tracked file listing unavailable: {exc}")

    tracked_files = [project_root / relative_path for relative_path in result.stdout.splitlines() if relative_path]
    return [path for path in tracked_files if path.is_file()]


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
        (DEFAULT_PROFILES_DIRECTORY / "mad_scientist_assistant" / "instructions.txt")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert prompts_mod.get_session_instructions() == expected
    assert read_instructions_for("mad_scientist_assistant") == expected


def test_session_voice_defaults_follow_selected_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session voice should fall back to the active backend default."""
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "gemini")
    monkeypatch.setattr(config, "MODEL_NAME", "gemini-3.1-flash-live-preview")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert prompts_mod.get_session_voice() == "Kore"


def test_packaged_profiles_win_outside_source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed builds should use packaged profiles, not an unrelated sibling folder."""
    unrelated_profiles = tmp_path / "profiles"
    unrelated_profiles.mkdir()
    packaged_profiles = tmp_path / "package_data" / "profiles"
    packaged_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "_packaged_profiles_directory", lambda: packaged_profiles)

    assert config_mod._resolve_default_profiles_directory() == packaged_profiles


def test_project_file_paths_stay_within_windows_budget() -> None:
    """Git-tracked project file paths should stay below the agreed Windows budget."""
    project_root = Path(__file__).parents[1].resolve()
    project_files = _git_tracked_files(project_root)

    violations = []
    for path in project_files:
        relative = str(Path(project_root.name) / path.relative_to(project_root))
        length = len(relative)
        if length > WINDOWS_PATH_BUDGET:
            violations.append(
                f"Windows path budget exceeded ({WINDOWS_PATH_BUDGET}): {relative} is {length} characters long"
            )

    assert not violations, "\n".join(violations)


def test_wheel_file_paths_stay_within_windows_budget(tmp_path: Path) -> None:
    """Built wheel paths should stay below the agreed Windows budget."""
    project_root = Path(__file__).parents[1].resolve()
    source_checkout = tmp_path / "checkout"
    dist_dir = tmp_path / "dist"

    for source_file in _git_tracked_files(project_root):
        target_file = source_checkout / source_file.relative_to(project_root)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            cwd=source_checkout,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        details = exc.stderr if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        pytest.fail(f"Wheel build failed while checking Windows path budget: {details}")

    wheel_files = list(dist_dir.glob("*.whl"))
    assert len(wheel_files) == 1, f"Expected exactly one built wheel in {dist_dir}, found: {wheel_files}"

    with zipfile.ZipFile(wheel_files[0]) as archive:
        archived_paths = [PurePosixPath(info.filename) for info in archive.infolist() if not info.is_dir()]

    violations = []
    for path in archived_paths:
        length = len(path.as_posix())
        if length > WINDOWS_WHEEL_PATH_BUDGET:
            violations.append(
                f"Windows wheel path budget exceeded ({WINDOWS_WHEEL_PATH_BUDGET}): "
                f"{path.as_posix()} is {length} characters long"
            )

    assert not violations, "\n".join(violations)
