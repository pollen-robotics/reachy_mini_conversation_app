"""Helpers for resolving profile paths with built-in compatibility aliases."""

from __future__ import annotations
from typing import Iterable
from pathlib import Path


BUILTIN_PROFILE_DIRECTORY_ALIASES: dict[str, str] = {
    "short_bored_teenager": "s_bored_teen",
    "short_captain_circuit": "s_capt_circuit",
    "short_chess_coach": "s_chess_coach",
    "short_hype_bot": "s_hype_bot",
    "short_mad_scientist_assistant": "s_mad_sci_asst",
    "short_nature_documentarian": "s_nat_doc",
    "short_noir_detective": "s_noir_det",
    "short_time_traveler": "s_time_travel",
    "short_victorian_butler": "s_vict_butler",
}
BUILTIN_PROFILE_PUBLIC_NAMES = {value: key for key, value in BUILTIN_PROFILE_DIRECTORY_ALIASES.items()}

INSTRUCTIONS_FILENAMES: tuple[str, ...] = ("inst.txt", "instructions.txt")
TOOLS_FILENAMES: tuple[str, ...] = ("tools.txt",)
VOICE_FILENAME = "voice.txt"


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def to_storage_profile_name(profile_name: str, *, profiles_root: Path, builtin_root: Path) -> str:
    """Map a public profile name to its on-disk directory name."""
    if _same_path(profiles_root, builtin_root):
        return BUILTIN_PROFILE_DIRECTORY_ALIASES.get(profile_name, profile_name)
    return profile_name


def to_public_profile_name(profile_name: str, *, profiles_root: Path, builtin_root: Path) -> str:
    """Map an on-disk built-in profile directory back to its public name."""
    if _same_path(profiles_root, builtin_root):
        return BUILTIN_PROFILE_PUBLIC_NAMES.get(profile_name, profile_name)
    return profile_name


def collect_profile_names(profiles_root: Path, *, builtin_root: Path) -> set[str]:
    """Return public profile names for all profile directories in a root."""
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {
        to_public_profile_name(path.name, profiles_root=profiles_root, builtin_root=builtin_root)
        for path in profiles_root.iterdir()
        if path.is_dir()
    }


def resolve_profile_dir(profile_name: str, *, profiles_root: Path, builtin_root: Path) -> Path:
    """Return the directory for a profile name, applying built-in aliases."""
    return profiles_root / to_storage_profile_name(profile_name, profiles_root=profiles_root, builtin_root=builtin_root)


def find_profile_file(profile_dir: Path, filenames: Iterable[str]) -> Path | None:
    """Return the first existing file from a list of candidate names."""
    for filename in filenames:
        candidate = profile_dir / filename
        if candidate.exists():
            return candidate
    return None


def resolve_profile_file(
    profile_name: str,
    *,
    profiles_root: Path,
    builtin_root: Path,
    filenames: Iterable[str],
) -> Path:
    """Resolve a profile file, preferring existing compact filenames."""
    profile_dir = resolve_profile_dir(profile_name, profiles_root=profiles_root, builtin_root=builtin_root)
    existing = find_profile_file(profile_dir, filenames)
    if existing is not None:
        return existing
    first_name = next(iter(filenames))
    return profile_dir / first_name
