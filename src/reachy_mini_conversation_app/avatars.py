"""Avatar (SVG) resolution for personality profiles.

Mirrors the front-end ``AVATAR_BY_PROFILE`` map (``static/js/constants.js``);
kept in sync by hand. Resolution order: a profile-local ``avatar.svg``, then
the built-in map, then the default.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

from .personality import DEFAULT_OPTION, resolve_profile_dir


DEFAULT_AVATAR_FILE = "default.svg"

# Built-in profile dir name -> avatar file under static/avatars/.
# Mirror of AVATAR_BY_PROFILE in static/js/constants.js.
AVATAR_BY_PROFILE: dict[str, str] = {
    "bored_teenager": "bored-teenager.svg",
    "captain_circuit": "captain-circuit.svg",
    "chess_coach": "chess-coach.svg",
    "cosmic_kitchen": "cosmic-kitchen.svg",
    "default": "default.svg",
    "hype_bot": "hype-bot.svg",
    "mad_scientist_assistant": "mad-scientist.svg",
    "mars_rover": "mars-rover.svg",
    "nature_documentarian": "nature-doc.svg",
    "noir_detective": "noir-detective.svg",
    "sorry_bro": "sorry-bro.svg",
    "time_traveler": "time-traveler.svg",
    "victorian_butler": "victorian-butler.svg",
}


def _avatars_dir() -> Path:
    return Path(__file__).parent / "static" / "avatars"


def _own_avatar_path(name: str) -> Optional[Path]:
    """Return a profile-local ``avatar.svg`` path when it exists, else None."""
    if name == DEFAULT_OPTION:
        return None
    try:
        candidate = resolve_profile_dir(name) / "avatar.svg"
        return candidate if candidate.is_file() else None
    except Exception:
        return None


def avatar_id_for(name: str) -> str:
    """Return a stable id for a selection's avatar, for client-side caching.

    Two selections that resolve to the same file share an id, so a client can
    fetch each distinct avatar once. User personas that carry their own
    ``avatar.svg`` get their (unique) selection as id.
    """
    if _own_avatar_path(name) is not None:
        return name
    key = "default" if name == DEFAULT_OPTION else name
    file = AVATAR_BY_PROFILE.get(key, DEFAULT_AVATAR_FILE)
    return file[:-4] if file.endswith(".svg") else file


def read_avatar_svg(name: str) -> Optional[str]:
    """Return the SVG markup for a selection, falling back to the default.

    Returns None only if even the default avatar is missing from disk.
    """
    own = _own_avatar_path(name)
    if own is not None:
        try:
            return own.read_text(encoding="utf-8")
        except Exception:
            pass

    key = "default" if name == DEFAULT_OPTION else name
    file = AVATAR_BY_PROFILE.get(key, DEFAULT_AVATAR_FILE)
    for candidate in (_avatars_dir() / file, _avatars_dir() / DEFAULT_AVATAR_FILE):
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception:
                continue
    return None
