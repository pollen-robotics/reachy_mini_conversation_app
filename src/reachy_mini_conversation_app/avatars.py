"""Avatar (SVG) resolution for personality profiles.

Mirrors the front-end ``AVATAR_BY_PROFILE`` map (``static/js/constants.js``);
kept in sync by hand. Resolution order: a profile-local ``avatar.svg``, then
the built-in map, then the default.
"""

import logging
from pathlib import Path

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME, canonical_profile_name


DEFAULT_AVATAR_FILE = "default.svg"
logger = logging.getLogger(__name__)

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


def _own_avatar_path(name: str) -> Path | None:
    """Return a profile-local ``avatar.svg`` path when it exists, else None."""
    profile_name = canonical_profile_name(name)
    if profile_name == DEFAULT_PROFILE_NAME:
        return None
    try:
        candidate = config.resolve_profile_dir(profile_name) / "avatar.svg"
        return candidate if candidate.is_file() else None
    except OSError as exc:
        logger.warning("Failed to inspect avatar for profile %r: %s", profile_name, exc)
        return None


def avatar_id_for(name: str) -> str:
    """Return a stable id for a selection's avatar, for client-side caching.

    Two selections that resolve to the same file share an id, so a client can
    fetch each distinct avatar once. User personas that carry their own
    ``avatar.svg`` get their (unique) selection as id.
    """
    if _own_avatar_path(name) is not None:
        return name
    key = canonical_profile_name(name)
    file = AVATAR_BY_PROFILE.get(key, DEFAULT_AVATAR_FILE)
    return file[:-4] if file.endswith(".svg") else file


def read_avatar_svg(name: str) -> str | None:
    """Return the SVG markup for a selection, falling back to the default.

    Returns None only if even the default avatar is missing from disk.
    """
    own = _own_avatar_path(name)
    if own is not None:
        try:
            return own.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            logger.warning("Failed to read avatar %s: %s", own, exc)

    key = canonical_profile_name(name)
    file = AVATAR_BY_PROFILE.get(key, DEFAULT_AVATAR_FILE)
    for candidate in (_avatars_dir() / file, _avatars_dir() / DEFAULT_AVATAR_FILE):
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                logger.warning("Failed to read avatar %s: %s", candidate, exc)
                continue
    return None
