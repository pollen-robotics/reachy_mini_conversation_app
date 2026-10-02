"""Encoding of a personality selection into the text written on an NFC tag.

The tag carries the personality itself, so it means the same thing on any robot:

    built-in ``noir_detective``            -> ``hf_noir_detective``
    user ``user_personalities/my_bot``     -> ``usr_my_bot``

Both sides are prefixed, not only the built-ins: the name sanitizer lets a user
call their personality ``hf_noir_detective``, which would otherwise shadow the
built-in ``noir_detective``.
"""

from __future__ import annotations
from typing import Optional

from reachy_mini_conversation_app.config import USER_PERSONALITIES_DIRNAME


BUILTIN_PREFIX = "hf_"
USER_PREFIX = "usr_"


def to_tag_token(selection: str) -> Optional[str]:
    """Return the text to write on a tag for a personality selection, or None for an empty name."""
    s = (selection or "").strip()
    if not s:
        return None
    if s.startswith(USER_PERSONALITIES_DIRNAME + "/"):
        name = s[len(USER_PERSONALITIES_DIRNAME) + 1 :].strip("/")
        return f"{USER_PREFIX}{name}" if name else None
    return f"{BUILTIN_PREFIX}{s}"


def from_tag_token(token: str) -> Optional[str]:
    """Return the personality selection a tag's text refers to, or None if it is not a token."""
    t = (token or "").strip()
    if t.startswith(USER_PREFIX):
        name = t[len(USER_PREFIX) :]
        return f"{USER_PERSONALITIES_DIRNAME}/{name}" if name else None
    if t.startswith(BUILTIN_PREFIX):
        name = t[len(BUILTIN_PREFIX) :]
        return name or None
    return None
