"""Per-persona voice-clone reference audio loader for Chatterbox TTS.

Storage convention
------------------
Place the reference audio clip at::

    profiles/<persona>/voice_clone_ref.wav   (preferred)
    profiles/<persona>/voice_clone_ref.mp3   (fallback)
    profiles/<persona>/voice_clone_ref.flac  (fallback)
    profiles/<persona>/voice_clone_ref.ogg   (fallback)

The first format that exists wins.  These files are gitignored (copyrighted
archival audio) and must be supplied locally on each machine that runs the
Chatterbox pipeline.

Usage
-----
::

    from robot_comic.chatterbox_voice_clone import load_voice_clone_ref
    from pathlib import Path

    ref_path = load_voice_clone_ref(Path("profiles/house_comedian"))
    if ref_path:
        # pass ref_path to the Chatterbox /tts endpoint as audio_prompt_path
        ...
"""

from __future__ import annotations
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# Candidate extensions in priority order.
_EXTENSIONS = ("wav", "mp3", "flac", "ogg")


def load_voice_clone_ref(profile_dir: Path) -> Path | None:
    """Return the path to the voice-clone reference audio for *profile_dir*.

    Searches for ``voice_clone_ref.<ext>`` inside *profile_dir* in extension
    priority order: ``.wav`` → ``.mp3`` → ``.flac`` → ``.ogg``.

    Returns the resolved :class:`~pathlib.Path` of the first match, or
    ``None`` when no candidate file is present.

    Logs at INFO level in both cases so operators can verify the right clip
    is being loaded (or diagnose why cloning isn't active).

    Args:
        profile_dir: Directory for the active persona (e.g.
            ``Path("profiles/house_comedian")``).  Need not be absolute — the
            path is resolved before the size check.

    """
    persona = profile_dir.name

    for ext in _EXTENSIONS:
        candidate = profile_dir / f"voice_clone_ref.{ext}"
        if candidate.exists():
            resolved = candidate.resolve()
            size_kb = resolved.stat().st_size / 1024
            logger.info(
                "chatterbox: voice clone ref for %r = %s (%.1f KB)",
                persona,
                resolved.name,
                size_kb,
            )
            return resolved

    logger.info(
        "chatterbox: no voice_clone_ref for %r — using generic Chatterbox voice",
        persona,
    )
    return None
