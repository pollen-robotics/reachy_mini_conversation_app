"""Optional kiosk-mode startup voice prompt (issue #41).

When ``REACHY_MINI_STARTUP_SCREEN=true``, :func:`run_startup_screen` plays a
two-part announcement before the app loads any persona:

1. An immediate welcome line via the generic Chatterbox baseline voice.
2. After a configurable wait (default 5 s), a persona listing derived from the
   profiles directory.

The wait is cancellable: if an external coroutine sets the supplied
``selection_event`` before the timeout fires, the listing is skipped and the
function returns immediately.

When the feature is disabled (the default), the function returns without doing
anything so existing deployments are not affected.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any
from pathlib import Path


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

_WELCOME_PROMPT = "Welcome to Robot Comic. Choose your comedian."
_LISTING_TEMPLATE = "You can hear the comedy of {names}."
_SELECTION_WAIT_S: float = 5.0

# Generic Chatterbox baseline — no voice clone reference needed.
_BASELINE_EXAGGERATION: float = 1.0
_BASELINE_CFG_WEIGHT: float = 0.5


# --------------------------------------------------------------------------- #
# Persona list builder                                                         #
# --------------------------------------------------------------------------- #


def build_persona_list(
    profiles_dir: Path,
    persona_order: str = "",
) -> list[str]:
    """Return a display-ready list of persona names from *profiles_dir*.

    Args:
        profiles_dir: Directory whose immediate subdirectories are profile names.
        persona_order: Optional comma-separated list of profile names used to
            curate the listing order.  Names that don't exist on disk are
            silently skipped.  Any discovered profiles not mentioned in
            *persona_order* are appended in alphabetical order.

    Returns:
        List of profile directory names (strings) in the resolved order.
        ``default`` and ``example`` profiles are excluded from the listing
        because they are not presentable comedian personas.

    """
    _EXCLUDED = {"default", "example"}

    if not profiles_dir.is_dir():
        logger.warning("startup_screen: profiles directory not found: %s", profiles_dir)
        return []

    discovered: set[str] = {p.name for p in profiles_dir.iterdir() if p.is_dir() and p.name not in _EXCLUDED}

    ordered: list[str] = []
    seen: set[str] = set()

    # Apply curated order first, honouring only names that exist on disk.
    if persona_order:
        for name in persona_order.split(","):
            name = name.strip()
            if name and name in discovered and name not in seen:
                ordered.append(name)
                seen.add(name)

    # Append remaining profiles alphabetically.
    for name in sorted(discovered):
        if name not in seen:
            ordered.append(name)

    return ordered


def _humanise_name(profile_name: str) -> str:
    """Convert a filesystem profile name to a display name.

    ``house_comedian`` → ``House Comedian``
    ``default`` → ``Default``
    """
    return " ".join(word.capitalize() for word in profile_name.replace("-", "_").split("_"))


def _build_listing_sentence(profiles: list[str]) -> str:
    """Build the spoken persona-listing sentence from a list of profile names."""
    if not profiles:
        return ""
    names = [_humanise_name(p) for p in profiles]
    if len(names) == 1:
        display = names[0]
    elif len(names) == 2:
        display = f"{names[0]} or {names[1]}"
    else:
        display = ", ".join(names[:-1]) + ", or " + names[-1]
    return _LISTING_TEMPLATE.format(names=display)


# --------------------------------------------------------------------------- #
# Chatterbox HTTP helper                                                       #
# --------------------------------------------------------------------------- #


async def _call_chatterbox(
    http_client: Any,
    url: str,
    text: str,
) -> bytes | None:
    """POST *text* to the Chatterbox /tts endpoint using the generic baseline.

    Returns raw WAV bytes on success, or ``None`` on failure (errors are
    logged as warnings so the app continues to start).
    """
    payload: dict[str, object] = {
        "text": text,
        "voice_mode": "clone",
        "output_format": "wav",
        "split_text": False,
        "exaggeration": _BASELINE_EXAGGERATION,
        "cfg_weight": _BASELINE_CFG_WEIGHT,
        # Use the server's default predefined voice (no reference file path).
        # Omitting ``audio_prompt_path`` and ``reference_audio_filename`` makes
        # Chatterbox fall back to its built-in generic voice.
    }
    try:
        r = await http_client.post(f"{url}/tts", json=payload, timeout=30.0)
        r.raise_for_status()
        content: bytes = r.content
        return content
    except Exception as exc:
        logger.warning("startup_screen: Chatterbox TTS call failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Audio playback helper                                                        #
# --------------------------------------------------------------------------- #


def _play_wav_bytes_sync(wav_bytes: bytes) -> None:
    """Play *wav_bytes* (WAV format) synchronously via the system player.

    Uses ``sounddevice`` if available (cross-platform, no subprocess overhead),
    otherwise falls back to the same subprocess approach used by warmup_audio.
    Errors are suppressed so a missing player never aborts startup.
    """
    try:
        import io
        import wave

        import numpy as np
        import sounddevice as sd

        with wave.open(io.BytesIO(wav_bytes)) as wf:
            rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
        else:
            audio = audio.reshape(-1, 1)

        sd.play(audio, samplerate=rate, blocking=True)
        return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("startup_screen: sounddevice playback failed: %s", exc)
        return

    # Fallback: write to a temp file and spawn a system player.
    import sys
    import shutil
    import tempfile
    import subprocess

    if sys.platform == "win32":
        logger.debug("startup_screen: no subprocess player on Windows; skipping playback")
        return

    player_cmd: list[str] | None = None
    for candidate in ("pw-play", "paplay", "aplay", "afplay"):
        path = shutil.which(candidate)
        if path:
            player_cmd = [path] + (["-q"] if candidate == "aplay" else [])
            break

    if player_cmd is None:
        logger.debug("startup_screen: no audio player found; skipping playback")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        subprocess.run([*player_cmd, tmp_path], check=False, timeout=60)
    except Exception as exc:
        logger.warning("startup_screen: subprocess playback failed: %s", exc)
    finally:
        try:
            import os as _os

            _os.unlink(tmp_path)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #


async def run_startup_screen(
    *,
    chatterbox_url: str,
    profiles_dir: Path,
    persona_order: str = "",
    selection_wait_s: float = _SELECTION_WAIT_S,
    selection_event: asyncio.Event | None = None,
) -> None:
    """Play the kiosk startup announcement, then return.

    Args:
        chatterbox_url: Base URL for the Chatterbox TTS server
            (e.g. ``http://astralplane.lan:8004``).
        profiles_dir: Filesystem path to the profiles directory used to build
            the dynamic persona listing.
        persona_order: Optional curated comma-separated persona order (value of
            ``REACHY_MINI_STARTUP_SCREEN_PERSONA_ORDER``).
        selection_wait_s: Seconds to wait for a persona selection before
            reading out the listing.  Defaults to 5.
        selection_event: Optional :class:`asyncio.Event` that external code can
            set to signal an early selection.  When set before the wait
            elapses, the listing is skipped.

    """
    import httpx

    logger.info("startup_screen: starting kiosk announcement")

    async with httpx.AsyncClient() as http:
        # Step 1: immediate welcome prompt.
        logger.info("startup_screen: synthesising welcome prompt")
        welcome_wav = await _call_chatterbox(http, chatterbox_url, _WELCOME_PROMPT)
        if welcome_wav:
            await asyncio.get_event_loop().run_in_executor(None, _play_wav_bytes_sync, welcome_wav)

        # Step 2: wait for a selection or timeout.
        if selection_event is not None:
            try:
                await asyncio.wait_for(selection_event.wait(), timeout=selection_wait_s)
                logger.info("startup_screen: selection received — skipping persona listing")
                return
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(selection_wait_s)

        # Step 3: no selection received — play persona listing.
        personas = build_persona_list(profiles_dir, persona_order)
        if not personas:
            logger.info("startup_screen: no personas found; skipping listing")
            return

        listing = _build_listing_sentence(personas)
        logger.info("startup_screen: synthesising persona listing (%d personas)", len(personas))
        listing_wav = await _call_chatterbox(http, chatterbox_url, listing)
        if listing_wav:
            await asyncio.get_event_loop().run_in_executor(None, _play_wav_bytes_sync, listing_wav)

    logger.info("startup_screen: announcement complete")
