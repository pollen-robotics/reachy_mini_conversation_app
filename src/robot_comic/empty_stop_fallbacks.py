"""Canned-line pool for Gemini empty-STOP fallbacks (issue #267).

Gemini 2.5 Flash sometimes returns ``finish_reason=STOP`` with **zero
parts** (no text, no function_call) when called with the combination of
(tools-enabled config + short user input + certain persona system
prompts). Today's composable orchestrator
(:meth:`robot_comic.composable_pipeline.ComposablePipeline._speak_assistant_text`)
warns and drops the turn — the robot silently hangs from the operator's
perspective.

This module is the persona-pluggable canned-line pool. The orchestrator
samples one line on each empty-STOP, avoids immediate repeats across
consecutive empties in a single session, and records the spoken line as
an ``assistant`` turn so the LLM has continuity on the next request.

Profile override
----------------

If ``<PROFILES_DIRECTORY>/<persona>/empty_stop_fallbacks.txt`` exists, its
non-blank, non-``#`` lines are used in place of the bundled default pool.
An empty or all-comment file falls back to the default pool (rather than
silently disabling fallbacks) so an operator doesn't accidentally
re-introduce the silent-hang behaviour by saving a blank file.

The loader caches per-profile-path so repeated empty-STOPs in one session
don't re-read the disk; the cache is keyed on the resolved path so
swapping personas mid-session loads the new profile's pool on demand.

The default pool is intentionally short and persona-neutral so any
persona without an override sounds at least minimally coherent.
Persona-tailored overrides live in external profile directories
(see ``REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY``).
"""

from __future__ import annotations
import random
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# Generic, persona-neutral fallback lines. Used when no profile override
# is present (missing file OR file present but all-blank/all-comment).
# Keep the pool >=3 entries so the no-immediate-repeat sampler has room
# to pick a different line on the second empty-STOP without falling back
# to the same string.
_DEFAULT_POOL: tuple[str, ...] = (
    "I'm here.",
    "Say that again?",
    "I missed that — one more time?",
    "Hmm. Go on.",
    "Hang on, try that again.",
)


# Cache of {resolved_profile_path: tuple_of_lines}. Keyed on the
# absolute path so two different profile names can coexist; also keyed
# on a sentinel ``None`` for the in-code default pool to keep the
# accessor's return type uniform.
_POOL_CACHE: dict[Path | None, tuple[str, ...]] = {}


def _read_pool_file(path: Path) -> tuple[str, ...]:
    """Return non-blank, non-comment lines from *path*.

    Returns an empty tuple on read errors or if the file contains only
    blank/comment lines. The caller is responsible for falling back to
    the default pool when this returns empty.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "empty_stop_fallbacks: could not read %s (%s); using default pool",
            path,
            exc,
        )
        return ()
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return tuple(lines)


def get_default_pool() -> tuple[str, ...]:
    """Return the in-code default fallback pool (read-only)."""
    return _DEFAULT_POOL


def load_pool(profile_dir: Path | None) -> tuple[str, ...]:
    """Return the active fallback pool for the given profile directory.

    Resolution order:

    1. If ``profile_dir`` is provided and
       ``profile_dir/empty_stop_fallbacks.txt`` exists and parses to a
       non-empty list of lines, return those.
    2. Otherwise return the bundled :data:`_DEFAULT_POOL`.

    Results are cached per *profile_dir*; pass ``profile_dir=None`` to
    skip the lookup and get the default pool directly. The cache is
    process-wide; ``clear_cache()`` is exposed for tests.
    """
    if profile_dir is None:
        return _DEFAULT_POOL
    cached = _POOL_CACHE.get(profile_dir)
    if cached is not None:
        return cached
    candidate = profile_dir / "empty_stop_fallbacks.txt"
    pool: tuple[str, ...]
    if candidate.is_file():
        from_file = _read_pool_file(candidate)
        if from_file:
            pool = from_file
        else:
            logger.info(
                "empty_stop_fallbacks: %s is empty or all comments; using default pool",
                candidate,
            )
            pool = _DEFAULT_POOL
    else:
        pool = _DEFAULT_POOL
    _POOL_CACHE[profile_dir] = pool
    return pool


def clear_cache() -> None:
    """Drop the per-profile pool cache. Intended for tests."""
    _POOL_CACHE.clear()


def pick_fallback(pool: tuple[str, ...], *, last_spoken: str | None) -> str:
    """Return a line from *pool*, avoiding *last_spoken* when possible.

    If *pool* has at least two distinct entries, the chosen line is
    guaranteed not to equal *last_spoken*. If the pool has only one
    distinct entry (or is exactly ``(last_spoken,)``) the only-available
    line is returned — the no-repeat invariant is best-effort.

    The pool is assumed non-empty; the caller guarantees this by going
    through :func:`load_pool`, which always falls back to the bundled
    default rather than returning an empty tuple.
    """
    if not pool:
        # Defensive: should not happen via load_pool, but never raise on
        # a path the operator cares about (silent hang is exactly what
        # we are trying to avoid).
        return _DEFAULT_POOL[0]
    if last_spoken is None or len(pool) == 1:
        return random.choice(pool)
    candidates = tuple(line for line in pool if line != last_spoken)
    if not candidates:
        # Pool exists but every entry equals last_spoken (e.g. a
        # one-line file). Return last_spoken — better than nothing.
        return pool[0]
    return random.choice(candidates)


def resolve_profile_dir() -> Path | None:
    """Return the active profile directory, or ``None`` if unset.

    Reads :data:`robot_comic.config.REACHY_MINI_CUSTOM_PROFILE` /
    :data:`robot_comic.config.PROFILES_DIRECTORY` defensively so an
    import-time misconfiguration can't prevent the fallback path from
    working — any failure logs and returns ``None`` (default pool).
    """
    try:
        from robot_comic import config

        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        if not profile:
            return None
        root = getattr(config, "PROFILES_DIRECTORY", None)
        if root is None:
            return None
        resolved: Path = Path(root) / profile
        return resolved
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("empty_stop_fallbacks: resolve_profile_dir failed: %s", exc)
        return None
