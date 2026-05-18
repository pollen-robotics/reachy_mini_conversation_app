"""GestureRegistry — maps gesture names to callable movement functions.

A gesture function has the signature::

    def my_gesture(manager: MovementManager) -> None: ...

It is expected to call ``manager.queue_move(...)`` with one or more
``Move`` instances.  Gestures are primary moves, so they run exclusively
and sequentially via the existing ``MovementManager`` queue.

Per-persona beat mappings
-------------------------
Each persona profile directory may contain a ``gestures.txt`` file that
maps *abstract beat names* (e.g. ``disapproval``) to canonical gesture
names (e.g. ``point_left``).  Use :func:`load_persona_beats` to read
that file, then :meth:`GestureRegistry.resolve_for_persona` to resolve
a beat to a canonical name before calling :meth:`GestureRegistry.play`.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Callable
from pathlib import Path


if TYPE_CHECKING:
    from robot_comic.moves import MovementManager


logger = logging.getLogger(__name__)

GestureFn = Callable[["MovementManager"], None]


# ---------------------------------------------------------------------------
# Persona-beat loader
# ---------------------------------------------------------------------------


def load_persona_beats(profile_dir: Path) -> Dict[str, str]:
    """Read a persona's ``gestures.txt`` and return a beat-to-gesture map.

    The file format is one ``beat=gesture`` pair per line.  Blank lines
    and lines beginning with ``#`` are ignored.

    Parameters
    ----------
    profile_dir:
        Path to the persona's profile directory (e.g.
        ``profiles/house_comedian``).

    Returns
    -------
    dict[str, str]
        Mapping of abstract beat name → canonical gesture name.
        Returns an **empty dict** if the file does not exist, so callers
        can treat a missing file as "no overrides" rather than an error.

    """
    gestures_file = profile_dir / "gestures.txt"
    if not gestures_file.exists():
        logger.debug("load_persona_beats: no gestures.txt in %s", profile_dir)
        return {}

    beats: Dict[str, str] = {}
    for raw_line in gestures_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            logger.warning("load_persona_beats: skipping malformed line %r", raw_line)
            continue
        beat, _, gesture = line.partition("=")
        beat = beat.strip()
        gesture = gesture.strip()
        if beat and gesture:
            beats[beat] = gesture
        else:
            logger.warning("load_persona_beats: skipping empty beat/gesture in %r", raw_line)

    logger.debug("load_persona_beats: loaded %d beats from %s", len(beats), gestures_file)
    return beats


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class GestureRegistry:
    """Registry of named gesture functions.

    Thread-safety note: ``register`` and ``list_gestures`` are typically
    called at startup/import time.  ``play`` is called from tool dispatch
    threads but only reads from the registry dict after registration is
    complete, so no locking is required for the current use-pattern.
    """

    def __init__(self) -> None:
        """Initialise an empty gesture registry."""
        self._gestures: Dict[str, GestureFn] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, fn: GestureFn) -> None:
        """Register a gesture function under *name*.

        Parameters
        ----------
        name:
            Canonical snake_case identifier (e.g. ``"shrug"``).
        fn:
            Callable that accepts a ``MovementManager`` and enqueues
            primary moves on it.

        """
        if name in self._gestures:
            logger.warning("GestureRegistry: overwriting existing gesture %r", name)
        self._gestures[name] = fn
        logger.debug("GestureRegistry: registered gesture %r", name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_gestures(self) -> List[str]:
        """Return a sorted list of all registered gesture names."""
        return sorted(self._gestures)

    # ------------------------------------------------------------------
    # Per-persona beat resolution
    # ------------------------------------------------------------------

    def resolve_for_persona(self, beat: str, persona_beats: Dict[str, str]) -> str:
        """Resolve an abstract beat name to a canonical gesture name.

        Parameters
        ----------
        beat:
            Abstract beat name (e.g. ``"disapproval"``).  See
            :mod:`robot_comic.gestures.beats` for the canonical set.
        persona_beats:
            Mapping returned by :func:`load_persona_beats`.

        Returns
        -------
        str
            The canonical gesture name for the beat.

        Raises
        ------
        KeyError
            If *beat* is not present in *persona_beats*.  The message
            lists the available beats so the LLM can correct itself.

        """
        if beat in persona_beats:
            gesture = persona_beats[beat]
            logger.debug("GestureRegistry: resolved beat %r → gesture %r", beat, gesture)
            return gesture

        available = ", ".join(sorted(persona_beats)) or "<none>"
        raise KeyError(f"Unknown beat {beat!r}. Available beats for this persona: {available}")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play(self, name: str, manager: "MovementManager") -> None:
        """Queue the named gesture's moves on *manager*.

        Parameters
        ----------
        name:
            Gesture name as returned by :meth:`list_gestures`.
        manager:
            The live ``MovementManager`` instance.

        Raises
        ------
        KeyError
            If *name* is not registered.  The message lists available
            gestures so callers can surface a helpful error to the LLM.

        """
        fn = self._gestures.get(name)
        if fn is None:
            available = ", ".join(self.list_gestures()) or "<none>"
            raise KeyError(f"Unknown gesture {name!r}. Available gestures: {available}")
        logger.info("GestureRegistry: playing gesture %r", name)
        fn(manager)
