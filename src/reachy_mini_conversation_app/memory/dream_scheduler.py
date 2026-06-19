"""Background dream runner."""

from __future__ import annotations
import logging
import threading
from typing import Callable
from dataclasses import dataclass

from reachy_mini_conversation_app.memory.dreamer import DEFAULT_DREAMER_MODEL, Dreamer, DreamLogStats
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager


logger = logging.getLogger(__name__)

__all__ = ["DreamScheduler", "DreamSummary", "DEFAULT_DREAMER_MODEL"]


@dataclass
class DreamSummary:
    """One-line outcome of a dream pass, used to phrase the awareness note."""

    logs_processed: int = 0
    created: int = 0
    updated: int = 0
    errored: bool = False

    @classmethod
    def from_stats(cls, stats: list[DreamLogStats]) -> "DreamSummary":
        """Fold the dreamer's per-log stats into a single summary."""
        return cls(
            logs_processed=len(stats),
            created=sum(s.created for s in stats),
            updated=sum(s.updated for s in stats),
            errored=any(s.errors for s in stats),
        )


class DreamScheduler:
    """Run a dream pass on a daemon thread."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        on_start: Callable[[], None],
        on_finish: Callable[[DreamSummary], None],
        self_reflect: bool = False,
        dreamer_factory: Callable[[], Dreamer] | None = None,
    ) -> None:
        """Initialize the scheduler. Pass ``dreamer_factory`` in tests to stub the dreamer."""
        self._manager = memory_manager
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._on_start = on_start
        self._on_finish = on_finish
        self._self_reflect = self_reflect
        self._dreamer_factory = dreamer_factory
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the dream thread, or return False when skipped."""
        if self._thread is not None and self._thread.is_alive():
            logger.info("[DREAM] A dream is already running; not starting another.")
            return False

        pending = self._manager.list_pending_logs(exclude_session=True)
        if not pending:
            logger.info("[DREAM] No pending logs; skipping background dream.")
            return False

        logger.info("[DREAM] Launching background dream over %d pending log(s).", len(pending))
        self._thread = threading.Thread(target=self._run, name="dream-scheduler", daemon=True)
        self._thread.start()
        return True

    def is_running(self) -> bool:
        """Whether a dream thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self._on_start()
        except Exception:
            logger.exception("[DREAM] on_start callback raised; continuing.")

        summary = DreamSummary(errored=True)
        try:
            if self._dreamer_factory:
                dreamer = self._dreamer_factory()
            else:
                dreamer = Dreamer(
                    self._manager,
                    model=self._model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    self_reflect=self._self_reflect,
                )
            stats = dreamer.run()
            summary = DreamSummary.from_stats(stats)
            logger.info(
                "[DREAM] Background dream finished: %d log(s), created %d, updated %d.",
                summary.logs_processed,
                summary.created,
                summary.updated,
            )
        except Exception:
            logger.exception("[DREAM] Background dream failed; conversation is unaffected.")
        finally:
            try:
                self._on_finish(summary)
            except Exception:
                logger.exception("[DREAM] on_finish callback raised.")
