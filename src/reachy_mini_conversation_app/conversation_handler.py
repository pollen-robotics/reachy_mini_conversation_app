from __future__ import annotations
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Literal, ClassVar, TypeAlias
from collections.abc import Callable

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from numpy.typing import NDArray

from reachy_mini_conversation_app.idle_policy import start_idle_tool_call
from reachy_mini_conversation_app.tools.core_tools import ToolSpec, ToolDependencies, get_active_tool_specs
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
HandlerOutput: TypeAlias = AudioFrame | AdditionalOutputs | None
QueueItem: TypeAlias = AudioFrame | AdditionalOutputs


class ConversationHandler(AsyncStreamHandler, ABC):
    """Shared app handler contract and idle behavior for realtime conversation backends."""

    IDLE_BEHAVIOR_THRESHOLD_S: ClassVar[float] = 180.0

    deps: ToolDependencies
    tool_manager: BackgroundToolManager
    output_queue: asyncio.Queue[QueueItem]
    last_activity_time: float
    last_idle_behavior_time: float
    _clear_queue: Callable[[], None] | None = None
    _activity_observer: Callable[[str], None] | None = None

    def __init__(
        self,
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int | None = None,
        input_sample_rate: int = 48000,
        fps: int = 30,
    ) -> None:
        """Initialize the stream handler and shared idle/activity tracking."""
        super().__init__(
            expected_layout=expected_layout,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            input_sample_rate=input_sample_rate,
            fps=fps,
        )
        self.last_activity_time = time.monotonic()
        self.last_idle_behavior_time = self.last_activity_time

    def set_activity_observer(self, observer: Callable[[str], None] | None) -> None:
        """Attach or detach an activity observer. Pass None to clear."""
        self._activity_observer = observer

    def _mark_activity(self, reason: str) -> None:
        """Record non-idle conversation activity for the idle timer."""
        self.last_activity_time = time.monotonic()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)
        if self._activity_observer is not None:
            try:
                self._activity_observer(reason)
            except Exception:
                logger.debug("activity observer raised (ignored)", exc_info=True)

    def _idle_behavior_ready(self) -> bool:
        """Return whether idle behavior may run now. Backends can add guards."""
        return True

    async def emit(self) -> HandlerOutput:
        """Emit the next queued output, triggering local idle behavior when due."""
        now = time.monotonic()
        idle_duration = now - self.last_activity_time
        idle_behavior_duration = now - self.last_idle_behavior_time
        if (
            idle_duration > self.IDLE_BEHAVIOR_THRESHOLD_S
            and idle_behavior_duration > self.IDLE_BEHAVIOR_THRESHOLD_S
            and self._idle_behavior_ready()
            and self.deps.movement_manager.is_idle()
        ):
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle tool skipped (connection closed?): %s", e)
                return None
            self.last_idle_behavior_time = now
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Run a locally selected idle tool without sending an idle turn to the model."""
        if not self._is_connected():
            logger.debug("No active session; cannot run idle tool")
            return

        available_tool_names = {spec["name"] for spec in self._get_active_tool_specs()}
        await start_idle_tool_call(
            deps=self.deps,
            tool_manager=self.tool_manager,
            output_queue=self.output_queue,
            available_tool_names=available_tool_names,
            idle_duration=idle_duration,
        )

    def _get_active_tool_specs(self) -> list[ToolSpec]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    @abstractmethod
    def _is_connected(self) -> bool:
        """Return whether the backend session/connection is currently open."""
        ...

    @abstractmethod
    def copy(self) -> ConversationHandler:
        """Create a copy of the handler."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile."""
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Return voices available for the active backend."""
        ...

    @abstractmethod
    def get_current_voice(self) -> str:
        """Return the current voice."""
        ...

    @abstractmethod
    async def change_voice(self, voice: str) -> str:
        """Change the current voice."""
        ...
