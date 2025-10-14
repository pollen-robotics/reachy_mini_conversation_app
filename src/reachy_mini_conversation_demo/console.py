"""Bidirectional local audio stream.

records mic frames to the handler and plays handler audio frames to the speaker.
"""

import asyncio
import logging

import librosa
from fastrtc import AdditionalOutputs, audio_to_int16, audio_to_float32

from reachy_mini import ReachyMini
from reachy_mini_conversation_demo.openai_realtime import OpenaiRealtimeHandler


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(self, handler: OpenaiRealtimeHandler, robot: ReachyMini):
        """Initialize the stream with an OpenAI realtime handler and pipelines."""
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue  # type: ignore[assignment]

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops."""
        self._stop_event.clear()
        self._robot.media.start_recording()
        self._robot.media.start_playing()

        async def runner() -> None:
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        - Stops audio recording and playback
        """
        logger.info("Stopping LocalStream...")
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        self._robot.media.stop_recording()
        self._robot.media.stop_playing()

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        logger.info("Starting receive loop")
        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                frame_mono = audio_frame.T[0]  # both channels are identical
                frame = audio_to_int16(frame_mono)
                await self.handler.receive((16000, frame))
                # await asyncio.sleep(0)  # yield to event loop
            else:
                await asyncio.sleep(0.01)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_frame = handler_output
                device_sample_rate = self._robot.media.get_audio_samplerate()
                audio_frame = audio_to_float32(audio_frame.squeeze())
                if input_sample_rate != device_sample_rate:
                    audio_frame = librosa.resample(
                        audio_frame, orig_sr=input_sample_rate, target_sr=device_sample_rate
                    )
                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
