"""Bidirectional local audio stream.

records mic frames to the handler and plays handler audio frames to the speaker.
"""

import time
import asyncio
import logging
from typing import List

import librosa
from fastrtc import AdditionalOutputs, audio_to_int16, audio_to_float32, AsyncStreamHandler
import numpy as np
import wave
from pathlib import Path
from datetime import datetime

from reachy_mini import ReachyMini


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(self, handler: AsyncStreamHandler, robot: ReachyMini):
        """Initialize the stream with a realtime handler and pipelines."""
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue

        # Recording of robot output (post-resample), stored as int16 mono
        self._recording_chunks: List[np.ndarray] = []
        self._session_started_at: datetime = datetime.now()

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops."""
        self._stop_event.clear()
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

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

        # Persist a single WAV file with the whole session robot audio
        try:
            if self._recording_chunks:
                recordings_dir = Path("recordings")
                recordings_dir.mkdir(parents=True, exist_ok=True)
                ts = self._session_started_at.strftime("%Y%m%d_%H%M%S")
                wav_path = recordings_dir / f"session_{ts}.wav"

                full_audio = np.concatenate(self._recording_chunks).astype(np.int16, copy=False)
                sample_rate = 44100

                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit PCM
                    wf.setframerate(sample_rate)
                    wf.writeframes(full_audio.tobytes())

                logger.info(
                    "Saved session output audio to %s (%.2f seconds)",
                    wav_path,
                    len(full_audio) / float(sample_rate),
                )
        except Exception:
            logger.exception("Failed to write session output recording WAV file")
        finally:
            self._recording_chunks.clear()

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
                try:  # Record what the robot hears (convert to float first, then resample)
                    frame_mono_float = audio_to_float32(frame_mono)/2  # Somehow it goes a bit above 1.0
                    frame_resampled = librosa.resample(
                        frame_mono_float, orig_sr=16000, target_sr=44100,
                    )
                    frame_int16 = audio_to_int16(frame_resampled)
                    self._recording_chunks.append(np.ascontiguousarray(frame_int16, dtype=np.int16))
                except Exception:
                    logger.exception("Failed to buffer output audio for recording")

                frame = audio_to_int16(frame_mono)
                await self.handler.receive((16000, frame))

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
                audio_frame_float = audio_to_float32(audio_frame.squeeze())

                if input_sample_rate != device_sample_rate:
                    audio_frame_float = librosa.resample(
                        audio_frame_float,
                        orig_sr=input_sample_rate,
                        target_sr=device_sample_rate,
                    )

                self._robot.media.push_audio_sample(audio_frame_float)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
