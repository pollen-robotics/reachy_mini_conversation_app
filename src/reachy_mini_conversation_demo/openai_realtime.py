from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Optional

import numpy as np
from openai import AsyncOpenAI
from websockets import ConnectionClosedError, ConnectionClosedOK

from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item

from reachy_mini_conversation_demo.prompts import SESSION_INSTRUCTIONS
from reachy_mini_conversation_demo.tools import (
    ToolDependencies,
    ALL_TOOL_SPECS,
    dispatch_tool_call,
)
from reachy_mini_conversation_demo.audio.audio_sway import AudioSync, pcm_to_b64
from reachy_mini_conversation_demo.config import config

logger = logging.getLogger(__name__)


SAMPLE_RATE = 24000  # keep same default as before
MODEL_NAME = config.MODEL_NAME
API_KEY = config.OPENAI_API_KEY


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """
    Async handler that bridges mic input -> OpenAI Realtime -> audio out (+ tool calls).
    Public API:
      - start_up()
      - receive(frame: bytes)
      - emit() -> tuple[int, np.ndarray] | AdditionalOutputs | None
      - shutdown()
    """

    def __init__(
        self,
        deps: ToolDependencies,
        audio_sync: AudioSync,
        robot_is_speaking: asyncio.Event,
        speaking_queue: asyncio.Queue,
        no_interruptions: bool = False,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        # deps
        self.deps = deps
        self.audio_sync = audio_sync
        self.robot_is_speaking = robot_is_speaking
        self.speaking_queue = speaking_queue
        self.no_interruptions = no_interruptions

        # runtime
        self.client: Optional[AsyncOpenAI] = None
        self.connection = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._stop = False
        self._started_audio = False
        self._connection_ready = False
        self._speech_start_time = 0.0

        # backoff
        self._backoff_start = 1.0
        self._backoff_max = 16.0
        self._backoff = self._backoff_start

    def copy(self) -> "OpenAIRealtimeHandler":
        return OpenAIRealtimeHandler(
            self.deps,
            self.audio_sync,
            self.robot_is_speaking,
            self.speaking_queue,
        )

    async def start_up(self):
        if not self._started_audio:
            self.audio_sync.start()
            self._started_audio = True

        if self.client is None:
            logger.info("Realtime start_up: creating AsyncOpenAI client…")
            self.client = AsyncOpenAI(
                websocket_base_url="wss://haixuantao-dora-openai-realtime.hf.space",
                api_key=API_KEY,
            )

        self._backoff = self._backoff_start
        while not self._stop:
            try:
                async with self.client.beta.realtime.connect(model=MODEL_NAME) as rtc:
                    self.connection = rtc
                    self._connection_ready = False

                    # Configure session
                    await rtc.session.update(
                        session={
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.6,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 800,
                            },
                            "voice": "ballad",
                            "instructions": SESSION_INSTRUCTIONS,
                            "input_audio_transcription": {
                                "model": "whisper-1",
                                "language": "en",
                            },
                            "tools": ALL_TOOL_SPECS,
                            "tool_choice": "auto",
                            "temperature": 0.7,
                        }
                    )

                    # Give the server a breath to apply config
                    await asyncio.sleep(0.2)

                    # Extra brevity instruction
                    await rtc.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        f"{SESSION_INSTRUCTIONS}\n\n"
                                        "IMPORTANT: Always keep responses under 25 words. Be extremely concise."
                                    ),
                                }
                            ],
                        }
                    )

                    self._connection_ready = True
                    self._backoff = self._backoff_start
                    logger.info(
                        "Session ready (tools=%d, voice=%s)",
                        len(ALL_TOOL_SPECS),
                        "ballad",
                    )

                    async for event in rtc:
                        et = getattr(event, "type", None)
                        logger.debug("RT event: %s", et)

                        # conversation / transcripts
                        if et == "input_audio_buffer.speech_started":
                            if not self.no_interruptions and not self.robot_is_speaking.is_set():
                                self.audio_sync.on_input_speech_started()
                                logger.info("User speech detected")
                        elif et == "response.started":
                            self._speech_start_time = time.time()
                            self.audio_sync.on_response_started()
                            logger.info("Robot started speaking")
                        elif et in (
                            "response.audio.completed",
                            "response.completed",
                            "response.audio.done",
                        ):
                            logger.info("Robot finished speaking (%s)", et)
                        elif (
                            et
                            == "conversation.item.input_audio_transcription.completed"
                        ):
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "user", "content": event.transcript}
                                )
                            )
                        elif et == "response.audio_transcript.done":
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "assistant", "content": event.transcript}
                                )
                            )

                        # streaming audio
                        if et == "response.audio.delta":
                            self.robot_is_speaking.set()
                            # block mic briefly per chunk to reduce echo
                            self.speaking_queue.put_nowait(0.25)
                            self.audio_sync.on_response_audio_delta(
                                getattr(event, "delta", b"")
                            )

                        # tool calls
                        elif et == "response.function_call_arguments.done":
                            tool_name = getattr(event, "name", None)
                            args_json_str = getattr(event, "arguments", None)
                            call_id = getattr(event, "call_id", None)

                            try:
                                tool_result = await dispatch_tool_call(
                                    tool_name, args_json_str, self.deps
                                )
                                logger.info("Tool result: %s", tool_result)
                            except Exception as e:
                                logger.exception("Tool %s failed", tool_name)
                                tool_result = {"error": str(e)}

                            await rtc.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(tool_result),
                                }
                            )

                            # Force LLM to speak after vision/camera tools
                            if tool_name and (
                                tool_name == "camera" or "scene" in tool_name
                            ):
                                await rtc.response.create()

                        # server error
                        if et == "error":
                            err = getattr(event, "error", None)
                            msg = getattr(
                                err, "message", str(err) if err else "unknown error"
                            )
                            logger.error("Realtime error: %s (raw=%s)", msg, err)
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {"role": "assistant", "content": f"[error] {msg}"}
                                )
                            )

            except (ConnectionClosedOK, ConnectionClosedError) as e:
                if self._stop:
                    break
                logger.warning(
                    "Connection closed (%s). Reconnecting…",
                    getattr(e, "code", "no-code"),
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Realtime loop error; will reconnect")
            finally:
                self.connection = None
                self._connection_ready = False

            # Exponential backoff before reconnect
            delay = min(self._backoff, self._backoff_max) + random.uniform(0, 0.5)
            logger.info("Reconnect in %.1fs…", delay)
            await asyncio.sleep(delay)
            self._backoff = min(self._backoff * 2.0, self._backoff_max)

    async def receive(self, frame: bytes) -> None:
        """Accept PCM16 mono frames from the mic pipeline (fastrtc)."""
        if (self.no_interruptions and self.robot_is_speaking.is_set()) or not self._connection_ready:
            return

        mic_samples = np.frombuffer(frame, dtype=np.int16).squeeze()
        audio_b64 = pcm_to_b64(mic_samples)

        try:
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except (ConnectionClosedOK, ConnectionClosedError):
            pass

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        """Return either audio to play (sr, np.int16 array) or chat outputs."""
        try:
            sample_rate, pcm_frame = self.audio_sync.playback_q.get_nowait()
            logger.debug(
                "Emitting playback frame (sr=%d, n=%d)", sample_rate, pcm_frame.size
            )
            return (sample_rate, pcm_frame)
        except asyncio.QueueEmpty:
            pass
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        logger.info("Shutdown: closing connections and audio")
        self._stop = True
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                logger.exception("Error closing realtime connection")
            finally:
                self.connection = None
                self._connection_ready = False
        await self.audio_sync.stop()
