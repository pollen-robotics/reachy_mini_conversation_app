import json
import base64
import asyncio
import logging
from datetime import datetime

import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item

from reachy_mini_conversation_demo.tools import (
    ALL_TOOL_SPECS,
    ToolDependencies,
    dispatch_tool_call,
)
from reachy_mini_conversation_demo.config import config


logger = logging.getLogger(__name__)


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,  # openai outputs
            input_sample_rate=16000,  # respeaker output
        )
        self.deps = deps

        self.connection = None
        self.output_queue = asyncio.Queue()

        self._pending_calls: dict[str, dict] = {}

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

    def copy(self):
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps)

    async def start_up(self):
        """Start the handler."""
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        async with self.client.beta.realtime.connect(model=config.MODEL_NAME) as conn:
            await conn.session.update(
                session={
                    "turn_detection": {
                        "type": "server_vad",
                    },
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "en",
                    },
                    "voice": "ballad",
                    "instructions": "We speak in English",
                    "tools": ALL_TOOL_SPECS,
                    "tool_choice": "auto",
                    "temperature": 0.7,
                }
            )

            # Manage event received from the openai server
            self.connection = conn
            async for event in self.connection:
                logger.debug(f"OpenAI event: {event.type}")
                if event.type == "input_audio_buffer.speech_started":
                    self.clear_queue()
                    self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)
                    logger.debug("user speech started")

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)
                    logger.debug("user speech stopped")

                if event.type in ("response.audio.completed", "response.completed"):
                    # Doesn't seem to be called
                    logger.debug("response completed")
                    self.deps.head_wobbler.reset()

                if event.type == "response.created":
                    logger.debug("response created")

                if event.type == "response.done":
                    # Doesn't mean the audio is done playing
                    logger.debug("response done")
                    pass
                    # self.deps.head_wobbler.reset()

                if event.type == "conversation.item.input_audio_transcription.completed":
                    logger.debug(f"user transcript: {event.transcript}")
                    await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                if event.type == "response.audio_transcript.done":
                    logger.debug(f"assistant transcript: {event.transcript}")
                    await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                if event.type == "response.audio.delta":
                    self.deps.head_wobbler.feed(event.delta)
                    self.last_activity_time = asyncio.get_event_loop().time()
                    logger.debug("last activity time updated to %s", self.last_activity_time)
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1),
                        ),
                    )

                # ---- tool-calling plumbing ----
                # 1) model announces a function call item; capture name + call_id
                if event.type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        call_id = getattr(item, "call_id", None)
                        name = getattr(item, "name", None)
                        if call_id and name:
                            self._pending_calls[call_id] = {
                                "name": name,
                                "args_buf": "",
                            }

                # 2) model streams JSON arguments; buffer them by call_id
                if event.type == "response.function_call_arguments.delta":
                    call_id = getattr(event, "call_id", None)
                    delta = getattr(event, "delta", "")
                    if call_id in self._pending_calls:
                        self._pending_calls[call_id]["args_buf"] += delta

                # 3) when args done, execute Python tool, send function_call_output, then trigger a new response
                if event.type == "response.function_call_arguments.done":
                    call_id = getattr(event, "call_id", None)
                    info = self._pending_calls.get(call_id)
                    if not info:
                        continue
                    tool_name = info["name"]
                    args_json_str = info["args_buf"] or "{}"

                    try:
                        tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                        logger.debug("[Tool %s executed]", tool_name)
                        logger.debug("Tool result: %s", tool_result)
                    except Exception as e:
                        logger.error("Tool %s failed", tool_name)
                        tool_result = {"error": str(e)}

                    # send the tool result back
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(tool_result),
                        }
                    )

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": json.dumps(tool_result),
                                "metadata": {"title": "🛠️ Used tool " + tool_name, "status": "done"},
                            },
                        )
                    )

                    if tool_name == "camera" and "b64_im" in tool_result:
                        b64_im = json.dumps(tool_result["b64_im"])
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_im}",
                                    }
                                ],
                            }
                        )
                        logger.info("additional input camera")

                        np_img = self.deps.camera_worker.get_latest_frame()
                        img = gr.Image(value=np_img)

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": img,
                                }
                            )
                        )

                    if not self.is_idle_tool_call:
                        await self.connection.response.create(
                            response={
                                "instructions": "Use the tool result just returned and answer concisely in speech."
                            }
                        )
                    else:
                        self.is_idle_tool_call = False

                    # re synchronize the head wobble after a tool call that may have taken some time
                    self.deps.head_wobbler.reset()
                    # cleanup
                    self._pending_calls.pop(call_id, None)

                # server error
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    logger.error("Realtime error: %s (raw=%s)", msg, err)
                    await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))

    # Microphone receive
    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Receive audio frame from the microphone and send it to the openai server."""
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        # Fills the input audio buffer to be sent to the server
        await self.connection.input_audio_buffer.append(audio=audio_message)  # type: ignore

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            await self.send_idle_signal(idle_duration)

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        if self.connection:
            await self.connection.close()
            self.connection = None

    def format_timestamp(self):
        """Format current timestamp with date, time and elapsed seconds."""
        current_time = asyncio.get_event_loop().time()
        elapsed_seconds = current_time - self.start_time
        dt = datetime.fromtimestamp(current_time)
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def send_idle_signal(self, idle_duration) -> None:
        """Send an idle signal to the openai server."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": timestamp_msg}],
            }
        )
        await self.connection.response.create(
            response={
                "modalities": ["text"],
                "instructions": "You MUST respond with function calls only - no speech or text. Choose appropriate actions for idle behavior.",
                "tool_choice": "required",
            }
        )
        # TODO additional inputs
