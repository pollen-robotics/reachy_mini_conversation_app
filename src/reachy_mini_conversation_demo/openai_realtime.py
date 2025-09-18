import asyncio  # noqa: D100
import base64
import json
import logging
from datetime import datetime

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from openai import AsyncOpenAI

from reachy_mini_conversation_demo.tools import (
    ALL_TOOL_SPECS,
    ToolDependencies,
    dispatch_tool_call,
)

logger = logging.getLogger(__name__)


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=24000,
        )
        self.deps = deps

        self.connection = None
        self.output_queue = asyncio.Queue()

        self._pending_calls: dict[str, dict] = {}

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        # Track last processed text to avoid duplicates on UI refresh
        self._last_text_sent: str | None = None
        # Track current response id and whether we should suppress audio playback
        self._current_response_id: str | None = None
        self._suppress_audio: bool = False

    def copy(self):
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps)

    async def start_up(self):
        """Start the handler."""
        self.client = AsyncOpenAI()
        async with self.client.beta.realtime.connect(model="gpt-realtime") as conn:
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
                # Attempt to capture current response id
                if event.type == "response.created":
                    rid = None
                    resp = getattr(event, "response", None)
                    # try common shapes: event.response.id or event.id
                    if resp is not None:
                        rid = getattr(resp, "id", None)
                    if rid is None:
                        rid = getattr(event, "id", None)
                    if isinstance(rid, str):
                        self._current_response_id = rid
                        logger.debug("captured response id: %s", rid)
                if event.type == "input_audio_buffer.speech_started":
                    self.clear_queue()
                    self.deps.head_wobbler.reset()
                    logger.debug("user speech started")

                if event.type == "input_audio_buffer.speech_stopped":
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
                    self._current_response_id = None
                    self._suppress_audio = False
                    # self.deps.head_wobbler.reset()

                if (
                    event.type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    logger.debug(f"user transcript: {event.transcript}")
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "user", "content": event.transcript})
                    )

                if event.type == "response.audio_transcript.done":
                    logger.debug(f"assistant transcript: {event.transcript}")
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": event.transcript}
                        )
                    )

                if event.type == "response.audio.delta":
                    if not self._suppress_audio:
                        self.deps.head_wobbler.feed(event.delta)
                        self.last_activity_time = asyncio.get_event_loop().time()
                        logger.debug(
                            "last activity time updated to %s", self.last_activity_time
                        )
                        await self.output_queue.put(
                            (
                                self.output_sample_rate,
                                np.frombuffer(
                                    base64.b64decode(event.delta), dtype=np.int16
                                ).reshape(1, -1),
                            ),
                        )
                    else:
                        # Drop audio frames while suppressed
                        logger.debug("audio suppressed; dropping delta chunk")

                # When the server confirms the audio buffer is cleared, allow future audio
                if event.type in ("output_audio_buffer.cleared", "output_audio_buffer.stopped"):
                    logger.debug("server output audio buffer cleared/stopped")
                    self._suppress_audio = False

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
                        tool_result = await dispatch_tool_call(
                            tool_name, args_json_str, self.deps
                        )
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
                                "metadata": dict(
                                    title="🛠️ Used tool " + tool_name, status="done"
                                ),
                            },
                        )
                    )

                    if tool_name == "camera":
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
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": f"[error] {msg}"}
                        )
                    )

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

            self.last_activity_time = (
                asyncio.get_event_loop().time()
            )  # avoid repeated resets

        # Handle text input coming from the WebRTC textbox variant or additional inputs
        # Stream wires input values via latest_args; when using the built-in textbox variant,
        # latest_args[0] is a WebRTCData-like object (or dict) with a 'textbox' field.
        try:
            if self.args_set.is_set():
                args = list(self.latest_args)
                # reset ready flag for next inputs
                self.reset()
                if args:
                    text_value = None
                    first = args[0]
                    # Case 1: direct string (from a plain Textbox additional input)
                    if isinstance(first, str):
                        text_value = first
                    # Case 2: WebRTCData model/dict with .textbox or ['textbox']
                    else:
                        # try attribute access chain
                        for candidate in [
                            getattr(first, "textbox", None),
                            getattr(getattr(first, "root", None), "textbox", None)
                            if hasattr(first, "root")
                            else None,
                        ]:
                            if isinstance(candidate, str) and candidate:
                                text_value = candidate
                                break
                        # try dict-style
                        if text_value is None and isinstance(first, dict):
                            root = first.get("root", first)
                            if isinstance(root, dict):
                                tv = root.get("textbox")
                                if isinstance(tv, str):
                                    text_value = tv

                    text = (text_value or "").strip()
                    if text and text != self._last_text_sent:
                        # Any incoming text interrupts current speech like a voice barge-in
                        self._suppress_audio = True
                        self.clear_queue()
                        self.deps.head_wobbler.reset()
                        # Best-effort server-side cancel and clear of audio buffer
                        try:
                            if self.connection:
                                if self._current_response_id:
                                    await self.connection.response.cancel(
                                        response_id=self._current_response_id
                                    )
                                await self.connection.output_audio_buffer.clear()
                        except Exception:
                            pass
                        # push to chatbot as a user message (non-blocking UI update)
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "user", "content": text})
                        )
                        # send to the realtime conversation
                        if self.connection:
                            await self.connection.conversation.item.create(
                                item={
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": text}
                                    ],
                                }
                            )
                            # Ask the assistant to respond (defaults to speaking)
                            await self.connection.response.create(response={})
                            self._last_text_sent = text
                            # refresh activity timestamp so idle logic pauses
                            self.last_activity_time = asyncio.get_event_loop().time()
        except Exception as e:
            logger.error("Failed handling text input: %s", e)
            await self.output_queue.put(
                AdditionalOutputs(
                    {"role": "assistant", "content": f"[error] {str(e)}"}
                )
            )

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
