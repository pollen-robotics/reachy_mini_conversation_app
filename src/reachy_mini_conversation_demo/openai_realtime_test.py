import asyncio
import base64
import json

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from openai import AsyncOpenAI

from reachy_mini_conversation_demo.tools import (
    ALL_TOOL_SPECS,
    ToolDependencies,
    dispatch_tool_call,
)


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
                    # "input_audio_transcription": {
                    #     "model": "whisper-1",
                    #     "language": "en",
                    # },
                    "voice": "ballad",
                    "instructions": "You are a helpful assistant.",
                    "tools": ALL_TOOL_SPECS,
                    "tool_choice": "auto",
                    "temperature": 0.7,
                }
            )

            # Manage event received from the openai server
            self.connection = conn
            async for event in self.connection:
                # print(f"[DEBUG] OpenAI event: {event.type}")
                if event.type == "input_audio_buffer.speech_started":
                    self.clear_queue()
                    self.deps.head_wobbler.reset()
                    print("[DEBUG] user speech started")

                if event.type == "input_audio_buffer.speech_stopped":
                    print("[DEBUG] user speech stopped")
                    pass

                if event.type == "response.created":
                    print("[DEBUG] response created")
                    pass

                if event.type == "response.done":
                    print("[DEBUG] response done")
                    # self.deps.head_wobbler.reset()

                # if (
                #     event.type
                #     == "conversation.item.input_audio_transcription.completed"
                # ):
                #     await self.output_queue.put(
                #         AdditionalOutputs({"role": "user", "content": event.transcript})
                #     )

                # if event.type == "response.audio_transcript.done":
                #     await self.output_queue.put(
                #         AdditionalOutputs(
                #             {"role": "assistant", "content": event.transcript}
                #         )
                #     )

                if event.type == "response.audio.delta":
                    self.deps.head_wobbler.feed(event.delta)
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(
                                base64.b64decode(event.delta), dtype=np.int16
                            ).reshape(1, -1),
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
                        tool_result = await dispatch_tool_call(
                            tool_name, args_json_str, self.deps
                        )
                        print("[Tool %s executed]", tool_name)
                        print("Tool result: %s", tool_result)
                    except Exception as e:
                        print("Tool %s failed", tool_name)
                        tool_result = {"error": str(e)}

                    # send the tool result back
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(tool_result),
                        }
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

                    await self.connection.response.create(
                        response={
                            "instructions": "Use the tool result just returned and answer concisely in speech."
                        }
                    )

                    # cleanup
                    self._pending_calls.pop(call_id, None)

                # server error
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    print("Realtime error: %s (raw=%s)", msg, err)
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
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        if self.connection:
            await self.connection.close()
            self.connection = None
