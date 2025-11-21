import io
import json
import base64
import asyncio
import logging
from typing import Any, Tuple
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
import PIL.Image
from fastrtc import AdditionalOutputs, wait_for_item
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.handlers.base import ConversationHandler
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)


class GeminiRealtimeHandler(ConversationHandler):
    """A Gemini Live API handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=16000,
        )
        self.deps = deps

        self.session: Any = None
        self.client: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Transcription buffers
        self.input_transcription_buffer = ""
        self.output_transcription_buffer = ""


    def copy(self) -> "GeminiRealtimeHandler":
        """Create a copy of the handler."""
        return GeminiRealtimeHandler(self.deps)


    def _convert_tool_specs_to_gemini_format(self) -> list[dict]:
        """Convert OpenAI tool specs to Gemini function declarations format."""
        openai_specs = get_tool_specs()
        function_declarations = []

        for spec in openai_specs:
            if spec["type"] == "function":
                function_declarations.append({
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                })
                logger.debug(f"Configured tool: {spec['name']}")

        return [{"function_declarations": function_declarations}]


    async def start_up(self) -> None:
        """Start the Gemini Live API session."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            logger.error(
                "google-genai package not installed. "
                "Install with: uv add google-genai"
            )
            raise RuntimeError(
                "google-genai package required for Gemini handler"
            ) from e

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.types = types  # Store for later use

        system_instruction = get_session_instructions()
        tools = self._convert_tool_specs_to_gemini_format()

        logger.info(f"System instruction length: {len(system_instruction)} chars")
        logger.info(f"Configured {len(tools[0]['function_declarations']) if tools else 0} tools")

        config_dict = {
            "response_modalities": ["AUDIO"],               # Note(David): see https://ai.google.dev/gemini-api/docs/live-guide?hl=en#response-modalities
            "system_instruction": system_instruction,
            "tools": tools,
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}  # Voices https://ai.google.dev/gemini-api/docs/speech-generation?hl=en#voices
            },
            "input_audio_transcription": {},   # Enable user speech transcription
            "output_audio_transcription": {},  # Enable assistant speech transcription
        }

        try:
            async with self.client.aio.live.connect(
                model=config.GEMINI_LIVE_MODEL_NAME,
                config=config_dict
            ) as session:
                self.session = session
                logger.info("Gemini Live session established")
                await self._run_session_loop()
        except Exception:
            logger.exception("Gemini Live session failed")
        finally:
            self.session = None


    async def _run_session_loop(self) -> None:
        """Process Gemini messages in main event loop."""
        # Keep the session alive indefinitely for multi-turn conversations
        while True:
            try:
                async for message in self.session.receive():
                    # Handle tool calls
                    if hasattr(message, 'tool_call') and message.tool_call:
                        await self._handle_tool_call(message.tool_call)

                    # Handle server content (audio, transcriptions, etc.)
                    if hasattr(message, 'server_content') and message.server_content:
                        await self._handle_server_content(message.server_content)

                # receive() completes after processing current messages, continue to next batch
                logger.debug("Session receive() batch completed, fetching next batch...")
                await asyncio.sleep(0.01)  # Small delay to avoid tight loop
            except asyncio.CancelledError:
                logger.info("Session loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Session loop error: {e}")
                raise


    async def _handle_tool_call(self, tool_call: Any) -> None:
        """Handle function calls from Gemini."""
        function_responses = []

        for fc in tool_call.function_calls:
            tool_name = fc.name
            args_json = json.dumps(fc.args)  # fc.args is dict

            logger.info(f"Tool call: {tool_name}")

            try:
                tool_result = await dispatch_tool_call(tool_name, args_json, self.deps)
                logger.debug(f"Tool '{tool_name}' result: {str(tool_result)[:500]}")   # truncate long logs
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {e}")
                tool_result = {"error": str(e)}

            # Prepare response:
            # Note(David): the b64_im is too large for tool response, so it will go later with _inject_camera_image
            response_to_send = tool_result.copy()
            camera_image_b64 = None
            if tool_name == "camera" and "b64_im" in response_to_send:
                camera_image_b64 = response_to_send.pop("b64_im")
                response_to_send["image_captured"] = True

            logger.debug(f"Tool response to Gemini: {str(response_to_send)[:200]}")

            # Build function response
            function_responses.append(
                self.types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response=response_to_send,
                )
            )

            # Send UI update (use original tool_result with image)
            await self.output_queue.put(
                AdditionalOutputs({
                    "role": "assistant",
                    "content": json.dumps(tool_result),
                    "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                })
            )

            # Send camera image
            if camera_image_b64:
                await self._inject_camera_image(camera_image_b64)
                await self._show_camera_image_in_ui()

        # Send function responses back (triggers model continuation)
        if function_responses:
            if not self.is_idle_tool_call:
                await self.session.send_tool_response(function_responses=function_responses)
            else:
                self.is_idle_tool_call = False

            # Re-sync head wobbler after tool execution
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()


    async def _inject_camera_image(self, b64_image: str) -> None:
        """Inject camera image into Gemini Live session since it is too big for tool response."""
        try:
            image_bytes = base64.b64decode(b64_image)
            image = PIL.Image.open(io.BytesIO(image_bytes))
            await self.session.send_realtime_input(media=image)
            logger.info("Camera image sent to Gemini")
        except Exception as e:
            logger.error(f"Failed to send camera image: {e}")


    async def _show_camera_image_in_ui(self) -> None:
        """Display camera image in the UI."""
        if self.deps.camera_worker is not None:
            np_img = self.deps.camera_worker.get_latest_frame()
            if np_img is not None:
                # Camera frames are BGR from OpenCV; convert so Gradio displays correct colors.
                rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = None
            img = gr.Image(value=rgb_frame)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": img})
            )


    async def _handle_server_content(self, server_content: Any) -> None:
        """Handle server content messages (audio, transcriptions)."""
        # Handle input transcription (user speech) - accumulate chunks
        if hasattr(server_content, 'input_transcription'):
            transcription = server_content.input_transcription
            if transcription and hasattr(transcription, 'text'):
                self.deps.movement_manager.set_listening(True)
                self.input_transcription_buffer += transcription.text

        # Handle output transcription (assistant speech) - accumulate chunks
        if hasattr(server_content, 'output_transcription'):
            transcription = server_content.output_transcription
            if transcription and hasattr(transcription, 'text'):
                self.output_transcription_buffer += transcription.text

        # Handle model turn (audio output)
        if hasattr(server_content, 'model_turn') and server_content.model_turn:
            for part in server_content.model_turn.parts:
                # Handle inline data (audio)
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    inline_data = part.inline_data
                    mime_type = getattr(inline_data, 'mime_type', None)
                    if mime_type and mime_type.startswith("audio/pcm"):
                        # Gemini sends audio as raw bytes
                        audio_bytes = inline_data.data
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)

                        # Feed to head wobbler
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed(base64.b64encode(audio_bytes).decode())

                        # Update activity time
                        self.last_activity_time = asyncio.get_event_loop().time()

                        # Send to output queue
                        await self.output_queue.put((self.output_sample_rate, audio_array))

        # Handle turn complete - flush transcription buffers
        if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
            self.deps.movement_manager.set_listening(False)

            # Flush input transcription buffer
            if self.input_transcription_buffer.strip():
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": self.input_transcription_buffer.strip()})
                )
                self.input_transcription_buffer = ""

            # Flush output transcription buffer
            if self.output_transcription_buffer.strip():
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": self.output_transcription_buffer.strip()})
                )
                self.output_transcription_buffer = ""

        # Handle interruption - clear buffers and queue
        if hasattr(server_content, 'interrupted') and server_content.interrupted:
            if hasattr(self, '_clear_queue') and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            # Clear transcription buffers on interruption
            self.input_transcription_buffer = ""
            self.output_transcription_buffer = ""


    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone and send to Gemini."""
        if not self.session:
            return

        _, array = frame
        array = array.squeeze()

        # Get raw PCM bytes & manually encode to Base64 string
        audio_bytes = array.tobytes()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        try:
            await self.session.send_realtime_input(
                audio={
                    "mime_type": "audio/pcm;rate=16000",
                    "data": b64_audio
                }
            )
        except Exception as e:
            logger.error(f"Failed to send audio to Gemini: {e}")
            raise


    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame or transcripts to be played/displayed."""
        # Handle idle detection
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning(f"Idle signal skipped: {e}")
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)


    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send idle signal to Gemini to trigger autonomous behavior."""
        self.is_idle_tool_call = True

        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - "
            f"No activity for {idle_duration:.1f}s] "
            "You've been idle for a while. Feel free to get creative - "
            "dance, show an emotion, look around, do nothing, or just be yourself!"
        )

        if not self.session:
            return

        # Send text message to trigger tool use
        await self.session.send(input=timestamp_msg, end_of_turn=True)


    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"


    async def shutdown(self) -> None:
        """Shutdown the handler."""
        if self.session:
            try:
                # Note: Gemini Live sessions auto-close on exit of async with block
                pass
            except Exception as e:
                logger.debug(f"Session cleanup error: {e}")
            finally:
                self.session = None

        # Clear transcription buffers
        self.input_transcription_buffer = ""
        self.output_transcription_buffer = ""

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
