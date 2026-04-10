"""Gemini Live API handler for real-time audio conversation.

Drop-in alternative to OpenaiRealtimeHandler. Uses the google-genai SDK's
Live API for bidirectional audio streaming with function calling support.

Audio formats (per Gemini Live API spec):
  Input:  16-bit PCM, 16 kHz, mono
  Output: 16-bit PCM, 24 kHz, mono
"""

import json
import uuid
import random
import asyncio
import logging
from typing import Any, Dict, List, Final, Tuple, Literal, Optional
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from google import genai
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from google.genai import types
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000
GEMINI_OUTPUT_SAMPLE_RATE: Final[int] = 24000

# Voices supported by Gemini Live API
GEMINI_AVAILABLE_VOICES: List[str] = [
    "Aoede",
    "Charon",
    "Fenrir",
    "Kore",
    "Leda",
    "Orus",
    "Puck",
    "Zephyr",
]

DEFAULT_GEMINI_VOICE = "Kore"


def _openai_tool_specs_to_gemini(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool specs to Gemini function_declarations format.

    OpenAI format:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}

    Gemini format:
        {"name": "...", "description": "...", "parameters": {...}}

    The parameters schema is mostly compatible (JSON Schema), but Gemini uses
    uppercase type names (STRING, NUMBER, OBJECT, ARRAY, BOOLEAN, INTEGER).
    """
    declarations = []
    for spec in specs:
        decl: Dict[str, Any] = {
            "name": spec["name"],
        }
        if "description" in spec:
            decl["description"] = spec["description"]
        if "parameters" in spec and spec["parameters"]:
            decl["parameters"] = _convert_schema_types(spec["parameters"])
        declarations.append(decl)
    return declarations


def _convert_schema_types(schema: Any) -> Any:
    """Recursively convert JSON Schema type strings to Gemini uppercase format."""
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)

    # Convert type field to uppercase
    if "type" in result:
        type_map = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }
        t = result["type"]
        if isinstance(t, str):
            result["type"] = type_map.get(t.lower(), t.upper())

    # Recurse into properties
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {k: _convert_schema_types(v) for k, v in result["properties"].items()}

    # Recurse into items (for arrays)
    if "items" in result:
        result["items"] = _convert_schema_types(result["items"])

    # Remove fields not supported by Gemini
    for unsupported_key in ("additionalProperties",):
        result.pop(unsupported_key, None)

    return result


def _resolve_gemini_voice(profile_voice: str) -> str:
    """Map a profile voice name to the closest Gemini voice.

    If the voice is already a valid Gemini voice (case-insensitive), use it.
    Otherwise fall back to the default.
    """
    voice_map = {v.lower(): v for v in GEMINI_AVAILABLE_VOICES}
    return voice_map.get(profile_voice.lower(), DEFAULT_GEMINI_VOICE)


class GeminiLiveHandler(AsyncStreamHandler):
    """Gemini Live API handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.session: Any = None  # google.genai live session
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Track API key source (env vs textbox)
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Stop event for the receive loop
        self._stop_event: asyncio.Event = asyncio.Event()

    def copy(self) -> "GeminiLiveHandler":
        """Create a copy of the handler."""
        return GeminiLiveHandler(self.deps, self.gradio_mode, self.instance_path)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime.

        For Gemini Live, we must restart the session since there's no
        session.update equivalent.
        """
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info("Set custom profile to %r", profile)

            try:
                _ = get_session_instructions()
                _ = get_session_voice()
            except BaseException as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Force a restart to apply new config
            if self.session is not None:
                try:
                    await self._restart_session()
                    return "Applied personality and restarted Gemini session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def start_up(self) -> None:
        """Start the handler with retries on unexpected closure."""
        gemini_api_key = config.GEMINI_API_KEY
        if self.gradio_mode and not gemini_api_key:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
            if textbox_api_key is not None:
                gemini_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                gemini_api_key = config.GEMINI_API_KEY
        else:
            if not gemini_api_key or not gemini_api_key.strip():
                logger.warning("GEMINI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                gemini_api_key = "DUMMY"

        self.client = genai.Client(api_key=gemini_api_key)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_live_session()
                return
            except Exception as e:
                logger.warning(
                    "Gemini Live session closed unexpectedly (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    base_delay = 2 ** (attempt - 1)
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                self.session = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one."""
        try:
            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    pass
                finally:
                    self.session = None

            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: Gemini client not initialized yet.")
                return

            try:
                self._connected_event.clear()
            except Exception:
                pass
            self._stop_event.set()  # Signal the old receive loop to stop
            await asyncio.sleep(0.1)
            self._stop_event.clear()
            asyncio.create_task(self._run_live_session(), name="gemini-live-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Gemini Live session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Gemini Live session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    def _build_live_config(self) -> types.LiveConnectConfig:
        """Build the LiveConnectConfig for a Gemini Live session."""
        instructions = get_session_instructions()
        voice = _resolve_gemini_voice(get_session_voice())

        # Convert OpenAI-style tool specs to Gemini function declarations
        tool_specs = get_tool_specs()
        function_declarations = _openai_tool_specs_to_gemini(tool_specs)

        tools_config: List[Dict[str, Any]] = []
        if function_declarations:
            tools_config.append({"function_declarations": function_declarations})

        live_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(parts=[types.Part(text=instructions)]),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    ),
                ),
            ),
            tools=tools_config,  # type: ignore[arg-type]
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        logger.info(
            "Gemini Live config: model=%r voice=%r tools=%d",
            config.MODEL_NAME,
            voice,
            len(function_declarations),
        )
        return live_config

    async def _handle_tool_call(self, response: Any) -> None:
        """Process a tool_call from Gemini and send the response back."""
        if not response.tool_call or not response.tool_call.function_calls:
            return

        for fc in response.tool_call.function_calls:
            tool_name = fc.name
            call_id = fc.id or str(uuid.uuid4())
            args_dict = dict(fc.args) if fc.args else {}
            args_json_str = json.dumps(args_dict)

            logger.info(
                "Gemini tool call: tool_name=%r, call_id=%s, is_idle=%s, args=%s",
                tool_name,
                call_id,
                self.is_idle_tool_call,
                args_json_str,
            )

            bg_tool = await self.tool_manager.start_tool(
                call_id=call_id,
                tool_call_routine=ToolCallRoutine(
                    tool_name=tool_name,
                    args_json_str=args_json_str,
                    deps=self.deps,
                ),
                is_idle_tool_call=self.is_idle_tool_call,
            )

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. Tool ID: {bg_tool.tool_id}",
                    },
                ),
            )

            if self.is_idle_tool_call:
                self.is_idle_tool_call = False

            logger.info("Started background tool: %s (id=%s, call_id=%s)", tool_name, bg_tool.tool_id, call_id)

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Process the result of a completed tool and send it back to Gemini."""
        if bg_tool.error is not None:
            logger.error("Tool '%s' (id=%s) failed: %s", bg_tool.tool_name, bg_tool.id, bg_tool.error)
            tool_result = {"error": bg_tool.error}
        elif bg_tool.result is not None:
            tool_result = bg_tool.result
            logger.info("Tool '%s' (id=%s) succeeded.", bg_tool.tool_name, bg_tool.id)
        else:
            logger.warning("Tool '%s' (id=%s) returned no result and no error", bg_tool.tool_name, bg_tool.id)
            tool_result = {"error": "No result returned from tool execution"}

        if not self.session:
            logger.warning("Connection closed during tool '%s' execution", bg_tool.tool_name)
            return

        try:
            # Send tool response back to Gemini
            function_response = types.FunctionResponse(
                id=bg_tool.id if isinstance(bg_tool.id, str) else str(bg_tool.id),
                name=bg_tool.tool_name,
                response=tool_result,
            )
            await self.session.send_tool_response(function_responses=[function_response])

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result),
                        "metadata": {
                            "title": f"🛠️ Used tool {bg_tool.tool_name}",
                            "status": "done",
                        },
                    },
                ),
            )

            # Handle camera images
            if bg_tool.tool_name == "camera" and bg_tool.result and "b64_im" in bg_tool.result:
                b64_im = bg_tool.result["b64_im"]
                if not isinstance(b64_im, str):
                    b64_im = str(b64_im)

                # Send image via realtime input as video frame
                import base64

                image_bytes = base64.b64decode(b64_im)
                await self.session.send_realtime_input(video=types.Blob(data=image_bytes, mime_type="image/jpeg"))
                logger.info("Sent camera image to Gemini via video input")

                if self.deps.camera_worker is not None:
                    np_img = self.deps.camera_worker.get_latest_frame()
                    if np_img is not None:
                        rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = None
                    img = gr.Image(value=rgb_frame)
                    await self.output_queue.put(
                        AdditionalOutputs({"role": "assistant", "content": img}),
                    )

        except Exception as e:
            logger.warning("Error sending tool result to Gemini: %s", e)

    async def _video_sender_loop(self) -> None:
        """Send camera frames to Gemini Live at ~1 FPS for continuous visual context.

        Only runs when a camera_worker is available. Frames are JPEG-encoded
        and sent via send_realtime_input(video=...).
        """
        logger.info("Video sender loop started (1 FPS)")
        while not self._stop_event.is_set():
            try:
                if self.session and self.deps.camera_worker is not None:
                    frame = self.deps.camera_worker.get_latest_frame()
                    if frame is not None:
                        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if success:
                            jpeg_bytes = buffer.tobytes()
                            await self.session.send_realtime_input(
                                video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                            )
            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.debug("Video sender error (will retry): %s", e)

            await asyncio.sleep(1.0)  # 1 FPS

        logger.info("Video sender loop stopped")

    async def _run_live_session(self) -> None:
        """Establish and manage a single Gemini Live session."""
        live_config = self._build_live_config()

        async with self.client.aio.live.connect(
            model=config.MODEL_NAME,
            config=live_config,
        ) as session:
            self.session = session
            try:
                self._connected_event.set()
            except Exception:
                pass

            logger.info("Gemini Live session connected successfully")

            video_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start video sender if camera is available
                if self.deps.camera_worker is not None:
                    video_task = asyncio.create_task(self._video_sender_loop(), name="gemini-video-sender")

                # session.receive() yields responses for the current turn then completes.
                # We loop so the session stays alive across multiple conversation turns.
                while not self._stop_event.is_set():
                    try:
                        async for response in session.receive():
                            if self._stop_event.is_set():
                                logger.info("Stop event set, breaking receive loop")
                                break

                            # Handle server content (audio, transcription, interruption)
                            if response.server_content:
                                content = response.server_content

                                # Handle interruption / barge-in
                                if content.interrupted is True:
                                    logger.debug("Gemini: user interrupted")
                                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                                        self._clear_queue()
                                    if self.deps.head_wobbler is not None:
                                        self.deps.head_wobbler.reset()

                                # Handle audio output from model
                                if content.model_turn and content.model_turn.parts:
                                    for part in content.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            audio_bytes = part.inline_data.data
                                            if isinstance(audio_bytes, str):
                                                import base64

                                                audio_bytes = base64.b64decode(audio_bytes)

                                            if len(audio_bytes) == 0:
                                                continue

                                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                                            if len(audio_array) == 0:
                                                continue

                                            if self.deps.head_wobbler is not None:
                                                import base64 as b64mod

                                                self.deps.head_wobbler.feed(
                                                    b64mod.b64encode(audio_bytes).decode("utf-8")
                                                )

                                            self.last_activity_time = asyncio.get_event_loop().time()

                                            await self.output_queue.put(
                                                (GEMINI_OUTPUT_SAMPLE_RATE, audio_array),
                                            )

                                # Handle input transcription (user speech)
                                if content.input_transcription and content.input_transcription.text:
                                    transcript = content.input_transcription.text
                                    logger.debug("User transcript: %s", transcript)
                                    await self.output_queue.put(
                                        AdditionalOutputs({"role": "user", "content": transcript})
                                    )

                                # Handle output transcription (model speech)
                                if content.output_transcription and content.output_transcription.text:
                                    transcript = content.output_transcription.text
                                    logger.debug("Assistant transcript: %s", transcript)
                                    await self.output_queue.put(
                                        AdditionalOutputs({"role": "assistant", "content": transcript})
                                    )

                                # Turn complete
                                if content.turn_complete:
                                    logger.debug("Gemini turn complete")
                                    self.deps.movement_manager.set_listening(True)

                            # Handle tool calls
                            if response.tool_call:
                                await self._handle_tool_call(response)

                    except Exception as e:
                        if self._stop_event.is_set():
                            break
                        logger.warning("Receive loop error (will continue): %s", e)
                        await asyncio.sleep(0.1)

            finally:
                if video_task is not None:
                    video_task.cancel()
                    try:
                        await video_task
                    except asyncio.CancelledError:
                        pass
                await self.tool_manager.shutdown()

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone and send to Gemini."""
        if not self.session:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample to 16kHz if needed
        if GEMINI_INPUT_SAMPLE_RATE != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sample_rate),
            )

        # Cast to int16
        audio_frame = audio_to_int16(audio_frame)

        # Send raw PCM bytes to Gemini
        try:
            pcm_bytes = audio_frame.tobytes()
            await self.session.send_realtime_input(audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000"))
        except Exception as e:
            logger.debug("Dropping audio frame: session not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._stop_event.set()

        await self.tool_manager.shutdown()

        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.debug("session.close() error: %s", e)
            finally:
                self.session = None

        # Clear remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to Gemini."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            "You've been idle for a while. Feel free to get creative - dance, show an emotion, "
            "look around, do nothing, or just be yourself!"
        )
        if not self.session:
            logger.debug("No session, cannot send idle signal")
            return

        await self.session.send_realtime_input(text=timestamp_msg)

    async def get_available_voices(self) -> list[str]:
        """Return the list of available Gemini voices."""
        return list(GEMINI_AVAILABLE_VOICES)
