"""Gemini Live API handler for real-time audio conversation.

Drop-in alternative to OpenaiRealtimeHandler. Uses the google-genai SDK's
Live API for bidirectional audio streaming with function calling support.

Audio formats (per Gemini Live API spec):
  Input:  16-bit PCM, 16 kHz, mono
  Output: 16-bit PCM, 24 kHz, mono
"""

import json
import uuid
import base64
import random
import asyncio
import logging
from typing import Any, Dict, List, Final, Tuple, Literal, Optional

import numpy as np
from google import genai
from fastrtc import AdditionalOutputs, audio_to_int16
from google.genai import types
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import (
    GEMINI_BACKEND,
    GEMINI_AVAILABLE_VOICES,
    DEFAULT_VOICE_BY_BACKEND,
    config,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_instructions,
    get_session_greeting_prompt,
)
from reachy_mini_conversation_app.tools.core_tools import (
    ToolSpec,
    ToolDependencies,
    initialize_tools,
)
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.camera_frame_encoding import encode_bgr_frame_as_jpeg
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000
GEMINI_OUTPUT_SAMPLE_RATE: Final[int] = 24000


def _openai_tool_specs_to_gemini(specs: list[ToolSpec]) -> List[Dict[str, Any]]:
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
            "description": spec["description"],
        }
        if spec["parameters"]:
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
    return voice_map.get(profile_voice.lower(), DEFAULT_VOICE_BY_BACKEND[GEMINI_BACKEND])


def _resolve_gemini_startup_voice(voice: str | None) -> str | None:
    """Return a valid persisted Gemini startup voice or None."""
    if voice is None:
        return None

    voice_map = {candidate.lower(): candidate for candidate in GEMINI_AVAILABLE_VOICES}
    resolved = voice_map.get(voice.lower())
    if resolved is None:
        logger.warning(
            "Ignoring persisted Gemini startup voice %r; expected one of %s",
            voice,
            GEMINI_AVAILABLE_VOICES,
        )
    return resolved


class GeminiLiveHandler(ConversationHandler):
    """Gemini Live API handler for fastrtc Stream."""

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.instance_path = instance_path
        self._voice_override: str | None = _resolve_gemini_startup_voice(startup_voice)

        self.session: Any = None  # google.genai live session
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        # Track API key source (env vs textbox)
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Stop event for the receive loop
        self._stop_event: asyncio.Event = asyncio.Event()
        self._pending_user_transcript_chunks: list[str] = []
        self._pending_assistant_transcript_chunks: list[str] = []
        self._listening_state = False
        self._startup_greeting_sent = False

    def copy(self) -> "GeminiLiveHandler":
        """Return a fresh handler, preserving deps and voice override."""
        return GeminiLiveHandler(
            self.deps,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    def _set_listening_state(self, listening: bool) -> None:
        """Avoid queueing redundant listening-state updates."""
        if self._listening_state == listening:
            return
        self._listening_state = listening
        self.deps.movement_manager.set_listening(listening)

    async def _flush_transcript_chunks(self, role: str, chunks: list[str]) -> None:
        """Emit one finalized transcript message for the current turn."""
        if not chunks:
            return

        transcript = "".join(chunks).strip()
        chunks.clear()
        if not transcript:
            return

        await self.output_queue.put(AdditionalOutputs({"role": role, "content": transcript}))

    def _is_connected(self) -> bool:
        """Return whether the Gemini Live session is open."""
        return self.session is not None

    async def _mark_model_response_started(self) -> None:
        """Switch out of user-listening mode when the model begins responding."""
        await self._flush_transcript_chunks("user", self._pending_user_transcript_chunks)
        self._set_listening_state(False)
        self._mark_activity("response_created")

    async def _handle_interruption(self) -> None:
        """Stop current playback and preserve any transcript already spoken."""
        logger.debug("Gemini: user interrupted")
        await self._flush_transcript_chunks("assistant", self._pending_assistant_transcript_chunks)
        if self._clear_queue:
            self._clear_queue()
        self._set_listening_state(True)

    async def _handle_turn_complete(self) -> None:
        """Finalize the current turn and restore post-speech motion state."""
        logger.debug("Gemini turn complete")
        await self._flush_transcript_chunks("user", self._pending_user_transcript_chunks)
        await self._flush_transcript_chunks("assistant", self._pending_assistant_transcript_chunks)
        self._set_listening_state(False)

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
                _ = get_session_instructions(self.instance_path)
                _ = get_session_voice()
            except BaseException as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Rebuild the tool registry
            initialize_tools(force=True)

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

    async def change_voice(self, voice: str) -> str:
        """Change only the voice and restart the session."""
        self._voice_override = voice
        if getattr(self, "client", None) is not None:
            try:
                await self._restart_session()
                return f"Voice changed to {voice}."
            except Exception as e:
                logger.warning("Failed to restart session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the resolved Gemini voice currently selected for this handler."""
        return _resolve_gemini_voice(self._voice_override or get_session_voice())

    async def start_up(self) -> None:
        """Start the handler with retries on unexpected closure."""
        gemini_api_key = config.GEMINI_API_KEY
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
            asyncio.create_task(self.start_up(), name="gemini-live-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Gemini Live session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Gemini Live session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    def _build_live_config(self) -> types.LiveConnectConfig:
        """Build the LiveConnectConfig for a Gemini Live session."""
        instructions = get_session_instructions(self.instance_path)
        voice = _resolve_gemini_voice(self._voice_override or get_session_voice())

        # Convert OpenAI-style tool specs to Gemini function declarations
        tool_specs = self._get_active_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
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
                "Gemini tool call: tool_name=%r, call_id=%s, args=%s",
                tool_name,
                call_id,
                args_json_str,
            )

            background_tool = await self.tool_manager.start_tool(
                call_id=call_id,
                tool_call_routine=ToolCallRoutine(
                    tool_name=tool_name,
                    args_json_str=args_json_str,
                    deps=self.deps,
                ),
                is_idle_tool_call=False,
            )

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. Tool ID: {background_tool.tool_id}",
                    },
                ),
            )

            logger.info("Started background tool: %s (id=%s, call_id=%s)", tool_name, background_tool.tool_id, call_id)

    async def _send_startup_greeting_prompt(self) -> None:
        """Prompt Gemini to open the conversation once the live session is ready."""
        if self._startup_greeting_sent or not self.session:
            return

        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            self._startup_greeting_sent = True
            return

        send_client_content = getattr(self.session, "send_client_content", None)
        if not callable(send_client_content):
            self._startup_greeting_sent = True
            logger.warning("Gemini session does not support send_client_content; startup greeting skipped")
            return

        try:
            await send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=greeting_prompt)],
                ),
                turn_complete=True,
            )
            self._startup_greeting_sent = True
            self._mark_activity("startup_greeting_prompt")
            logger.info("Queued Gemini startup greeting prompt")
        except Exception as e:
            logger.warning("Failed to queue Gemini startup greeting prompt: %s", e)

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Process the result of a completed tool and send it back to Gemini."""
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed: %s", completed_tool.tool_name, completed_tool.id, completed_tool.error
            )
            tool_result = {"error": completed_tool.error}
        elif completed_tool.result is not None:
            tool_result = completed_tool.result
            logger.info("Tool '%s' (id=%s) succeeded.", completed_tool.tool_name, completed_tool.id)
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            tool_result = {"error": "No result returned from tool execution"}

        if not self.session:
            logger.warning("Connection closed during tool '%s' execution", completed_tool.tool_name)
            return

        try:
            send_result_to_model = not completed_tool.is_idle_tool_call

            if (
                send_result_to_model
                and completed_tool.tool_name == "camera"
                and isinstance(tool_result, dict)
                and "b64_im" in tool_result
            ):
                b64_im = tool_result.pop("b64_im")
                if not tool_result:
                    tool_result = {"status": "image_captured"}

                try:
                    if isinstance(b64_im, str):
                        image_bytes = base64.b64decode(b64_im)
                    else:
                        image_bytes = bytes(b64_im)
                    await self.session.send_realtime_input(video=types.Blob(data=image_bytes, mime_type="image/jpeg"))
                    logger.info("Pushed camera snapshot to Gemini via realtime video input")
                except Exception as ve:
                    logger.warning("Failed to push camera snapshot to Gemini: %s", ve)

            console_content = json.dumps(tool_result)

            if send_result_to_model:
                self._mark_activity("tool_result_ready")
                function_response = types.FunctionResponse(
                    id=completed_tool.id if isinstance(completed_tool.id, str) else str(completed_tool.id),
                    name=completed_tool.tool_name,
                    response=tool_result,
                )
                await self.session.send_tool_response(function_responses=[function_response])

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": console_content,
                    },
                ),
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
                        jpeg_bytes = encode_bgr_frame_as_jpeg(frame)
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

                await self._send_startup_greeting_prompt()

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
                                    await self._handle_interruption()

                                # Handle audio output from model
                                if content.model_turn and content.model_turn.parts:
                                    has_audio_part = any(
                                        part.inline_data and part.inline_data.data for part in content.model_turn.parts
                                    )
                                    if has_audio_part:
                                        await self._mark_model_response_started()

                                    for part in content.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            audio_bytes = part.inline_data.data
                                            if isinstance(audio_bytes, str):
                                                audio_bytes = base64.b64decode(audio_bytes)

                                            if len(audio_bytes) == 0:
                                                continue

                                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                                            if len(audio_array) == 0:
                                                continue

                                            self._mark_activity("assistant_audio_delta")

                                            await self.output_queue.put(
                                                (GEMINI_OUTPUT_SAMPLE_RATE, audio_array),
                                            )

                                # Handle input transcription (user speech)
                                if content.input_transcription and content.input_transcription.text:
                                    transcript = content.input_transcription.text
                                    logger.debug("User transcript chunk: %s", transcript)
                                    self._pending_user_transcript_chunks.append(transcript)
                                    self._set_listening_state(True)
                                    self._mark_activity("user_transcription_delta")

                                # Handle output transcription (model speech)
                                if content.output_transcription and content.output_transcription.text:
                                    transcript = content.output_transcription.text
                                    logger.debug("Assistant transcript chunk: %s", transcript)
                                    await self._mark_model_response_started()
                                    self._pending_assistant_transcript_chunks.append(transcript)

                                # Turn complete
                                if content.turn_complete:
                                    self._mark_activity("assistant_transcript_done")
                                    await self._handle_turn_complete()

                            # Handle tool calls
                            if response.tool_call:
                                self._mark_activity("tool_call_received")
                                await self._handle_tool_call(response)

                    except Exception as e:
                        if self._stop_event.is_set():
                            break
                        logger.warning("Receive loop error, restarting Gemini session: %s", e)
                        raise

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

    async def get_available_voices(self) -> list[str]:
        """Return the list of available Gemini voices."""
        return list(GEMINI_AVAILABLE_VOICES)
