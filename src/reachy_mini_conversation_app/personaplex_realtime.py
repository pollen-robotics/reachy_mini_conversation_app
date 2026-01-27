"""PersonaPlex realtime handler for Reachy Mini conversation app."""

import json
import base64
import asyncio
import logging
import websockets
from typing import Any, Final, Tuple, Literal, Optional
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# PersonaPlex/Moshi uses 24kHz audio
PERSONAPLEX_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
PERSONAPLEX_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class PersonaPlexHandler(AsyncStreamHandler):
    """A PersonaPlex/Moshi realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=PERSONAPLEX_OUTPUT_SAMPLE_RATE,
            input_sample_rate=PERSONAPLEX_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        self.websocket: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # PersonaPlex specific
        self.server_url = config.PERSONAPLEX_SERVER_URL
        self.current_persona: str | None = None
        self.current_voice_prompt: str | None = None

    def copy(self) -> "PersonaPlexHandler":
        """Create a copy of the handler."""
        return PersonaPlexHandler(self.deps, self.gradio_mode, self.instance_path)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime.

        Updates the persona configuration and restarts the session if connected.

        Returns a short status message for UI feedback.
        """
        try:
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            # Get persona instructions
            try:
                instructions = get_session_instructions()
                self.current_persona = instructions
            except BaseException as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Restart session if connected
            if self.websocket is not None:
                try:
                    await self._restart_session()
                    return "Applied personality and restarted PersonaPlex session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with connection to Moshi server."""
        logger.info(f"Connecting to PersonaPlex/Moshi server at {self.server_url}")

        # Load persona instructions
        self.current_persona = get_session_instructions()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_moshi_session()
                return
            except ConnectionClosedError as e:
                logger.warning("Moshi websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    delay = 2 ** (attempt - 1)  # exponential backoff
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                self.websocket = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one."""
        try:
            if self.websocket is not None:
                try:
                    await self.websocket.close()
                except Exception:
                    pass
                finally:
                    self.websocket = None

            try:
                self._connected_event.clear()
            except Exception:
                pass

            asyncio.create_task(self._run_moshi_session(), name="personaplex-moshi-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Moshi session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Moshi session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _run_moshi_session(self) -> None:
        """Establish and manage a single Moshi session."""
        try:
            async with websockets.connect(self.server_url) as ws:
                self.websocket = ws
                self._connected_event.set()
                logger.info("Connected to Moshi server")

                # Send initial configuration if needed
                await self._send_config()

                # Start receiving task
                receive_task = asyncio.create_task(self._receive_loop())

                try:
                    await receive_task
                except asyncio.CancelledError:
                    logger.debug("Receive loop cancelled")
                    raise

        except Exception as e:
            logger.error(f"Moshi session error: {e}")
            raise

    async def _send_config(self) -> None:
        """Send initial configuration to Moshi server."""
        config_msg = {
            "type": "config",
            "persona": self.current_persona or "You are a helpful robot assistant.",
            "sample_rate": self.input_sample_rate,
        }
        await self.websocket.send(json.dumps(config_msg))
        logger.info("Sent configuration to Moshi server")

    async def _receive_loop(self) -> None:
        """Receive and process messages from Moshi server."""
        async for message in self.websocket:
            try:
                if isinstance(message, bytes):
                    # Binary message - audio data
                    await self._handle_audio_data(message)
                else:
                    # Text message - JSON event
                    event = json.loads(message)
                    await self._handle_event(event)
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _handle_audio_data(self, audio_bytes: bytes) -> None:
        """Handle incoming audio data from Moshi."""
        self.last_activity_time = asyncio.get_event_loop().time()

        # Convert bytes to int16 audio
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)

        # Feed to head wobbler if available
        if self.deps.head_wobbler is not None:
            # Convert to base64 for head wobbler (mimicking OpenAI format)
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            self.deps.head_wobbler.feed(b64_audio)

        # Queue for output
        await self.output_queue.put((self.output_sample_rate, audio_array))

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Handle JSON events from Moshi server."""
        event_type = event.get("type")
        logger.debug(f"Moshi event: {event_type}")

        if event_type == "speech_started":
            # User started speaking
            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            logger.debug("User speech started")

        elif event_type == "speech_stopped":
            # User stopped speaking
            self.deps.movement_manager.set_listening(False)
            logger.debug("User speech stopped")

        elif event_type == "transcript_partial":
            # Partial transcript (user speaking)
            transcript = event.get("text", "")
            logger.debug(f"User partial transcript: {transcript}")

            self.partial_transcript_sequence += 1
            current_sequence = self.partial_transcript_sequence

            if self.partial_transcript_task and not self.partial_transcript_task.done():
                self.partial_transcript_task.cancel()
                try:
                    await self.partial_transcript_task
                except asyncio.CancelledError:
                    pass

            self.partial_transcript_task = asyncio.create_task(
                self._emit_debounced_partial(transcript, current_sequence)
            )

        elif event_type == "transcript_complete":
            # Complete transcript (user finished speaking)
            transcript = event.get("text", "")
            logger.debug(f"User transcript: {transcript}")

            if self.partial_transcript_task and not self.partial_transcript_task.done():
                self.partial_transcript_task.cancel()
                try:
                    await self.partial_transcript_task
                except asyncio.CancelledError:
                    pass

            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        elif event_type == "response_transcript":
            # Assistant transcript
            transcript = event.get("text", "")
            logger.debug(f"Assistant transcript: {transcript}")
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": transcript}))

        elif event_type == "tool_call":
            # Tool call request from Moshi
            await self._handle_tool_call(event)

        elif event_type == "error":
            error_msg = event.get("message", "Unknown error")
            logger.error(f"Moshi error: {error_msg}")
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {error_msg}"}))

    async def _handle_tool_call(self, event: dict[str, Any]) -> None:
        """Handle tool call from Moshi server."""
        tool_name = event.get("name")
        args_json = event.get("arguments", "{}")
        call_id = event.get("call_id")

        if not tool_name:
            logger.error("Invalid tool call: no tool name")
            return

        try:
            tool_result = await dispatch_tool_call(tool_name, args_json, self.deps)
            logger.debug(f"Tool '{tool_name}' executed successfully")
            logger.debug(f"Tool result: {tool_result}")
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            tool_result = {"error": str(e)}

        # Send tool result back to Moshi
        result_msg = {
            "type": "tool_result",
            "call_id": call_id,
            "result": json.dumps(tool_result),
        }
        await self.websocket.send(json.dumps(result_msg))

        # Display tool result in UI
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(tool_result),
                    "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                },
            ),
        )

        # Handle camera tool specially
        if tool_name == "camera" and "b64_im" in tool_result:
            b64_im = tool_result["b64_im"]
            if not isinstance(b64_im, str):
                logger.warning(f"Unexpected type for b64_im: {type(b64_im)}")
                b64_im = str(b64_im)

            # Send image to Moshi
            image_msg = {
                "type": "image",
                "data": f"data:image/jpeg;base64,{b64_im}",
            }
            await self.websocket.send(json.dumps(image_msg))
            logger.info("Added camera image to conversation")

            # Display image in Gradio
            if self.deps.camera_worker is not None:
                np_img = self.deps.camera_worker.get_latest_frame()
                if np_img is not None:
                    rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = None
                img = gr.Image(value=rgb_frame)

                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": img,
                        },
                    ),
                )

        # Re-sync head wobble after tool call
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to Moshi server."""
        if not self.websocket:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast to int16
        audio_frame = audio_to_int16(audio_frame)

        # Send to Moshi as binary data
        try:
            await self.websocket.send(audio_frame.tobytes())
        except Exception as e:
            logger.debug(f"Dropping audio frame: connection not ready ({e})")
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning(f"Idle signal skipped (connection closed?): {e}")
                return None

            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            try:
                await self.websocket.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"websocket.close() ignored: {e}")
            finally:
                self.websocket = None

        # Clear output queue
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
        """Send an idle signal to trigger robot behavior."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True

        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"

        if not self.websocket:
            logger.debug("No connection, cannot send idle signal")
            return

        idle_msg = {
            "type": "idle_signal",
            "message": timestamp_msg,
        }
        await self.websocket.send(json.dumps(idle_msg))
