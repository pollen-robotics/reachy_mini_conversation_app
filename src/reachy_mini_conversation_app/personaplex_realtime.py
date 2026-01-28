"""PersonaPlex realtime handler for Reachy Mini conversation app."""

import json
import base64
import asyncio
import logging
import websockets
from typing import Any, Final, Tuple, Literal, Optional
from datetime import datetime
from urllib.parse import quote

import io
import cv2
import opuslib
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
        self.partial_debounce_delay = 0.0  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()
        self._handshake_received: asyncio.Event = asyncio.Event()

        # PersonaPlex specific
        self.server_url = config.PERSONAPLEX_SERVER_URL
        self.current_persona: str | None = None
        self.current_voice_prompt: str | None = None

        # Opus encoder/decoder for audio (24kHz mono, 80ms frame size)
        # Mimi uses 12.5Hz frame rate = 80ms frames = 1920 samples at 24kHz
        self.opus_frame_size = 1920  # 80ms frames at 24kHz (matches Mimi's 12.5Hz frame rate)

        # Use opuslib for raw Opus encoding and decoding
        self.opus_encoder = opuslib.Encoder(24000, 1, opuslib.APPLICATION_AUDIO)
        try:
            self.opus_encoder.bitrate = 128000  # 128 kbps
        except Exception:
            pass  # Use default if setting fails

        # Use opuslib for decoding raw Opus packets from server
        self.opus_decoder = opuslib.Decoder(24000, 1)

        # Buffer for accumulating audio before encoding
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_samples = 0
        self.target_buffer_samples = 1920  # Encode in 80ms chunks

        # Latency tracking (similar to web client)
        self.total_audio_sent = 0.0  # Total seconds of audio sent to server
        self.total_audio_received = 0.0  # Total seconds of audio received from server
        self.last_latency_log_time = 0.0

    def copy(self) -> "PersonaPlexHandler":
        """Create a copy of the handler."""
        return PersonaPlexHandler(self.deps, self.gradio_mode, self.instance_path)

    def _ms_to_samples(self, ms: float) -> int:
        """Convert milliseconds to samples at 24kHz."""
        return int(ms * 24000 / 1000)

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
                self._handshake_received.clear()
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
            # Add required query parameters: voice_prompt and text_prompt
            # voice_prompt: select voice (NATF2 - natural female voice)
            # text_prompt: persona/system instructions (must be URL-encoded)
            persona = self.current_persona or "You are Reachy Mini, a friendly, compact robot assistant with a calm voice and a subtle sense of humor. Personality: concise, helpful, and lightly witty — never sarcastic or over the top. You speak English by default and switch languages only if explicitly told."
            ws_url = f"{self.server_url}?voice_prompt=NATF2.pt&text_prompt={quote(persona)}"
            async with websockets.connect(ws_url) as ws:
                self.websocket = ws
                self._connected_event.set()
                logger.info("Connected to Moshi server with persona and voice")

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

    async def _receive_loop(self) -> None:
        """Receive and process messages from Moshi server."""
        async for message in self.websocket:
            try:
                if isinstance(message, bytes):
                    # Binary message - check type byte
                    if len(message) == 0:
                        continue

                    msg_type = message[0]
                    payload = message[1:]

                    if msg_type == 0x00:
                        # Handshake - signal that we can now send audio
                        logger.info("Received handshake from Moshi server - ready to send audio")
                        self._handshake_received.set()
                    elif msg_type == 0x01:
                        # Audio data (Opus-encoded)
                        await self._handle_audio_data(payload)
                    elif msg_type == 0x02:
                        # Text tokens (UTF-8)
                        await self._handle_text_tokens(payload)
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                else:
                    # Text message - JSON event (if server sends any)
                    event = json.loads(message)
                    await self._handle_event(event)
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _handle_audio_data(self, opus_bytes: bytes) -> None:
        """Handle incoming raw Opus-encoded audio data from Moshi."""
        self.last_activity_time = asyncio.get_event_loop().time()

        # Decode raw Opus packets using opuslib (matching the modified server)
        try:
            logger.debug(f"Received Opus packet: {len(opus_bytes)} bytes")

            # The server sends concatenated raw Opus packets
            # We need to decode all of them
            # Opus packets can vary in size, but we'll try to decode the entire buffer

            # Decode the raw Opus packet
            # Use a large frame size to handle variable packet lengths
            pcm_int16_bytes = self.opus_decoder.decode(opus_bytes, 5760, decode_fec=False)

            # Convert bytes to int16 array
            audio_int16 = np.frombuffer(pcm_int16_bytes, dtype=np.int16)

            # Check if we got data
            if len(audio_int16) == 0:
                logger.debug("No PCM data decoded from Opus packet")
                return

            # Ensure shape is (1, samples) for mono
            audio_array = audio_int16.reshape(1, -1)

            logger.debug(f"Decoded to {audio_array.shape[1]} samples")

        except Exception as e:
            logger.error(f"Failed to decode Opus audio: {e}, packet size: {len(opus_bytes)} bytes")
            # Return silence to prevent ZeroDivisionError in playback
            audio_array = np.zeros((1, 1920), dtype=np.int16)

        # Track received audio for latency measurement
        audio_duration = audio_array.shape[1] / 24000.0  # seconds
        self.total_audio_received += audio_duration

        # Calculate and log latency periodically
        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_latency_log_time > 2.0:  # Log every 2 seconds
            queue_size = self.output_queue.qsize()

            # Calculate buffered audio time (how much audio is waiting to play)
            # Each packet in queue represents ~80ms of audio
            buffered_seconds = queue_size * 0.08  # approximate

            logger.info(
                f"Audio Buffer: {buffered_seconds:.3f}s ({queue_size} packets) | "
                f"Total: sent={self.total_audio_sent:.1f}s, received={self.total_audio_received:.1f}s"
            )
            self.last_latency_log_time = current_time

        # Feed to head wobbler if available
        if self.deps.head_wobbler is not None:
            # Convert to base64 for head wobbler (mimicking OpenAI format)
            pcm_bytes = audio_array.tobytes()
            b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
            self.deps.head_wobbler.feed(b64_audio)

        # Queue for output
        await self.output_queue.put((self.output_sample_rate, audio_array))

    async def _handle_text_tokens(self, text_bytes: bytes) -> None:
        """Handle incoming text tokens from Moshi."""
        try:
            text = text_bytes.decode("utf-8")
            logger.debug(f"Received text tokens: {text}")

            # Display as transcript
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": text}),
            )
        except Exception as e:
            logger.error(f"Failed to decode text tokens: {e}")

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

        # Note: Moshi doesn't have a JSON protocol for tool results
        # Tool results are handled internally by the model
        # If Moshi supports tool calling, it would be via binary protocol
        logger.debug(f"Tool {tool_name} executed, result: {tool_result}")

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

            # Note: Moshi doesn't have a JSON protocol for images
            # Vision would need to be handled differently (e.g., via local VLM)
            logger.info("Camera tool executed (image not sent to Moshi - vision not yet supported)")

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

        # Drop audio frames until handshake is received
        # This prevents accumulating latency during connection setup
        if not self._handshake_received.is_set():
            logger.debug("Handshake not yet received, dropping audio frame to prevent latency buildup")
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

        # Encode to Opus using opuslib (raw Opus packets)
        try:
            # Add to buffer
            self.audio_buffer.append(audio_frame)
            self.buffer_samples += len(audio_frame)

            # Process complete frames (1920 samples = 80ms at 24kHz)
            while self.buffer_samples >= self.target_buffer_samples:
                # Concatenate buffer
                full_buffer = np.concatenate(self.audio_buffer)

                # Extract one frame
                frame = full_buffer[:self.target_buffer_samples]
                remaining = full_buffer[self.target_buffer_samples:]

                # Encode to raw Opus
                opus_bytes = self.opus_encoder.encode(frame.tobytes(), self.opus_frame_size)

                logger.debug(f"Encoded raw Opus: {len(opus_bytes)} bytes from {len(frame)} samples")

                # Format: first byte = 1 (audio message type), then raw Opus payload
                message = bytes([1]) + opus_bytes
                await self.websocket.send(message)

                # Track audio sent for latency measurement
                audio_duration = len(frame) / 24000.0  # seconds
                self.total_audio_sent += audio_duration

                # Update buffer
                if len(remaining) > 0:
                    self.audio_buffer = [remaining]
                    self.buffer_samples = len(remaining)
                else:
                    self.audio_buffer = []
                    self.buffer_samples = 0

        except Exception as e:
            logger.error(f"Failed to encode/send audio: {e}")
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

        # Simple approach: just play audio as it arrives
        # The audio device will handle buffering naturally
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

        # Note: Moshi doesn't have a JSON protocol for idle signals
        # The model should handle idle behavior based on lack of user speech
        logger.debug(f"Idle condition detected: {timestamp_msg}")
