import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
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

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates to match OpenAI's requirements
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        # Override type annotations for OpenAI strict typing (only for values used in API)
        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        # Track how the API key was provided (env vs textbox) and its value
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0  # sequence counter to prevent stale emissions
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = get_session_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session={
                            "type": "realtime",
                            "instructions": instructions,
                            "audio": {"output": {"voice": voice}},
                        },
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
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
            # Only emit if this is still the latest partial (by sequence number)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        openai_api_key = config.OPENAI_API_KEY
        if self.gradio_mode and not openai_api_key:
            # api key was not found in .env or in the environment variables
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args[3]) > 0 else None
            if textbox_api_key is not None:
                openai_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                openai_api_key = config.OPENAI_API_KEY
        else:
            if not openai_api_key or not openai_api_key.strip():
                # In headless console mode, LocalStream now blocks startup until the key is provided.
                # However, unit tests may invoke this handler directly with a stubbed client.
                # To keep tests hermetic without requiring a real key, fall back to a placeholder.
                logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                openai_api_key = "DUMMY"

        self.client = AsyncOpenAI(api_key=openai_api_key)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            asyncio.create_task(self._run_realtime_session(), name="openai-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            try:
                await conn.session.update(
                    session={
                        "type": "realtime",
                        "instructions": get_session_instructions(),
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.input_sample_rate,
                                },
                                "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "interrupt_response": True,
                                },
                            },
                            "output": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.output_sample_rate,
                                },
                                "voice": get_session_voice(),
                            },
                        },
                        "tools": get_tool_specs(),  # type: ignore[typeddict-item]
                        "tool_choice": "auto",
                    },
                )
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    get_session_voice(),
                )
                # If we reached here, the session update succeeded which implies the API key worked.
                # Persist the key to a newly created .env (copied from .env.example) if needed.
                self._persist_api_key_if_needed()
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                return

            logger.info("Realtime session updated successfully")

            # Manage event received from the openai server
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass
            async for event in self.connection:
                logger.debug(f"OpenAI event: {event.type}")
                if event.type == "input_audio_buffer.speech_started":
                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                        self._clear_queue()
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)
                    logger.debug("User speech started")

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)
                    logger.debug("User speech stopped - server will auto-commit with VAD")

                if event.type in (
                    "response.audio.done",  # GA
                    "response.output_audio.done",  # GA alias
                    "response.audio.completed",  # legacy (for safety)
                    "response.completed",  # text-only completion
                ):
                    logger.debug("response completed")

                if event.type == "response.created":
                    logger.debug("Response created")

                if event.type == "response.done":
                    # Doesn't mean the audio is done playing
                    logger.debug("Response done")

                # Handle partial transcription (user speaking in real-time)
                if event.type == "conversation.item.input_audio_transcription.partial":
                    logger.debug(f"User partial transcript: {event.transcript}")

                    # Increment sequence
                    self.partial_transcript_sequence += 1
                    current_sequence = self.partial_transcript_sequence

                    # Cancel previous debounce task if it exists
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    # Start new debounce timer with sequence number
                    self.partial_transcript_task = asyncio.create_task(
                        self._emit_debounced_partial(event.transcript, current_sequence)
                    )

                # Handle completed transcription (user finished speaking)
                if event.type == "conversation.item.input_audio_transcription.completed":
                    logger.debug(f"User transcript: {event.transcript}")

                    # Cancel any pending partial emission
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                # Handle assistant transcription
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    logger.debug(f"Assistant transcript: {event.transcript}")
                    await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                # Handle audio delta
                if event.type in ("response.audio.delta", "response.output_audio.delta"):
                    if self.deps.head_wobbler is not None:
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
                if event.type == "response.function_call_arguments.done":
                    tool_name = getattr(event, "name", None)
                    args_json_str = getattr(event, "arguments", None)
                    call_id = getattr(event, "call_id", None)

                    if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                        logger.error("Invalid tool call: tool_name=%s, args=%s", tool_name, args_json_str)
                        continue

                    try:
                        tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                        logger.debug("Tool '%s' executed successfully", tool_name)
                        logger.debug("Tool result: %s", tool_result)
                    except Exception as e:
                        logger.error("Tool '%s' failed", tool_name)
                        tool_result = {"error": str(e)}

                    # send the tool result back
                    if isinstance(call_id, str):
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(tool_result),
                            },
                        )

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": json.dumps(tool_result),
                                "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                            },
                        ),
                    )

                    if tool_name == "camera" and "b64_im" in tool_result:
                        # use raw base64, don't json.dumps (which adds quotes)
                        b64_im = tool_result["b64_im"]
                        if not isinstance(b64_im, str):
                            logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                            b64_im = str(b64_im)
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_im}",
                                    },
                                ],
                            },
                        )
                        logger.info("Added camera image to conversation")

                        if self.deps.camera_worker is not None:
                            np_img = self.deps.camera_worker.get_latest_frame()
                            if np_img is not None:
                                # Camera frames are BGR from OpenCV; convert so Gradio displays correct colors.
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

                    # if this tool call was triggered by an idle signal, don't make the robot speak
                    # for other tool calls, let the robot reply out loud
                    if self.is_idle_tool_call:
                        self.is_idle_tool_call = False
                    else:
                        await self.connection.response.create(
                            response={
                                "instructions": "Use the tool result just returned and answer concisely in speech.",
                            },
                        )

                    # re synchronize the head wobble after a tool call that may have taken some time
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()

                # server error
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    code = getattr(err, "code", "")

                    logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                    # Only show user-facing errors, not internal state errors
                    if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                        )

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the OpenAI server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for OpenAI's API. Resamples if the input sample rate differs
        from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to OpenAI (guard against races during reconnect)
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped (connection closed?): %s", e)
                return None

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

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

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        # Conservative fallback list with default first
        fallback = [
            "cedar",
            "alloy",
            "aria",
            "ballad",
            "verse",
            "sage",
            "coral",
        ]
        try:
            # Best effort discovery; safe-guarded for unexpected shapes
            model = await self.client.models.retrieve(config.MODEL_NAME)
            # Try common serialization paths
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None
            # Scan for voice candidates
            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if "voice" in kl and isinstance(v, (list, tuple)):
                                for item in v:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                                        candidates.add(item["name"])
                            else:
                                _collect(v)
                    elif isinstance(obj, (list, tuple)):
                        for it in obj:
                            _collect(it)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)
            # Ensure default present and stable order
            voices = sorted(candidates) if candidates else fallback
            if "cedar" not in voices:
                voices = ["cedar", *[v for v in voices if v != "cedar"]]
            return voices
        except Exception:
            return fallback

    async def send_idle_signal(self, idle_duration: float) -> None:
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
            },
        )
        await self.connection.response.create(
            response={
                "instructions": "You MUST respond with function calls only - no speech or text. Choose appropriate actions for idle behavior.",
                "tool_choice": "required",
            },
        )

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key into `.env` inside `instance_path/` when appropriate.

        - Only runs in Gradio mode when key came from the textbox and is non-empty.
        - Only saves if `self.instance_path` is not None.
        - Writes `.env` to `instance_path/.env` (does not overwrite if it already exists).
        - If `instance_path/.env.example` exists, copies its contents while overriding OPENAI_API_KEY.
        """
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided via textbox; skipping persistence.")
                return
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            # Update the current process environment for downstream consumers
            try:
                import os

                os.environ["OPENAI_API_KEY"] = key
            except Exception:  # best-effort
                pass

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                # Respect existing user configuration
                logger.info(".env already exists at %s; not overwriting.", env_path)
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example at %s: %s", example_path, e)

            # Replace or append the OPENAI_API_KEY line
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    content_lines[i] = f"OPENAI_API_KEY={key}"
                    replaced = True
                    break
            if not replaced:
                content_lines.append(f"OPENAI_API_KEY={key}")

            # Ensure file ends with newline
            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created %s and stored OPENAI_API_KEY for future runs.", env_path)
        except Exception as e:
            # Never crash the app for QoL persistence; just log.
            logger.warning("Could not persist OPENAI_API_KEY to .env: %s", e)
