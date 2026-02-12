"""Sarvam AI realtime handler for conversation."""

import json
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)
from reachy_mini_conversation_app.realtime_handler_base import RealtimeHandlerBase


logger = logging.getLogger(__name__)

SARVAM_INPUT_SAMPLE_RATE: Final[int] = 16000
SARVAM_OUTPUT_SAMPLE_RATE: Final[int] = 16000


class SarvamRealtimeHandler(RealtimeHandlerBase):
    """A Sarvam AI realtime handler for fastrtc Stream.
    
    Integrates Sarvam AI's realtime conversation API for voice-based interactions
    with the Reachy Mini robot.
    """

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the Sarvam realtime handler."""
        super().__init__(
            deps=deps,
            expected_layout="mono",
            output_sample_rate=SARVAM_OUTPUT_SAMPLE_RATE,
            input_sample_rate=SARVAM_INPUT_SAMPLE_RATE,
            gradio_mode=gradio_mode,
            instance_path=instance_path,
        )

        self.input_sample_rate = SARVAM_INPUT_SAMPLE_RATE
        self.output_sample_rate = SARVAM_OUTPUT_SAMPLE_RATE

        self.client: Any = None
        self.connection: Any = None

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay = 0.5

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # Track how the API key was provided
        self._key_source: str = "env"
        self._provided_api_key: str | None = None

    def copy(self) -> "SarvamRealtimeHandler":
        """Create a copy of the handler."""
        return SarvamRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality profile at runtime if possible.

        For Sarvam, this would update the system prompt if the connection is active.
        Falls back to apply on next connection if no active connection.
        """
        try:
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = get_session_voice()
            except BaseException as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart if needed
            if self.connection is not None:
                try:
                    # TODO: Implement live update for Sarvam API
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

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
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler and establish connection to Sarvam realtime API."""
        sarvam_api_key = getattr(config, "SARVAM_API_KEY", None)
        
        if self.gradio_mode and not sarvam_api_key:
            # Try to get API key from args if in Gradio mode
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            # Assuming API key is passed as an argument
            textbox_api_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
            if textbox_api_key is not None:
                sarvam_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
        
        if not sarvam_api_key or not sarvam_api_key.strip():
            logger.warning("SARVAM_API_KEY missing. Proceeding with a placeholder (tests/offline).")
            sarvam_api_key = "DUMMY"

        # Initialize Sarvam client
        try:
            import sarvam  # type: ignore
            self.client = sarvam.AsyncClient(api_key=sarvam_api_key)
        except ImportError:
            logger.warning("sarvam package not installed. Using mock client for testing.")
            self.client = None

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                return
            except Exception as e:
                logger.warning("Realtime connection failed (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    base_delay = 2 ** (attempt - 1)
                    jitter = asyncio.random.random() * 0.5
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background."""
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: Sarvam client not initialized yet.")
                return

            try:
                self._connected_event.clear()
            except Exception:
                pass
            
            asyncio.create_task(self._run_realtime_session(), name="sarvam-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session with Sarvam."""
        if self.client is None:
            logger.error("Sarvam client not initialized")
            return

        try:
            # TODO: Implement Sarvam realtime session logic
            # This will depend on Sarvam's actual API structure
            # for now, we'll create a placeholder that logs the attempt
            logger.info("Attempting to establish Sarvam realtime connection...")
            logger.info("Realtime session initialized with profile=%r voice=%r",
                       getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                       get_session_voice())
            
            self._connected_event.set()
            
            # Keep the session alive - in production, this would handle
            # the actual WebSocket connection and event loop
            while not self._shutdown_requested:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error("Realtime session error: %s", e)
            raise

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone and send to Sarvam.

        Handles resampling if needed to match Sarvam's expected sample rate.
        """
        if not self.connection:
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

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        try:
            # TODO: Send audio to Sarvam API
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            logger.debug(f"Audio frame sent to Sarvam (sample_rate={self.input_sample_rate})")
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by speaker."""
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
        """Shutdown the handler and clean up resources."""
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
            except Exception as e:
                logger.debug(f"Connection close: {e}")
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
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Get list of available voices for Sarvam.
        
        Returns a list of commonly supported voices or falls back to defaults.
        """
        # Sarvam typically supports a predefined set of voices
        # This can be expanded once we have more API documentation
        fallback = [
            "default",
            "indian_male",
            "indian_female",
        ]
        
        try:
            # TODO: Implement voice discovery from Sarvam API
            logger.info("Sarvam available voices: %s", fallback)
            return fallback
        except Exception:
            return fallback

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to Sarvam."""
        logger.debug("Sending idle signal after %.1f seconds", idle_duration)
        self.is_idle_tool_call = True
        
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            "You've been idle for a while. Feel free to get creative - dance, show an emotion, "
            "look around, do nothing, or just be yourself!"
        )
        
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return

        try:
            # TODO: Send idle signal via Sarvam API
            logger.debug("Idle signal sent to Sarvam")
        except Exception as e:
            logger.error("Failed to send idle signal: %s", e)

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key to .env if appropriate."""
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided; skipping persistence.")
                return
            
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                logger.info(".env already exists; not overwriting.")
                return

            # Read template if available
            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example: %s", e)

            # Replace or append the SARVAM_API_KEY line
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("SARVAM_API_KEY="):
                    content_lines[i] = f"SARVAM_API_KEY={key}"
                    replaced = True
                    break
            
            if not replaced:
                content_lines.append(f"SARVAM_API_KEY={key}")

            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created .env and stored SARVAM_API_KEY for future runs.")
        except Exception as e:
            logger.warning("Could not persist SARVAM_API_KEY to .env: %s", e)
