import os
import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal

import sphn
import numpy as np
import websockets
from unmute import openai_realtime_api_events as events
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError
from unmute.llm.system_prompt import GenericToolInstructions

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    ALL_TOOLS,
    dispatch_tool_call,
    get_tool_specs,
)


logger = logging.getLogger(__name__)

UNMUTE_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
UNMUTE_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class UnmuteRealtimeHandler(AsyncStreamHandler):
    """An Unmute realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=UNMUTE_OUTPUT_SAMPLE_RATE,
            input_sample_rate=UNMUTE_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.opus_writer = sphn.OpusStreamWriter(self.input_sample_rate)
        self.opus_reader = sphn.OpusStreamReader(self.output_sample_rate)

        self.deps = deps

        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay = 0.5  # seconds

        # User transcript accumulation
        self.current_user_transcript = ""

        # Event handling task
        self.event_handler_task: asyncio.Task[None] | None = None

    def copy(self) -> "UnmuteRealtimeHandler":
        """Create a copy of the handler."""
        return UnmuteRealtimeHandler(self.deps)

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            # Only emit if this is still the latest partial (by sequence number)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user_partial", "content": transcript})
                )
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                logger.warning(
                    "Realtime websocket closed unexpectedly (attempt %d/%d): %s",
                    attempt, max_attempts, e
                )
                if attempt < max_attempts:
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                self.websocket = None

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        url = os.getenv("UNMUTE_BASE_URL")
        if url is None:
            url = "wss://gradium.ai/unmute/api/v1/realtime"
        api_key = os.getenv("GRADIUM_API_KEY")

        if not api_key:
            logger.error("GRADIUM_API_KEY environment variable not set")
            return

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        async with websockets.connect(
            url,
            additional_headers=headers,
            subprotocols=["realtime"],  # type: ignore
        ) as ws:

            self.websocket = ws
            logger.info("WebSocket connected to Unmute")

            tool_specs = get_tool_specs()
            fixed_tool_specs = []
            for tool_spec in tool_specs:
                if "function" not in tool_spec:
                    tool_spec = dict(tool_spec)
                    tool_spec.pop("type", None)
                    tool_spec = {
                        'type': 'function',
                        'function': tool_spec,
                    }
                fixed_tool_specs.append(tool_spec)
            instructions = GenericToolInstructions(
                text=get_session_instructions(),
                tools=fixed_tool_specs,
            )
            logger.debug("Sending Instructions to Gradium: %r", instructions)
            # Send session update
            try:
                session_update = events.SessionUpdate(
                    session=events.SessionConfig(
                        instructions=instructions,
                        voice="LFZvm12tW_z0xfGo",
                        allow_recording=False,
                    )
                )
                await ws.send(session_update.model_dump_json())
                logger.info("Session update sent")
            except Exception:
                logger.exception("Failed to send session update; aborting startup")
                return

            # Start event handler task
            self.event_handler_task = asyncio.create_task(self._handle_events())

            try:
                await self.event_handler_task
            except asyncio.CancelledError:
                logger.info("Event handler task cancelled")
            except Exception:
                logger.exception("Event handler task failed")

    async def _handle_events(self) -> None:
        """Handle incoming events from the websocket."""
        if not self.websocket:
            return

        try:

            pending_function_calls: list[tuple[int, Any]] = []
            pending_words: int = 0
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("type")


                    if event_type == "error":
                        error_event = events.Error(**data)
                        logger.error(
                            "Unmute error [%s]: %s",
                            error_event.error.code,
                            error_event.error.message
                        )
                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "assistant",
                                "content": f"[error] {error_event.error.message}"
                            })
                        )

                    elif event_type == "session.updated":
                        logger.info("Session updated successfully")
                        logger.debug(data)

                    elif event_type == "input_audio_buffer.speech_started":
                        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                            self._clear_queue()
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()
                        self.deps.movement_manager.clear_move_queue()
                        self.deps.movement_manager.set_listening(True)
                        # DON'T reset transcript here - first delta may have already arrived!
                        logger.info("User speech started")

                    elif event_type == "input_audio_buffer.speech_stopped":
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped")

                        # Emit the complete user transcript when speech stops
                        if self.current_user_transcript:
                            # Cancel any pending partial emission
                            if self.partial_transcript_task and not self.partial_transcript_task.done():
                                self.partial_transcript_task.cancel()
                                try:
                                    await self.partial_transcript_task
                                except asyncio.CancelledError:
                                    pass

                            await self.output_queue.put(
                                AdditionalOutputs({"role": "user", "content": self.current_user_transcript})
                            )

                            # Reset transcript AFTER emitting it
                            self.current_user_transcript = ""

                    elif event_type == "response.created":
                        logger.debug("Response created")

                    elif event_type == "conversation.item.input_audio_transcription.delta":
                        # Most likely the TTS got interrupted, so clearing any pending function calls.
                        pending_function_calls.clear()
                        pending_words = 0
                        delta_event = events.ConversationItemInputAudioTranscriptionDelta(**data)
                        logger.debug(f"User partial transcript delta: {delta_event.delta}")

                        # Add space before concatenating if we already have content and delta doesn't start with punctuation/space
                        if self.current_user_transcript and delta_event.delta:
                            # Check if we need to add a space
                            # Don't add space if delta starts with punctuation or whitespace
                            if delta_event.delta[0] not in '.,!?\'" \n\t':
                                self.current_user_transcript += " "

                        # Accumulate the transcript
                        self.current_user_transcript += delta_event.delta

                        # Increment sequence
                        self.partial_transcript_sequence += 1
                        current_sequence = self.partial_transcript_sequence

                        # Cancel previous debounce task
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                            try:
                                await self.partial_transcript_task
                            except asyncio.CancelledError:
                                pass

                        # Start new debounce timer
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(self.current_user_transcript, current_sequence)
                        )

                    elif event_type == "response.text.done":
                        # Assitant's complete response text
                        text_event = events.ResponseTextDone(**data)
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": text_event.text})
                        )

                    elif event_type == "response.audio.delta":
                        audio_event = events.ResponseAudioDelta(**data)

                        # Decode the Opus audio to PCM
                        opus_bytes = base64.b64decode(audio_event.delta)
                        pcm_audio = self.opus_reader.append_bytes(opus_bytes)

                        if pcm_audio.size > 0:
                            # Feed decoded PCM to head wobbler if available
                            if self.deps.head_wobbler is not None:
                                # head_wobbler expects base64-encoded PCM int16
                                pcm_b64 = base64.b64encode(audio_to_int16(pcm_audio).tobytes()).decode('utf8')
                                self.deps.head_wobbler.feed(pcm_b64)

                            self.last_activity_time = asyncio.get_event_loop().time()

                            # Send decoded PCM to output queue
                            await self.output_queue.put(
                                (
                                    self.output_sample_rate,
                                    pcm_audio,
                                ),
                            )

                    elif event_type == "response.audio.done":
                        logger.debug("Response audio done")

                    # Handle Unmute-specific events
                    elif event_type == "unmute.additional_outputs":
                        additional_event = events.UnmuteAdditionalOutputs(**data)
                        logger.debug(f"Additional outputs: {additional_event.args}")

                    elif event_type == "unmute.interrupted_by_vad":
                        logger.debug("Interrupted by VAD")

                    elif event_type == "unmute.response.function_call":
                        # Legacy function call, used for the on stage demo where we can sync movement
                        # with speech.
                        if data['name'] in ['play_emotion']:
                            call = (data['name'], data['parameters'])
                            ahead = 5
                            logger.info("Pending function call %s in %d: %r",
                                        call[0], pending_words - ahead, call[1])
                            pending_function_calls.append((pending_words, call))

                    elif event_type == "response.function_call_arguments.done":
                        # Synchronous function call supporting any function call.
                        # This will block the text generation and TTS until all function calls have
                        # been resolved.
                        tool_name = data["name"]
                        args_json_str = data["arguments"]
                        call_id = data["call_id"]

                        try:
                            tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                            logger.debug("Tool '%s' executed successfully", tool_name)
                            logger.debug("Tool result: %s", tool_result)
                        except Exception as e:
                            logger.error("Tool '%s' failed", tool_name)
                            tool_result = {"error": str(e)}

                        if tool_name == "camera" and "b64_im" in tool_result:
                            logger.warning("Trying to use camera tool, but not supported yet.")
                            tool_result = {"error": "Camera tool not supported yet!"}

                        # send the tool result back
                        if isinstance(call_id, str):

                            result_message = {
                                'type': "unmute.response.function_call.result",
                                'call_id': call_id,
                                'content': json.dumps(tool_result)
                            }
                            await self.websocket.send(json.dumps(result_message))

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(tool_result),
                                    "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                                },
                            ),
                        )
                        # re synchronize the head wobble after a tool call that may have taken some time
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()

                    elif event_type == "unmute.response.text.delta.ready":
                        logger.debug("Pending text:  %s", data['delta'])
                        pending_words += 1
                    elif event_type == "response.text.delta":
                        logger.debug("Text read: %s", data['delta'])
                        for idx, (count, call) in enumerate(pending_function_calls):
                            pending_function_calls[idx] = (count - 1, call)
                        while pending_function_calls and pending_function_calls[0][0] <= 0:
                            count, call = pending_function_calls.pop(0)
                            logger.info("Playing function call %s: %r", *call)
                            await ALL_TOOLS[call[0]](self.deps, **call[1])
                    else:
                        logger.debug(f"Other event type: {event_type}")
                        logger.debug(data)

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message: {message}")
                except Exception:
                    logger.exception(f"Error processing event: {message}")

        except asyncio.CancelledError:
            logger.info("Event handler cancelled")
            raise
        except Exception:
            logger.exception("Event handler error")

    async def receive(self, frame: Tuple[int, NDArray[np.float32]]) -> None:
        """Receive audio frame from the microphone and send it to the server.

        Args:
            frame: A tuple containing the sample rate and the audio frame.

        """
        if not self.websocket:
            logger.debug("Websocket is down")
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Use one channel for mono
            if audio_frame.shape[1] == 2:
                audio_frame = audio_frame[:, 0]
            else:
                audio_frame = audio_frame.squeeze()
        if audio_frame.dtype == np.int16:
            audio_frame = audio_frame.astype(np.float32) / 32768.0

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * self.input_sample_rate / input_sample_rate)
            )

        opus_bytes = self.opus_writer.append_pcm(audio_frame)

        if opus_bytes:
            # Encode and send
            audio_message = base64.b64encode(opus_bytes).decode("utf-8")
            append_event = events.InputAudioBufferAppend(audio=audio_message)
            await self.websocket.send(append_event.model_dump_json())


    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        # Cancel event handler task
        if self.event_handler_task and not self.event_handler_task.done():
            self.event_handler_task.cancel()
            try:
                await self.event_handler_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            try:
                await self.websocket.close()
            except ConnectionClosedError as e:
                logger.debug(f"WebSocket already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"websocket.close() ignored: {e}")
            finally:
                self.websocket = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
