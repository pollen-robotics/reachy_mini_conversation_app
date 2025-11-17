"""Gradio UI for cascade mode."""

from __future__ import annotations
import json
import wave
import asyncio
import logging
import threading

# import Queue
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import gradio as gr
import sounddevice as sd  # Only used for microphone input recording
import numpy.typing as npt


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

from reachy_mini_conversation_app.cascade.asr import StreamingASRProvider


logger = logging.getLogger(__name__)


class CascadeGradioUI:
    """Gradio interface for cascade pipeline."""

    def __init__(self, cascade_handler: CascadeHandler, robot: Optional['ReachyMini'] = None) -> None:
        """Initialize Gradio UI.

        Args:
            cascade_handler: Cascade pipeline handler
            robot: Robot instance (if running on robot hardware, enables robot speaker output)

        """
        self.handler = cascade_handler
        self.robot = robot
        # Tell handler to skip audio playback, since it will be done in gradio UI.
        self.handler.skip_audio_playback = True

        self.chat_history: List[Dict[str, Any]] = []
        self.recording = False
        self.audio_frames: List[npt.NDArray[np.int16]] = []
        self.recorded_file: Optional[str] = None
        self.audio_data: Optional[npt.NDArray[np.int16]] = None  # Keep audio in memory
        self.record_thread: Optional[threading.Thread] = None
        self.sample_rate = 16000
        self._last_click_time: float = 0.0

        # Pre-warmed audio playback system
        self.audio_queue: Queue[Any] = Queue(maxsize=100)
        self.wobbler_queue: Queue[Any] = Queue(maxsize=100)
        self.playback_thread: Optional[threading.Thread] = None
        self.wobbler_thread: Optional[threading.Thread] = None
        self.playback_active = False
        self.shutdown_event = threading.Event()

        # Initialize persistent playback threads
        self._init_playback_threads()

    def _is_streaming_asr(self) -> bool:
        """Check if the ASR provider supports streaming."""
        return isinstance(self.handler.asr, StreamingASRProvider)

    def _init_playback_threads(self) -> None:
        """Initialize persistent audio playback and wobbler threads (pre-warmed)."""
        # Determine playback mode based on system's default audio output device
        # Use robot.media ONLY if:
        # 1. Robot hardware is available (not simulation)
        # 2. Default output device is a robot speaker (reSpeaker, etc.)

        robot_available = (
            self.robot is not None
            and hasattr(self.robot, 'media')
            and not self.robot.client.get_status().get("simulation_enabled", False)
        )

        # Check if default output is a robot speaker
        default_is_robot_speaker = False
        if robot_available:
            try:
                default_device = sd.query_devices(kind='output')
                device_name = default_device['name'].lower()
                # Common robot speaker names
                robot_speaker_keywords = ['respeaker', 'xvf3800', 'reachy']
                default_is_robot_speaker = any(keyword in device_name for keyword in robot_speaker_keywords)
                logger.info(f"AUDIO PREWARM: Default output device: {default_device['name']}")
                logger.debug(f"AUDIO PREWARM: Is robot speaker? {default_is_robot_speaker}")
            except Exception as e:
                logger.warning(f"Failed to detect default audio device: {e}")

        # Use robot.media only if both robot is available AND default output is robot speaker
        self.use_robot_media = robot_available and default_is_robot_speaker

        if self.use_robot_media:
            logger.info("AUDIO PREWARM: Using robot.media for playback (robot speaker is default)")
            self._init_robot_playback_threads()
        else:
            reason = "laptop/other speaker" if robot_available else "simulation/no robot"
            logger.info(f"AUDIO PREWARM: Using sounddevice for playback ({reason})")
            self._init_sounddevice_playback_threads()

    def _init_sounddevice_playback_threads(self) -> None:
        """Initialize sounddevice playback threads (laptop speakers)."""
        import time

        def persistent_playback_thread() -> None:
            """Run persistent audio playback thread (pre-warmed and ready)."""
            from reachy_mini_conversation_app.cascade.timing import tracker

            try:
                # Pre-initialize sounddevice stream (happens once at startup), it helps avoid delays.
                tracker.mark("audio_stream_prewarm_start")

                all_devices = sd.query_devices()
                default_output_idx = sd.default.device[1]
                logger.info(f"AUDIO PREWARM: Total {len(all_devices)} devices available")
                for i, dev in enumerate(all_devices):
                    if dev['max_output_channels'] > 0:
                        default_marker = " (DEFAULT OUTPUT)" if i == default_output_idx else ""
                        logger.info(f"  [{i}] {dev['name']} - {dev['max_output_channels']} channels{default_marker}")

                default_device = sd.query_devices(kind='output')
                logger.info(f"AUDIO PREWARM: Using default device: {default_device['name']}")

                # Create stream once (pre-warmed)
                stream = sd.OutputStream(
                    samplerate=24000,
                    channels=1,
                    dtype=np.int16,
                    blocksize=4096,
                    latency='low',
                )
                stream.start()
                actual_latency_ms = stream.latency * 1000
                logger.info(f"AUDIO PREWARM: Stream ready. Latency: {actual_latency_ms:.0f}ms")
                tracker.mark("audio_stream_prewarm_complete", {
                    "stream_latency_ms": round(actual_latency_ms, 1)
                })


                # Main playback loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        chunk = self.audio_queue.get(timeout=0.1)

                        if chunk is None:  # Sentinel - end of current playback session
                            continue

                        # Write chunk to stream (immediate playback)
                        stream.write(chunk)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in persistent playback thread: {e}")
            finally:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                logger.info("Playback thread shutdown")


        def persistent_wobbler_thread() -> None:
            """Run persistent wobbler thread (pre-warmed and ready)."""
            try:
                logger.info("WOBBLER PREWARM: Thread ready")

                # Main wobbler loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        chunk = self.wobbler_queue.get(timeout=0.1)

                        if chunk is None:  # Sentinel - end of current playback session
                            # Reset wobbler between turns
                            if self.handler.deps.head_wobbler:
                                self.handler.deps.head_wobbler.reset()
                            continue

                        # Feed to wobbler
                        if self.handler.deps.head_wobbler:
                            self.handler.deps.head_wobbler.feed_pcm(chunk)

                        # Rate limit to match playback
                        chunk_duration = len(chunk) / (2 * 24000)
                        time.sleep(chunk_duration)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in persistent wobbler thread: {e}")
            finally:
                logger.info("Wobbler thread shutdown")


        # Start persistent threads
        self.playback_thread = threading.Thread(target=persistent_playback_thread, daemon=True, name="AudioPlayback")
        self.wobbler_thread = threading.Thread(target=persistent_wobbler_thread, daemon=True, name="Wobbler")

        self.playback_thread.start()
        self.wobbler_thread.start()

        # Give threads time to initialize
        time.sleep(0.1)

        logger.info("Pre-warmed audio playback system initialized (sounddevice)")

    def _init_robot_playback_threads(self) -> None:
        """Initialize robot.media playback threads (robot speakers)."""
        import time

        import librosa

        # Start robot media playback (must be called before pushing audio)
        if self.robot is not None and hasattr(self.robot, 'media'):
            logger.info("Starting robot.media playback system...")
            self.robot.media.start_playing()
            time.sleep(0.5)  # Give pipeline time to initialize
            logger.info("Robot.media playback system started")

        def persistent_playback_thread() -> None:
            """Run persistent audio playback thread using robot.media."""
            from reachy_mini_conversation_app.cascade.timing import tracker

            # Type guard: ensure robot and media are available
            if self.robot is None or not hasattr(self.robot, 'media'):
                logger.error("Robot media not available for playback")
                return

            try:
                # Pre-initialize robot media
                tracker.mark("audio_stream_prewarm_start")

                # Get robot audio sample rate
                device_sample_rate = self.robot.media.get_audio_samplerate()
                logger.info(f"AUDIO PREWARM: Robot speaker sample rate: {device_sample_rate}Hz")
                tracker.mark("audio_stream_prewarm_complete", {
                    "device": "robot.media",
                    "sample_rate": device_sample_rate
                })

                # Main playback loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        chunk = self.audio_queue.get(timeout=0.1)

                        if chunk is None:  # Sentinel - end of current playback session
                            continue

                        # Convert int16 to float32 for robot.media
                        audio_float = chunk.astype(np.float32) / 32768.0

                        # Resample if needed (TTS outputs 24kHz)
                        if device_sample_rate != 24000:
                            audio_float = librosa.resample(
                                audio_float,
                                orig_sr=24000,
                                target_sr=device_sample_rate,
                            )

                        # Push to robot speaker
                        self.robot.media.push_audio_sample(audio_float)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in robot playback thread: {e}")
            finally:
                logger.info("Robot playback thread shutdown")

        def persistent_wobbler_thread() -> None:
            """Run persistent wobbler thread (pre-warmed and ready)."""
            try:
                logger.info("WOBBLER PREWARM: Thread ready")

                # Main wobbler loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        chunk = self.wobbler_queue.get(timeout=0.1)

                        if chunk is None:  # Sentinel - end of current playback session
                            # Reset wobbler between turns
                            if self.handler.deps.head_wobbler:
                                self.handler.deps.head_wobbler.reset()
                            continue

                        # Feed to wobbler
                        if self.handler.deps.head_wobbler:
                            self.handler.deps.head_wobbler.feed_pcm(chunk)

                        # Rate limit to match playback
                        chunk_duration = len(chunk) / (2 * 24000)
                        time.sleep(chunk_duration)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in persistent wobbler thread: {e}")
            finally:
                logger.info("Wobbler thread shutdown")

        # Start persistent threads
        self.playback_thread = threading.Thread(target=persistent_playback_thread, daemon=True, name="RobotAudioPlayback")
        self.wobbler_thread = threading.Thread(target=persistent_wobbler_thread, daemon=True, name="Wobbler")

        self.playback_thread.start()
        self.wobbler_thread.start()

        # Give threads time to initialize
        time.sleep(0.1)

        logger.info("Pre-warmed audio playback system initialized (robot.media)")

    def _record_audio(self) -> None:
        """Record audio from microphone in background thread."""
        import io

        self.audio_frames = []

        # Initialize streaming ASR session if supported
        if self._is_streaming_asr():
            logger.info("Initializing streaming ASR session...")
            future = asyncio.run_coroutine_threadsafe(
                self.handler.process_audio_streaming_start(),
                self.handler.loop
            )
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to start streaming ASR: {e}")

        try:
            # Create audio stream (note : use small blocksize for lower latency)
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.int16,
                blocksize=1024,  # 1024 for lower latency (64ms)
            ) as stream:
                logger.info("Recording started...")
                while self.recording:
                    data, overflowed = stream.read(1024)
                    if overflowed:
                        logger.warning("Audio buffer overflowed")
                    self.audio_frames.append(data.copy())

                    # Send chunk to streaming ASR if supported
                    if self._is_streaming_asr():
                        # Convert chunk to WAV bytes
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, "wb") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(self.sample_rate)
                            wav_file.writeframes(data.tobytes())

                        chunk_wav = wav_buffer.getvalue()

                        # Send to handler (non-blocking)
                        asyncio.run_coroutine_threadsafe(
                            self.handler.process_audio_streaming_chunk(chunk_wav),
                            self.handler.loop
                        )

            # Thread cleanup - only set audio_data if main thread hasn't already done it
            # (main thread may have concatenated early for low latency)
            if self.audio_data is None and self.audio_frames:
                audio_data = np.concatenate(self.audio_frames)
                self.audio_data = audio_data
                logger.debug(f"Recording thread finished: {len(audio_data)} frames")

        except Exception as e:
            logger.exception(f"Error recording audio: {e}")
            self.recording = False


    def _start_recording(self) -> str:
        """Start recording audio."""
        if self.recording:
            return "STOP Recording (already recording)"

        self.recording = True
        self.audio_frames = []
        self.audio_data = None  # Reset audio data
        self.recorded_file = None

        # Start recording in background thread
        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()

        logger.info("Recording started...")
        return "STOP Recording (recording...)"


    def _stop_recording(self) -> tuple[str, str]:
        """Stop recording audio."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Reset tracker at the start of the full pipeline
        tracker.reset("user_conversation_turn")
        tracker.mark("user_stop_click")

        if not self.recording:
            return "START Recording", "Not recording. Click START first."

        self.recording = False
        tracker.mark("recording_stop_signal_sent")

        # OPTIMIZATION: Don't wait for thread to finish !
        # We already have all the audio frames in self.audio_frames
        # Concatenate immediately and start ASR without delay
        if self.audio_frames and len(self.audio_frames) > 0:
            audio_data = np.concatenate(self.audio_frames)
            self.audio_data = audio_data

            logger.info(f"Recording captured immediately: {len(audio_data)} frames ({len(audio_data)/self.sample_rate:.2f}s)")
            tracker.mark("recording_captured", {
                "frames": len(audio_data),
                "duration_s": round(len(audio_data) / self.sample_rate, 2)
            })
            tracker.mark("recording_ready")

            # Let thread finish cleanup in background (non-blocking)
            # The thread will exit cleanly since self.recording = False

            status = "Processing your message..."
            return "START Recording", status
        else:
            # No frames captured - wait for thread to see if it has data
            if self.record_thread:
                self.record_thread.join(timeout=0.5)
            tracker.mark("recording_thread_joined")

            if self.audio_data is not None and len(self.audio_data) > 0:
                tracker.mark("recording_ready")
                status = "Processing your message..."
                return "START Recording", status
            else:
                return "START Recording", "Failed to capture recording."



    def create_interface(self) -> gr.Blocks:
        """Create and return Gradio interface."""
        # Custom CSS to change button color based on recording state
        custom_css = """
        button:has(span:contains("STOP")) {
            background: linear-gradient(to bottom right, #dc2626, #991b1b) !important;
            border-color: #991b1b !important;
        }
        button:has(span:contains("START")) {
            background: linear-gradient(to bottom right, #16a34a, #15803d) !important;
            border-color: #15803d !important;
        }
        """

        with gr.Blocks(title="Reachy Mini - Cascade Mode", css=custom_css) as demo:
            gr.Markdown("# Reachy Mini Conversation (Cascade Mode)")
            gr.Markdown(
                "**Instructions:** Click the button to **START** recording, speak your message, "
                "then click again to **STOP** and process automatically!"
            )

            # Chat display
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=400,
            )

            # Recording toggle button (single button for start/stop)
            with gr.Row():
                record_btn = gr.Button(
                    "🎤 START Recording",
                    scale=1,
                    variant="primary",
                    size="lg",
                )

            # Status display
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to record...",
                placeholder="Status updates will appear here",
            )

            # Clear button
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear History", scale=1)

            # Toggle recording button logic
            def toggle_recording_wrapper(chat_history: List[Dict[str, Any]]) -> tuple[str, str, List[Dict[str, Any]]]:
                """Toggle between start and stop recording."""
                import time

                # Prevent rapid double-clicks (debounce)
                current_time = time.time()
                if current_time - self._last_click_time < 0.5:  # 500ms debounce
                        # Return current state without changes
                        if self.recording:
                            return "STOP Recording", "Recording... (speak now)", chat_history
                        else:
                            return "START Recording", "Ready to record...", chat_history
                self._last_click_time = current_time

                if not self.recording:
                    # Start recording
                    self._start_recording()
                    return "STOP Recording", "Recording... (speak now)", chat_history
                else:
                    # Stop recording and process
                    _, save_status = self._stop_recording()

                    # Auto-process if recording was successful
                    if self.audio_data is not None and len(self.audio_data) > 0:
                        # Convert to WAV bytes
                        import io
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, "wb") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(self.sample_rate)
                            wav_file.writeframes(self.audio_data.tobytes())

                        wav_bytes = wav_buffer.getvalue()
                        chat_history, process_status = self._process_audio_sync(wav_bytes, chat_history)
                        return "START Recording", process_status, chat_history
                    else:
                        return "START Recording", save_status, chat_history

            record_btn.click(
                fn=toggle_recording_wrapper,
                inputs=[chatbot],
                outputs=[record_btn, status_box, chatbot],
            )

            # Clear button
            clear_btn.click(
                fn=self._clear_history,
                inputs=None,
                outputs=[chatbot, status_box],
            )

        return demo  # type: ignore[no-any-return]


    def _process_audio_sync(
        self, audio_bytes: bytes, chat_history: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], str]:
        """Process audio bytes (sync wrapper for async handler).

        Args:
            audio_bytes: WAV audio bytes
            chat_history: Current chat history

        Returns:
            Tuple of (updated_chat_history, status_message)

        """
        if not audio_bytes or len(audio_bytes) == 0:
            return chat_history, "No audio data. Please record first."

        logger.debug(f"Processing {len(audio_bytes)} bytes of audio")

        try:

            # Use handler's event loop via run_coroutine_threadsafe
            # This avoids conflicts with the handler's background event loop
            if not self.handler.loop:
                return chat_history, "Error: Event loop not available"

            future = asyncio.run_coroutine_threadsafe(
                self._process_audio_async(audio_bytes),
                self.handler.loop
            )

            # Wait for result with timeout
            result = future.result(timeout=60)

            # Update chat history
            if result["success"]:
                # Add user message
                if result.get("transcript"):
                    chat_history.append(
                        {"role": "user", "content": result["transcript"]}
                    )

                # Add assistant responses
                if result.get("responses"):
                    for response in result["responses"]:
                        chat_history.append({"role": "assistant", "content": response})

                status = "Message processed successfully!"
            else:
                status = f"Error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.exception(f"Error processing audio: {e}")
            status = f"Error processing audio: {str(e)}"

        return chat_history, status


    async def _process_audio_async(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Process audio through cascade pipeline (async).

        Args:
            audio_bytes: Audio file bytes

        Returns:
            Dictionary with processing results

        """
        result: Dict[str, Any] = {
            "success": False,
            "transcript": None,
            "responses": [],
            "error": None,
        }

        try:
            # Store the current conversation length to track new messages
            initial_history_length = len(self.handler.conversation_history)

            # Choose streaming or batch processing based on ASR provider
            if self._is_streaming_asr():
                # Streaming path: Finalize stream and get transcript
                # Note: Streaming session was already started in _record_audio()
                # and chunks were sent during recording
                logger.info("Finalizing streaming ASR session...")
                transcript = await self.handler.process_audio_streaming_end()
            else:
                # Batch path: Use handler's process_audio_manual which handles ASR→LLM→TTS pipeline
                # Note: This will execute speak tools but won't play audio (that's done separately below)
                transcript = await self.handler.process_audio_manual(audio_bytes)

            result["transcript"] = transcript

            if not transcript.strip():
                result["error"] = "Empty transcript"
                return result

            # Collect all speech messages first (to play them all in one stream)
            speak_messages = []

            # Check if there are any speak tool calls (to avoid duplicate messages)
            has_speak_tool = any(
                msg.get("role") == "tool" and msg.get("name") == "speak"
                for msg in self.handler.conversation_history[initial_history_length:]
            )

            # Extract responses from conversation history that were added during processing
            # The handler adds: user message, assistant message(s), and tool results
            for message in self.handler.conversation_history[initial_history_length:]:
                role = message.get("role")

                if role == "assistant":
                    # Only add assistant text if there's NO speak tool (to avoid duplicates)
                    # When speak tool is present, we'll show that message instead
                    if not has_speak_tool and "content" in message and message["content"]:
                        result["responses"].append(message["content"])

                elif role == "tool":

                    tool_name = message.get("name")
                    if tool_name == "speak":
                        # Extract speak tool results for display, since this tool is used in place of a message
                        try:
                            tool_content = json.loads(message.get("content", "{}"))
                            if "message" in tool_content:
                                result["responses"].append(f"{tool_content['message']}")
                                # Collect speak messages to synthesize all at once
                                speak_messages.append(tool_content['message'])
                        except json.JSONDecodeError:
                            pass
                    elif tool_name == "camera":
                        # Display camera image in chat
                        # TODO : fix this ?
                        try:
                            tool_content = json.loads(message.get("content", "{}"))
                            if "b64_im" in tool_content and self.handler.deps.camera_worker is not None:
                                # Get the latest camera frame as numpy array
                                np_img = self.handler.deps.camera_worker.get_latest_frame()

                                # Save to temporary file for Gradio Chatbot display
                                import tempfile

                                from PIL import Image

                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp")
                                # Convert RGB numpy array to PIL Image and save
                                pil_img = Image.fromarray(np_img)
                                pil_img.save(temp_file.name, format="JPEG", quality=85)
                                temp_file.close()

                                # Add file path as string (Gradio Chatbot accepts file paths)
                                result["responses"].append(temp_file.name)
                                logger.info(f"Added camera image to chat display: {temp_file.name}")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse camera tool result")
                        except Exception as e:
                            logger.exception(f"Failed to add camera image to chat: {e}")
                    elif tool_name:
                        # Display other tool executions
                        result["responses"].append(
                            f"Used tool: {tool_name}"
                        )

            # In case there are several speech messages in a single stream (it shouldn't happen too much), concatenate them.
            if speak_messages:
                combined_text = ". ".join(speak_messages)
                logger.info(f"Synthesizing {len(speak_messages)} speak message(s) as one stream")
                await self._synthesize_for_gradio(combined_text)
            else:
                # No speech - print summary here since _synthesize_for_gradio won't be called
                from reachy_mini_conversation_app.cascade.timing import tracker
                logger.info("No speech output - printing latency summary")
                tracker.print_summary()

            result["success"] = True

        except Exception as e:
            logger.exception(f"Error in async processing: {e}")
            result["error"] = str(e)

        return result


    async def _synthesize_for_gradio(self, text: str) -> None:
        """Synthesize speech and play through pre-warmed audio system.

        1. Generates TTS audio chunks
        2. Plays audio immediately through pre-warmed sounddevice (robot speaker)
        3. Feeds chunks to head_wobbler for animation (synchronized with playback)

        Args:
            text: Text to synthesize

        """
        import time

        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.info(f"Synthesizing speech: '{text[:50]}...'")

        audio_chunks = []
        first_chunk_queued = False

        try:
            # Split text into sentences for streaming, so TTS can send a first audio faster.
            sentences = self._split_into_sentences(text)
            logger.debug(f"Split text into {len(sentences)} sentence chunks for streaming TTS")
            for i, s in enumerate(sentences):
                logger.debug(f"  Sentence {i+1}: '{s}'")

            # Use pre-warmed persistent queues (no thread creation overhead!)
            logger.debug("Using pre-warmed audio playback system")

            # Generate TTS with PARALLEL sentence generation and STREAMING playback
            # Key optimization: Start playing sentence 1 while generating sentence 2, etc.
            total_chunks = 0
            import wave
            import asyncio
            import tempfile
            sentence_audio_files = []

            async def generate_and_queue_sentence(idx: int, sentence: str) -> List[npt.NDArray[np.int16]]:
                """Generate TTS for one sentence and queue chunks immediately."""
                nonlocal total_chunks, first_chunk_queued

                logger.debug(f"TTS sentence {idx+1}/{len(sentences)}: '{sentence}' (PARALLEL)")
                sentence_chunks = []
                sentence_start = time.time()

                # TODO: Check if sentence 1 is very long and sentence 2 very short, we might have sentence 2 played first ?
                async for chunk in self.handler.tts.synthesize(sentence):
                    total_chunks += 1

                    # Convert to numpy array
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    sentence_chunks.append(audio_array)

                    # Collect for Gradio output
                    audio_chunks.append(audio_array)

                    # Queue for immediate playback (pre-warmed thread picks it up instantly!)
                    self.audio_queue.put(audio_array)

                    # Queue for wobbler (pre-warmed thread picks it up too)
                    self.wobbler_queue.put(chunk)

                    # Track first chunk queued (should be nearly instant playback)
                    if not first_chunk_queued:
                        first_chunk_queued = True
                        tracker.mark("audio_first_chunk_queued")
                        tracker.mark("audio_playback_started")  # With pre-warmed system, queueing = playing
                        tracker.mark("wobbler_first_chunk")
                        logger.info("First audio chunk playing - playback started while TTS continues in background")

                # For DEBUG : Save sentence debug file
                if sentence_chunks:
                    sentence_audio = np.concatenate(sentence_chunks)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_sentence_{idx+1}.wav', dir='/tmp')
                    with wave.open(temp_file.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(24000)
                        wf.writeframes(sentence_audio.tobytes())
                    sentence_audio_files.append(temp_file.name)

                    duration = time.time() - sentence_start
                    logger.debug(f"Sentence {idx+1} complete: {len(sentence_chunks)} chunks ({len(sentence_audio)} samples, {len(sentence_audio)/24000:.2f}s) generated in {duration:.2f}s -> {temp_file.name}")

                return sentence_chunks

            # Strategy: Generate sentences with intelligent overlap
            # - Sentence 1: Start immediately, play as soon as first chunk arrives
            # - Sentence 2: Start while sentence 1 is still generating/playing
            # - Sentence 3+: Start with small delay to avoid overwhelming TTS API

            tasks = []
            for idx, sentence in enumerate(sentences):
                if idx == 0:
                    # Sentence 1: Start immediately
                    task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                    tasks.append(task)
                elif idx == 1:
                    # Sentence 2: Start after a small delay (let sentence 1 begin streaming)
                    await asyncio.sleep(0.3)  # 300ms delay - allows sentence 1 to start playing
                    task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                    tasks.append(task)
                else:
                    # Sentence 3+: Start after previous task completes (avoid API overload)
                    # But playback is already happening from sentence 1 & 2!
                    if idx >= 2 and tasks:
                        await tasks[idx - 1]  # Wait for previous sentence to finish generating
                    task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                    tasks.append(task)

            # Wait for all sentence generations to complete
            await asyncio.gather(*tasks)
            logger.info(f"Parallel TTS complete: All {len(sentences)} sentences generated")

            logger.info(f"Generated {total_chunks} total audio chunks from {len(sentences)} sentences")
            logger.debug(f"Debug files: {sentence_audio_files}")

            # Signal end of this playback session to persistent threads
            self.audio_queue.put(None)
            self.wobbler_queue.put(None)

            # Wait for audio to finish playing (estimate based on total duration)
            if audio_chunks:
                total_samples = sum(len(chunk) for chunk in audio_chunks)
                duration_seconds = total_samples / 24000
                logger.info(f"Waiting {duration_seconds:.1f}s for playback to complete...")
                await asyncio.sleep(duration_seconds + 0.5)  # Add 500ms buffer

            logger.info("Playback complete (using pre-warmed system)")

            # Print latency summary now that full pipeline is complete (speech path)
            from reachy_mini_conversation_app.cascade.timing import tracker
            tracker.print_summary()

            # Save final combined audio for debugging
            if audio_chunks:
                import wave
                import tempfile
                combined_audio = np.concatenate(audio_chunks)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='_combined.wav', dir='/tmp')
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(combined_audio.tobytes())
                logger.debug(f"Combined debug file: Saved {len(audio_chunks)} chunks ({len(combined_audio)} samples, {len(combined_audio)/24000:.2f}s) to {temp_file.name}")

        except Exception as e:
            logger.exception(f"Error synthesizing speech: {e}")


    def _split_into_sentences(self, text: str, min_length: int = 8) -> list[str]:
        """Split text into sentence-like chunks for streaming TTS.

        Splits on: . ! ? , ; — (but keeps punctuation with the sentence)
        This allows starting TTS synthesis earlier for long texts.

        Args:
            text: Text to split
            min_length: Minimum characters per segment (default 8)
                       Segments shorter than this are combined with the next segment.

        Returns:
            List of text segments, each at least min_length characters (except possibly the last)

        """
        import re

        # Split on sentence boundaries but keep the punctuation
        # Pattern: split after punctuation + optional whitespace
        pattern = r'([.!?,;—]\s+)'
        parts = re.split(pattern, text)

        # Recombine parts to keep punctuation with sentences
        raw_sentences = []
        current = ""
        for part in parts:
            current += part
            # If this part ends with punctuation + space, it's a sentence boundary
            if re.match(pattern, part):
                if current.strip():
                    raw_sentences.append(current.strip())
                current = ""

        # Add any remaining text
        if current.strip():
            raw_sentences.append(current.strip())

        # If no splits, return original text
        if not raw_sentences:
            return [text]

        # Merge short segments to meet minimum length requirement
        merged_sentences = []
        accumulator = ""

        for sentence in raw_sentences:
            # Add to accumulator
            if accumulator:
                accumulator += " " + sentence
            else:
                accumulator = sentence

            # If accumulator is long enough, add it as a segment
            if len(accumulator) >= min_length:
                merged_sentences.append(accumulator)
                accumulator = ""

        # Add any remaining text (last segment can be shorter)
        if accumulator:
            # If we have previous segments, try to append to last one if it's not too long
            # Otherwise, add as separate segment
            if merged_sentences and len(merged_sentences[-1]) < min_length * 2:
                merged_sentences[-1] += " " + accumulator
            else:
                merged_sentences.append(accumulator)

        return merged_sentences if merged_sentences else [text]


    def _clear_history(self) -> tuple[List[Dict[str, Any]], str]:
        """Clear conversation history."""
        self.handler.conversation_history = []
        self.chat_history = []
        self.recorded_file = None
        return [], "History cleared"

    def launch(self, **kwargs: Any) -> None:
        """Launch Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)

    def close(self) -> None:
        """Close Gradio interface."""
        # Stop recording if still active
        self.recording = False
        if self.record_thread:
            self.record_thread.join(timeout=2)

        # Shutdown persistent playback threads
        logger.info("Shutting down pre-warmed audio system...")
        self.shutdown_event.set()

        if self.playback_thread:
            self.playback_thread.join(timeout=2)
        if self.wobbler_thread:
            self.wobbler_thread.join(timeout=2)

        # Stop robot media playback if using robot.media
        if self.use_robot_media and self.robot is not None and hasattr(self.robot, 'media'):
            logger.info("Stopping robot.media playback system...")
            self.robot.media.stop_playing()

        logger.info("Audio system shutdown complete")
