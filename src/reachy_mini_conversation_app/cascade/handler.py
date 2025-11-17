"""Main cascade handler orchestrating ASR → LLM → TTS pipeline.

Note: This implementation is designed for Gradio UI mode only.
"""

from __future__ import annotations
import json
import asyncio
import logging
import threading

# Type hints only (conditional import to avoid circular dependency)
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from reachy_mini_conversation_app.tools import ToolDependencies, get_tool_specs, dispatch_tool_call
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.cascade.asr import ASRProvider, OpenAIWhisperASR, StreamingASRProvider
from reachy_mini_conversation_app.cascade.llm import OpenAILLM, LLMProvider
from reachy_mini_conversation_app.cascade.tts import OpenAITTS, TTSProvider


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.transcript_analysis import TranscriptAnalysisManager


logger = logging.getLogger(__name__)


class CascadeHandler:
    """Main handler for cascade pipeline mode (Gradio UI only)."""

    def __init__(self, deps: ToolDependencies, skip_audio_playback: bool = False):
        """Initialize cascade handler.

        Args:
            deps: Tool dependencies for robot control
            skip_audio_playback: If True, don't play audio in _speak() (for Gradio mode)

        """
        self.deps = deps
        self.skip_audio_playback = skip_audio_playback

        # Initialize providers based on config
        self.asr = self._init_asr_provider()
        self.llm = self._init_llm_provider()
        self.tts = self._init_tts_provider()

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.processing_lock = asyncio.Lock()
        self.running = False

        # Event loop for async operations
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None

        # Transcript analysis (initialized below, may be None if no demo reactions)
        self.transcript_manager: Optional[TranscriptAnalysisManager] = None

        # Track last partial transcript to avoid log spam
        self._last_partial_transcript = ""

        # Get tool specs and convert to Chat Completions format
        # Note : get_tool_specs() returns Realtime API format, so we need Chat Completions format
        # Exclude speak tool from specs since it's handled internally
        self.tool_specs = self._convert_tool_specs_to_chat_format(get_tool_specs(exclusion_list=["speak"]))

        # Initialize transcript analysis (if demo has reactions)
        self.transcript_manager = self._init_transcript_analysis()

        logger.info(f"Cascade handler initialized (skip_audio_playback={skip_audio_playback})")


    def _convert_tool_specs_to_chat_format(self, realtime_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool specs from Realtime API format to Chat Completions API format.

        Realtime API format:
            {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Chat Completions API format:
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        chat_specs = []
        for spec in realtime_specs:
            if spec.get("type") == "function":
                chat_spec = {
                    "type": "function",
                    "function": {
                        "name": spec.get("name"),
                        "description": spec.get("description"),
                        "parameters": spec.get("parameters"),
                    }
                }
                chat_specs.append(chat_spec)
        return chat_specs


    def _init_asr_provider(self) -> ASRProvider:
        """Initialize ASR provider based on config."""
        provider = config.CASCADE_ASR_PROVIDER

        if provider == "openai_whisper":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            return OpenAIWhisperASR(api_key=config.OPENAI_API_KEY)

        elif provider == "parakeet":
            from reachy_mini_conversation_app.cascade.asr import ParakeetMLXASR
            return ParakeetMLXASR(
                model_name=config.PARAKEET_MODEL,
                precision=config.PARAKEET_PRECISION,
            )

        elif provider == "deepgram_streaming":
            from reachy_mini_conversation_app.cascade.asr import DeepgramStreamingASR
            if not config.DEEPGRAM_API_KEY:
                raise ValueError("DEEPGRAM_API_KEY not set in .env file")
            return DeepgramStreamingASR(
                api_key=config.DEEPGRAM_API_KEY,
                model=config.DEEPGRAM_MODEL,
                language="en",
            )

        elif provider == "parakeet_streaming":
            from reachy_mini_conversation_app.cascade.asr import ParakeetMLXStreamingASR
            return ParakeetMLXStreamingASR(
                model_name=config.PARAKEET_MODEL,
                precision=config.PARAKEET_PRECISION,
                context_size=config.PARAKEET_STREAMING_CONTEXT,
                depth=config.PARAKEET_STREAMING_DEPTH,
            )

        else:
            raise ValueError(f"Unknown ASR provider: {provider}")

    def _init_llm_provider(self) -> LLMProvider:
        """Initialize LLM provider based on config."""
        provider = config.CASCADE_LLM_PROVIDER

        # Add cascade-specific instructions about using speak tool instead of direct messaging
        # This is because some LLM cannot at the same time call tools and send a message
        # This allows the robot to for instance both speak and dance in one tool call.
        cascade_instructions = (
            get_session_instructions()
            + "\n\nIMPORTANT: Use the 'speak' tool to talk to the user. "
            "You can call 'speak' along with other tools in the same response."
        )

        if provider == "openai_gpt":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            return OpenAILLM(
                api_key=config.OPENAI_API_KEY,
                model=config.CASCADE_LLM_MODEL,
                system_instructions=cascade_instructions,
            )

        elif provider == "gemini":
            from reachy_mini_conversation_app.cascade.llm import GeminiLLM
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in .env file")
            return GeminiLLM(
                api_key=config.GEMINI_API_KEY,
                model=config.GEMINI_MODEL,
                system_instructions=cascade_instructions,
            )

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")


    def _init_tts_provider(self) -> TTSProvider:
        """Initialize TTS provider based on config."""
        provider = config.CASCADE_TTS_PROVIDER

        if provider == "openai_tts":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            return OpenAITTS(
                api_key=config.OPENAI_API_KEY,
                voice=config.CASCADE_TTS_VOICE,
                response_format="pcm",
            )

        elif provider == "kokoro":
            from reachy_mini_conversation_app.cascade.tts import KokoroTTS
            return KokoroTTS(voice=config.KOKORO_VOICE)

        elif provider == "elevenlabs":
            from reachy_mini_conversation_app.cascade.tts import ElevenLabsTTS
            if not config.ELEVENLABS_API_KEY:
                raise ValueError("ELEVENLABS_API_KEY not set in .env file")
            return ElevenLabsTTS(
                api_key=config.ELEVENLABS_API_KEY,
                voice_id=config.ELEVENLABS_VOICE_ID,
                model=config.ELEVENLABS_MODEL,
                output_format="pcm_24000",
            )

        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

    def _init_transcript_analysis(self) -> Optional[TranscriptAnalysisManager]:
        """Initialize transcript analysis from demo reactions.

        Returns:
            TranscriptAnalysisManager if demo has reactions, None otherwise

        """
        from reachy_mini_conversation_app.cascade.transcript_analysis import (
            KeywordAnalyzer,
            TranscriptAnalyzer,
            TranscriptAnalysisManager,
            get_demo_reactions,
        )

        reactions = get_demo_reactions()
        if not reactions:
            logger.info("No demo reactions configured, transcript analysis disabled")
            return None

        analyzers: List[TranscriptAnalyzer] = []

        # Add keyword analyzer if keywords defined
        if reactions.get("keywords"):
            analyzers.append(KeywordAnalyzer(reactions["keywords"]))
            logger.info(f"  Keyword analyzer: {len(reactions['keywords'])} keywords")

        # Add entity analyzer if entities defined (requires optional gliner extra)
        if reactions.get("entities"):
            try:
                from reachy_mini_conversation_app.cascade.transcript_analysis import EntityAnalyzer

                # Get GLiNER model (configurable per demo)
                gliner_model = reactions.get("gliner_model", "urchade/gliner_small-v2.1")

                analyzers.append(EntityAnalyzer(reactions["entities"], model_name=gliner_model))
                logger.info(f"  Entity analyzer: {len(reactions['entities'])} entity types (model: {gliner_model})")
            except ImportError:
                logger.warning(
                    "GLiNER not installed, skipping entity analyzer. "
                    "Install with: pip install 'reachy_mini_conversation_app[cascade_gliner]'"
                )

        if not analyzers:
            logger.info("No analyzers configured (keywords and entities both empty)")
            return None

        return TranscriptAnalysisManager(analyzers=analyzers, deps=self.deps)

    async def process_audio_manual(self, audio_bytes: bytes) -> str:
        """Process recorded audio through the cascade pipeline.

        Called manually from Gradio UI.

        Args:
            audio_bytes: WAV audio bytes from Gradio recording

        Returns:
            Transcript of user's speech

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Note: tracker.reset() is called in gradio_ui._stop_recording()
        # to capture user_stop_click in the same timeline

        async with self.processing_lock:
            try:
                # Update robot state - user is speaking
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(True)

                # 1. ASR: Audio → Text
                logger.info("Transcribing...")
                tracker.mark("transcribing_start")
                transcript = await self.asr.transcribe(audio_bytes, language="en")
                logger.info(f"User said: {transcript}")

                if not transcript.strip():
                    logger.warning("Empty transcript, ignoring")
                    if self.deps.movement_manager:
                        self.deps.movement_manager.set_listening(False)
                    return ""

                # Add user message to history
                self.conversation_history.append({"role": "user", "content": transcript})

                # Update robot state - done listening
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)

                # Analyze final transcript (parallel with LLM, fire-and-forget)
                if self.transcript_manager:
                    asyncio.create_task(self.transcript_manager.analyze_final(transcript))

                # 2. LLM: Text → Response + Tool Calls
                logger.info("Generating LLM response...")
                tracker.mark("llm_start")
                await self._process_llm_response()
                tracker.mark("llm_complete")

                # Reset transcript analysis for next conversation
                if self.transcript_manager:
                    self.transcript_manager.reset()

                # Note: summary will be printed in gradio_ui after TTS completes

                return transcript

            except Exception as e:
                logger.exception(f"Error processing audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise


    async def process_audio_streaming_start(self) -> None:
        """Initialize streaming ASR session.

        Called from Gradio UI when user starts recording with a streaming ASR provider.
        """
        if isinstance(self.asr, StreamingASRProvider):
            logger.info("Starting streaming ASR session")
            await self.asr.start_stream()

            # Update robot state - user is about to speak
            if self.deps.movement_manager:
                self.deps.movement_manager.set_listening(True)
        else:
            logger.warning("ASR provider does not support streaming")


    async def process_audio_streaming_chunk(self, chunk: bytes) -> Optional[str]:
        """Send audio chunk to streaming ASR and get partial transcript.

        Called from Gradio UI during recording to stream audio in real-time.

        Args:
            chunk: Audio chunk bytes (WAV format)

        Returns:
            Partial transcript if available, None otherwise

        """
        if isinstance(self.asr, StreamingASRProvider):
            await self.asr.send_audio_chunk(chunk)
            partial = await self.asr.get_partial_transcript()

            # Analyze partial transcript (debounced, fire-and-forget)
            # IMPORTANT: Only use stable text for entity extraction to avoid noisy draft tokens
            if partial and self.transcript_manager:
                # Only log if transcript changed (reduce spam)
                if partial != self._last_partial_transcript:
                    logger.debug(f"🎤 Got partial transcript: '{partial[:60]}...'")
                    self._last_partial_transcript = partial

                # Get stable text for analysis (if provider supports it)
                stable_text = partial  # Default to full partial
                if hasattr(self.asr, 'get_stable_text'):
                    stable_text = self.asr.get_stable_text()
                    if stable_text and stable_text != partial:
                        logger.debug(f"📌 Using stable text for analysis: '{stable_text[:60]}...'")

                await self.transcript_manager.analyze_partial(stable_text)

            if partial and partial != self._last_partial_transcript:
                logger.debug(f"Partial transcript: {partial}")
            return partial
        return None


    async def process_audio_streaming_end(self) -> str:
        """Finalize streaming session, get final transcript, and run LLM pipeline.

        Called from Gradio UI when user stops recording with a streaming ASR provider.

        Returns:
            Final complete transcript

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        async with self.processing_lock:
            try:
                # Get final transcript from streaming ASR
                if isinstance(self.asr, StreamingASRProvider):
                    logger.info("Finalizing streaming ASR session")
                    tracker.mark("transcribing_start")
                    transcript = await self.asr.end_stream()
                    tracker.mark("asr_complete", {"transcript_len": len(transcript)})
                else:
                    # Fallback to batch (shouldn't happen if UI checks properly)
                    logger.warning("ASR provider does not support streaming, this shouldn't happen")
                    return ""

                logger.info(f"User said: {transcript}")

                if not transcript.strip():
                    logger.warning("Empty transcript, ignoring")
                    if self.deps.movement_manager:
                        self.deps.movement_manager.set_listening(False)
                    return ""

                # Add user message to history
                self.conversation_history.append({"role": "user", "content": transcript})

                # Update robot state - done listening
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)

                # Analyze final transcript (parallel with LLM, fire-and-forget)
                if self.transcript_manager:
                    asyncio.create_task(self.transcript_manager.analyze_final(transcript))

                # 2. LLM: Text → Response + Tool Calls
                logger.info("Generating LLM response...")
                tracker.mark("llm_start")
                await self._process_llm_response()
                tracker.mark("llm_complete")

                # Reset transcript analysis for next conversation
                if self.transcript_manager:
                    self.transcript_manager.reset()

                # Reset partial transcript tracking
                self._last_partial_transcript = ""

                return transcript

            except Exception as e:
                logger.exception(f"Error processing streaming audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise


    async def _process_llm_response(self) -> None:
        """Process LLM response with streaming, tool calls, and TTS."""
        try:
            # Generate streaming response
            text_chunks: List[str] = []
            tool_calls: List[Dict[str, Any]] = []

            async for chunk in self.llm.generate(
                messages=self.conversation_history,
                tools=self.tool_specs,
                temperature=1.0,            # TODO: maybe move temperature parameter in config ?
            ):
                if chunk.type == "text_delta" and chunk.content:
                    text_chunks.append(chunk.content)
                    logger.debug(f"LLM text delta: {chunk.content}")

                elif chunk.type == "tool_call" and chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                    logger.info(f"LLM tool call: {chunk.tool_call}")

                elif chunk.type == "done":
                    logger.debug("LLM generation complete")
                    break

            # Create assistant message with text, tool calls...
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            full_text = ""
            if text_chunks:
                full_text = "".join(text_chunks)
                assistant_message["content"] = full_text
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if text_chunks or tool_calls:
                self.conversation_history.append(assistant_message)

            # Handle text-only responses: auto-inject speak tool call
            # This handles cases where LLM returns text without using the speak tool
            # In principle it should not happen thanks to the extra instructions.
            # If it happens, we create a synthetic tool call for speaking
            if full_text and not tool_calls:
                logger.info("❓LLM returned text without speak tool - auto-injecting speak call")

                synthetic_tool_call = {
                    "id": "auto_speak",
                    "type": "function",
                    "function": {
                        "name": "speak",
                        "arguments": json.dumps({"message": full_text})
                    }
                }
                await self._execute_tool_calls([synthetic_tool_call])
            elif tool_calls:
                # Process normal tool calls
                await self._execute_tool_calls(tool_calls)

        except Exception as e:
            logger.exception(f"Error processing LLM response: {e}")


    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Execute tool calls and handle camera and speak tool specially."""
        has_camera_tool = False

        for tool_call in tool_calls:
            try:
                call_id, tool_name, arguments = self.llm.parse_tool_call(tool_call)

                logger.info(f"Executing tool: {tool_name}({arguments})")

                # Execute tool
                result = await dispatch_tool_call(
                    tool_name,
                    json.dumps(arguments),
                    self.deps,
                )

                # Do not log result if the tool_name was camera
                if tool_name == "camera":
                    logger.info("Tool result: [camera image in base64, now shown]")
                else:
                    logger.info(f"Tool result: {result}")

                # Add tool result to conversation
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )

                # Special handling for camera tool
                if tool_name == "camera" and "b64_im" in result:
                    has_camera_tool = True
                    b64_im = result["b64_im"]
                    logger.info("Camera tool executed - adding image to conversation for LLM analysis")

                    # Add image to conversation as a user message (for LLM to analyze)
                    # Decode base64 to raw bytes for Gemini inline_data format
                    # TODO : shouldn't we inject this as a response to the LLM tool call, with the tool call id, etc.
                    # TODO : maybe we should differentiate images that are requested by the LLM and those that we give it because of logic on the app side ?
                    # TODO : This is Gemini. Check how other LLM providers like OpenAI require this ?
                    import base64

                    image_bytes = base64.b64decode(b64_im)

                    self.conversation_history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_bytes,  # Will be converted to Gemini format in LLM
                                }
                            ]
                        }
                    )

                # Special handling for speak tool
                elif tool_name == "speak" and "message" in result:
                    message = result["message"]
                    logger.info(f"Speaking: {message}")

                    # Only synthesize audio if not in Gradio mode (Gradio UI handles audio playback separately)
                    if not self.skip_audio_playback:
                        await self._speak(message)
                    else:
                        logger.debug("Skipping audio playback (Gradio mode will handle it)")

            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}: {e}")

                # Add error to conversation
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps({"error": str(e)}),
                    }
                )

        # If camera tool was used, call LLM again to analyze the image
        if has_camera_tool:
            logger.info("Camera image added to conversation - calling LLM to analyze it")
            await self._process_llm_response()


    async def _speak(self, text: str) -> None:
        """Synthesize speech and feed to head wobbler for animation.

        Audio Playback Strategy:
        - Console mode (future): Would play audio directly via sounddevice
        - Gradio mode: Audio is NOT played here - Gradio UI handles playback via browser

        This method only:
        1. Generates TTS audio chunks
        2. Feeds chunks to head_wobbler for synchronized head animation
        3. Rate-limits to match real-time audio playback speed
        """
        try:
            # Start head wobbler if available
            if self.deps.head_wobbler:
                self.deps.head_wobbler.reset()

            # Stream TTS audio for head wobbler animation
            audio_chunks = []
            async for chunk in self.tts.synthesize(text):
                audio_chunks.append(chunk)

                # Feed to head wobbler for motion
                if self.deps.head_wobbler:
                    # Note: OpenAI TTS outputs PCM int16 at 24kHz
                    self.deps.head_wobbler.feed_pcm(chunk)

                # Rate limiting: match audio generation speed
                # PCM int16 at 24kHz: 2 bytes per sample
                chunk_duration = len(chunk) / (2 * 24000)
                # Sleep for 95% of chunk duration to stay slightly ahead
                await asyncio.sleep(chunk_duration * 0.95)

            logger.info(f"Generated {len(audio_chunks)} audio chunks for head animation")

            # Wait for animation to finish (estimate based on audio length)
            total_bytes = sum(len(chunk) for chunk in audio_chunks)
            duration_seconds = total_bytes / (2 * 24000)
            await asyncio.sleep(duration_seconds + 0.5)  # Add buffer

            # Reset head wobbler
            if self.deps.head_wobbler:
                self.deps.head_wobbler.reset()

        except Exception as e:
            logger.exception(f"Error speaking: {e}")


    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logger.debug("Event loop started in background thread")
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()


    def start(self) -> None:
        """Start the cascade handler (Gradio mode)."""
        if self.running:
            logger.warning("Cascade handler already running")
            return

        logger.info("Starting cascade handler (Gradio mode)...")
        self.running = True

        # Start event loop in background thread for async operations
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        # Wait for event loop to start
        import time
        time.sleep(0.5)

        # Warmup LLM connection
        if hasattr(self.llm, 'warmup') and self.loop:
            logger.info("Pre-warming LLM connection...")
            asyncio.run_coroutine_threadsafe(
                self.llm.warmup(tools=self.tool_specs),
                self.loop
            )

        logger.info("Cascade handler started")


    def stop(self) -> None:
        """Stop the cascade handler."""
        if not self.running:
            return

        logger.info("Stopping cascade handler...")
        self.running = False

        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.loop_thread:
            self.loop_thread.join(timeout=5)

        logger.info("Cascade handler stopped")


    def copy(self) -> CascadeHandler:
        """Create a copy of the handler (for compatibility with stream interface)."""
        return CascadeHandler(self.deps)

    # Methods for compatibility with fastrtc interface
    async def start_up(self) -> None:
        """Start up handler (async)."""
        self.start()

    async def shutdown(self) -> None:
        """Shutdown handler (async)."""
        self.stop()

    async def receive(self, frame: tuple[int, npt.NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone (fastrtc interface).

        In cascade mode with VAD, this would buffer audio and trigger processing.
        For now, this is a no-op since we use Gradio's manual recording.
        """
        pass

    async def emit(self) -> tuple[int, npt.NDArray[np.int16]] | None:
        """Emit audio frame for playback (fastrtc interface).

        Returns audio chunks from the output queue for WebRTC streaming.
        This allows cascade mode to work with fastrtc.Stream for robot speaker output.
        """
        # For now, return None since cascade mode doesn't stream audio output
        # Audio is played via sounddevice in Gradio UI
        return None

