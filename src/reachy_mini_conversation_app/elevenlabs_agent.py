"""ElevenLabs Agents integration for Reachy Mini.

Provides a custom AudioInterface and handler for ElevenLabs conversational AI.
"""

import time
import threading
import logging
from typing import Any, Optional, Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from fastrtc import audio_to_int16, audio_to_float32

try:
    import librosa
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    AudioInterface = object  # type: ignore[misc,assignment]

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.tools import ToolDependencies


logger = logging.getLogger(__name__)


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs agent."""

    agent_id: str
    api_key: Optional[str] = None
    requires_auth: bool = False


class ReachyAudioInterface(AudioInterface):  # type: ignore[misc]
    """Custom audio interface that connects ElevenLabs to Reachy Mini's speaker/mic."""

    # ElevenLabs expects chunks of this size for optimal performance
    INPUT_FRAMES_PER_BUFFER = 4000  # 250ms @ 16kHz (matches ElevenLabs default)
    
    def __init__(self, robot: ReachyMini):
        """Initialize the audio interface with robot hardware."""
        self.robot = robot
        # ElevenLabs uses 16kHz PCM16 audio for both input and output
        self.output_sample_rate = 16000
        self.input_sample_rate = 16000
        self.device_sample_rate = robot.media.get_audio_samplerate()
        
        # Threading control
        self.should_stop = threading.Event()
        self.input_thread: Optional[threading.Thread] = None
        self.input_callback: Optional[Callable[[bytes], None]] = None
        
        # Buffer for accumulating audio frames
        self.audio_buffer: list[bytes] = []
        self.buffer_size = 0
        
        logger.info(
            f"ReachyAudioInterface initialized: device={self.device_sample_rate}Hz, "
            f"input={self.input_sample_rate}Hz, output={self.output_sample_rate}Hz, "
            f"buffer={self.INPUT_FRAMES_PER_BUFFER} frames"
        )

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Start audio input/output streams.
        
        Args:
            input_callback: Callback function to send audio to the agent
        """
        logger.info("Starting audio streams with callback pattern")
        self.input_callback = input_callback
        self.should_stop.clear()
        
        # Start robot audio
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        logger.info("Waiting for audio streams to initialize...")
        time.sleep(1.0)  # Wait longer for pipelines to fully initialize
        
        # Test if we're getting audio
        test_frame = self.robot.media.get_audio_sample()
        if test_frame is not None:
            logger.info(f"Test audio frame received: shape={test_frame.shape}, dtype={test_frame.dtype}")
            test_mono = test_frame.T[0]
            test_rms = np.sqrt(np.mean(test_mono.astype(np.float32)**2))
            logger.info(f"Test audio RMS: {test_rms:.2f}")
        else:
            logger.warning("No test audio frame received!")
        
        # Start input thread to read microphone and call the callback
        self.input_thread = threading.Thread(target=self._input_thread, daemon=True)
        self.input_thread.start()
        logger.info("Input thread started")

    def stop(self) -> None:
        """Stop audio input/output streams."""
        logger.info("Stopping audio streams")
        self.should_stop.set()
        
        # Wait for input thread to finish
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=2.0)
        
        self.robot.media.stop_recording()
        self.robot.media.stop_playing()

    def _input_thread(self) -> None:
        """Thread that reads from microphone and calls the input callback.
        
        This runs in a separate thread and continuously reads audio from the robot's
        microphone, then calls the input_callback with the audio data.
        """
        logger.info("Input thread started - reading from microphone")
        frame_count = 0
        chunks_sent = 0
        
        while not self.should_stop.is_set():
            try:
                audio_frame = self.robot.media.get_audio_sample()
                if audio_frame is not None:
                    frame_count += 1
                    
                    # Convert stereo to mono (both channels are identical)
                    frame_mono = audio_frame.T[0]
                    
                    # Convert to int16 PCM format
                    frame_int16 = audio_to_int16(frame_mono)
                    audio_bytes = frame_int16.tobytes()
                    
                    # Add to buffer
                    self.audio_buffer.append(audio_bytes)
                    self.buffer_size += len(frame_int16)
                    
                    # Send when we have enough samples (4000 frames = 250ms)
                    if self.buffer_size >= self.INPUT_FRAMES_PER_BUFFER:
                        # Concatenate buffered audio
                        combined_audio = b''.join(self.audio_buffer)
                        
                        # Debug: Check audio levels
                        if chunks_sent < 3:  # Log first few chunks
                            audio_array_check = np.frombuffer(combined_audio, dtype=np.int16)
                            rms = np.sqrt(np.mean(audio_array_check.astype(np.float32)**2))
                            max_val = np.max(np.abs(audio_array_check))
                            logger.info(
                                f"Audio chunk {chunks_sent + 1}: {len(combined_audio)} bytes, "
                                f"{self.buffer_size} samples, RMS={rms:.1f}, Max={max_val}"
                            )
                        
                        # Call the callback with the buffered audio
                        if self.input_callback:
                            self.input_callback(combined_audio)
                            chunks_sent += 1
                            
                            if chunks_sent % 10 == 0:
                                logger.debug(f"Sent {chunks_sent} audio chunks to ElevenLabs")
                        
                        # Clear buffer
                        self.audio_buffer = []
                        self.buffer_size = 0
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in input thread: {e}", exc_info=True)
                break
        
        logger.info(f"Input thread ended. Frames captured: {frame_count}, Chunks sent: {chunks_sent}")

    def output(self, audio: bytes) -> None:
        """Play audio chunks from ElevenLabs through the speaker.
        
        This is called by ElevenLabs SDK to send audio to play.
        
        Args:
            audio: Raw PCM audio bytes from ElevenLabs (16kHz, int16, mono)
        """
        try:
            # Convert bytes to numpy array (int16 PCM)
            audio_array = np.frombuffer(audio, dtype=np.int16)
            
            # Convert to float32 for the robot's audio system
            audio_float = audio_to_float32(audio_array)
            
            # Resample if device sample rate differs from ElevenLabs output
            if self.output_sample_rate != self.device_sample_rate:
                logger.debug(f"Resampling from {self.output_sample_rate}Hz to {self.device_sample_rate}Hz")
                audio_float = librosa.resample(
                    audio_float,
                    orig_sr=self.output_sample_rate,
                    target_sr=self.device_sample_rate,
                )
            
            # Push to robot speaker
            self.robot.media.push_audio_sample(audio_float)
            
        except Exception as e:
            logger.error(f"Error playing audio output: {e}")

    def interrupt(self) -> None:
        """Interrupt the current audio playback."""
        logger.info("Audio interrupt requested")
        # Clear any queued audio by flushing the player
        # Note: The robot's media player doesn't have a built-in flush,
        # so we'll just let the queue drain naturally


class ElevenLabsStream:
    """Stream manager for ElevenLabs agent using Reachy Mini's audio hardware."""

    def __init__(self, config: ElevenLabsConfig, robot: ReachyMini, deps: ToolDependencies):
        """Initialize the ElevenLabs stream.
        
        Args:
            config: ElevenLabs configuration (agent_id, api_key)
            robot: ReachyMini instance
            deps: Tool dependencies for robot control
        """
        if not ELEVENLABS_AVAILABLE:
            raise ImportError(
                "ElevenLabs SDK not available. Install with: pip install elevenlabs"
            )
        
        self.config = config
        self.robot = robot
        self.deps = deps
        
        # Create ElevenLabs client
        self.client = ElevenLabs(api_key=config.api_key) if config.api_key else ElevenLabs()
        
        # Create custom audio interface
        self.audio_interface = ReachyAudioInterface(robot)
        
        # Create conversation instance
        self.conversation: Optional[Conversation] = None
        self._conversation_id: Optional[str] = None

    def _on_agent_response(self, response: str) -> None:
        """Callback when agent speaks."""
        logger.info(f"Agent: {response}")

    def _on_agent_response_correction(self, original: str, corrected: str) -> None:
        """Callback when agent corrects its response."""
        logger.info(f"Agent correction: {original} -> {corrected}")

    def _on_user_transcript(self, transcript: str) -> None:
        """Callback when user speaks."""
        logger.info(f"User: {transcript}")

    def _on_latency_measurement(self, latency: int) -> None:
        """Callback for latency measurements."""
        logger.debug(f"Latency: {latency}ms")

    def launch(self) -> None:
        """Start the conversation and run until interrupted."""
        logger.info(f"Starting ElevenLabs conversation with agent {self.config.agent_id}")
        
        # Initialize conversation
        self.conversation = Conversation(
            self.client,
            self.config.agent_id,
            requires_auth=self.config.requires_auth,
            audio_interface=self.audio_interface,
            
            # Callbacks for logging
            callback_agent_response=self._on_agent_response,
            callback_agent_response_correction=self._on_agent_response_correction,
            callback_user_transcript=self._on_user_transcript,
            callback_latency_measurement=self._on_latency_measurement,
        )
        
        try:
            # Start the conversation session
            self.conversation.start_session()
            
            # Wait for conversation to end (blocks until user interrupts or session ends)
            self._conversation_id = self.conversation.wait_for_session_end()
            
            logger.info(f"Conversation ended. ID: {self._conversation_id}")
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.close()
        except Exception as e:
            logger.exception(f"Error during conversation: {e}")
            self.close()

    def close(self) -> None:
        """Clean up resources and end the conversation."""
        logger.info("Closing ElevenLabs stream")
        
        if self.conversation:
            try:
                self.conversation.end_session()
            except Exception as e:
                logger.warning(f"Error ending conversation: {e}")
        
        # Stop audio interface
        if self.audio_interface:
            self.audio_interface.stop()
        
        logger.info("ElevenLabs stream closed")

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._conversation_id

