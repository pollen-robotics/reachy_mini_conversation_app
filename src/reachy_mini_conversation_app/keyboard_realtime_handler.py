import asyncio
import logging
import json

import numpy as np
from fastrtc import AdditionalOutputs
from numpy.typing import NDArray
from typing import Tuple
import soundfile as sf
from pathlib import Path
import reachy_mini_conversation_app
from pynput import keyboard

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call

from reachy_mini.motion.recorded_move import RecordedMoves
RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item

logger = logging.getLogger(__name__)

class KeyboardListener:
    """Non-blocking keyboard listener using pynput that enqueues all keys except escape."""
    
    def __init__(self, key_queue, event_loop_ref):
        self.running = False
        self.key_queue = key_queue
        self.event_loop_ref = event_loop_ref
        self._listener = None
    
    def start(self):
        """Start the keyboard listener using pynput."""
        if self.running:
            return
        
        self.running = True
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=None  # We only care about key presses
        )
        self._listener.start()
        logger.info("Keyboard listener started.")
    
    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
    
    def _on_key_press(self, key):
        """Handle key press events from pynput."""
        try:
            if not self.running:
                return False  # Stop listener
                
            # Handle all keys generically
            try:
                # Try to get character representation first
                if hasattr(key, 'char') and key.char is not None:
                    key_str = key.char
                else:
                    # Use string representation for all other keys
                    key_str = str(key).replace('Key.', '')
                self._enqueue_key(key_str)
                
            except AttributeError:
                # Fallback for any unexpected key types
                self._enqueue_key(str(key))

        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
            return False  # Stop listener on error
    
    def _enqueue_key(self, key):
        """Enqueue a key press to the async queue."""
        try:
            if self.event_loop_ref() and not self.event_loop_ref().is_closed():
                self.event_loop_ref().call_soon_threadsafe(self.key_queue.put_nowait, key)
        except Exception as e:
            logger.error(f"Error enqueuing key '{key}': {e}")


class KeyboardRealtimeHandler(AsyncStreamHandler):
    """A custom realtime handler that allows for keyboard event triggering + fastrtc Stream."""

    def __init__(self, deps: ToolDependencies):
        """Initialize the handler with keyboard support."""
        super().__init__(
            expected_layout="mono",
            input_sample_rate=24000,
            output_sample_rate=24000,
        )
        self.deps = deps

        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Initialize key queue and event loop reference
        self.key_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._event_loop = None  # Will be set in start_up
        
        # Initialize keyboard listener (will be created in start_up when event loop is available)
        self.keyboard_listener = None

        audio_folder = Path(reachy_mini_conversation_app.__file__).parent / "audio_data"
        self.actions_sequence = [
            (audio_folder / "response_audio_2.wav", None, audio_folder / "sway_results_2.json"),
            (audio_folder / "response_audio_3.wav", None, audio_folder / "sway_results_3.json"),
            (audio_folder / "response_audio_4.wav", "laughing1", audio_folder / "sway_results_4.json"),
            (audio_folder / "response_audio_5.wav", "scared1", audio_folder / "sway_results_5.json"),
        ]
        self.action_index = 0

    def copy(self) -> "KeyboardRealtimeHandler":
        """Create a copy of the handler."""
        return KeyboardRealtimeHandler(self.deps)
    
    async def start_up(self) -> None:
        """Start the handler."""
        # Store reference to the current event loop
        self._event_loop = asyncio.get_running_loop()
        
        # Create keyboard listener with event loop reference
        self.keyboard_listener = KeyboardListener(
            self.key_queue, 
            lambda: self._event_loop  # Pass event loop as a callable
        )
        
        # Start keyboard listener
        self.keyboard_listener.start()
        
        # Process keyboard events in a loop
        while True:
            try:
                # Wait for a key press (this will block until a key is pressed)
                key = await self.key_queue.get()
                logger.debug(f"Pressed key: '{key}' (ASCII: {ord(key) if len(key) == 1 else 'N/A'})")
                
                # Process any key press
                await self._process_key(key)

            except asyncio.CancelledError:
                logger.info("start_up cancelled")
                break
            except Exception as e:
                logger.error(f"Error in start_up: {e}")
                break
    
    async def _process_key(self, key: str):
        """Process any key press - you can customize this method."""
        logger.info(f"Processing key: '{key}'")
        
        if key == 's' and self.action_index < len(self.actions_sequence):
            next_action = self.actions_sequence[self.action_index]
            await self._queue_audio_delta(next_action[0], next_action[1], next_action[2])
            self.action_index += 1
    
    async def _queue_audio_delta(self, audio_file: str, emotion: str | None = None, sway_results_file: str | None = None):
        """Queue audio delta event to output_queue."""
        try:
            #Load audio file
            audio_data, sample_rate = sf.read(audio_file, dtype="int16")

            emotion_move = RECORDED_MOVES.get(emotion) if emotion is not None else None

            # Feed to head wobbler if available
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
                if sway_results_file is not None:
                    with open(sway_results_file, "r") as f:
                        sway_results = json.load(f)
                    self.deps.head_wobbler.feed_array(sway_results)
                else:
                    self.deps.head_wobbler.feed_raw(audio_data)
                if emotion_move is not None:
                    self.deps.movement_manager.queue_move(emotion_move)
            
            # Update activity time
            self.last_activity_time = asyncio.get_event_loop().time()
            
            # Queue audio for playback
            await self.output_queue.put(
                (
                    sample_rate,
                    audio_data.reshape(1, -1),
                )
            )
            logger.info("Audio delta event queued successfully")
            
        except Exception as e:
            logger.error(f"Failed to queue audio delta: {e}")
    
    async def _queue_function_call(self, emotion: str):
        """Queue function call execution to output_queue."""
        try:
            # Execute the dance tool directly (simulates function_call_arguments.done)
            tool_name = "play_emotion"
            args_json_str = '{"emotion": "' + emotion + '"}'
            logger.info(f"Executing tool: {tool_name}")
            
            # Execute the tool
            tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
            logger.debug("Tool '%s' executed successfully", tool_name)
            logger.debug("Tool result: %s", tool_result)
            
            # Queue the tool result for display
            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result),
                        "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                    }
                )
            )
            logger.info("Function call event queued successfully")

        except Exception as e:
            logger.error(f"Failed to queue function call: {e}")
            # Queue error message
            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant", 
                        "content": f"Tool execution failed: {str(e)}",
                        "metadata": {"title": "🛠️ Tool Error", "status": "error"},
                    }
                )
            )

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

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone (not used in keyboard mode)."""
        # In keyboard mode, we don't process microphone input
        pass

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Stop keyboard listener if it exists
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        from datetime import datetime
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal (simulate idle behavior)."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"
        
        # Queue an idle message
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": timestamp_msg,
                    "metadata": {"title": "💤 Idle Signal", "status": "info"},
                }
            )
        )