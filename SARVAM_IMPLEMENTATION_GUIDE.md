# Sarvam AI Integration - Implementation Guide

This guide provides step-by-step instructions for completing the Sarvam AI realtime API integration.

## Current Status

The branch `feat/sarvam-integration` includes:
✅ Architecture and provider abstraction layer
✅ SarvamRealtimeHandler scaffold
✅ Configuration and dependency management
✅ Integration with existing pipeline

## Next Steps

### 1. Research Sarvam Realtime API

**File**: Study Sarvam's API documentation
- Endpoint: Check the Sarvam API docs for the WebSocket endpoint
- Authentication: How to pass the API key
- Audio format: Expected sample rate, encoding, format
- Event types: What events are emitted by the API
- Request format: How to send audio and receive responses

**Key questions to answer:**
- What is the WebSocket endpoint URL?
- How is the API key sent (header, query param, or in the connection)?
- What audio format does it accept (sample rate, bit depth, codec)?
- What events are emitted during a conversation?
- How are transcripts provided (streaming or final)?
- Does it support tool calling/function calling?

### 2. Implement WebSocket Connection

**File**: `src/reachy_mini_conversation_app/sarvam_realtime.py`

Replace the placeholder in `_run_realtime_session()`:

```python
async def _run_realtime_session(self) -> None:
    """Establish and manage a single realtime session with Sarvam."""
    
    # Example structure (adapt based on actual Sarvam API):
    import websockets
    
    url = "wss://api.sarvam.ai/realtime"  # Replace with actual endpoint
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with websockets.connect(url, extra_headers=headers) as ws:
        self.connection = ws
        self._connected_event.set()
        
        # Send initial configuration
        await ws.send(json.dumps({
            "type": "session.start",
            "config": {
                "mode": "realtime",
                "language": "en",
                "voice": get_session_voice(),
            }
        }))
        
        # Event processing loop
        async for message in ws:
            event = json.loads(message)
            await self._handle_sarvam_event(event)
```

### 3. Implement Event Handling

**File**: `src/reachy_mini_conversation_app/sarvam_realtime.py`

Create a method to process events from Sarvam:

```python
async def _handle_sarvam_event(self, event: dict) -> None:
    """Process events from Sarvam API."""
    
    event_type = event.get("type", "")
    
    if event_type == "audio.delta":
        # Handle audio output
        delta_audio = event.get("delta", "")
        audio_bytes = base64.b64decode(delta_audio)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        self.last_activity_time = asyncio.get_event_loop().time()
        await self.output_queue.put(
            (self.output_sample_rate, audio_data.reshape(1, -1))
        )
    
    elif event_type == "transcription.partial":
        # Handle partial user transcript
        transcript = event.get("text", "")
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
    
    elif event_type == "transcription.final":
        # Handle completed user transcript
        transcript = event.get("text", "")
        await self.output_queue.put(
            AdditionalOutputs({"role": "user", "content": transcript})
        )
    
    elif event_type == "function_call":
        # Handle tool calling (if Sarvam supports it)
        await self._handle_tool_call(event)
    
    elif event_type == "error":
        logger.error("Sarvam error: %s", event.get("message", "unknown"))
```

### 4. Implement Audio Sending

**File**: Update `receive()` method

```python
async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
    """Receive audio from microphone and send to Sarvam."""
    
    if not self.connection:
        return
    
    input_sample_rate, audio_frame = frame
    
    # Reshape if needed
    if audio_frame.ndim == 2:
        if audio_frame.shape[1] > audio_frame.shape[0]:
            audio_frame = audio_frame.T
        if audio_frame.shape[1] > 1:
            audio_frame = audio_frame[:, 0]
    
    # Resample to 16kHz for Sarvam
    if self.input_sample_rate != input_sample_rate:
        audio_frame = resample(
            audio_frame, 
            int(len(audio_frame) * self.input_sample_rate / input_sample_rate)
        )
    
    audio_frame = audio_to_int16(audio_frame)
    
    try:
        # Send audio to Sarvam
        audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
        await self.connection.send(json.dumps({
            "type": "audio.data",
            "audio": audio_message,
            "sample_rate": self.input_sample_rate,
        }))
    except Exception as e:
        logger.debug("Failed to send audio: %s", e)
```

### 5. Implement Tool Calling (Optional)

If Sarvam supports function/tool calling, implement:

```python
async def _handle_tool_call(self, event: dict) -> None:
    """Handle tool/function calls from Sarvam."""
    
    tool_name = event.get("function_name", "")
    args_json_str = event.get("arguments", "{}")
    call_id = event.get("call_id", "")
    
    try:
        tool_result = await dispatch_tool_call(tools_json_str, self.deps)
        logger.debug("Tool '%s' executed successfully", tool_name)
    except Exception as e:
        logger.error("Tool '%s' failed", tool_name)
        tool_result = {"error": str(e)}
    
    # Send tool result back to Sarvam
    await self.connection.send(json.dumps({
        "type": "function_call_result",
        "call_id": call_id,
        "result": json.dumps(tool_result),
    }))
```

### 6. Test the Implementation

Create a test file:

```python
# tests/test_sarvam_realtime.py

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from reachy_mini_conversation_app.sarvam_realtime import SarvamRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

@pytest.fixture
def mock_deps():
    deps = MagicMock(spec=ToolDependencies)
    deps.movement_manager = MagicMock()
    deps.movement_manager.is_idle.return_value = False
    return deps

@pytest.mark.asyncio
async def test_sarvam_handler_initialization(mock_deps):
    handler = SarvamRealtimeHandler(mock_deps)
    assert handler.input_sample_rate == 16000
    assert handler.output_sample_rate == 16000
    assert handler.connection is None

@pytest.mark.asyncio
async def test_copy_handler(mock_deps):
    handler = SarvamRealtimeHandler(mock_deps)
    handler_copy = handler.copy()
    assert isinstance(handler_copy, SarvamRealtimeHandler)
    assert handler_copy.input_sample_rate == handler.input_sample_rate
```

### 7. Add Provider-Specific Tests

```python
# tests/conftest.py - Add Sarvam fixtures

@pytest.fixture(params=["openai", "sarvam"])
def realtime_provider(request):
    """Parametrized fixture to test both providers."""
    import os
    os.environ["REALTIME_PROVIDER"] = request.param
    yield request.param
```

### 8. Documentation and Examples

Create example files:
- `examples/sarvam_basic.py` - Simple usage example
- `examples/sarvam_with_robot.py` - Integration with Reachy Mini

### 9. Integration Testing

Create an integration test:

```python
# tests/test_sarvam_integration.py

@pytest.mark.asyncio
async def test_sarvam_conversation_flow():
    """Test a basic conversation flow with Sarvam."""
    handler = SarvamRealtimeHandler(mock_deps)
    
    # Start the handler
    # For testing, mock the WebSocket connection
    with patch('sarvam.AsyncClient') as mock_client:
        await handler.start_up()
        
        # Test receiving audio
        audio = (16000, np.zeros((16000,), dtype=np.int16))
        await handler.receive(audio)
        
        # Test emitting
        output = await asyncio.wait_for(handler.emit(), timeout=1.0)
        assert output is not None
```

## Debugging Tips

1. **Enable Debug Logging**:
   ```bash
   python -m reachy_mini_conversation_app --debug
   ```

2. **Monitor WebSocket Traffic**:
   ```python
   # Add detailed logging in _run_realtime_session()
   logger.debug("Sending: %s", json.dumps(message)[:100])
   logger.debug("Received: %s", event)
   ```

3. **Test Audio Stream**:
   ```python
   # Check if audio is being sent/received correctly
   logger.info("Audio frame size: %d bytes", len(audio_bytes))
   logger.info("Sample rate: %d", input_sample_rate)
   ```

4. **Check Event Types**:
   ```python
   # Log all events to understand the API
   logger.info("Event type: %s", event.get("type"))
   logger.info("Event payload: %s", event)
   ```

## Common Issues & Solutions

### Issue: WebSocket connection fails
- Check if the API endpoint URL is correct
- Verify API credentials and authentication method
- Check network connectivity and firewall

### Issue: No audio output
- Verify sample rate matches expectations
- Check audio encoding (PCM, 16-bit)
- Ensure audio queue is not stuck

### Issue: Tool calling not working
- Verify Sarvam API supports function calling
- Check if event types match documentation
- Debug tool dispatch with logging

### Issue: Memory leak or connection hangs
- Ensure connection cleanup in `shutdown()`
- Check for unclosed asyncio tasks
- Review event handling loop for blocking operations

## Resources

- Sarvam API Docs: https://docs.sarvam.ai/api-reference-docs/beta-apis
- WebSockets Library: https://websockets.readthedocs.io/
- AsyncIO Best Practices: https://docs.python.org/3/library/asyncio.html
- OpenAI Realtime Example: Study the OpenaiRealtimeHandler implementation

## Checklist

- [ ] Read Sarvam API documentation thoroughly
- [ ] Implement WebSocket connection
- [ ] Implement event handling
- [ ] Implement audio sending
- [ ] Test basic connection
- [ ] Implement tool calling (if supported)
- [ ] Add error handling and recovery
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Test with actual Reachy Mini robot
- [ ] Performance optimization
- [ ] Code review and cleanup

## Questions?

Refer to:
1. The Sarvam API documentation
2. The OpenAI implementation as reference
3. The base handler interface for required methods
4. The tool calling system in `tools/core_tools.py`
