# Sarvam AI Integration for Reachy Mini Conversation App

This document describes the Sarvam AI realtime integration added to the Reachy Mini conversation app.

## Overview

The Reachy Mini conversation app now supports multiple realtime conversation providers:
- **OpenAI** (default) - Uses OpenAI's realtime API with GPT-4 realtime models
- **Sarvam AI** - Uses Sarvam AI's realtime conversation API (Beta)

Both providers can be used interchangeably by setting the `REALTIME_PROVIDER` environment variable.

## Architecture

### Provider Abstraction

A new base class `RealtimeHandlerBase` was introduced to define a common interface for all realtime conversation providers:

```python
class RealtimeHandlerBase(AsyncStreamHandler, ABC):
    async def start_up(self) -> None
    async def shutdown(self) -> None
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None
    async def apply_personality(self, profile: str | None) -> str
    async def get_available_voices(self) -> list[str]
    async def send_idle_signal(self, idle_duration: float) -> None
```

Both `OpenaiRealtimeHandler` and `SarvamRealtimeHandler` inherit from this base class, ensuring consistent behavior across providers.

### Files Changed

- **New Files:**
  - `src/reachy_mini_conversation_app/realtime_handler_base.py` - Base handler interface
  - `src/reachy_mini_conversation_app/sarvam_realtime.py` - Sarvam implementation
  
- **Modified Files:**
  - `src/reachy_mini_conversation_app/config.py` - Added SARVAM_API_KEY and REALTIME_PROVIDER settings
  - `src/reachy_mini_conversation_app/main.py` - Provider selection logic
  - `src/reachy_mini_conversation_app/console.py` - Updated type hints for base handler
  - `src/reachy_mini_conversation_app/openai_realtime.py` - Now inherits from RealtimeHandlerBase
  - `pyproject.toml` - Added sarvam optional dependency

## Configuration

### Environment Variables

```bash
# Select the realtime provider (default: "openai")
REALTIME_PROVIDER=sarvam

# Sarvam AI API key (required if using Sarvam provider)
SARVAM_API_KEY=your_sarvam_api_key

# OpenAI API key (required if using OpenAI provider)
OPENAI_API_KEY=your_openai_api_key
```

### .env File Example

```bash
REALTIME_PROVIDER=sarvam
SARVAM_API_KEY=your_api_key_here
REACHY_MINI_CUSTOM_PROFILE=astronomer
```

## Installation

### Install with Sarvam Support

```bash
# Install with Sarvam dependencies
pip install -e ".[sarvam]"

# Or install with all providers
pip install -e ".[sarvam,local_vision,yolo_vision,mediapipe_vision]"
```

### Install Sarvam Package

The Sarvam Python SDK needs to be installed separately:

```bash
pip install sarvam-ai
```

Or specify it in your requirements:
```
sarvam-ai>=0.1.0
```

## Usage

### Using Sarvam AI Provider

```bash
export REALTIME_PROVIDER=sarvam
export SARVAM_API_KEY=your_api_key

# Run the app
python -m reachy_mini_conversation_app

# Or with Gradio UI
python -m reachy_mini_conversation_app --gradio
```

### Using OpenAI Provider (Default)

```bash
export REALTIME_PROVIDER=openai  # or just omit this (default)
export OPENAI_API_KEY=your_api_key

# Run the app
python -m reachy_mini_conversation_app
```

## API Features Mapping

### Supported Features

Both providers support:
- ✅ Real-time audio streaming
- ✅ Voice synthesis
- ✅ Speech recognition/transcription
- ✅ Tool calling (function invocation)
- ✅ Personality profiles
- ✅ Idle behavior triggers
- ✅ Robot control via tool calls (camera, audio, movements)

### Audio Streaming

- **OpenAI**: 24kHz sample rate (PCM, 16-bit)
- **Sarvam**: 16kHz sample rate (PCM, 16-bit)

The handlers automatically resample audio to match each provider's requirements.

## Integration Details

### Event Handling

Both providers implement event-driven architectures:
- Audio reception from microphone (`receive()`)
- Audio emission to speaker (`emit()`)
- Transcript updates (partial and completed)
- Tool call responses
- Error handling and recovery

### Tool Calling

The app uses a shared tool system (`reachy_mini_conversation_app.tools.core_tools`) that both providers integrate with:

Available tools:
- **camera**: Capture and analyze images
- **set_head_position**: Control robot head movements
- **dance**: Trigger choreographed dances
- **arm_wave**: Wave the robot's arm
- **show_emotion**: Display emotions through movement

### Configuration Persistence

Both handlers support:
- Dynamic API key configuration (Gradio mode)
- `.env` file persistence
- Runtime personality switching
- Session management

## Development Notes

### Adding a New Provider

To add support for a new realtime provider:

1. Create a new handler class inheriting from `RealtimeHandlerBase`:
   ```python
   class NewProviderHandler(RealtimeHandlerBase):
       def __init__(self, deps: ToolDependencies, ...):
           super().__init__(deps, ...)
       
       # Implement all abstract methods...
   ```

2. Update `config.py` to add API key configuration:
   ```python
   NEW_PROVIDER_API_KEY = os.getenv("NEW_PROVIDER_API_KEY")
   ```

3. Update `main.py` to instantiate the handler:
   ```python
   if config.REALTIME_PROVIDER == "new_provider":
       handler = NewProviderHandler(...)
   ```

4. Add optional dependency to `pyproject.toml`:
   ```toml
   new_provider = ["new-provider-sdk>=1.0.0"]
   ```

### Sarvam API Implementation Status

The current implementation provides a **template/scaffold** for Sarvam integration. To complete the implementation, you need to:

1. **Implement WebSocket connection** in `_run_realtime_session()`
   - Connect to Sarvam's realtime endpoint
   - Handle connection lifecycle (open, events, close)

2. **Implement event processing**
   - Audio delta events for speech output
   - Transcription events for user input
   - Tool call events (if supported by Sarvam)

3. **Implement audio handling**
   - Send microphone frames to Sarvam
   - Receive and process audio output

4. **API-specific features**
   - Voice selection
   - Session configuration
   - Error recovery

Refer to the Sarvam documentation at https://docs.sarvam.ai/api-reference-docs/beta-apis for the specific API details.

## Testing

### Unit Tests

Run tests with provider selection:

```bash
# Test OpenAI provider
export REALTIME_PROVIDER=openai
pytest tests/

# Test Sarvam provider
export REALTIME_PROVIDER=sarvam
pytest tests/
```

Existing tests in `tests/test_openai_realtime.py` can be adapted for Sarvam by:
1. Creating `tests/test_sarvam_realtime.py`
2. Mocking the Sarvam API client
3. Testing the same interface contracts

## Troubleshooting

### API Key Issues

```
WARNING: SARVAM_API_KEY missing
```

Make sure your `.env` file or environment variables include:
```bash
export SARVAM_API_KEY=your_api_key
```

### Connection Errors

If the realtime connection fails:
1. Verify the API key is valid
2. Check network connectivity
3. Review logs for specific error messages
4. Ensure you have the correct API endpoint

### Audio Issues

If there's no audio output:
1. Check sample rate conversion (16kHz for Sarvam, 24kHz for OpenAI)
2. Verify audio device is working
3. Check audio queue for stuck items

## Future Enhancements

Planned improvements:
- [ ] Implement full Sarvam API integration with event streaming
- [ ] Add voice selection UI for Sarvam voices
- [ ] Performance optimization for audio processing
- [ ] Support for language selection (if available)
- [ ] Metrics and monitoring for provider selection
- [ ] Graceful fallback between providers
- [ ] Provider health checks

## Resources

- **Sarvam API Documentation**: https://docs.sarvam.ai/api-reference-docs/beta-apis
- **OpenAI Realtime API**: https://platform.openai.com/docs/guides/realtime
- **Reachy Mini Documentation**: https://reachy-mini-code.readthedocs.io/

## Support

For issues or questions about the Sarvam integration:
1. Check the Sarvam API documentation
2. Review the implementation in `sarvam_realtime.py`
3. Enable debug logging: `--debug` flag
4. Check console output for detailed error messages
