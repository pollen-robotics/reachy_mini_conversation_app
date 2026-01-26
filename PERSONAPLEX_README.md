# PersonaPlex Handler for Reachy Mini

This directory now includes a PersonaPlex handler as an alternative to the OpenAI handler for speech-to-speech conversations.

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install app dependencies (includes websockets for PersonaPlex)
uv pip install -e .
```

### 2. Set Up PersonaPlex/Moshi

```bash
# Clone PersonaPlex
git clone https://github.com/NVIDIA/personaplex.git ~/personaplex
cd ~/personaplex

# Create virtual environment with uv
uv venv

# Activate and install
source .venv/bin/activate
uv pip install moshi/.

# Accept the model license at https://huggingface.co/nvidia/personaplex-7b-v1
# Then set your HuggingFace token
export HF_TOKEN=your_huggingface_token
```

### 3. Start Moshi Server

**Option A: Using the helper script (recommended)**

```bash
# Auto-detects your platform and device
./start_moshi_server.sh

# Or specify options
./start_moshi_server.sh --device mps --port 8998
```

**Option B: Manually**

```bash
# On macOS
export HF_TOKEN=your_huggingface_token
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd ~/personaplex
source .venv/bin/activate
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --device mps --port 8998

# On Linux with GPU
export HF_TOKEN=your_huggingface_token
cd ~/personaplex
source .venv/bin/activate
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --device cuda --port 8998
```

### 4. Configure Reachy App

Create or edit `.env`:

```bash
HANDLER_TYPE=personaplex
PERSONAPLEX_SERVER_URL=ws://localhost:8998
PERSONAPLEX_DEVICE=mps  # or cuda, or cpu
```

### 5. Run the App

```bash
reachy-mini-conversation-app --gradio
```

## What's New

### Files Added

- **[personaplex_realtime.py](src/reachy_mini_conversation_app/personaplex_realtime.py)** - PersonaPlex handler implementation
- **[PERSONAPLEX_SETUP.md](PERSONAPLEX_SETUP.md)** - Detailed setup and troubleshooting guide
- **[start_moshi_server.sh](start_moshi_server.sh)** - Helper script to start Moshi server
- **PERSONAPLEX_README.md** - This file

### Files Modified

- **[config.py](src/reachy_mini_conversation_app/config.py)** - Added handler selection and PersonaPlex config
- **[main.py](src/reachy_mini_conversation_app/main.py)** - Added handler switching logic
- **[pyproject.toml](pyproject.toml)** - Added websockets dependency
- **[.env.example](.env.example)** - Added PersonaPlex configuration examples

## Features

The PersonaPlex handler includes:

✅ **Speech-to-speech conversation** - Full duplex audio streaming at 24kHz
✅ **Tool calling** - Support for all robot actions (movements, camera, emotions, dances)
✅ **Personality customization** - Custom profiles and persona instructions
✅ **Vision integration** - Camera feed for visual context in conversations
✅ **Idle behavior** - Automatic robot actions during idle periods
✅ **Transcript display** - Real-time user and assistant transcripts in UI
✅ **Runtime switching** - Easy switch between OpenAI and PersonaPlex

## Switching Between Handlers

### Use OpenAI (default)

```bash
export HANDLER_TYPE=openai
export OPENAI_API_KEY=your_key
reachy-mini-conversation-app --gradio
```

### Use PersonaPlex

```bash
export HANDLER_TYPE=personaplex
export PERSONAPLEX_SERVER_URL=ws://localhost:8998
reachy-mini-conversation-app --gradio
```

## Architecture

```
┌─────────────────────────────────────┐
│   Reachy Mini Conversation App      │
│                                     │
│  ┌────────────────────────────┐    │
│  │   main.py                  │    │
│  │   (selects handler)        │    │
│  └────────────┬───────────────┘    │
│               │                     │
│      ┌────────┴────────┐           │
│      │                 │           │
│  ┌───▼────┐      ┌────▼─────┐     │
│  │ OpenAI │      │PersonaPlex│     │
│  │Handler │      │ Handler   │     │
│  └───┬────┘      └────┬─────┘     │
│      │                │           │
└──────┼────────────────┼───────────┘
       │                │
       │         ┌──────▼────────┐
       │         │ Moshi Server  │
       │         │ (PersonaPlex) │
       │         └───────────────┘
       │
  ┌────▼──────┐
  │OpenAI API │
  └───────────┘
```

## Implementation Details

The `PersonaPlexHandler` class:

- Extends `AsyncStreamHandler` (same base as `OpenaiRealtimeHandler`)
- Connects to Moshi server via WebSocket
- Handles binary audio data (24kHz PCM int16)
- Processes JSON events for transcripts, tool calls, and errors
- Integrates with all Reachy dependencies (movement manager, camera, vision, head wobbler)

### Communication Protocol

**Audio**: Binary WebSocket messages containing raw PCM audio (int16, 24kHz, mono)

**Events**: JSON WebSocket messages for:
- Configuration (persona, sample rate)
- Transcripts (partial and complete)
- Tool calls and results
- Images (base64 encoded)
- Errors

## Performance Notes

- **macOS**: Runs slower due to MPS backend limitations (expected behavior)
- **Linux GPU**: Best performance with CUDA
- **Remote Server**: Can run Moshi on remote GPU and connect via WebSocket

## Troubleshooting

See [PERSONAPLEX_SETUP.md](PERSONAPLEX_SETUP.md) for:
- Connection issues
- Audio problems
- Performance optimization
- MPS fallback errors
- Tool calling debugging

## Resources

- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [PersonaPlex Paper](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
- [Mac Setup Guide](https://github.com/NVIDIA/personaplex/issues/11)
- [Moshi Framework](https://github.com/kyutai-labs/moshi)

## License

PersonaPlex is licensed under NVIDIA Open Model License and CC-BY-4.0.

## Contributing

To improve the PersonaPlex handler:
1. Test with different personas and tools
2. Report issues specific to PersonaPlex integration
3. Optimize the WebSocket protocol for better performance
4. Add support for additional PersonaPlex features

## Support

For questions or issues:
- PersonaPlex/Moshi: [GitHub Issues](https://github.com/NVIDIA/personaplex/issues)
- Reachy Integration: Open an issue on this repository
- General Reachy: Contact Pollen Robotics
