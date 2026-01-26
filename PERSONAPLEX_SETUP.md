# PersonaPlex Handler Setup Guide

This guide explains how to set up and use the PersonaPlex handler for Reachy Mini conversation app.

## Overview

PersonaPlex is an NVIDIA speech-to-speech conversational model that provides an alternative to the OpenAI handler. The implementation uses the Moshi server as the backend for PersonaPlex.

## Prerequisites

### Install uv

PersonaPlex installation uses `uv` for faster package management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on macOS with Homebrew
brew install uv
```

For other installation methods, see [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### For macOS (Apple Silicon)

1. **Install system dependencies**:
   ```bash
   brew install opus
   ```

2. **Set environment variable for PyTorch MPS fallback**:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```
   Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### For Linux

Standard PyTorch with CUDA support should work out of the box.

## Installation

### 1. Clone PersonaPlex Repository

```bash
cd ~
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex
```

### 2. Install PersonaPlex Dependencies

Follow the installation instructions from the PersonaPlex repository:

```bash
# Navigate to PersonaPlex directory
cd ~/personaplex

# Create a virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Moshi (the PersonaPlex backend)
uv pip install moshi/.
```

**Note**: You need to accept the PersonaPlex model license on HuggingFace:
1. Go to https://huggingface.co/nvidia/personaplex-7b-v1
2. Log in and accept the license
3. Set your HuggingFace token:
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

### 3. Fix macOS Compatibility Issue

There's a bug in the Moshi code that prevents loading voice prompts on non-CUDA devices. Apply this fix:

```bash
cd ~/personaplex
# Edit moshi/moshi/models/lm.py, line 979
# Change:     state = torch.load(path)
# To:         state = torch.load(path, map_location=self.lm_model.device)
```

Or apply the fix automatically:
```bash
cd ~/personaplex
sed -i.bak 's/state = torch.load(path)/state = torch.load(path, map_location=self.lm_model.device)/' moshi/moshi/models/lm.py
```

### 4. Install Reachy Mini Conversation App Dependencies

```bash
cd /path/to/reachy_mini_conversation_app
uv pip install -e .
```

The PersonaPlex handler requires `websockets>=12.0` which is now included in the dependencies.

## Running the Moshi Server

### On macOS

```bash
cd ~/personaplex
source .venv/bin/activate  # Activate PersonaPlex environment

# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Create SSL directory (required by Moshi server)
SSL_DIR=$(mktemp -d)

# Set MPS fallback environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start the Moshi server
python -m moshi.server --ssl "$SSL_DIR" --device mps --port 8998
```

**Note**: The server runs slower on Mac than on GPU-accelerated Linux systems due to MPS limitations.

### On Linux (with CUDA)

```bash
cd ~/personaplex
source .venv/bin/activate

# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Create SSL directory
SSL_DIR=$(mktemp -d)

# Start the Moshi server with CUDA
python -m moshi.server --ssl "$SSL_DIR" --device cuda --port 8998
```

### Server Options

- `--port`: WebSocket port (default: 8998)
- `--device`: Device to use (`mps` for Mac, `cuda` for GPU, `cpu` for CPU-only)
- `--ssl`: SSL certificate directory (required)

## Configuring Reachy Mini Conversation App

### Option 1: Environment Variables

Create or edit `.env` file in your Reachy Mini conversation app directory:

```bash
# Handler selection
HANDLER_TYPE=personaplex

# PersonaPlex configuration
PERSONAPLEX_SERVER_URL=ws://localhost:8998
PERSONAPLEX_DEVICE=mps  # or cuda, or cpu

# Optional: Custom profile
REACHY_MINI_CUSTOM_PROFILE=friendly
```

### Option 2: Export Environment Variables

```bash
export HANDLER_TYPE=personaplex
export PERSONAPLEX_SERVER_URL=ws://localhost:8998
export PERSONAPLEX_DEVICE=mps
```

## Running the App

Once the Moshi server is running and configuration is set:

```bash
cd /path/to/reachy_mini_conversation_app
reachy-mini-conversation-app --gradio
```

Or for headless mode:

```bash
reachy-mini-conversation-app
```

## Switching Between Handlers

You can easily switch between OpenAI and PersonaPlex handlers:

### Use OpenAI Handler

```bash
export HANDLER_TYPE=openai
export OPENAI_API_KEY=your_api_key_here
reachy-mini-conversation-app --gradio
```

### Use PersonaPlex Handler

```bash
export HANDLER_TYPE=personaplex
export PERSONAPLEX_SERVER_URL=ws://localhost:8998
reachy-mini-conversation-app --gradio
```

## Features

The PersonaPlex handler supports:

✅ **Speech-to-speech conversation** - Full duplex audio streaming
✅ **Tool calling** - All robot actions (movement, camera, emotions, dances)
✅ **Personality customization** - Custom profiles and personas
✅ **Vision integration** - Camera feed for visual context
✅ **Idle behavior** - Automatic robot actions during idle time
✅ **Transcript display** - Real-time user and assistant transcripts

## Troubleshooting

### Connection Errors

**Problem**: `Connection refused` or `Failed to connect to Moshi server`

**Solution**:
- Ensure the Moshi server is running: `ps aux | grep moshi.server`
- Check the port matches: default is 8998
- Verify the URL: `ws://localhost:8998` (not `wss://` or `http://`)

### Audio Issues

**Problem**: No audio output or distorted audio

**Solution**:
- Check sample rate matches (24kHz for both PersonaPlex and Reachy)
- Verify microphone and speaker permissions
- Check audio device settings in your system

### Performance Issues on Mac

**Problem**: Slow response or laggy audio

**Solution**:
- This is expected on Mac due to MPS backend limitations
- Consider running the Moshi server on a remote Linux machine with GPU
- Reduce background processes to free up resources
- Update `PERSONAPLEX_SERVER_URL` to point to remote server:
  ```bash
  export PERSONAPLEX_SERVER_URL=ws://remote-server-ip:8998
  ```

### MPS Fallback Errors

**Problem**: `RuntimeError: MPS backend not supported for operation aten::index_copy.out`

**Solution**:
- Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before starting Moshi server
- Add to shell profile for persistence:
  ```bash
  echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc  # or ~/.bashrc
  source ~/.zshrc
  ```

### Tool Calling Issues

**Problem**: Tools not being called or errors during tool execution

**Solution**:
- Check logs for tool call events
- Verify the Moshi server supports tool calling (may need custom implementation)
- Test with simple tools first (like `do_nothing` or `move_head`)

## Architecture Notes

The PersonaPlex handler (`PersonaPlexHandler`) is designed to match the interface of `OpenaiRealtimeHandler`:

- Extends `AsyncStreamHandler` from fastrtc
- Manages WebSocket connection to Moshi server
- Handles audio streaming (24kHz PCM)
- Processes JSON events for transcripts and tool calls
- Integrates with robot dependencies (movement, camera, vision, head wobbler)

### Protocol

The handler communicates with the Moshi server using:

- **Binary messages**: Raw audio data (int16 PCM at 24kHz)
- **JSON messages**: Events for configuration, transcripts, tool calls, errors

Example JSON messages:

```json
// Configuration
{
  "type": "config",
  "persona": "You are a helpful robot assistant.",
  "sample_rate": 24000
}

// Tool result
{
  "type": "tool_result",
  "call_id": "call_123",
  "result": "{\"status\": \"success\"}"
}

// Image data
{
  "type": "image",
  "data": "data:image/jpeg;base64,..."
}
```

## Additional Resources

- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [PersonaPlex Paper](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
- [PersonaPlex Demo](https://research.nvidia.com/labs/adlr/personaplex/)
- [Moshi Framework](https://github.com/kyutai-labs/moshi)
- [Mac Setup Issue](https://github.com/NVIDIA/personaplex/issues/11)

## License

PersonaPlex is released under the NVIDIA Open Model License and CC-BY-4.0. Please review the license before use.

## Support

For issues specific to:
- **PersonaPlex/Moshi**: Open an issue on the [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex/issues)
- **Reachy Mini Integration**: Open an issue on the Reachy Mini conversation app repository
- **General Reachy Mini**: Contact Pollen Robotics support
