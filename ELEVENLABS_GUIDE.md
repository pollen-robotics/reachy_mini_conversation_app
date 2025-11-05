# ElevenLabs Agent Integration Guide

This guide explains how to use ElevenLabs conversational AI agents with your Reachy Mini robot.

## Overview

The Reachy Mini conversation app now supports two AI agent providers:

- **OpenAI Realtime API** (default) - GPT-powered conversations with tool calling
- **ElevenLabs Agents** - Natural voice AI with ultra-low latency

## Setup

### 1. Install ElevenLabs Dependencies

```bash
# Using pip
pip install -e .[elevenlabs]

# Using uv
uv sync --extra elevenlabs
```

### 2. Create an ElevenLabs Agent

1. Go to the [ElevenLabs Platform](https://elevenlabs.io/app/conversational-ai)
2. Create a new conversational AI agent
3. Configure your agent's:
   - **Voice** - Choose from 100+ natural voices
   - **Prompt** - Define your agent's personality and behavior
   - **Knowledge Base** (optional) - Add custom information
   - **Tools** (optional) - Currently not integrated with robot tools
4. Copy your **Agent ID** from the agent settings

### 3. Configure Environment Variables

Copy and update your `.env` file:

```bash
cp .env.example .env
```

Add your ElevenLabs credentials:

```env
# ElevenLabs Configuration
ELEVENLABS_AGENT_ID=your_agent_id_here
ELEVENLABS_API_KEY=your_api_key_here  # Optional for public agents
```

**Note:** `ELEVENLABS_API_KEY` is only required if your agent has authentication enabled. Public agents don't need it.

## Usage

### Basic Usage

Run the app with ElevenLabs agent:

```bash
reachy-mini-conversation-app --agent elevenlabs
```

The audio will automatically play through the **Reachy Mini's built-in speaker** and record from its microphone.

### With Camera and Face Tracking

```bash
# With MediaPipe face tracking
reachy-mini-conversation-app --agent elevenlabs --head-tracker mediapipe

# With YOLO face tracking
reachy-mini-conversation-app --agent elevenlabs --head-tracker yolo
```

### Audio-Only Mode

Disable camera for audio-only conversations:

```bash
reachy-mini-conversation-app --agent elevenlabs --no-camera
```

### Debug Mode

Enable verbose logging:

```bash
reachy-mini-conversation-app --agent elevenlabs --debug
```

## Features

### ✅ Supported Features

- **Real-time audio streaming** through robot's speaker/microphone
- **Natural voice conversations** with ultra-low latency
- **Face tracking** with MediaPipe or YOLO
- **Robot movements** during conversation (breathing, idle poses)
- **Conversation transcripts** in console logs
- **Graceful shutdown** with Ctrl+C

### ❌ Current Limitations

- **No Gradio UI support** - Console mode only
- **No robot tool calling** - ElevenLabs tools not yet integrated with robot functions
- **No camera vision** - Camera tool not exposed to ElevenLabs agents
- **No head wobbler** - Audio-reactive head motion not implemented for ElevenLabs

## Architecture

### Audio Flow

```
User speaks → Reachy Mic → ElevenLabs API → Response → Reachy Speaker
```

The `ReachyAudioInterface` class implements ElevenLabs' `AudioInterface` protocol to:

1. **Capture audio** from Reachy's microphone (16kHz, mono)
2. **Stream to ElevenLabs** for processing
3. **Receive audio response** from ElevenLabs (24kHz, mono)
4. **Resample and play** through Reachy's speaker

### Key Components

- **`elevenlabs_agent.py`** - Main integration module

  - `ElevenLabsConfig` - Configuration dataclass
  - `ReachyAudioInterface` - Custom audio interface for robot hardware
  - `ElevenLabsStream` - Stream manager for conversation lifecycle

- **`main.py`** - Updated to support both OpenAI and ElevenLabs
- **`config.py`** - Environment variable management
- **`utils.py`** - CLI argument parsing with `--agent` flag

## Troubleshooting

### No Audio Output

**Problem:** Agent speaks but no sound from robot

**Solution:** Check that:

```bash
# Verify robot media is working
python -c "from reachy_mini import ReachyMini; r = ReachyMini(); print(r.media.get_audio_samplerate())"
```

### Agent ID Not Found

**Problem:** `ELEVENLABS_AGENT_ID is not set in .env file`

**Solution:**

1. Create an agent in the [ElevenLabs dashboard](https://elevenlabs.io/app/conversational-ai)
2. Copy the Agent ID
3. Add to `.env`: `ELEVENLABS_AGENT_ID=your_agent_id`

### Authentication Error

**Problem:** Agent requires authentication

**Solution:**

1. Get your API key from [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
2. Add to `.env`: `ELEVENLABS_API_KEY=your_api_key`

### Import Error

**Problem:** `ModuleNotFoundError: No module named 'elevenlabs'`

**Solution:**

```bash
pip install -e .[elevenlabs]
```

## Comparison: OpenAI vs ElevenLabs

| Feature                   | OpenAI Realtime   | ElevenLabs Agents       |
| ------------------------- | ----------------- | ----------------------- |
| **Latency**               | ~300-500ms        | ~200-300ms              |
| **Voice Quality**         | Good              | Excellent               |
| **Robot Tools**           | ✅ Full support   | ❌ Not yet integrated   |
| **Camera Vision**         | ✅ Supported      | ❌ Not available        |
| **Gradio UI**             | ✅ Supported      | ❌ Console only         |
| **Audio Reactive Motion** | ✅ Head wobbler   | ❌ Not implemented      |
| **Face Tracking**         | ✅ Supported      | ✅ Supported            |
| **Pricing**               | Per-token + Audio | Per-minute conversation |

## Future Enhancements

Planned improvements for ElevenLabs integration:

- [ ] **Tool calling integration** - Expose robot functions to ElevenLabs agents
- [ ] **Gradio UI support** - Web interface for ElevenLabs conversations
- [ ] **Audio-reactive motion** - Head wobbler synchronized with speech
- [ ] **Conversation history** - Save and review past conversations
- [ ] **Custom voices** - Upload and use custom voice models
- [ ] **Multi-language support** - Conversations in different languages

## Additional Resources

- [ElevenLabs Conversational AI Documentation](https://elevenlabs.io/docs/agents-platform/libraries/python)
- [ElevenLabs Agent Dashboard](https://elevenlabs.io/app/conversational-ai)
- [Reachy Mini Documentation](https://docs.pollen-robotics.com/)

## Support

For issues or questions:

- **ElevenLabs issues:** [ElevenLabs Support](https://elevenlabs.io/support)
- **Reachy Mini issues:** [Pollen Robotics GitHub](https://github.com/pollen-robotics/reachy-mini-conversation-app/issues)
