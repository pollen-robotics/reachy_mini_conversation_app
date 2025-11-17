# Cascade Mode Architecture

## Overview

The cascade mode implements a traditional **ASR → LLM → TTS** pipeline for robot conversation. 
For now, it's designed specifically for Gradio UI with manual push-to-talk recording.

---

## Components

### 1. **GradioUI** (`gradio_ui.py`)
**Role:** User interface and audio I/O management

- Display chat interface
- Record audio from microphone (push-to-talk)
- Convert recorded audio to WAV bytes
- Send audio to Handler for processing
- Extract responses from Handler's conversation history
- Synthesize speech (TTS) and play through robot speaker
- Synchronize head wobbler animation with audio playback

**Key Methods:**
- `_start_recording()` / `_stop_recording()`: Manage microphone recording
- `_process_audio_sync()`: Wrapper to call Handler's async processing
- `_synthesize_for_gradio()`: Generate TTS and play audio

**Audio Playback System:**
- Pre-warmed persistent threads for zero-latency playback
- Parallel sentence synthesis (sentence 1 plays while sentence 2 generates)
- Direct sounddevice integration for robot speaker

---

### 2. **Handler** (`handler.py`)
**Role:** Pipeline orchestrator and conversation manager

- Initialize ASR/LLM/TTS providers from config
- Manage conversation history (messages, tool calls, results)
- Orchestrate ASR → LLM → Tool Execution pipeline
- Execute tool calls (speak, camera, robot actions)
- Handle multi-modal inputs (text + images from camera tool)
- **Does NOT play audio** (UI handles that)

**Key Methods:**
- `process_audio_manual()`: Main entry point for processing recorded audio
- `_process_llm_response()`: Stream LLM response and handle tool calls
- `_execute_tool_calls()`: Execute tool functions and update conversation
- `_speak()`: Placeholder method (skipped in Gradio mode)

**Important:** Handler has `skip_audio_playback=True` in Gradio mode, so it doesn't play audio itself.

---

## Data Flow

### **Recording → Transcription → Response → Playback**

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. RECORDING PHASE (Gradio UI)                                  │
└──────────────────────────────────────────────────────────────────┘
User clicks "START Recording"
    ↓
gradio_ui._start_recording()
    ↓
gradio_ui._record_audio() [background thread]
    ↓ (captures audio chunks via sounddevice)
    │
User clicks "STOP Recording"
    ↓
gradio_ui._stop_recording()
    ↓
Concatenate audio chunks → WAV bytes
    ↓
gradio_ui.toggle_recording_wrapper()
    ↓
gradio_ui._process_audio_sync(wav_bytes)
    ↓
[Async wrapper to Handler]


┌──────────────────────────────────────────────────────────────────┐
│ 2. HANDLER PIPELINE (Handler)                                    │
└──────────────────────────────────────────────────────────────────┘
handler.process_audio_manual(audio_bytes)
    ↓
┌─────────────────────────────┐
│ ASR: Audio → Text          │
└─────────────────────────────┘
    asr.transcribe(audio_bytes) → transcript
    ↓
    Add to conversation_history: {"role": "user", "content": transcript}
    ↓
┌─────────────────────────────┐
│ LLM: Generate Response     │
└─────────────────────────────┘
    handler._process_llm_response()
        ↓
        llm.generate(messages, tools) [streaming]
        ↓
        Collect: text_chunks, tool_calls
        ↓
        Add to conversation_history: {"role": "assistant", "content": ..., "tool_calls": ...}
        ↓
┌─────────────────────────────┐
│ TOOLS: Execute Actions     │
└─────────────────────────────┘
    handler._execute_tool_calls(tool_calls)
        ↓
        For each tool:
            dispatch_tool_call(tool_name, args, deps)
            ↓
            Add to conversation_history: {"role": "tool", "name": tool_name, "content": result}
            ↓
            Special handling:
            • speak tool: Extract message (UI will synthesize later)
            • camera tool: Add image to conversation → Re-call LLM to analyze
            • other tools: Log execution
        ↓
    Return transcript to UI


┌──────────────────────────────────────────────────────────────────┐
│ 3. RESPONSE EXTRACTION (Gradio UI)                              │
└──────────────────────────────────────────────────────────────────┘
gradio_ui._process_audio_async()
    ↓
    Scan handler.conversation_history for new messages:
        • role="tool" + name="speak" → Extract message for TTS
        • role="tool" + name="camera" → Extract image for display
        • role="tool" + other → Show "Used tool: <name>"
    ↓
    Collect all speak messages
    ↓
┌─────────────────────────────┐
│ TTS: Text → Audio          │
└─────────────────────────────┘
    gradio_ui._synthesize_for_gradio(combined_text)
        ↓
        Split text into sentences
        ↓
        For each sentence (parallel generation):
            tts.synthesize(sentence) [streaming chunks]
            ↓
            For each audio chunk:
                audio_queue.put(chunk)      → Pre-warmed playback thread
                wobbler_queue.put(chunk)    → Pre-warmed wobbler thread
        ↓
    [Playback happens in parallel with TTS generation]
    ↓
    Update chat_history for display
```

---

## Provider Interface

### **ASR Provider** (`cascade/asr/base.py`)
```python
async def transcribe(audio_bytes: bytes, language: Optional[str]) -> str
```
- Input: WAV audio bytes
- Output: Transcribed text
- Implementations: `OpenAIWhisperASR`, `ParakeetMLXASR`

### **LLM Provider** (`cascade/llm/base.py`)
```python
async def generate(messages, tools, temperature) -> AsyncIterator[StreamChunk]
```
- Input: Conversation history, available tools
- Output: Streaming chunks (text deltas, tool calls, done signal)
- Implementations: `OpenAILLM`, `GeminiLLM`

### **TTS Provider** (`cascade/tts/base.py`)
```python
async def synthesize(text: str) -> AsyncIterator[bytes]
```
- Input: Text to speak
- Output: Streaming audio chunks (PCM int16)
- Implementations: `OpenAITTS`, `KokoroTTS`, `ElevenLabsTTS`

---

## Key Design Decisions

### **Why Handler doesn't play audio:**
- Gradio UI needs full control over audio playback for:
  - Pre-warmed playback system (zero-latency start)
  - Parallel sentence synthesis (start playing sentence 1 while generating sentence 2)
  - Head wobbler synchronization
  - User experience timing (can't block handler's async loop)

### **Why TTS is called by UI, not Handler:**
- Handler executes `speak` tool but doesn't synthesize
- UI extracts speak messages from conversation history
- UI calls TTS provider directly for synthesis + playback
- Allows UI to optimize playback strategy independently

### **Why conversation history is in Handler:**
- Handler needs full context for LLM generation
- Tool results must be added to conversation for multi-turn reasoning
- Camera tool adds images to conversation → LLM analyzes them
- UI displays messages by reading Handler's conversation history

---

## Sequence Diagram

```
User         Gradio UI              Handler              ASR    LLM    TTS
 │               │                      │                 │      │      │
 │──START────────>│                     │                 │      │      │
 │               │──record audio───┐    │                 │      │      │
 │               │<─────────────────┘    │                 │      │      │
 │──STOP─────────>│                      │                 │      │      │
 │               │──wav_bytes───────────>│                 │      │      │
 │               │                       │──audio_bytes───>│      │      │
 │               │                       │<──transcript────│      │      │
 │               │                       │──messages───────────>│ │      │
 │               │                       │<──response+tools─────│ │      │
 │               │                       │──execute tools──┐     │ │      │
 │               │                       │<────────────────┘     │ │      │
 │               │<──returns transcript──│                       │ │      │
 │               │──extract speak msgs───┐                       │ │      │
 │               │<──────────────────────┘                       │ │      │
 │               │──text─────────────────────────────────────────────>│
 │               │<──audio chunks────────────────────────────────────│
 │<──hears robot speak──────────────────────────────────────────────┘
```

---

## Configuration

Providers are selected via `config.py`:
```python
CASCADE_ASR_PROVIDER = "parakeet"  # or "openai_whisper"
CASCADE_LLM_PROVIDER = "openai_gpt"  # or "gemini"
CASCADE_TTS_PROVIDER = "kokoro"  # or "openai_tts", "elevenlabs"
```

Handler initializes providers at startup based on these settings.

---

## Future Considerations (Not Implemented)

- **Streaming ASR**: Would require UI to send audio chunks during recording, Handler to manage streaming transcription
- **Voice Activity Detection (VAD)**: Could trigger pipeline before user clicks STOP
- **Streaming TTS**: Already implemented in UI's parallel sentence synthesis
- **Console Mode**: Handler's `_speak()` method exists but is disabled in Gradio mode
