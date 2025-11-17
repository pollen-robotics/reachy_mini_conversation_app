# ASR Streaming Strategies

## Current Limitation

**Batch-only ASR:** User must finish speaking → entire audio sent to Parakeet → wait for complete transcription

**Latency breakdown:**
- Recording ends: 0ms
- Write to temp file: ~50-100ms
- Parakeet inference: 500-2000ms (depends on audio length)
- **Total ASR latency: 550-2100ms**

**Goal:** Reduce time-to-transcript by processing audio while user is still speaking.

---

## Option 1: Deepgram Streaming ASR ⚡

### **Description**
Cloud-based real-time ASR with WebSocket streaming. Audio chunks sent continuously during recording, partial transcripts received in real-time.

### **Performance**
- **First partial transcript:** ~300-500ms after speech starts
- **Final transcript:** ~200-400ms after user stops speaking
- **Latency improvement:** 60-80% reduction vs. current Parakeet

### **Pros**
- ✅ Excellent latency (best option)
- ✅ True streaming with incremental results
- ✅ Very accurate (state-of-the-art models)
- ✅ Minimal code complexity
- ✅ Production-ready, well-documented SDK

### **Cons**
- ❌ Cloud-based (requires internet)
- ❌ Costs money (~$0.0043/min, ~$12.90 for 50 hours)
- ❌ Privacy: audio sent to Deepgram servers
- ❌ Dependency on third-party service

---

### **Architecture Impact**

#### **New Files**
```
cascade/asr/
  ├── base_streaming.py          # New abstract base for streaming ASR
  └── deepgram_streaming.py      # Deepgram implementation (~200 lines)
```

#### **Modified Files**

**`handler.py`** - **MEDIUM changes** (~50 lines modified/added)
```python
# Current: Single method for batch ASR
async def process_audio_manual(self, audio_bytes: bytes) -> str:
    transcript = await self.asr.transcribe(audio_bytes)
    # ... rest of pipeline

# New: Add streaming support (optional path)
async def process_audio_streaming_start(self) -> None:
    """Initialize streaming ASR session."""
    if isinstance(self.asr, StreamingASRProvider):
        await self.asr.start_stream()

async def process_audio_streaming_chunk(self, chunk: bytes) -> Optional[str]:
    """Send audio chunk, get partial transcript."""
    if isinstance(self.asr, StreamingASRProvider):
        await self.asr.send_audio_chunk(chunk)
        return await self.asr.get_partial_transcript()
    return None

async def process_audio_streaming_end(self) -> str:
    """Finalize stream, get final transcript, run LLM pipeline."""
    if isinstance(self.asr, StreamingASRProvider):
        transcript = await self.asr.end_stream()
    else:
        # Fallback to batch (shouldn't happen)
        transcript = await self.asr.transcribe(self._buffered_audio)

    # Continue with existing LLM pipeline
    self.conversation_history.append({"role": "user", "content": transcript})
    await self._process_llm_response()
    return transcript

# Existing process_audio_manual() stays unchanged for backward compatibility
```

**Key changes:**
- Add 3 new methods for streaming workflow
- Keep existing `process_audio_manual()` for batch ASR (Parakeet, Whisper)
- No changes to LLM/TTS pipeline
- Handler checks ASR provider type to choose path

---

**`gradio_ui.py`** - **MEDIUM changes** (~80 lines modified/added)

```python
# Current: Record all audio, then send batch
def _record_audio(self) -> None:
    """Record audio in background thread."""
    # Records to self.audio_frames list
    while self.recording:
        data, _ = stream.read(1024)
        self.audio_frames.append(data)

# New: Also send chunks to handler during recording
def _record_audio(self) -> None:
    """Record audio + stream to handler."""
    # Initialize streaming ASR session
    if self._is_streaming_asr():
        asyncio.run_coroutine_threadsafe(
            self.handler.process_audio_streaming_start(),
            self.handler.loop
        )

    while self.recording:
        data, _ = stream.read(1024)
        self.audio_frames.append(data)  # Keep for fallback

        # Send chunk to streaming ASR
        if self._is_streaming_asr():
            chunk_wav = self._convert_to_wav_bytes(data)
            future = asyncio.run_coroutine_threadsafe(
                self.handler.process_audio_streaming_chunk(chunk_wav),
                self.handler.loop
            )
            # Optional: Get partial transcript for live display
            partial = future.result(timeout=0.1)
            if partial:
                self._update_partial_transcript(partial)

# Modified: Stop recording → finalize stream
def _stop_recording(self) -> tuple[str, str]:
    """Stop recording and finalize streaming."""
    self.recording = False

    if self._is_streaming_asr():
        # Finalize stream (gets final transcript + runs LLM)
        future = asyncio.run_coroutine_threadsafe(
            self.handler.process_audio_streaming_end(),
            self.handler.loop
        )
        transcript = future.result(timeout=60)
    else:
        # Existing batch path (Parakeet)
        audio_data = np.concatenate(self.audio_frames)
        wav_bytes = self._convert_to_wav(audio_data)
        # ... existing code

    # Rest stays the same (extract responses, synthesize TTS)
```

**Key changes:**
- Detect if ASR provider supports streaming
- Send audio chunks during recording (non-blocking)
- Optionally display partial transcripts in UI
- Finalize stream when user clicks STOP
- Keep batch path for backward compatibility

**UI Enhancement (optional):**
- Add status indicator showing partial transcript while speaking
- Visual feedback that ASR is processing in real-time

---

**`config.py`** - **MINIMAL changes** (~5 lines)
```python
# Add new provider option
CASCADE_ASR_PROVIDER = "deepgram_streaming"  # or "parakeet", "openai_whisper"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
```

---

### **Summary: Option 1 Impact**

| File | Lines Changed | Complexity | Breaking Changes |
|------|---------------|------------|------------------|
| `base_streaming.py` | +60 (new) | Low | None |
| `deepgram_streaming.py` | +200 (new) | Medium | None |
| `handler.py` | +50 | Low | None (additive) |
| `gradio_ui.py` | +80 | Medium | None (additive) |
| `config.py` | +5 | Low | None |
| **Total** | **~395 lines** | **Medium** | **None** |

**Backward compatibility:** ✅ Fully maintained. Batch ASR (Parakeet) still works unchanged.

**Testing strategy:**
1. Test Deepgram with `CASCADE_ASR_PROVIDER=deepgram_streaming`
2. Verify Parakeet still works with `CASCADE_ASR_PROVIDER=parakeet`
3. Both paths coexist cleanly

---

## Option 2: Faster-Whisper with VAD (Local Streaming) 🏠

### **Description**
Local ASR using Faster-Whisper (optimized Whisper) + Voice Activity Detection. Audio segmented in real-time using VAD, each segment transcribed immediately.

### **Performance**
- **First segment transcript:** ~800-1500ms after segment ends
- **Per-segment latency:** 500-1000ms
- **Latency improvement:** 30-50% reduction vs. current Parakeet

**Note:** Not true streaming - still batch processing per segment, but faster feedback.

### **Pros**
- ✅ Fully local (no internet required)
- ✅ Free and private
- ✅ Good accuracy (Whisper-based)
- ✅ Runs on Apple Silicon (MLX or CoreML)

### **Cons**
- ❌ Slower than Deepgram (not true streaming)
- ❌ VAD tuning required (false positives/negatives)
- ❌ More complex implementation
- ❌ Medium latency improvement only

---

### **Architecture Impact**

#### **New Files**
```
cascade/asr/
  ├── base_streaming.py          # Same as Option 1
  ├── vad.py                      # VAD logic (WebRTC VAD or Silero VAD)
  └── faster_whisper_vad.py      # Faster-Whisper + VAD (~250 lines)
```

#### **Modified Files**

**`handler.py`** - **MEDIUM changes** (~50 lines, same as Option 1)
- Same streaming methods as Deepgram option
- Handler doesn't care about ASR implementation details

---

**`gradio_ui.py`** - **MEDIUM-HIGH changes** (~100 lines)

Similar to Option 1, but with additional VAD logic:

```python
def _record_audio(self) -> None:
    """Record audio + VAD segmentation."""
    # Initialize streaming
    if self._is_streaming_asr():
        asyncio.run_coroutine_threadsafe(
            self.handler.process_audio_streaming_start(),
            self.handler.loop
        )

    vad = VAD()  # Initialize VAD
    audio_buffer = []

    while self.recording:
        data, _ = stream.read(1024)
        self.audio_frames.append(data)

        # VAD processing
        audio_buffer.append(data)
        is_speech = vad.is_speech(data)

        if not is_speech and len(audio_buffer) > 0:
            # Speech ended → transcribe segment
            segment_audio = np.concatenate(audio_buffer)
            segment_wav = self._convert_to_wav_bytes(segment_audio)

            # Send segment to ASR
            asyncio.run_coroutine_threadsafe(
                self.handler.process_audio_streaming_chunk(segment_wav),
                self.handler.loop
            )

            audio_buffer = []  # Reset for next segment
```

**Key differences from Option 1:**
- VAD logic adds complexity (~20 extra lines)
- Segment buffering required
- Tuning VAD sensitivity parameters

---

**`config.py`** - **MINIMAL changes** (~8 lines)
```python
CASCADE_ASR_PROVIDER = "faster_whisper_vad"
FASTER_WHISPER_MODEL = "base"  # or "small", "medium"
VAD_THRESHOLD = 0.5  # Speech detection sensitivity
```

---

### **Summary: Option 2 Impact**

| File | Lines Changed | Complexity | Breaking Changes |
|------|---------------|------------|------------------|
| `base_streaming.py` | +60 (new) | Low | None |
| `vad.py` | +80 (new) | Medium | None |
| `faster_whisper_vad.py` | +250 (new) | High | None |
| `handler.py` | +50 | Low | None (additive) |
| `gradio_ui.py` | +100 | Medium-High | None (additive) |
| `config.py` | +8 | Low | None |
| **Total** | **~548 lines** | **Medium-High** | **None** |

**Backward compatibility:** ✅ Fully maintained.

**Testing strategy:**
- VAD tuning required for good results
- Test with different speech patterns/pauses
- More integration testing needed than Option 1

---

## Option 3: Hybrid VAD + Parakeet (Minimal Changes) 🔄

### **Description**
Keep existing Parakeet ASR but add VAD to detect speech pauses. Transcribe segments immediately instead of waiting for full recording.

### **Performance**
- **First segment transcript:** ~600-1200ms after segment ends
- **Per-segment latency:** Similar to current Parakeet
- **Latency improvement:** 20-40% reduction (only helps with long utterances)

**Note:** Smallest improvement but easiest to implement.

### **Pros**
- ✅ Minimal code changes
- ✅ Fully local (keeps Parakeet)
- ✅ No new dependencies
- ✅ Easy to test and tune

### **Cons**
- ❌ Smallest latency improvement
- ❌ Still batch-based per segment
- ❌ Parakeet overhead per segment (temp file writes)
- ❌ Only helps with multi-sentence utterances

---

### **Architecture Impact**

#### **New Files**
```
cascade/asr/
  └── vad.py                      # VAD logic (~80 lines)
```

#### **Modified Files**

**`handler.py`** - **MINIMAL changes** (~20 lines)

```python
# Add optional segment buffering
async def process_audio_manual(self, audio_bytes: bytes) -> str:
    """Process audio (with optional VAD pre-segmentation)."""
    # If audio_bytes is pre-segmented by VAD, process normally
    transcript = await self.asr.transcribe(audio_bytes)

    if not transcript.strip():
        return ""

    # Continue with existing pipeline
    self.conversation_history.append({"role": "user", "content": transcript})
    await self._process_llm_response()
    return transcript

# Optional: Add method to accumulate multi-segment transcripts
async def process_audio_segment(self, audio_bytes: bytes) -> str:
    """Process one VAD segment."""
    segment_transcript = await self.asr.transcribe(audio_bytes)
    # Accumulate in temporary buffer, return when full utterance complete
    return segment_transcript
```

**Key changes:**
- Minimal modification to existing method
- Optional segment accumulation logic
- No streaming infrastructure needed

---

**`gradio_ui.py`** - **LOW-MEDIUM changes** (~60 lines)

```python
def _record_audio(self) -> None:
    """Record audio with VAD segmentation."""
    vad = VAD()
    audio_buffer = []
    segments = []

    while self.recording:
        data, _ = stream.read(1024)
        self.audio_frames.append(data)

        # VAD processing
        audio_buffer.append(data)
        is_speech = vad.is_speech(data)

        if not is_speech and len(audio_buffer) > MIN_SEGMENT_SIZE:
            # Speech pause detected → save segment
            segment = np.concatenate(audio_buffer)
            segments.append(segment)

            # Optionally: Transcribe immediately in background
            # (fire-and-forget, don't block recording)
            asyncio.run_coroutine_threadsafe(
                self._transcribe_segment_async(segment),
                self.handler.loop
            )

            audio_buffer = []

def _stop_recording(self) -> tuple[str, str]:
    """Stop recording and combine segment transcripts."""
    self.recording = False

    # Wait for all segment transcriptions to complete
    # Combine into final transcript
    # Continue with existing LLM pipeline
```

**Key changes:**
- Add VAD to detect pauses during recording
- Transcribe segments in background (don't block user)
- Combine segment transcripts when user clicks STOP
- Most existing code unchanged

---

**`config.py`** - **MINIMAL changes** (~3 lines)
```python
VAD_ENABLED = True  # Toggle VAD on/off
VAD_THRESHOLD = 0.5
```

---

### **Summary: Option 3 Impact**

| File | Lines Changed | Complexity | Breaking Changes |
|------|---------------|------------|------------------|
| `vad.py` | +80 (new) | Medium | None |
| `handler.py` | +20 | Low | None (additive) |
| `gradio_ui.py` | +60 | Medium | None (additive) |
| `config.py` | +3 | Low | None |
| **Total** | **~163 lines** | **Low-Medium** | **None** |

**Backward compatibility:** ✅ Fully maintained. VAD can be toggled off.

**Testing strategy:**
- Simple A/B test with VAD on/off
- Tune VAD threshold for good segment detection
- Minimal risk (smallest changes)

---

## Comparison Summary

| Option | Latency Improvement | Code Changes | Complexity | Cost | Local/Cloud |
|--------|-------------------|--------------|------------|------|-------------|
| **1. Deepgram** | ⭐⭐⭐⭐⭐ (60-80%) | ~395 lines | Medium | ~$13/50hrs | Cloud |
| **2. Faster-Whisper VAD** | ⭐⭐⭐ (30-50%) | ~548 lines | Medium-High | Free | Local |
| **3. VAD + Parakeet** | ⭐⭐ (20-40%) | ~163 lines | Low-Medium | Free | Local |

---

## Recommendation

### **Start with Option 1 (Deepgram)** if:
- Latency is top priority
- Internet connection reliable
- Budget allows ($13/50hrs is acceptable)
- Want production-ready solution quickly

### **Choose Option 2 (Faster-Whisper VAD)** if:
- Must be fully local/private
- Have time for implementation + tuning
- Want good latency without cloud dependency

### **Choose Option 3 (VAD + Parakeet)** if:
- Want minimal risk / quick experiment
- Latency improvement nice-to-have, not critical
- Want to validate VAD approach before committing to Options 1/2

---

## Implementation Strategy

**Phased approach (recommended):**

1. **Phase 1:** Implement Option 1 (Deepgram)
   - Fast path to major latency gains
   - Validate streaming architecture
   - Learn what works in production

2. **Phase 2:** Add Option 2 (Faster-Whisper VAD) as alternative
   - Local fallback when internet unavailable
   - Use lessons from Deepgram implementation
   - Reuse streaming infrastructure

3. **Phase 3 (optional):** Keep Option 3 as ultra-lightweight mode
   - Minimal latency help for constrained environments
   - Simplest implementation as reference

All three can coexist via `CASCADE_ASR_PROVIDER` config setting.
