# Console Mode Strategy for Cascade Pipeline

## Current State

**Console mode exists ONLY for OpenAI Realtime API**, not for Cascade mode.

### Existing Console Implementation (`console.py`)
- **Mode:** OpenAI Realtime API only
- **Architecture:** `LocalStream` class with `record_loop()` / `play_loop()`
- **Audio I/O:** Direct integration with `robot.media` (GStreamer pipelines)
- **VAD:** Built-in server-side VAD from OpenAI Realtime API
- **Flow:** Continuous bidirectional audio streaming (mic → OpenAI → speaker)

### Current Cascade Limitation
From `main.py:92-97`:
```python
if args.cascade:
    if not args.gradio:
        logger.error("Cascade mode requires --gradio flag. Console mode with VAD is not yet implemented.")
        sys.exit(1)
```

**Reason:** Cascade pipeline currently depends on Gradio's push-to-talk button for recording start/stop.

---

## Why Console Mode is Challenging for Cascade

### **Problem 1: No Manual Start/Stop Trigger**
- **Gradio:** User clicks START/STOP button → clear recording boundaries
- **Console:** Continuous audio stream → need automatic speech detection

**Solution required:** VAD to detect speech start/end

---

### **Problem 2: Orchestration Responsibilities**
Gradio UI currently handles:
1. Recording management (start/stop/buffer)
2. TTS synthesis and playback
3. Chat history display
4. Response extraction from Handler's conversation history
5. Pre-warmed audio playback system
6. Head wobbler queue management

**Console mode must replicate:**
- Items 1, 2, 5, 6 (audio I/O, TTS, playback)
- Items 3, 4 can be simplified (no visual chat needed)

---

### **Problem 3: Shared Audio Device**
- **Recording loop:** Captures mic input continuously
- **Playback loop:** Outputs TTS audio to speaker
- **Challenge:** Prevent feedback (mic hearing speaker output)

**Gradio solution:** Sequential recording (user clicks STOP before robot speaks)

**Console solution:**
- Full-duplex audio with echo cancellation, OR
- Half-duplex (pause recording during robot speech)

---

## Proposed Console Architecture

```
┌──────────────────────────────────────────────────────┐
│  CascadeConsoleStream (new)                         │
│  ├── record_loop() + VAD                            │
│  ├── play_loop() + TTS synthesis                    │
│  └── handler orchestration                          │
└──────────────────────────────────────────────────────┘
           │                    │
           ↓                    ↓
    ┌─────────────┐      ┌─────────────┐
    │  Handler    │      │ robot.media │
    │  (ASR/LLM)  │      │ (audio I/O) │
    └─────────────┘      └─────────────┘
```

### Components to Create

#### **1. CascadeConsoleStream** (`cascade/console_stream.py`)
**Role:** Console equivalent of `CascadeGradioUI`

**Responsibilities:**
- Manage audio recording with VAD-based segmentation
- Send audio segments to Handler for ASR→LLM processing
- Extract speak messages from Handler's conversation history
- Synthesize TTS and play through robot speaker
- Synchronize head wobbler animation

**Key differences from Gradio UI:**
- No visual interface
- No push-to-talk button → VAD-triggered recording
- Direct `robot.media` integration (like existing `LocalStream`)
- Continuous operation (no manual start/stop)

---

## Implementation Plan

### **New Files**

```
cascade/
  ├── console_stream.py          # Main console orchestrator (~300 lines)
  └── vad.py                      # VAD logic (shared with streaming ASR) (~80 lines)
```

### **Modified Files**

#### **`main.py`** - **MINIMAL changes** (~15 lines)

```python
# Current:
if args.cascade:
    if not args.gradio:
        logger.error("Cascade mode requires --gradio flag.")
        sys.exit(1)
    cascade_ui = CascadeGradioUI(handler)
    stream_manager = cascade_ui.create_interface()

# New:
if args.cascade:
    if args.gradio:
        cascade_ui = CascadeGradioUI(handler)
        stream_manager = cascade_ui.create_interface()
    else:
        # Console mode for cascade
        from reachy_mini_conversation_app.cascade.console_stream import CascadeConsoleStream
        stream_manager = CascadeConsoleStream(handler, robot)
```

**Changes:**
- Remove error message blocking console mode
- Add conditional import for `CascadeConsoleStream`
- Branch on `args.gradio` to choose UI vs console

---

#### **`handler.py`** - **MINIMAL changes** (~20 lines)

Add a helper method for console mode to process segments:

```python
async def process_audio_segment_console(self, audio_bytes: bytes) -> Optional[str]:
    """Process audio segment in console mode (VAD-triggered).

    Returns transcript if segment contains speech, None otherwise.
    Does NOT trigger LLM yet - waits for full utterance.
    """
    # Transcribe segment
    segment_transcript = await self.asr.transcribe(audio_bytes)

    if not segment_transcript.strip():
        return None

    return segment_transcript

async def process_full_utterance_console(self, full_transcript: str) -> None:
    """Process complete user utterance (after VAD detects end of speech).

    Runs LLM pipeline on accumulated transcript.
    """
    # Add to conversation history
    self.conversation_history.append({"role": "user", "content": full_transcript})

    # Run LLM pipeline (existing code)
    await self._process_llm_response()
```

**Changes:**
- Two new methods for console's VAD-based workflow
- No changes to existing methods
- Fully backward compatible

---

## Detailed Implementation: `CascadeConsoleStream`

### **Structure** (based on existing `LocalStream`)

```python
class CascadeConsoleStream:
    """Console stream for cascade pipeline with VAD."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini):
        self.handler = handler
        self.robot = robot
        self.vad = VAD(threshold=0.5)
        self.stop_event = asyncio.Event()

        # Speech buffering
        self.speech_buffer: List[np.ndarray] = []
        self.is_in_speech = False
        self.silence_frames = 0
        self.SILENCE_THRESHOLD = 30  # ~500ms at 16kHz/1024 samples

    def launch(self) -> None:
        """Start console stream."""
        self.robot.media.start_recording()
        self.robot.media.start_playing()

        async def runner():
            tasks = [
                asyncio.create_task(self.record_loop()),
                asyncio.create_task(self.play_loop()),
            ]
            await asyncio.gather(*tasks)

        asyncio.run(runner())

    async def record_loop(self) -> None:
        """Record audio with VAD segmentation."""
        logger.info("Starting cascade console record loop with VAD")

        while not self.stop_event.is_set():
            # Get audio frame from robot mic
            audio_frame = self.robot.media.get_audio_sample()
            if audio_frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_mono = audio_frame.T[0]
            frame_int16 = audio_to_int16(frame_mono)

            # VAD detection
            is_speech = self.vad.is_speech(frame_int16)

            if is_speech:
                # Speech detected
                self.speech_buffer.append(frame_int16)
                self.is_in_speech = True
                self.silence_frames = 0

                # Update robot state: listening
                if self.handler.deps.movement_manager:
                    self.handler.deps.movement_manager.set_listening(True)

            elif self.is_in_speech:
                # Potential end of speech (silence after speech)
                self.silence_frames += 1
                self.speech_buffer.append(frame_int16)  # Keep buffering during silence

                if self.silence_frames > self.SILENCE_THRESHOLD:
                    # End of speech confirmed → process utterance
                    logger.info("End of speech detected (VAD)")

                    # Update robot state: done listening
                    if self.handler.deps.movement_manager:
                        self.handler.deps.movement_manager.set_listening(False)

                    # Concatenate speech buffer
                    full_audio = np.concatenate(self.speech_buffer)

                    # Convert to WAV bytes
                    wav_bytes = self._audio_to_wav(full_audio, sample_rate=16000)

                    # Process through cascade pipeline (ASR→LLM→Tools)
                    asyncio.create_task(self._process_utterance(wav_bytes))

                    # Reset for next utterance
                    self.speech_buffer = []
                    self.is_in_speech = False
                    self.silence_frames = 0

            await asyncio.sleep(0)  # Yield to event loop

    async def _process_utterance(self, audio_bytes: bytes) -> None:
        """Process one complete user utterance."""
        try:
            # Use handler's existing pipeline
            transcript = await self.handler.process_audio_manual(audio_bytes)

            if transcript.strip():
                logger.info(f"User: {transcript}")

                # Extract speak messages for TTS
                speak_messages = self._extract_speak_messages()

                if speak_messages:
                    combined_text = ". ".join(speak_messages)
                    logger.info(f"Robot: {combined_text}")

                    # Synthesize and play
                    await self._synthesize_and_play(combined_text)

        except Exception as e:
            logger.exception(f"Error processing utterance: {e}")

    def _extract_speak_messages(self) -> List[str]:
        """Extract speak tool messages from handler's conversation history."""
        messages = []
        # Scan recent messages for speak tool calls
        for msg in reversed(self.handler.conversation_history[-10:]):
            if msg.get("role") == "tool" and msg.get("name") == "speak":
                try:
                    content = json.loads(msg.get("content", "{}"))
                    if "message" in content:
                        messages.insert(0, content["message"])
                except json.JSONDecodeError:
                    pass
        return messages

    async def _synthesize_and_play(self, text: str) -> None:
        """Synthesize speech and play through robot speaker."""
        # Similar to Gradio's _synthesize_for_gradio but simplified

        # Start head wobbler
        if self.handler.deps.head_wobbler:
            self.handler.deps.head_wobbler.reset()

        # Stream TTS
        async for chunk in self.handler.tts.synthesize(text):
            # Convert to float for robot.media
            audio_float = audio_to_float32(np.frombuffer(chunk, dtype=np.int16))

            # Resample if needed (TTS outputs 24kHz, robot may need different rate)
            device_sample_rate = self.robot.media.get_audio_samplerate()
            if device_sample_rate != 24000:
                audio_float = librosa.resample(
                    audio_float,
                    orig_sr=24000,
                    target_sr=device_sample_rate,
                )

            # Play through robot speaker
            self.robot.media.push_audio_sample(audio_float)

            # Feed to head wobbler
            if self.handler.deps.head_wobbler:
                self.handler.deps.head_wobbler.feed_pcm(chunk)

        # Reset head wobbler
        if self.handler.deps.head_wobbler:
            self.handler.deps.head_wobbler.reset()

    async def play_loop(self) -> None:
        """Placeholder play loop (most work done in _synthesize_and_play)."""
        # Could be used for background tasks, monitoring, etc.
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)

    def close(self) -> None:
        """Stop console stream."""
        self.stop_event.set()
        self.robot.media.stop_recording()
        self.robot.media.stop_playing()

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio to WAV bytes."""
        import io
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        return wav_buffer.getvalue()
```

---

## Architecture Impact Summary

| Component | Changes Required | Complexity | Lines of Code |
|-----------|-----------------|------------|---------------|
| `cascade/console_stream.py` | New file | Medium | ~300 |
| `cascade/vad.py` | New file | Low-Medium | ~80 |
| `handler.py` | Add 2 helper methods | Low | +20 |
| `main.py` | Update cascade mode branching | Low | +15 |
| **Total** | | **Medium** | **~415 lines** |

**Breaking changes:** None (fully backward compatible)

---

## Key Differences: Console vs Gradio

| Aspect | Gradio UI | Console Mode |
|--------|-----------|--------------|
| **Recording trigger** | Manual button (START/STOP) | VAD automatic |
| **Audio I/O** | sounddevice (local laptop) | robot.media (GStreamer) |
| **Visual feedback** | Chat interface, partial transcripts | Logs only |
| **Playback system** | Pre-warmed persistent threads | Direct streaming to robot.media |
| **User control** | Full (can interrupt anytime) | Limited (VAD-driven) |
| **Complexity** | Higher (UI + audio) | Lower (audio only) |

---

## Synergies and Conflicts with ASR Streaming

### **Synergies** 🤝

#### **1. Shared VAD Infrastructure**
- **Console mode requires VAD** for speech detection
- **ASR streaming Option 2 & 3 require VAD** for segmentation
- **Win:** Implement VAD once (`cascade/vad.py`), reuse for both features

**Impact:** Implementing console mode first would reduce ASR streaming effort by ~80 lines

---

#### **2. Continuous Audio Flow Architecture**
- **Console mode:** Continuous recording with frame-by-frame processing
- **ASR streaming:** Continuous transcription with chunk-by-chunk processing
- **Win:** Console mode establishes streaming mindset, makes ASR streaming more natural

**Example:**
```python
# Console record_loop (continuous frames)
while not self.stop_event.is_set():
    audio_frame = self.robot.media.get_audio_sample()
    # Process frame with VAD

# ASR streaming (continuous chunks)
while recording:
    audio_chunk = stream.read(1024)
    # Send chunk to streaming ASR
```

**Similar patterns!**

---

#### **3. Handler Streaming Methods**
- **Console mode needs:** `process_audio_segment_console()` for VAD segments
- **ASR streaming needs:** `process_audio_streaming_chunk()` for real-time chunks
- **Win:** Both require similar handler modifications (segment processing)

**Overlap:** ~50% of handler changes are common

---

### **Conflicts** ⚠️

#### **1. Different Audio Sources**
- **Console mode:** `robot.media.get_audio_sample()` (GStreamer)
- **Gradio mode (where ASR streaming goes):** `sounddevice.InputStream` (local laptop)

**Impact:** Console and Gradio have different recording loops → ASR streaming integration must be done **per UI mode**

**Consequence:** Implementing console first doesn't reduce ASR streaming Gradio work

---

#### **2. Different Segmentation Strategies**
- **Console VAD:** Detect speech **end** → send full utterance to batch ASR
- **ASR streaming VAD (Option 2/3):** Detect speech **pauses** → send segments to ASR immediately
- **Deepgram streaming (Option 1):** No VAD needed → send raw chunks continuously

**Impact:** VAD code can be shared, but **usage patterns differ**

**Example:**
```python
# Console VAD: Buffer until end of speech
if self.silence_frames > THRESHOLD:
    full_audio = concatenate(buffer)
    await handler.process_audio_manual(full_audio)

# ASR streaming VAD: Transcribe each segment immediately
if self.silence_frames > SHORT_PAUSE:
    segment = concatenate(buffer)
    await handler.process_audio_streaming_chunk(segment)
```

---

#### **3. Scope Creep Risk**
- **Console mode:** Adds new execution mode (console vs Gradio)
- **ASR streaming:** Adds new ASR processing mode (streaming vs batch)
- **Combining both:** Two dimensions of complexity

**Risk:** Implementing both simultaneously could lead to:
- 4 code paths (Console+Batch, Console+Streaming, Gradio+Batch, Gradio+Streaming)
- Harder testing and debugging
- More integration points

---

## Realistic Difficulty Assessment

### **Console Mode Alone**
- **Difficulty:** 6/10
- **Time estimate:** 2-3 days (with testing)
- **Main challenges:**
  - VAD tuning for good speech detection
  - Avoiding audio feedback (mic picking up speaker)
  - Testing on real robot hardware (simulation won't work)
- **Dependencies:** None (standalone feature)

---

### **ASR Streaming Alone (Deepgram - Option 1)**
- **Difficulty:** 5/10
- **Time estimate:** 2-3 days (with testing)
- **Main challenges:**
  - WebSocket integration with Deepgram
  - Modifying Gradio recording loop to send chunks
  - Handling partial transcripts in UI
- **Dependencies:** None (works with existing Gradio)

---

### **Both Console + ASR Streaming**
- **Difficulty:** 8/10 (not 11/10, but increased)
- **Time estimate:** 5-7 days (testing + integration)
- **Main challenges:**
  - Managing 4 code paths (Console×Streaming matrix)
  - VAD behavior differs between modes
  - Double the testing surface
  - Risk of coupling issues

---

## Honest Recommendation: Order Matters 🎯

### **Option A: ASR Streaming FIRST** ⭐ **RECOMMENDED**

**Reasoning:**
1. **Higher impact on user experience**
   - ASR latency is critical for conversation flow
   - Console mode is a "nice-to-have" alternative interface
   - Gradio is your primary interface (per current usage)

2. **Lower risk**
   - Gradio already works well
   - ASR streaming is additive (doesn't break existing flows)
   - Can A/B test easily (Parakeet vs Deepgram)

3. **Clearer path**
   - Deepgram streaming is well-documented
   - No hardware dependencies (works in simulation + real robot)
   - Faster iteration cycle (no need for robot audio testing)

4. **VAD synergy still captured**
   - If you later do ASR streaming Option 2 (Faster-Whisper VAD), you'll build VAD infrastructure
   - This VAD can then be reused for console mode
   - If you go with Deepgram (Option 1), console mode's VAD is independent anyway

**Plan:**
```
Week 1: Implement Deepgram streaming (Option 1)
Week 2: Test + optimize latency
Week 3 (optional): Add console mode using lessons learned
```

---

### **Option B: Console Mode FIRST**

**Reasoning:**
1. **Learn streaming patterns**
   - Console forces you to think about continuous audio
   - VAD experience helps with ASR streaming Options 2/3
   - Smaller scope (easier to debug)

2. **When this makes sense:**
   - If you need console mode for deployment (e.g., robot runs standalone)
   - If you want to test cascade pipeline without Gradio overhead
   - If you prefer incremental complexity

**Plan:**
```
Week 1: Implement console mode with VAD
Week 2: Test on robot hardware + tune VAD
Week 3: Implement ASR streaming (reuse VAD if applicable)
```

---

### **Option C: Parallel Implementation** ❌ **NOT RECOMMENDED**

**Reasoning:**
- Too many moving parts
- 4 code paths to maintain
- Testing becomes exponentially harder
- High risk of bugs and coupling issues

**Only consider if:** You have 2+ developers working independently

---

## Final Verdict

### **Recommended Order: ASR Streaming → Console Mode**

**Why:**
1. **ASR streaming (Deepgram) is faster to implement and test**
2. **Higher ROI** (latency improvements visible immediately)
3. **Console mode can leverage streaming infrastructure** if you later add ASR streaming Option 2/3
4. **Lower risk** (Gradio is stable, console mode needs robot hardware)

**Timeline:**
- **Phase 1 (Week 1-2):** Deepgram streaming in Gradio → Big latency wins
- **Phase 2 (Week 3-4):** Console mode → Alternative interface for robot
- **Phase 3 (optional):** Local streaming ASR (Faster-Whisper VAD) → Reuses VAD from console

**Path of least resistance:** Start where you have the most stable foundation (Gradio), then expand.

---

## Summary Table

| Factor | ASR Streaming First | Console Mode First |
|--------|---------------------|-------------------|
| **Impact** | High (latency critical) | Medium (alternative UI) |
| **Risk** | Low (Gradio stable) | Medium (robot hardware needed) |
| **Testing** | Easy (simulation works) | Harder (needs real robot) |
| **Dependencies** | None | None |
| **Synergy with other** | Moderate | Moderate |
| **Time to value** | Faster (2-3 days) | Slower (2-3 days + hardware testing) |
| **Recommendation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Best path:** ASR Streaming (Deepgram) → Console Mode → Local Streaming ASR (if needed)
