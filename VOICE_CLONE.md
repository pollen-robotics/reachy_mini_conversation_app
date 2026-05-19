# Voice Clone — Local Infrastructure & Don Rickles Clone

> **Living planning document.** Update decisions and check off tasks as they complete.
> Last updated: 2026-05-19. Decisions #1–#8 re-checked and still current.
> Consolidated architecture snapshot lives in `FORK_STATUS.md`.

---

## Per-Persona Voice-Clone Reference Audio (Issue #44)

### Storage Convention

Place the reference clip for each persona at:

```
profiles/<persona>/voice_clone_ref.wav    ← preferred
profiles/<persona>/voice_clone_ref.mp3    ← fallback
profiles/<persona>/voice_clone_ref.flac   ← fallback
profiles/<persona>/voice_clone_ref.ogg    ← fallback
```

The loader (`src/robot_comic/chatterbox_voice_clone.py`) tries each extension in
that order and returns the first match.  Clips are **gitignored** — they are
copyrighted archival audio that must be provided locally on each machine.

### Clip Guidelines

- Duration: 5–10 seconds of clean speech (no audience, no music bed, no reverb).
- Sample rate: 22 050 Hz preferred; the Chatterbox server resamples automatically.
- Format: WAV (mono, 16-bit) gives the most predictable results; MP3/FLAC/OGG
  also work if that is what you have.
- Chatterbox zero-shot uses approximately the first 10 s of the reference clip —
  longer clips are silently truncated.

### How the Loader Wires Into the Pipeline

At handler startup (`_prepare_startup_credentials`), `ChatterboxTTSResponseHandler`
calls `load_voice_clone_ref(config.PROFILES_DIRECTORY / profile)` and caches the
result as `self._voice_clone_ref_path`.  On every `/tts` request:

- **Ref found** → payload uses `"audio_prompt_path": "<absolute path>"` so the
  Chatterbox server reads the local file directly.
- **Ref absent** → falls back to `"reference_audio_filename": "<voice>.wav"` (the
  existing server-side predefined-voice behaviour).

### Items 3 & 4 — Intentionally Deferred

- **Item 3 (per-persona baseline calibration)** — requires listening to Chatterbox
  output with each cloned voice; cannot be done without hardware + ears.
- **Item 4 (Turbo vs. standard decision)** — also requires on-robot A/B comparison.

---

## Session Notes

### Session 2 — 2026-05-12

**Completed:**
- Chatterbox pipeline is fully wired in code: Moonshine STT → Ollama `/api/generate` (with context token conversation history) → Chatterbox TTS
- `chatterbox_tag_translator.translate()` is called in `chatterbox_tts.py` — LLM delivery tags ([fast], [annoyance], etc.) are converted to per-segment `exaggeration`/`cfg_weight` params and never spoken aloud
- Chatterbox is now selectable in the admin settings UI (was env-var only before)
- Test suite fixed after merged persona commit moved shared tools from `profiles/don_rickles/` to `src/robot_comic/tools/`
- All changes committed and pushed to main

**Critical lesson — Ollama GUI vs. direct serve:**
The Ollama tray/GUI app (`ollama app.exe`) is a wrapper that partially exposes the Ollama API over LAN. It works for simple GET requests (`/api/tags`) but blocks `/api/chat`, `/v1/chat/completions`, and appears to silently drop or mishandle POST requests with large bodies (e.g. our full Rickles system prompt in the `system` field of `/api/generate`). **Never start Ollama by clicking the tray icon.** Always use the desktop shortcut → `Start-RobotServices.ps1`, which runs `ollama.exe serve` directly with `OLLAMA_HOST=0.0.0.0`.

> **Status 2026-05-19:** Ollama has been replaced by llama-server (see `LOCAL_LLM.md`). The cautionary tale above is preserved for historical reference but no longer applies — the active LLM endpoint is `llama-server`'s `/v1/chat/completions` on port 11434.

**Current state at end of Session 2:**
- Code is on main and correct
- Pi has the code but end-to-end audio test was NOT confirmed — blocked by the Ollama GUI issue above
- To resume: use desktop shortcut to start services, then test from Pi

**Pi cleanup needed before testing:**
- `startup_settings.json` on Pi may have `"voice": "Kore"` — apply personality from admin UI or clear the file so `chatterbox.txt` takes effect
- `src/robot_comic/.env` on Pi: remove `MODEL_NAME=gemini-3.1-flash-live-preview` if still present

**Deferred to next session:**
- Issue #39: Tool call support for Chatterbox/Ollama pipeline (dance, camera, emotions) — explicitly next after pipeline confirmed working
- Task 9: Thermal baseline — still deferred
- Phase 2: Don Rickles voice clone — not started

---

### Session 1 — 2026-05-09/10

**Completed:** Tasks 1–6 fully done. Infrastructure is installed and working.

**Gotchas discovered (for future reference):**

- **Chatterbox install is three separate steps, not one.** `requirements.txt` is CPU-only and explicitly omits the `chatterbox` package to avoid ONNX/protobuf conflicts. Correct install order for CUDA:
  1. `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
  2. `uv pip install -r requirements.txt` (installs supporting libs only)
  3. `uv pip install --no-deps git+https://github.com/devnen/chatterbox-v2.git@master s3tokenizer==0.3.0 onnx==1.16.0`
  4. `uv pip install "protobuf>=3.20.2"` (onnx 1.16.0 needs this; requirements.txt pins an older version)
  5. `uv pip install "huggingface_hub[hf_xet]"` (optional, faster HF downloads)

- **Always target venv explicitly with uv** — PowerShell resets CWD between tool calls, so `cd D:\Projects\chatterbox-tts && uv pip install ...` doesn't work reliably. Use `--python D:\Projects\chatterbox-tts\.venv\Scripts\python.exe` flag on every `uv pip install`.

- **Chatterbox model weights (~4.5 GB) are downloaded on first server launch**, not at install time. Cached to `D:\Projects\chatterbox-tts\model_cache\` after that.

- **Ollama is v0.23.1, model is `hermes3:8b-llama3.1-q4_K_M` (4.9 GB).** Already pulled and cached.

- **Power plan:** OEM "Extreme" plan was already active (better than Balanced). Ultimate Performance couldn't be set as active due to OEM restriction, but Extreme is equivalent. Sleep/hibernate timeouts set to never on AC; manual hibernate via Start menu still works.

- **TDR registry change requires a reboot** to take effect. Has been written to registry but not yet rebooted.

- **Port is 8004** (not 8080 as initially assumed). `config.yaml` already had `host: 0.0.0.0` — no changes needed.

**Deferred:**
- Tasks 7 (network/static DHCP) and 8 (Wake-on-LAN) — requires router + Pi access, paused intentionally.
- Task 9 (thermal baseline) — paused, to run next session.

**Next session starts at:** Task 9 (thermal baseline), then Phase 2 (Don Rickles voice clone).

---

## Decisions Log

Answers to the key architectural questions, made once so they don't get re-litigated.

### 1. Platform: Native Windows (not WSL2, not Docker)

**Decision:** Run both servers as native Windows processes.

**Rationale:**
- WSL2 mirrored networking still has documented conflicts with Docker Desktop and other services as of April 2026. The Pi needs a *stable IP/hostname* — WSL2 NAT adds unnecessary complexity.
- Native Windows gives first-class CUDA via PyTorch CUDA wheels and llama.cpp prebuilt binaries — no translation layer, no overhead.
- Disk I/O for model loading is faster without the virtual filesystem. PyTorch `.safetensors` model loads on NVMe are noticeably slower through the WSL2 VirtioFS.
- Services running as Windows SYSTEM survive lock screen (Win+L) and logout — the laptop can be locked while serving the Pi.

### 2. LLM Serving: Ollama for Windows

**Decision:** Ollama (`ollama serve`) over llama.cpp native build or LM Studio.

**Rationale:**
- One-command install, already ships with an OpenAI-compatible `/v1/chat/completions` SSE endpoint.
- For a single robot client making sequential requests, Ollama's multi-user penalty (VRAM spillover at 5+ parallel requests) is irrelevant.
- llama.cpp native build offers ~10-15% better throughput but requires a manual MSVC + CUDA Toolkit compile; not worth the maintenance burden here.
- KoboldCpp / TabbyAPI ruled out: narrower ecosystem, less tested on Ada (sm_89).

**Installed:** `hermes3:8b-llama3.1-q4_K_M` (4.9 GB, cached). Ollama auto-start shortcut removed; controlled via desktop shortcut instead.

### 3. TTS Serving: Chatterbox via `devnen/Chatterbox-TTS-Server`

**Decision:** [`devnen/Chatterbox-TTS-Server`](https://github.com/devnen/Chatterbox-TTS-Server) over rolling a custom FastAPI wrapper or using XTTS v2.

**Rationale:**
- Ships with: Web UI, OpenAI-compatible `/v1/audio/speech` endpoint, streaming chunked WAV/Opus output, voice cloning via reference audio upload, and predefined voice slots.
- CUDA/ROCm/CPU fallback handled internally.
- XTTS v2 (Coqui) requires `espeak-ng` on Windows PATH and has a harder native install; Chatterbox avoids this entirely.
- Chatterbox Turbo supports `[laugh]`, `[cough]`, `[chuckle]` tags — useful for a comedian persona.
- Zero-shot voice cloning: 5–10 s clean reference clip → clone, no fine-tuning.
- ~4.5 GB VRAM.

**Installed:** `D:\Projects\chatterbox-tts`, venv at `.venv\`, port 8004. Chatterbox model weights cached in `model_cache\`.

### 4. GPU Sharing — 12 GB VRAM Budget

| Process | Est. VRAM |
|---|---|
| Ollama (8B Q4_K_M) | ~5.0 GB |
| Chatterbox TTS | ~4.5 GB |
| Windows compositor + Chrome | ~0.5–1.0 GB |
| **Total** | **~10–10.5 GB** |

Leaves ~1.5 GB headroom. Both processes can be co-resident.

**Settings:**
- **HAGS (Hardware-Accelerated GPU Scheduling): OFF.** Disable in Settings → System → Display → Graphics → Change default graphics settings.
- **NVIDIA Control Panel → Manage 3D settings → Power management mode → "Prefer maximum performance"** (global, not per-app).
- **TDR:** `TdrDelay=60s`, `TdrDdiDelay=60s` written to registry — **reboot pending** to take effect.
- `CUDA_VISIBLE_DEVICES` irrelevant on single-GPU machine; leave unset.

### 5. Service Control: Manual Start/Stop Scripts with Desktop Shortcuts

**Decision:** PowerShell start/stop scripts pinned to desktop — no NSSM, no auto-start on boot.

**Installed:**
- `C:\Services\scripts\Start-RobotServices.ps1` — starts Ollama + Chatterbox, logs to `C:\Logs\`
- `C:\Services\scripts\Stop-RobotServices.ps1` — kills both
- Desktop shortcuts: "Start Robot Services" and "Stop Robot Services"

### 6. Power & Presence

- **Power plan:** OEM "Extreme" plan active (equivalent to Ultimate Performance; OEM restriction prevented switching). Sleep/hibernate auto-timeout disabled on AC. Manual hibernate (Start → Power → Hibernate) still works.
- **TDR:** Registry written, reboot needed to apply.
- **Wake-on-LAN:** ⏳ Deferred (Task 8).

### 7. Pi-to-Laptop Discovery: Static DHCP + mDNS

**Decision:** Static DHCP reservation on router *plus* Windows mDNS `.local` hostname as the primary name. ⏳ Deferred (Task 7).

### 8. Failure Modes

| Failure | Mitigation |
|---|---|
| Laptop sleeps mid-session | Sleep disabled on AC; manual hibernate only. |
| TDR crash (CUDA kernel timeout) | TdrDelay → 60 s (registry set, reboot pending). |
| Thermal throttle under sustained load | Task 9 thermal baseline will determine if undervolting needed. |
| Chrome pushes model out of VRAM | HAGS off + "Prefer max performance" reduces risk. Accept ~2 s reload latency on first post-eviction request. |

---

## Phase 1: Infrastructure Setup

### Task 1: Install Ollama and Pull Target Model ✅

- [x] Install Ollama v0.23.1 via winget
- [x] Pull `hermes3:8b-llama3.1-q4_K_M` (4.9 GB, cached)
- [x] Remove auto-created Startup folder shortcut
- [ ] **Verify GPU usage** — open Task Manager → Performance → GPU while `ollama run` is active, confirm ~5 GB used *(do next session)*

### Task 2: Install Chatterbox-TTS-Server ✅

- [x] Clone to `D:\Projects\chatterbox-tts`
- [x] Create venv (Python 3.12 via uv)
- [x] Install PyTorch 2.5.1+cu124, requirements, chatterbox package, protobuf fix, hf_xet
- [x] First launch — model weights downloaded and cached (~4.5 GB)
- [x] Verified: `PyTorch 2.5.1+cu124  CUDA=True  GPU=NVIDIA GeForce RTX 4080 Laptop GPU`
- [x] Server confirmed running: `Uvicorn running on http://0.0.0.0:8004`

### Task 3: Verify Chatterbox Config for Streaming + LAN ✅

- [x] `config.yaml` confirmed: `host: 0.0.0.0`, `port: 8004` — no changes needed
- [ ] **Test from Pi** (when Pi is available — Task 7 prereq):
  ```bash
  curl http://astralplane.lan:8004/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"tts-1","input":"Hello from the Pi","voice":"Michael.wav"}' \
    --output test.wav
  ```
  Note: `voice` must be a predefined voice filename (e.g., `Michael.wav`), not `"default"`.
  Check available voices: `GET http://localhost:8004/get_predefined_voices`
  Check model status: `GET http://localhost:8004/api/model-info` (no `/health` endpoint)

### Task 4: Create Start/Stop Scripts with Desktop Shortcuts ✅

- [x] `C:\Services\scripts\Start-RobotServices.ps1` created
- [x] `C:\Services\scripts\Stop-RobotServices.ps1` created
- [x] Desktop shortcuts created: "Start Robot Services" / "Stop Robot Services"
- [x] Ollama auto-start shortcut removed from Startup folder
- [ ] **Full shortcut smoke test** *(do next session — stop current services, restart via shortcut, confirm both logs appear)*

### Task 5: Power Plan ✅

- [x] OEM "Extreme" plan confirmed active
- [x] `standby-timeout-ac 0` and `hibernate-timeout-ac 0` applied (auto-sleep disabled, manual hibernate retained)
- [ ] **NVIDIA Control Panel** → Power management mode → "Prefer maximum performance" *(manual step, confirm done)*
- [ ] **HAGS off** → Settings → System → Display → Graphics → Change default graphics settings *(manual step, confirm done)*

### Task 6: Increase TDR Timeout ✅

- [x] `TdrDelay=60`, `TdrDdiDelay=60` written to `HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (elevated PowerShell)
- [ ] **Reboot to apply** *(pending — reboot when convenient before next heavy CUDA session)*
- [ ] Verify after reboot: `Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" | Select TdrDelay, TdrDdiDelay`

### Task 7: Configure Network Discovery ⏳ Deferred

- [ ] On router: assign static DHCP lease for the laptop's MAC
- [ ] On Pi: add to `/etc/hosts`: `<laptop-ip>  astralplane.lan`
- [ ] On Pi: verify Ollama reachable: `curl http://astralplane.lan:11434/api/tags`
- [ ] On Pi: verify Chatterbox reachable: `curl http://astralplane.lan:8004/health`

### Task 8: Enable Wake-on-LAN ⏳ Deferred

- [ ] Device Manager → Network Adapters → NIC → Power Management → "Wake on Magic Packet"
- [ ] BIOS/UEFI: enable "Wake on LAN" for S4/S5
- [ ] Test from Pi: `wakeonlan <laptop-MAC>`

### Task 9: Thermal Monitoring Baseline ⏳ Next session

- [ ] Install [HWiNFO64](https://www.hwinfo.com/) (free sensor monitor)
- [ ] Optionally install [MSI Afterburner](https://www.msi.com/page/afterburner) (only needed if undervolting required)
- [ ] Start both services via desktop shortcut
- [ ] Run 10-minute stress loop (Claude will provide curl commands):
  - Simultaneous Chatterbox TTS requests (long text, repeated)
  - Simultaneous Ollama inference requests
- [ ] Record peaks: GPU edge temp, GPU hotspot, VRAM total, GPU power draw
- [ ] **Decision:** edge < 85°C → done; 85–90°C → consider undervolt; > 90°C → undervolt recommended
- [ ] Document results: **`Edge: ___°C | Hotspot: ___°C | VRAM: ___ GB | Power: ___ W | Undervolt: ___ mV`**

---

## Phase 2: Don Rickles Voice Clone

> **Status 2026-05-19:** Individual `profiles/don_rickles/` was consolidated
> into the `profiles/house_comedian/` chimera during the profile cleanup
> (commit 23583e5). Tasks 10–14 below predate that consolidation. If you
> want a dedicated Rickles persona again, restore the profile directory
> first; otherwise treat tasks 10–14 as describing the per-persona clone
> workflow generically and apply them to whichever persona needs cloning.

### Task 10: Collect and Prepare Reference Audio

**Goal:** 5–10 s clean Don Rickles audio — no audience noise, no music bed, no reverb. Chatterbox zero-shot uses only the first ~10 s of reference clip.

- [ ] Source clean audio via yt-dlp + ffmpeg:
  - Suggested: Dean Martin Celebrity Roast recordings, Carson Tonight Show appearances
  ```powershell
  yt-dlp -x --audio-format wav -o "raw_rickles.wav" "<URL>"
  ffmpeg -i raw_rickles.wav -ss 00:01:23 -t 8 -ar 22050 -ac 1 rickles_ref.wav
  ```
- [ ] Confirm: mono, 22050 Hz, no clipping, no background noise
- [ ] Store at: `profiles/don_rickles/voice_ref.wav` (local only — `profiles/*/voice_ref.wav`
      is gitignored; archival source audio must not be committed to this public fork)

### Task 11: Test Zero-Shot Clone via Chatterbox Web UI

- [ ] Open `http://localhost:8004`
- [ ] Upload `rickles_ref.wav` as reference audio, set voice mode to "clone"
- [ ] Test phrases:
  - `"You are the most beautiful audience I've ever seen, and that's not saying much."`
  - `"[laugh] I kid! I kid, sweetheart."`
- [ ] If timbre matches → proceed. If not → try a different reference clip segment.
- [ ] Save best output as `rickles_test_output.wav`

### Task 12: Register Reference Voice via API

- [ ] Copy `rickles_ref.wav` into `D:\Projects\chatterbox-tts\reference_audio\don_rickles.wav`
  (the server's `reference_audio_path` config serves files from this directory)
- [ ] Test via OpenAI-compatible endpoint (voice = filename in `predefined_voices/` dir):
  ```powershell
  curl http://localhost:8004/v1/audio/speech `
    -H "Content-Type: application/json" `
    -d '{"model":"tts-1","input":"You are a hockey puck!","voice":"don_rickles.wav"}' `
    --output rickles_api_test.wav
  ```
  Alternative via custom /tts endpoint (voice cloning mode):
  ```powershell
  curl http://localhost:8004/tts `
    -H "Content-Type: application/json" `
    -d '{"text":"You are a hockey puck!","voice_mode":"clone","reference_audio_filename":"don_rickles.wav"}' `
    --output rickles_api_test.wav
  ```
- [ ] Confirm audio plays back with Rickles voice

### Task 13: Wire Rickles Voice into robot_comic

- [ ] In `src/robot_comic/config.py`: add:
  ```python
  CHATTERBOX_URL: str = os.getenv("CHATTERBOX_URL", "http://astralplane.lan:8004")
  CHATTERBOX_VOICE: str = os.getenv("CHATTERBOX_VOICE", "don_rickles")
  ```
- [ ] Add to Pi `.env`:
  ```
  CHATTERBOX_URL=http://astralplane.lan:8004
  CHATTERBOX_VOICE=don_rickles
  ```
- [ ] End-to-end test: speech → STT → LLM → Chatterbox Rickles voice → playback

### Task 14: Latency Profiling

- [ ] Instrument Pi code with `time.perf_counter()` at: end-of-speech, LLM first token, TTS first byte, audio playback start
- [ ] Target: < 1–2 s end-to-end
- [ ] If TTS latency high: reduce `chunk_size` in `D:\Projects\chatterbox-tts\config.yaml`
- [ ] **Results: `STT→LLM: ___ ms | LLM→TTS: ___ ms | TTS→audio: ___ ms`**

---

## Open Questions

- [ ] Does `/v1/audio/speech` return chunked streaming or a complete file? (Check devnen issues — affects TTFA latency design)
- [ ] Does Reachy Mini audio stack support streaming HTTP WAV, or need a complete file?
- [ ] Best Don Rickles source clip? *(fill in)*
- [ ] Stable undervolt offset after Task 9? *(fill in)*

---

## Reference Links

- [devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server)
- [devnen/chatterbox-v2](https://github.com/devnen/chatterbox-v2) — the actual `chatterbox` Python package (fork)
- [Chatterbox repo (Resemble AI)](https://github.com/resemble-ai/chatterbox)
- [Chatterbox Turbo announcement](https://www.resemble.ai/chatterbox-turbo/)
- [travisvn/chatterbox-tts-api](https://github.com/travisvn/chatterbox-tts-api) — alternative if streaming endpoint needed
- [Ollama Windows docs](https://docs.ollama.com/windows)
- [WSL2 advanced networking (Microsoft Learn)](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- [TDR registry settings (Acceleware)](https://training.acceleware.com/blog/timeout-detection-windows-display-driver-model-when-running-cuda-kernels-symptoms-solutions-and-registry-modifications)
- [Chatterbox Windows install walkthrough (Jan 2026)](https://emanueleferonato.com/2026/01/07/text-to-speech-on-your-pc-running-chatterbox-turbo-locally-on-windows-clean-setup-known-pitfalls/)
