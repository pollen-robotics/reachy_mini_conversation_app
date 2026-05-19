# LOCAL_LLM.md — Local LLM Backend Planning & Tracking

> **Living planning document.** Update decisions and check off tasks as they complete.
> Last updated: 2026-05-19. Tasks 1–7 ✅; remaining open item is Qwen3 14B vs 35B benchmark (#78).
> Consolidated architecture snapshot lives in `FORK_STATUS.md`.

---

## Current State

- **Active model:** `Qwen3-14B-UD-Q4_K_XL.gguf` via llama-server on port 11434
- **Previous model:** Hermes3 8B (Ollama) — broke on system prompts > 2500 tokens
- **Scrapped:** Qwen3 35B-A3B MoE — CPU MoE offload bottlenecked prefill at 86 tok/s,
  decode at 17 tok/s; MTP PR #22673 crashes after first request (checkpoint size mismatch)
- **Stack:** llama.cpp (no MTP), OpenAI-compatible `/v1/chat/completions` endpoint

---

## Decisions Log

### 1. Platform: Native Windows (matching Session 1 VOICE_CLONE.md Decision #1)

**Decision:** Build llama.cpp as a native Windows process. No WSL2.

**Rationale:** WSL2 adds CUDA passthrough latency, slower VirtioFS model loads, and
breaks the "services survive lock screen" property. Native CUDA on RTX 4080 Mobile is
first-class. Single client (Pi), sequential requests — Ollama's multi-user penalty never
applied, and llama.cpp's single-slot mode is a better fit anyway.

**Build toolchain:** MSVC (Visual Studio Build Tools) + CMake. No GCC cross-compile.
**Service wrapper:** PowerShell script added to `Start-RobotServices.ps1`. No systemd.
**Build location:** `D:\Projects\llama.cpp\` (Dev Drive, NVMe-optimized).

### 2. Model: Qwen3 14B Dense (UD-Q4_K_XL)

**Decision:** Qwen3-14B-UD-Q4_K_XL over the original Qwen3 35B-A3B MoE.

**Rationale:**
- 35B MoE was scrapped: CPU MoE offload bottlenecked prefill at 86 tok/s; MTP
  PR #22673 crashes after first request (checkpoint size mismatch bug).
- 14B dense fits entirely in 12 GB VRAM — no CPU offload, no prefill penalty.
  Expected ~600-1000 tok/s prefill, ~25-35 tok/s decode.
- UD (Unsloth Dynamic) quantization uses mixed precision per layer for better
  accuracy at identical file size vs standard Q4_K_M (~9.16 GB).
- Strong instruction following and tool use (Tau2-Bench 65.1). Beats Gemma 2
  27B and Phi-4 14B on 2026 leaderboards.
- **CUDA 13.2 is known to produce gibberish with Qwen3.** Pin CUDA 12.4.
  Do not upgrade.

**Source:** `unsloth/Qwen3-14B-GGUF` on HuggingFace.
**Download command:**
```powershell
curl.exe -L -o "D:\Projects\models\Qwen3-14B-UD-Q4_K_XL.gguf" `
    "https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-UD-Q4_K_XL.gguf"
```

**Next in queue if 14B underperforms:**
1. **Llama 3.3 8B** (~5 GB at Q4) — specifically cited for structured output + tool-call patterns; smaller size leaves more headroom and runs faster
2. **Gemma 3 12B** (~7 GB at Q4) — strong prose/alignment, good fallback if Llama 3.3 lacks persona depth
- Gemma 3 27B (~15-16 GB) and Gemma 4 26B won't fit in 12 GB VRAM without CPU offload — skip those tiers

### 3. LLM Server: llama.cpp + MTP PR #22673

**Decision:** llama.cpp with Multi-Token Prediction speculative decoding.

**Rationale:**
- MTP draft heads (bundled in the GGUF) let the server speculatively draft 3
  tokens per step. On short conversational outputs, expect ~1.4–1.5× decode
  speedup vs. non-speculative — not 2×, because schema-constrained tool-call
  JSON drafts worse than free text (acceptance rate ~40–55% expected vs.
  headline ~75%).
- If MTP acceptance rate is below ~40% on our actual workload, fall back to
  running without `--spec-type mtp`. The overhead is not worth it below that.
- PR #22673 is beta / not yet merged to master. Pin the commit hash used at
  build time in this doc.

**Fetched PR branch:** `ggml-org/llama.cpp` PR #22673 — `spec-mtp` branch or equivalent.
**Commit pinned:** `3bdc61fe03299a84ade6f185819738c9a05200e7`
Branch top commits: `3bdc61fe` cont: simplify, `fea55085` rename files, `cfb386c6` fix batch size, `a7813c71` spec: support MTP

### 4. Context Window: 8192 tokens

**Decision:** `-c 8192` (not 32K).

**Rationale:** MTP draft branch consumes additional VRAM proportional to context length.
Our workload is 3–4K input + <200 tokens output. 8K provides comfortable headroom.
Expanding to 16K later is easy; shrinking after OOM is not fun.

### 5. CPU MoE Offload: `--n-cpu-moe`

**Decision:** Start with `--n-cpu-moe 30` and tune downward by watching `nvidia-smi`.

**Rationale:** The 35B MoE model won't fit in 12 GB VRAM with all layers on GPU.
MoE expert layers are the main spill candidate — they're large, used sparsely.
Attention and shared layers stay on GPU. CPU RAM on this laptop is ample for expert
spill; the RTX 4080 Mobile's 12 GB carries the hot path.

### 6. TTS Pairing: Gemini TTS (Issue #60 — tracked separately)

Chatterbox is **parked** (not removed). The planned TTS pairing for the llama-server LLM
path is **Gemini TTS** (`gemini-3.1-flash-tts-preview`), already implemented in `gemini_tts.py`.

Plan: create a `LlamaServerGeminiTTSHandler` that combines:
- LLM: llama-server `_call_llm()` from `ChatterboxTTSResponseHandler`
- TTS: `_call_tts_with_retry()` from `GeminiTTSResponseHandler`

ElevenLabs is deprioritised (parked, not closed) — may revisit for voice cloning fidelity.
Both are cloud APIs — 0 VRAM overhead.

---

## VRAM Budget (New Stack)

| Process                          | Est. VRAM     |
|----------------------------------|---------------|
| llama.cpp (14B dense, all GPU)   | ~8–9 GB       |
| KV cache (q8_0, 16K context)     | ~0.5–1.0 GB   |
| Windows compositor + Chrome      | ~0.5–1.0 GB   |
| ElevenLabs TTS                   | 0 (cloud API) |
| Chatterbox (parked)              | 0             |
| **Total**                        | **~9–11 GB**  |

~1–3 GB headroom vs. the 12 GB cap. Can expand context to 32K if VRAM holds.

---

## Launch Script (Windows/PowerShell)

```powershell
# llama-server launch — in Start-RobotServices.ps1
$env:CUDA_VISIBLE_DEVICES = "0"  # Single GPU, explicit
& "D:\Projects\llama.cpp\build\bin\llama-server.exe" `
    -m "D:\Projects\models\Qwen3-14B-UD-Q4_K_XL.gguf" `
    -c 16384 `
    -ngl 999 `
    -fa on `
    --cache-type-k q8_0 `
    --cache-type-v q8_0 `
    -t 8 `
    -b 2048 `
    -ub 2048 `
    --jinja `
    --host 0.0.0.0 `
    --port 11434 `
    >> "C:\Logs\llama-server.log" 2>&1
```

Note: Using port 11434 to maintain backward compatibility with existing Pi `.env` files.
llama-server's `/v1/chat/completions` is OpenAI-compatible.

---

## Code Changes Required in robot_comic

### 1. `config.py` — Add `LLAMA_CPP_URL`

The current `_ollama_base_url` property in `chatterbox_tts.py` derives the LLM URL
from `CHATTERBOX_URL` by swapping the port to 11434 — fragile. Add a dedicated var:

```python
LLAMA_CPP_URL_ENV = "LLAMA_CPP_URL"
LLAMA_CPP_DEFAULT_URL = "http://astralplane.lan:11434"
# in Config class:
LLAMA_CPP_URL = os.getenv(LLAMA_CPP_URL_ENV, LLAMA_CPP_DEFAULT_URL)
# also add to LOCAL_STT_RESPONSE_BACKEND_CHOICES:
LLAMA_CPP_OUTPUT = "llama_cpp"
```

### 2. `chatterbox_tts.py` — Switch from `/api/chat` to `/v1/chat/completions`

Current: `POST {ollama_base_url}/api/chat` (Ollama-specific format)
New: `POST {llama_cpp_url}/v1/chat/completions` (OpenAI-compatible)

Response shape changes: `data["message"]` → `data["choices"][0]["message"]`

### 3. Hermes3-Specific Workarounds to Remove/Audit

These exist solely because Hermes3 8B has unreliable tool call formatting:
- `_trim_tool_spec()` — truncates enum lists and descriptions to keep prompt under 2500 tokens. **Remove** once verified Qwen3 handles full specs.
- `_coerce_text_tool_call()` — detects `{function: name, ...}` text-format calls. **Remove** after 20-turn sanity check confirms Qwen3 uses structured tool_calls.
- `_nudge_llm()` — retries with a "please use a tool call now" nudge. **Deprecate** (keep as fallback initially, remove after validation).
- `_TEXT_TOOL_CALL_RE`, `_parse_text_tool_args()` — regex parsing for text-format calls. **Remove** with `_coerce_text_tool_call`.
- `_parse_json_content_tool_call()` — JSON-in-content fallback. **Remove** after validation.

Do **not** remove before the 20-turn sanity check in Task 5.

### 4. Rename handler references

The class and module are named `ChatterboxTTS*` but will be serving llama.cpp LLM + ElevenLabs TTS. Rename in a follow-up (Issue #60 scope).

---

## Task Checklist

### Task 1: Clone llama.cpp and merge MTP PR #22673 ✅

- [x] VS 2022 BuildTools already installed (cmake + ninja bundled)
- [x] Clone: `git clone https://github.com/ggml-org/llama.cpp D:\Projects\llama.cpp`
- [x] Fetch PR branch and checkout `spec-mtp` — commit `3bdc61fe03299a84ade6f185819738c9a05200e7`
- [x] CUDA 12.4 installed via full local installer `cuda_12.4.0_551.61_windows.exe` (cuBLAS separate component required)
- [x] Built with Ninja + MSVC + CUDA 12.4: `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`
- [x] Verified: `llama-server version 9117 (3bdc61fe0)` — RTX 4080 Laptop GPU 12281 MiB sm_8.9 ✓
- **Binary:** `D:\Projects\llama.cpp\build\bin\llama-server.exe`
- **Note:** Run with `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` on PATH (handled in Start-RobotServices.ps1)

### Task 2: Download model GGUF ✅

#### 35B MoE (scrapped — too slow)
- [x] `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — 22.7 GB, downloaded to `D:\Projects\models\`
- Prefill: 86 tok/s (CPU MoE bottleneck), Decode: 17 tok/s, MTP: crashes after 1st request

#### 14B Dense (current target)
- [ ] `Qwen3-14B-UD-Q4_K_XL.gguf` — 9.16 GB, download to `D:\Projects\models\`
  ```powershell
  curl.exe -L -o "D:\Projects\models\Qwen3-14B-UD-Q4_K_XL.gguf" `
      "https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-UD-Q4_K_XL.gguf"
  ```

### Task 3: Wire up Start/Stop-RobotServices.ps1 ✅

- [x] llama-server launch in `Start-RobotServices.ps1` — updated for Qwen3-14B, dropped MTP/MoE flags, added `--cache-type-v q8_0`, context 16384
- [x] Ollama launch removed; Chatterbox commented out (parked)
- [x] `Stop-RobotServices.ps1` updated to kill `llama-server.exe` (was still killing Ollama)
- [ ] Test: stop all services, start via shortcut, confirm llama-server log shows model loaded (requires Task 2 download)

### Task 4: Benchmark on actual workload ⏸️ SHOW RESULTS BEFORE PROCEEDING

Run with representative Don Rickles system prompt (full, no trimming):
- [x] **Results (Qwen3-14B-UD-Q4_K_XL, 2026-05-12):**
  ```
  Prompt tokens:   3276
  Prefill:         1675 tok/s  |  TTFT: ~1956ms
  Decode:            29 tok/s  |  Total turn: 5.5s
  vs 35B MoE:      19× faster prefill, 1.7× faster decode
  ```
- No MTP (scrapped — PR #22673 checkpoint bug). No `--n-cpu-moe` (dense model, full GPU).
- **Decision: adopt 14B as production model.** Meets ≤5s warm-turn target.

### Task 5: Sanity-check tool-call JSON well-formedness ✅

- [x] 20/20 PASS — all turns valid tool_call JSON, greedy decoding, temperature=0
- [x] Consistent 3.2–4.4s per turn; 2 turns used text-only (correct behaviour for those prompts)
- [x] Hermes3 workarounds already removed in Task 6

### Task 6: Update config.py and chatterbox_tts.py ✅

- [x] Add `LLAMA_CPP_URL` to `config.py` — see `config.py:213` (`LLAMA_CPP_URL_ENV`) and `config.py:1070` (default).
- [x] Update `_call_llm()` in `chatterbox_tts.py` to use `/v1/chat/completions` — see `chatterbox_tts.py:314`.
- [x] Update `_ollama_base_url` property → `_llama_cpp_url` reading new config var — see `chatterbox_tts.py:205`.
- [x] Remove Hermes3 workarounds — `_trim_tool_spec`, `_coerce_text_tool_call`, `_nudge_llm`, `_TEXT_TOOL_CALL_RE`, `_parse_text_tool_args`, `_parse_json_content_tool_call` all gone from the tree (grep returns no hits).
- [x] Run existing test suite: `pytest tests/ -v`.
- [x] Update Pi `.env`: `LLAMA_CPP_URL=http://astralplane.lan:11434`.

### Task 7: Update OLLAMA_MODEL references ✅

- [x] `config.py`: `LLAMA_CPP_URL` is the live env var; legacy `OLLAMA_MODEL_DEFAULT` cleanup done as part of the BACKEND_PROVIDER retirement (Phase 4f, PR #381).
- [x] Issue #46 validation now targets llama-server.

---

## Open Questions

- [x] ~~Exact Qwen3 35B-A3B GGUF name and Unsloth repo URL?~~ — resolved at Task 2; 35B MoE scrapped, 14B dense adopted.
- [x] ~~Does PR #22673 build cleanly on MSVC?~~ — moot; MTP scrapped after checkpoint-mismatch crash.
- [x] ~~`--n-cpu-moe` sweet spot?~~ — n/a for 14B dense (full GPU, no CPU offload).
- [x] ~~MTP acceptance rate on tool-call JSON?~~ — n/a (MTP scrapped).
- [x] ~~Qwen3 `--jinja` template compatible with our tool schema?~~ — confirmed at Task 5 (20/20 pass).
- [ ] Benchmark Qwen3 14B dense vs 35B MoE on production prompt set (#78) — open; needs both models loaded on a GPU box.

---

## Reference Links

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [PR #22673 — MTP speculative decoding](https://github.com/ggml-org/llama.cpp/pull/22673)
- [Unsloth Qwen3 GGUFs on HuggingFace](https://huggingface.co/unsloth)
- [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)
- [CUDA 12.4 PyTorch wheels](https://download.pytorch.org/whl/cu124) — keep pinned, do NOT upgrade to 13.2
- GitHub Issue #59 — Switch local LLM from Hermes3 8B to Qwen
- GitHub Issue #60 — ElevenLabs TTS integration (tracks TTS side of this work)
