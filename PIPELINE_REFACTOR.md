# Pipeline Refactor — Epic #337 (Phase 4 closed) + Phase 5 (in flight)

**Status:** Phase 4 ✅ **COMPLETE** (epic #337 closed end-to-end on 2026-05-16). **Phase 5 in flight** — see `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` for the exploration memo and sub-phase DAG.

**Approach (Phase 4):** Option C (incremental retirement) from `docs/superpowers/specs/2026-05-15-phase-4-exploration.md`.

This document is the operating manual for both phases. Per-sub-phase specs and TDD plans live under `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Phase 4 status table (closed)

| Sub-phase | Goal | Status | PR / branch |
|-----------|------|--------|-------------|
| 4a | `ComposableConversationHandler(ConversationHandler)` wrapper over `ComposablePipeline` | ✅ Done | #355 (commit c8de597) |
| 4b | Factory dual path behind `REACHY_MINI_FACTORY_PATH` for `(moonshine, llama, elevenlabs)` | ✅ Done | #359 (commit 8f94691) |
| 4c | Expand composable to remaining triples + build `ChatterboxTTSAdapter`, `GeminiLLMAdapter`, `GeminiTTSAdapter` | ✅ Done (5/5 triples) | — |
| 4c.1 | `ChatterboxTTSAdapter` + `(moonshine, chatterbox, llama)` routing | ✅ Done | #361 (commit aa59ea1) |
| 4c.2 | `GeminiLLMAdapter` + `(moonshine, chatterbox, gemini)` routing | ✅ Done | #362 (commit a91acb2) |
| 4c.3 | `(moonshine, elevenlabs, gemini)` routing (reuses 4c.2 adapter) | ✅ Done | #364 (commit eb45e14) |
| 4c.4 | `(moonshine, elevenlabs, gemini-fallback)` routing | ✅ Done | #365 (commit ac1232e) |
| 4c.5 | `GeminiTTSAdapter` + `GeminiBundledLLMAdapter` + `(moonshine, gemini_tts)` routing | ✅ Done | #366 (commit 808da59) |
| 4c-tris | `HybridRealtimePipeline` for `LocalSTTOpenAIRealtimeHandler` / `LocalSTTHuggingFaceRealtimeHandler` | ⏭ Skipped (Option B per memo `docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md`) | #369 |
| 4d | Flip default `FACTORY_PATH=composable` | ✅ Done | #378 (commit 814efd8) |
| 4e | Delete legacy concrete handlers + the dual-path dial + rewrite tests | ✅ Done | #379 (commit 4bb06d1) |
| 4f | Retire `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` config dials | ✅ Done | #381 (commit 8873fa2) |

## Phase 5 status table

**Tracking epic:** #391.
Source-of-truth plan: `docs/superpowers/specs/2026-05-16-phase-5-exploration.md`.

| Sub-phase | Goal | Status | PR / branch |
|-----------|------|--------|-------------|
| 5a | Surviving TODO cleanup (composable_conversation_handler.py:158, composable_pipeline.py:278, chatterbox_tts_adapter.py:47, OTel `gen_ai.system` artifact, test-fixture `coroutine never awaited` smell) | ✅ Done (5a.1 echo-guard persona reset; 5a.2 delivery tags + first-audio marker via #393) | a4d0452 |
| 5b | Wire `ComposablePipeline.tool_dispatcher` in factory + `tool.execute` span (was latent prod bug — tool-triggered turns silently dropped on 4 of 5 composable triples) | ✅ Done | #388 (commit a8754d5) |
| 5c | Voice/personality method redesign (`apply_personality` ABC + wrapper forwarding) | ✅ Done (5c-1 wrapper forwards via `pipeline.tts`; 5c-2 `apply_personality` lives on `ComposablePipeline`) | #399 (101d07b), #401 (e3ab697) |
| 5d | `ConversationHandler` ABC: shrink (FastRTC-shim only) or collapse | ✅ Done — shrunk to FastRTC-shim role (72 lines in `conversation_handler.py`) | #403 (c8fef7d) |
| 5e | Factory STT decouple — make new STT backends pluggable cleanly (retire `LocalSTTInputMixin` from composable triples) | ✅ Done (6/6 sub-phases) | — |
| 5e.1 | Standalone `MoonshineSTTAdapter` + `MoonshineListener` (foundational) | ✅ Done | #405 (commit 847e6ac) |
| 5e.2 | Migrate `(moonshine, llama, elevenlabs)` off `LocalSTTInputMixin` + land `ComposablePipeline` host-concern wiring | ✅ Done | #407 (commit 03d55e2) |
| 5e.3 | Migrate `(moonshine, llama, chatterbox)` | ✅ Done | #409 (commit 35d1f7a) |
| 5e.4 | Migrate `(moonshine, gemini, chatterbox)` | ✅ Done | #411 (commit 8da54c4) |
| 5e.5 | Migrate `(moonshine, gemini, elevenlabs)` | ✅ Done | #412 (commit c6b3d40) |
| 5e.6 | Migrate `(moonshine, gemini-bundled, gemini_tts)` + retire legacy `_clear_queue` `_tts_handler` mirror; mark `LocalSTTInputMixin` as hybrid-only (still serves `LocalSTT*RealtimeHandler` hybrids per Phase 4c-tris Option B) | ✅ Done | `claude/phase-5e-6-gemini-tts-decouple-and-cleanup` |
| 5f | `FasterWhisperSTTAdapter` (closes #387) — alternate to Moonshine; A/B via `AUDIO_INPUT_BACKEND=faster_whisper` + `[faster_whisper_stt]` extra | ✅ Done | `claude/phase-5f-faster-whisper-stt-adapter` |

## Post-Phase-4 follow-up PRs (between Phase 4 close and Phase 5 5b)

| PR | What | Notes |
|----|------|-------|
| #382 | Hook #3b — `GeminiTTSAdapter` emits `first_greeting.tts_first_audio` | Parity restoration; legacy already emitted at `gemini_tts.py:415`, composable adapter was bypassing |
| #389 | Telemetry housekeeping + `_prepare_startup_credentials` triple-init guard | 4 orphan OTel attrs allowlisted; 2 dead counters wired (Moonshine errors + welcome-WAV underrun); idempotency guard; `handler.start_up.complete` event moved to post-delegate via `try/finally` |

## Post-Phase-5 follow-up PRs (after 5f close, all landed)

| PR | What | Notes |
|----|------|-------|
| #416 | `FasterWhisperSTTAdapter` (closes #387) — alternate to Moonshine | A/B via `AUDIO_INPUT_BACKEND=faster_whisper` |
| #420 | Phase-5f-1: swap `silero-vad` for `webrtcvad` to fit chassis eMMC | |
| #422 | Phase-5f-2: drop short transcripts + raise echo cooldown to break self-echo cascade | |
| #425 | Phase-5: widen transcript dedup + SequenceMatcher similarity for hypothesis-burst | |
| #426 | Phase-5f-3: content-similarity echo filter to close cascade gap left by 5f.2 | |
| #442 | `XttsTTSAdapter` — LAN xtts-v2 as composable TTSBackend + admin UI selector (#438) | Native adapter, not legacy-wrapping; raw PCM stream over chunked HTTP |
| #447 | Record `robot.tts.time_to_first_audio` metric per turn (#271) | |
| #463 | Scope chatterbox TTS probes to `AUDIO_OUTPUT_BACKEND=chatterbox` | XTTS-deployment boot reliability |
| #466 | TTS-down cooldown suppresses STT triggers on service outage | |
| #467 | Wire `speaking_until_setter` to close self-echo barge-in gap on XTTS path | |
| #468 | Post-TTS cooldown + stricter `no_speech_prob` to stop hallucination cascade | |
| #473 | Standalone `stt_service/` FastAPI package | Shipped; not yet wired into main app |

See `FORK_STATUS.md` for the consolidated post-fork architecture view.

## Research memos (2026-05-16, all merged)

- `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` (PR #386) — Phase 5 sub-phase DAG, surviving TODOs, recommendations. **Surfaced the 5b latent bug.**
- `docs/superpowers/specs/2026-05-16-reachy-boot-optimization-review.md` (PR #383) — full boot path survey + 10 attack vectors. Top-3-PRs-to-ship: instrument-first → parallelise-prepares → speak-during-load. Confirms ~42s cold-boot, ~20s Moonshine load is rank-1 cost.
- `docs/superpowers/specs/2026-05-16-instrumentation-audit.md` (PR #385) — telemetry inventory + 10 ranked gaps. Top gap (the missing `tool.execute` span) closed by Phase 5 5b. Three orphan attrs / two dead counters closed by #389.
- `docs/superpowers/specs/2026-05-16-moonshine-reliability-alternates-sibling-daemon.md` (PR #384) — **low confidence** on permanent Moonshine fix (Mode 1 rearm-N-then-die). Top alternate: faster-whisper `tiny.en` (~2s vs 20s cold-load). Sibling-daemon: **defer**.

---

## How the sub-phases run

This work is now **automated through a manager session**. See "Automation
note" at the bottom of this file for the loop shape.

Each sub-phase still follows the same pattern as 4a/4b:

1. Write a spec at `docs/superpowers/specs/<date>-phase-Nx-<slug>.md`.
2. Write a TDD plan at `docs/superpowers/plans/<date>-phase-Nx-<slug>.md`.
3. Execute TDD: failing test → minimum implementation → green → commit, one
   task at a time. Match the 4a/4b commit cadence.
4. Run `.venv/bin/ruff check` + `.venv/bin/ruff format --check` from the
   **repo root**, then `.venv/bin/mypy --pretty` on changed files, then
   `.venv/bin/pytest tests/ -q`.
5. Push to a branch named like `claude/phase-Nx-<slug>`.
6. Open a single PR. No stacking. Squash-merge with `--delete-branch`.
7. Pause for operator hardware validation **before 4d** (default flip) and
   **before 4e** (legacy deletion). Other sub-phases can ride straight
   through.

Memory files to honour every session (under `docs/superpowers/memory/`):

- `feedback_ruff_check_whole_repo_locally.md` — always ruff from repo root,
  never per-file (I001 only triggers on whole-repo).
- `feedback_ci_runs_against_pull_request_merge_commit.md` — diagnose green
  push + red pull_request as main being broken, fix main first.
- `feedback_pr_354_merged_red_branch_protection_check.md` — branch
  protection audit reminder.
- `project_session_2026_05_15_phase4a_landed_plan_for_phase4.md` — the
  expanded plan, deferred lifecycle hooks, container bootstrap recipe.

Container bootstrap (no `/venvs/apps_venv` in remote execution containers):

```
apt-get install -y libgirepository1.0-dev
uv sync --frozen --all-extras --group dev
```

---

## Sub-phase 4c — Expand composable to remaining triples

**Goal:** route every composable triple through `ComposableConversationHandler`
behind the `REACHY_MINI_FACTORY_PATH=composable` flag, so the only legacy
classes still reachable from the factory are the bundled-realtime handlers
and the two `LocalSTT*RealtimeHandler` hybrids (those are 4c-tris).

**Triples to migrate (one PR per triple, no stacking):**

| Triple | Today's class | Adapter work needed |
|--------|---------------|---------------------|
| `(moonshine, chatterbox, llama)` | `LocalSTTChatterboxHandler` | **`ChatterboxTTSAdapter`** (new ~120 LOC); reuse `LlamaLLMAdapter` + `MoonshineSTTAdapter` |
| `(moonshine, chatterbox, gemini)` | `GeminiTextChatterboxHandler` | **`GeminiLLMAdapter`** (new ~150 LOC) + `ChatterboxTTSAdapter` |
| `(moonshine, elevenlabs, gemini)` | `GeminiTextElevenLabsHandler` | `GeminiLLMAdapter` + `ElevenLabsTTSAdapter` (broaden annotation to accept the duck-typed surface — 4b's TODO) |
| `(moonshine, elevenlabs, gemini-fallback)` | `LocalSTTGeminiElevenLabsHandler` | Same as above; the Gemini-hardcoded `_prepare_startup_credentials` is what the adapter delegates into |
| `(moonshine, gemini_tts)` | `LocalSTTGeminiTTSHandler` | **`GeminiTTSAdapter`** (new ~120 LOC). Gemini-native LLM+TTS uses one `genai` client; the adapter wraps both the LLM call and the TTS stream. |

**New adapters (each in its own commit before the routing flip):**

- `ChatterboxTTSAdapter` — wraps `ChatterboxTTSResponseHandler._stream_tts_to_queue`. Pattern mirrors `ElevenLabsTTSAdapter` (temp queue + sentinel + task cleanup). First-audio marker and tags TODO per Phase 4 plan.
- `GeminiLLMAdapter` — wraps `GeminiTextResponseHandler._call_llm`. Mirrors `LlamaLLMAdapter` (history swap, tool-call conversion). Gemini's tool-call shape differs from llama-server's — convert at the adapter boundary, don't leak it to the orchestrator.
- `GeminiTTSAdapter` — wraps `GeminiTTSResponseHandler`. Both LLM and TTS run through the same `genai.Client`; the adapter handles the temp-queue pattern for the TTS half and exposes the LLM half via a partner `GeminiLLMAdapter` instance (or a single `GeminiBundledAdapter` — design decision for the sub-phase).

**Factory wiring:** one new `_build_composable_<output>_<llm>` helper per
triple in `handler_factory.py`, gated on `FACTORY_PATH=composable` inside
each existing per-triple branch. Pattern is exactly 4b's
`_build_composable_llama_elevenlabs`.

**Out of scope for 4c:**

- The `LocalSTT*RealtimeHandler` hybrids (4c-tris).
- Lifecycle-hook plumbing (telemetry, boot-timeline events, joke history,
  history trim, echo-guard). Each is a small follow-up PR between phases.
- Default flip (4d).
- BACKEND_PROVIDER retirement (4f).

**Open design questions for the 4c sub-agent to answer in its spec:**

1. Does `GeminiTTSAdapter` cover LLM+TTS as one adapter, or do we ship a
   separate `GeminiLLMAdapter` even for the Gemini-native-bundled case?
2. How does `ChatterboxTTSAdapter` surface the auto-gain / target-dBFS
   knobs? The legacy handler reads them from `config.py`; the adapter
   should keep doing that.
3. `change_voice` / `get_available_voices` for the Gemini and Chatterbox
   variants — the wrapper forwards to `_tts_handler`. Confirm that
   `GeminiTTSResponseHandler` and `ChatterboxTTSResponseHandler` both
   expose those methods today (they should — they inherit from
   `ConversationHandler`).

**Acceptance criteria:**

- All five triples above return `ComposableConversationHandler` under
  `FACTORY_PATH=composable`, the same concrete legacy class under
  `FACTORY_PATH=legacy`.
- Per-triple integration tests for transcript→audio round trip via real
  `ComposablePipeline` + the new adapters (stubbed network).
- All existing tests still pass.
- New `ruff check`, `ruff format`, `mypy`, `pytest` green from repo root.

**Estimated PRs:** five — one per triple. Plus one each for any new adapter
that ships separately (if the operator wants finer granularity, the
manager can split adapter-build PRs from routing PRs).

---

## Sub-phase 4c-tris — `HybridRealtimePipeline`

**Goal:** introduce a sibling pipeline class so `LocalSTTOpenAIRealtimeHandler`
and `LocalSTTHuggingFaceRealtimeHandler` can be routed through the same
`ComposableConversationHandler`-shaped wrapper, even though their LLM+TTS
half lives inside a single websocket session.

**Why a sibling class:** the STT/LLM/TTS Protocol from `backends.py` doesn't
fit — the realtime endpoint owns LLM+TTS as one unit. `HybridRealtimePipeline`
exposes `STTBackend` for the Moonshine half and a single `RealtimeBackend`
Protocol for the bundled half. The wrapper's interface stays the same so
the factory doesn't need a third dispatch path beyond
`FACTORY_PATH=composable`.

**Predecessor:** 4c must land first (we need the wrapper proven on the
non-hybrid composable triples).

**Required deliverables:**

1. A separate **design memo** at
   `docs/superpowers/specs/<date>-phase-4c-tris-hybrid-realtime-design.md`
   exploring whether this class is worth shipping or whether the two
   hybrid handlers should stay legacy forever. The exploration memo
   §6.6 already flags the question. **Do not implement until the
   operator signs off on the memo.**
2. If the memo says ship: one PR per hybrid handler migration.

**Acceptance criteria (only if the memo green-lights implementation):**

- `RealtimeBackend` Protocol defined alongside the existing three.
- `HybridRealtimePipeline` class with the same lifecycle surface as
  `ComposablePipeline` (`start_up` / `feed_audio` / `output_queue`).
- Factory routes the two hybrid triples through `ComposableConversationHandler`
  under `FACTORY_PATH=composable`.
- All existing tests still pass.

---

## Sub-phase 4d — Flip default to composable

**Prerequisites:**

- All 4c sub-phases merged and operator-validated on hardware.
- 4c-tris memo answered (ship or skip). If shipping, those PRs merged.
- All deferred lifecycle-hook follow-up PRs that the operator deemed
  blocking for the default flip are merged.

**Change:**

- Single-line: `DEFAULT_FACTORY_PATH = FACTORY_PATH_COMPOSABLE` in
  `src/robot_comic/config.py`.
- Update `.env.example` to match (`REACHY_MINI_FACTORY_PATH=composable`).
- Update `test_config_factory_path.py::test_config_field_default` to assert
  the new default.

**Soak window:** operator runs the robot for at least one full session
(estimated ≥1 hour of real conversation) on the new default before
moving to 4e. **Hard pause** between 4d's merge and 4e's start — the
manager session must wait for explicit operator green-light.

**Acceptance criteria:**

- `config.FACTORY_PATH` defaults to `composable` under unset env-var.
- All existing tests pass with the new default; any test that hard-coded
  `legacy` is parametrised over both values.

---

## Sub-phase 4e — Delete legacy concrete handlers + retire `FACTORY_PATH`

**Prerequisites:** operator has explicitly green-lit 4d's soak, with no
regressions in the issue tracker for at least one robot session.

**Files to delete:**

- `src/robot_comic/llama_elevenlabs_tts.py::LocalSTTLlamaElevenLabsHandler` (lines 359–366).
- `src/robot_comic/chatterbox_tts.py::LocalSTTChatterboxHandler`.
- `src/robot_comic/elevenlabs_tts.py::LocalSTTGeminiElevenLabsHandler` + the legacy alias `LocalSTTElevenLabsHandler`.
- `src/robot_comic/gemini_tts.py::LocalSTTGeminiTTSHandler`.
- `src/robot_comic/gemini_text_handlers.py::GeminiTextChatterboxHandler` and `GeminiTextElevenLabsHandler` + their `*ResponseHandler` diamond bases (if no longer referenced).
- `src/robot_comic/llama_gemini_tts.py::LocalSTTLlamaGeminiTTSHandler` (the "orphan" — already unreachable from the factory; confirm with a final grep before deletion).

**Files to keep:**

- `BaseLlamaResponseHandler` *internals* (`_call_llm`, `_prepare_startup_credentials`, etc.) — adapters call these.
- `ElevenLabsTTSResponseHandler` internals (`_stream_tts_to_queue`, etc.) — same.
- `ChatterboxTTSResponseHandler` internals — same.
- `GeminiTTSResponseHandler` internals — same.
- `GeminiTextResponseHandler` internals — same.
- `LocalSTTInputMixin` — `MoonshineSTTAdapter` depends on it.

**Factory cleanup:**

- Remove `FACTORY_PATH` constants from `config.py` and the dial logic.
- Remove the `if FACTORY_PATH == composable:` branches in
  `handler_factory.py`; the composable path becomes the only path.
- Delete the `LLM_BACKEND_LLAMA` / `LLM_BACKEND_GEMINI` branching in the
  factory if the per-triple `_build_*` helpers absorb it.

**Test rewrites:**

The exploration memo §5 has the full list. Big ones:

- `test_handler_factory.py` / `test_handler_factory_llama_llm.py` /
  `test_handler_factory_gemini_llm.py` — rewrite to assert
  `ComposableConversationHandler` is the return value with the right
  adapter chain.
- `test_handler_factory_factory_path.py` — delete (the dial is gone).
- `test_llama_base.py` (28.7 KB) — keep ~50%, the orchestration tests go.
- `test_elevenlabs_tts.py` (45.8 KB) — keep ~70%, orchestration goes.
- `test_llama_elevenlabs_tts.py` — most of this goes.
- `test_elevenlabs_start_up_supporting_events.py` — migrate to the wrapper
  (or to whichever component fires the events post-4e).

**Risk:** this is the biggest PR of the lot — 2k+ LOC delta. Manager
should consider splitting per-handler-deletion into separate PRs (one for
chatterbox, one for elevenlabs, etc.) if the diff gets too unwieldy for
review. The exploration memo recommended that, and the operator's
"one-PR-per-sub-phase" rule has wiggle room here.

**Acceptance criteria:**

- No file under `src/robot_comic/` still defines a class named
  `LocalSTT*Handler` (except the underlying mixin / response-handler
  bases that survive).
- The factory's source-of-truth dispatch matrix lives entirely in the
  composable helpers.
- All tests still pass.
- `git grep "LocalSTTLlamaElevenLabsHandler\|LocalSTTChatterboxHandler\|LocalSTTGeminiElevenLabsHandler\|LocalSTTGeminiTTSHandler\|GeminiTextChatterboxHandler\|GeminiTextElevenLabsHandler\|LocalSTTLlamaGeminiTTSHandler"` returns nothing inside `src/`.

---

## Sub-phase 4f — Retire `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND`

**Prerequisites:** 4e merged. All legacy class names are gone from `src/`.

**Why this is its own sub-phase:** the exploration memo §3 calls this out
explicitly — `BACKEND_PROVIDER` is woven into profile validation, model-name
derivation (`_resolve_model_name`), operator-facing `.env.example`, deploy
scripts under `deploy/`, and `main.py:240-251`'s HF-specific logging paths.
Cutting it is its own mini-refactor.

**Scope:**

- `config.py` — delete `BACKEND_PROVIDER`, `LOCAL_STT_RESPONSE_BACKEND`,
  `DEFAULT_BACKEND_PROVIDER`, `_normalize_backend_provider`, the
  `derive_audio_backends` / `resolve_audio_backends` machinery if it
  becomes orphaned.
- `main.py:240-251` — remove the HF-specific logging branches that
  switch on `config.BACKEND_PROVIDER`.
- `profiles/` — audit every profile's `instructions.txt` and any per-profile
  config for `BACKEND_PROVIDER` references; rewrite.
- `.env.example` — remove the dial documentation.
- `deploy/` — audit the systemd unit, the `reachy-app-autostart` install
  script, and any env-template files for the dial.
- `external_content/` — same audit if external profiles reference the dial.

**Acceptance criteria:**

- `git grep "BACKEND_PROVIDER\|LOCAL_STT_RESPONSE_BACKEND"` returns nothing
  in `src/`, `profiles/`, `deploy/`, `.env.example`, `pyproject.toml`,
  or any test file.
- Existing deployment still boots on the robot without those env vars set.
- All tests still pass.

---

## Deferred lifecycle hooks — follow-up PRs between sub-phases

All five hooks landed before 4d (default flip), so the composable path
reaches behavioural parity with legacy at the moment the default flips.

| Hook | Status | PR | Notes |
|------|--------|----|-------|
| `_speaking_until` echo-guard (`elevenlabs_tts.py:471-473`) | ✅ Done | #372 | Doc's "for free via adapter delegation" claim was **wrong**: the legacy write site lived in `emit()`, which the composable wrapper bypasses. Real fix moved the derivation into `_enqueue_audio_frame`. Also caught two subclass put-site bypasses (`llama_elevenlabs_tts.py`, `chatterbox_tts.py`). |
| `telemetry.record_llm_duration` | ✅ Done | #373 | NOT preserved through delegation. Fix wraps timing inside `LlamaLLMAdapter.chat`, `GeminiLLMAdapter.chat`, `GeminiBundledLLMAdapter.chat` (`try/finally` — exception parity with legacy). Bundled-Gemini gets new telemetry the legacy never emitted. |
| Boot-timeline supporting events (#321) | ✅ Done | #374 | Of #321's four events, only `handler.start_up.complete` was dropped (others preserved via `main.py` emission or adapter delegation). Fix emits in `ComposableConversationHandler.start_up()` before `pipeline.start_up()`. **Follow-up flagged:** `GeminiTTSAdapter` doesn't emit `first_greeting.tts_first_audio`; same bug class, separate test surface, deferred. |
| `record_joke_history` (`llama_base.py:578-594`, `gemini_tts.py:380-394`) | ✅ Done | #375 | NOT preserved. Orchestrator-level fix in `ComposablePipeline._speak_assistant_text`. New public `joke_history.record_joke_history(text)` helper for testability; legacy sites untouched until 4e. |
| `history_trim.trim_history_in_place` | ✅ Done | #376 | NOT preserved (doc's hint that delegation might cover it was incorrect — legacy trim lives in `_dispatch_completed_transcript`, not `_call_llm`). Orchestrator-level fix at top of `ComposablePipeline._run_llm_loop_and_speak`. |

**Lessons learned:** Doc-table "for free via adapter delegation" claims are
suspect by default. 5/5 hooks required real code changes; 0/5 were
test-only confirmations. Every future hook briefing should assume a fix is
needed and verify with a regression test before claiming preservation.

---

## Phase 5 cadence — sub-phase at a time

Phase 5 does NOT run as a continuous manager session. Operator's preference is to ship one sub-phase at a time and decide whether to chain another based on outcome. The exploration memo (`docs/superpowers/specs/2026-05-16-phase-5-exploration.md`) is the source-of-truth plan; this document tracks status only.

Sub-agent dispatch pattern stays the same as Phase 4 (spec + plan + TDD commits + one PR per sub-phase). The Phase 4 manager-session pattern below is preserved for reference and may be revived if the operator opts back into it.

## Automation note — Phase 4 manager session pattern (historical, for reference)

Starting at 4c, this work is driven by a **manager session** that does not
do the implementation itself. The pattern:

1. Manager session opens (operator launches it from `claude.ai/code`).
2. Manager reads `PIPELINE_REFACTOR.md` to find the next pending sub-phase.
3. Manager spawns a sub-agent via the `Agent` tool with
   `subagent_type=general-purpose` (or `claude`), briefed with the full
   sub-phase scope from this document, the relevant memory files, and
   the predecessor specs/plans. The sub-agent owns the per-sub-phase
   context.
4. Manager **monitors** the sub-agent via the `Monitor` tool (or
   `run_in_background`) so notifications surface as the sub-agent makes
   progress. The manager's own context window stays light — sub-agent
   results return as a single summary at the end.
5. When the sub-agent reports "branch pushed, ready to PR," manager
   uses the GitHub MCP tools to:
   - Verify CI is green on the branch (no green push + red pull_request
     mismatch — see the memory file).
   - Open the PR with the sub-agent's summary as the body.
   - Watch for CI completion via `subscribe_pr_activity`.
   - On green CI, squash-merge with `--delete-branch`.
   - Sync the manager's local main (`git fetch origin && git checkout main && git reset --hard origin/main`).
6. Manager **pauses** for operator hardware validation **only at
   designated checkpoints** (currently: before 4d, before 4e). For other
   sub-phases the manager proceeds straight to the next one.
7. Manager loops to step 2 until the status table is all ✅.

**Hard rules for the manager:**

- One PR per sub-phase. **No stacking.** If a sub-phase has multiple
  triples (4c does), it becomes multiple PRs — one per triple, merged
  sequentially.
- Never run `git push --force` to main. Never disable CI checks. Never
  merge with red CI. (Branch protection should enforce this, but the
  feedback memory shows it doesn't always — be paranoid.)
- Never bypass the `--no-verify` flag or the pre-commit hook.
- Never touch the upstream fork (`pollen-robotics/reachy_mini_conversation_app`).
- Memory files take precedence over this document if they conflict.

**Sub-agent context to inject every spawn:**

- The full text of this section of `PIPELINE_REFACTOR.md`.
- The four memory files under `docs/superpowers/memory/`.
- The exploration memo `docs/superpowers/specs/2026-05-15-phase-4-exploration.md`.
- The 4a + 4b specs and plans (as style templates).
- The current branch / commit on origin/main.
- The container bootstrap recipe.

**When to stop the loop and check with the operator:**

- Before 4d's default flip — **mandatory hardware validation**.
- Before 4e's legacy deletion — **mandatory operator green-light** after 4d soak.
- Any sub-phase whose acceptance criteria can't be met without an
  architecture decision the spec doesn't already answer.
- Any sub-phase that produces a 2k+ LOC delta in a single PR (4e is a
  candidate — split per-handler-deletion if so).

---

## Out of scope for Phase 4 (the whole epic, not just 4c)

- Bundled-realtime handlers (`OpenaiRealtimeHandler`, `HuggingFaceRealtimeHandler`, `GeminiLiveHandler`) — they stay as-is.
- The `LocalSTTInputMixin` survives untouched; `MoonshineSTTAdapter` depends on it.
- Any new STT backend (Whisper, Deepgram) — Phase 5 territory.
- Tool-system refactors. The adapter-level "ignore the orchestrator's tools arg" pattern stays; the orchestrator's `tool_dispatcher` callback wiring is a separate refactor (also Phase 5).
- Voice / personality method redesign. The wrapper forwards to the legacy TTS handler; that contract survives into 4e and only changes if `ConversationHandler`'s ABC is rewritten (Phase 5).
