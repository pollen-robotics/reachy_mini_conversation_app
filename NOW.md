# Now

## Current focus

Infrastructure polish and persona completeness. The core comedian pipeline is
stable; current effort is on reliability (boot sequencing, hardware safety,
echo suppression) and rounding out the remaining persona-specific assets and
guardrails.

## In progress

### Hardware-validation queue (needs real robot)

- [#155] wake_up animation physical smoothing — IK solver / motion profile gentling
- [#138] test: admin Restart end-to-end click-test (exit-75 → systemd relaunch)
- [#121] boot: investigate 11.5s gap between movement manager init and handler ready
- [#108] Welcome Gate: finish remaining welcome_name.wav assets (5 comedians + 6 characters)

## Next up

- **S2S smoke-test pass** — run the checklist in `FORK_STATUS.md` §3 against Gemini Live and the `(moonshine, llama, xtts)` triple before the next persona session.
- [#54] Modular audio pipeline — separate `AUDIO_INPUT_BACKEND` + `AUDIO_OUTPUT_BACKEND` config
- [#78] Benchmark Qwen3 14B dense vs 35B MoE for production suitability
- [#20] Basic L-R movement testing
- [#19] Servo calibration: avoid body collisions
- [#473 follow-up] Wire standalone `stt_service/` package into `main.py` / handler factory

## Recently shipped

*(rolling 30-day snapshot — last updated 2026-05-19)*

- [perf] (PR #474) Defer `av` import in `tools/{camera,roast}.py` — cold-boot win
- [hw]   (PR #475) IK tuning for head trajectories
- [boot] (PR #470) Camera startup fix
- [pkg]  (PR #473) Standalone `stt_service/` FastAPI package (not yet wired)
- [#271] (PR #447) Record `robot.tts.time_to_first_audio` metric per turn
- [cascade] (PR #468) Post-TTS cooldown + stricter `no_speech_prob` to stop hallucination cascade
- [xtts] (PR #467) Wire `speaking_until_setter` to close self-echo barge-in gap
- [cascade] (PR #466) TTS-down cooldown suppresses STT triggers on service outage
- [chatterbox] (PR #463) Skip TTS probes when `AUDIO_OUTPUT_BACKEND != chatterbox`
- [#438] (PR #442) Wire LAN xtts-v2 as composable TTSBackend + admin UI selector
- [phase-5f] (PRs #420/#422/#425/#426) Cascade hardening — webrtcvad swap, echo cooldown, transcript dedup, content-similarity filter
- [phase-5f] (PR #416) Add `FasterWhisperSTTAdapter` as alternate STT to Moonshine
- [phase-5e] (PRs #405/#407/#409/#411/#412/#413) Migrate all composable triples off `LocalSTTInputMixin`
- [phase-5d] (PR #403) Shrink `ConversationHandler` ABC to FastRTC-shim role (72 lines)
- [phase-5c] (PRs #399/#401) `TTSBackend` voice methods; `apply_personality` on `ComposablePipeline`
- [#110] (PR #204) Welcome Gate state machine gates handler until persona name is heard
- [#31]  (PR #203) Boot: send Wake-on-LAN magic packet when llama-server is asleep
- [#91]  (PR #202) Echo suppression: tighter playback-end estimate
- [#77]  (PR #201) Startup: pre-warm llama-server KV cache to cut first-turn latency
- [#113] (PR #200) Warmup: Windows winsound player + optional fast-blip cue
- [#53]  (PR #199) Chatterbox: auto-normalize voice clone output gain
- [#42]  (PR #198) Bill Hicks disengagement guardrail
- [#41]  (PR #197) Startup: optional kiosk-mode voice prompt before persona selection
- [#132] (PR #196) Battery: surface Reachy battery status in admin UI and monitor
- [#63]  (PR #195) Profiles: complete George Carlin persona
- [#21]  (PR #194) STT: Moonshine .ort fast-load + page-cache prewarm
- [#34]  (PR #193) Profiles: complete Rodney Dangerfield persona
- [#8]   (PR #191) Persona: persist joke history across sessions to avoid repeats
- [#135] (PR #189) Gemini Live: exponential-backoff presence check on user silence
- [#139] (PR #188) Profiles: add gemini_live.txt delivery styling for each persona
- [#6]   (PR #187) Boot: replace 30s sleep with active Reachy-daemon ready-poll
- [#98]  (PR #186) Trigger: switch primary stop-word prefix to 'robot' with backward compat
- [#93]  (PR #185) Monitor: show transcript excerpt on pending row

---

*Updated: 2026-05-19. Architecture overview and fork-divergence summary in `FORK_STATUS.md`.*
