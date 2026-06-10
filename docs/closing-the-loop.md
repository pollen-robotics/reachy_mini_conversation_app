# Closing the loop: autonomous robot observation (design)

**Status: design only — no code shipped yet.** This documents how an autonomous
coding agent (Claude Code running non-interactively, or the Hermes Pi agent)
could *verify the robot actually did something* — spoke, moved, reacted — so we
can iterate on "what good looks like" without a human watching every run.

The premise: a coding laptop drives changes; the robot (Richie) runs the app.
Today the only feedback is the maintainer's eyes and ears. We want a cheap,
machine-readable feedback channel that closes the perceive→act→verify loop.

## Why this is tractable today

Most of the signals already exist; we mostly need to *expose* them:

- `telemetry.py` already emits structured spans as `RCSPAN <json>` lines to
  stdout (`CompactLineExporter`) with attributes like `turn.outcome`,
  `turn.excerpt`, `tool.name`, `tool.id`, `vad.duration_ms`, `stt.type`, plus the
  `robot.tts.time_to_first_audio` metric and a `welcome.wav.completed` span when
  playback actually finishes. There's also an OTLP export path
  (`docs/signoz-on-pi.md`) that already targets a Pi.
- `audio/speech_tapper.py` already computes RMS dBFS (`_rms_dbfs`, thresholds
  `VAD_DB_ON = -35`, `VAD_DB_OFF = -45`) — the math for "is sound happening".
- `camera_worker.py` already owns a camera thread producing frames.
- `docs/references/audio-playback-recipe.md` already documents the
  laptop-speaker → robot-mic → STT→LLM→TTS loop; `docs/ws-protocol.md` documents
  a Pi↔laptop WebSocket channel.

## The cost/complexity ladder

Build bottom-up; each tier stands alone and is useful on its own.

### Tier 0 — robot event log (cheapest) — ✅ shipped (#515)

A JSONL sink fed by the spans the app already emits. Each line is a timestamped
fact: TTS started/ended, peak dBFS for the turn, the transcript excerpt, which
tool fired (`dance`, `play_emotion`, …), turn outcome. An agent reads the tail
and answers *"did the robot speak/move in the last N seconds, and what did it
say/do?"* — no new ML, no new sensors.

Shipped as `telemetry.JsonlEventExporter` (enable with
`ROBOT_INSTRUMENTATION=trace` + `ROBOT_EVENT_LOG=/path/events.jsonl`) and the
`robot_comic.observer.events` reader.

### Tier 1 — independent audio witness — ✅ shipped

A tiny always-on process on the *laptop* reads its own mic, computes RMS dBFS in
~100 ms blocks (the `dbfs()` helper mirrors `audio/speech_tapper.py`), and logs
"sound present" intervals with timestamps. This **independently confirms the
robot's speaker actually produced sound** — not just that the app *claims* it
did. Pairs with Tier 0: app says "TTS at t=12.3s", witness says "sound at
12.4–14.1s" ⇒ verified.

Shipped as `robot_comic.observer.audio_witness` (run it natively where the mic
lives — WSL2 mic capture is unreliable):

```
ROBOT_AUDIO_LOG=/path/audio.jsonl python -m robot_comic.observer.audio_witness   # needs: uv pip install sounddevice numpy
```

The `robot_comic.observer.audio_activity` reader + the `robot_get_audio_activity`
MCP tool surface it to the agent. A hysteresis state machine (`DB_ON` / `DB_OFF`,
defaults −35 / −45 dBFS) keeps brief dips from chopping one utterance into many.

Later option: hand the active windows to a local Whisper/Moonshine to capture
*what* was heard — but STT is a known pain point (see issues in
`lane/improvement` + the `Local voice clone + STT` milestone), so start with
level-only.

### Tier 2 — visual witness (on-demand only)

Triggered still capture from a webcam aimed at the robot (or reuse
`camera_worker` frames), described by a vision model to confirm a gesture ("did
the antennas move? is the head centered?"). Expensive and laggy, so **on-demand**
(when the agent needs to confirm a specific motion), never a continuous stream.

## The MCP

Wrap the tiers as a thin **local MCP server** the agent can call. Tools:

| Tool | Tier | Returns / does |
| --- | --- | --- |
| `robot_get_recent_events(window_s)` | 0 | Recent events from the JSONL (speech, tools, outcomes) |
| `robot_get_audio_activity(window_s)` | 1 | Sound-present intervals + peak dBFS from the mic witness |
| `robot_capture_frame()` | 2 | A still image from the robot-facing camera |
| `robot_describe_scene(prompt)` | 2 | Vision-model answer about the current frame |
| `robot_play_prompt(wav \| text)` | actuator | Plays audio from the laptop speaker so the robot's mic hears it |

`robot_play_prompt` is the autonomous *trigger*: it plays a sound (or TTS of
`text`) into the room, exactly the `audio-playback-recipe.md` loop, so behaviors
that depend on hearing audio can be exercised without a human talking.

Shipped as `robot_comic.observer.play_prompt` (run the MCP server where the
speaker lives; `uv pip install sounddevice numpy piper-tts`). Provide exactly one
of `wav` (a file path) or `text` (synthesized via piper —
`ROBOT_PLAY_PROMPT_PIPER_MODEL` points at a `.onnx` voice). Guarded per the
*Tradeoffs & guardrails* below: opt-in via `ROBOT_PLAY_PROMPT_ENABLED=1`, a
`ROBOT_PLAY_PROMPT_MAX_DURATION_S` clip cap (default 15s), and rate limiting
(`ROBOT_PLAY_PROMPT_COOLDOWN_S` gap, default 3s; `ROBOT_PLAY_PROMPT_MAX_PLAYS`
per-process cap, default 20 — the stop condition for an unattended loop). On
refusal the tool returns `{"played": false, "error": ...}` rather than raising.

The server is a small local process reading the Tier-0 JSONL + driving the
mic/camera; register it in the agent's MCP config (Claude Code or Hermes). It can
read signals locally, or over the existing `docs/ws-protocol.md` WebSocket when
the robot and laptop are separate hosts.

## Example autonomous loop

```
1. robot_play_prompt(text="Hey Richie, tell me a joke")   # trigger
2. wait ~6s
3. robot_get_audio_activity(window_s=8)                    # did sound come back?
4. robot_get_recent_events(window_s=8)                     # what did it say / which tools fired?
5. (optional) robot_capture_frame() + robot_describe_scene("did the head/antennas move?")
6. assert: TTS occurred AND audio witnessed AND (if expected) a motion tool fired
```

A green run is evidence the end-to-end behavior works; a red run hands the agent
the events to debug.

This loop is packaged as `robot_comic.observer.loop_check` — one call snapshots
the robot's event log, fires the prompt, listens on the laptop mic, diffs the
*newly appended* events (a line-count diff, so a robot/laptop clock skew can't
hide the turn), and returns a `{passed, robot_spoke, heard_response, excerpt,
tools_fired, ...}` verdict. Run it as a CLI
(`python -m robot_comic.observer.loop_check --text "Hey Richie, tell me a joke"
--expect-tool play_emotion`; exit 0/1) or via the `robot_run_loop_check` MCP
tool. Topology via `LOOP_CHECK_SSH_HOST` / `LOOP_CHECK_REMOTE_EVENT_LOG`.

## Tradeoffs & guardrails

- **Cost/latency:** Tier 0/1 are nearly free and fast; Tier 2 vision calls are
  the expensive, laggy ones — gate them behind explicit need.
- **Echo:** the robot's mic hears its own TTS; the app's echo-guard
  (`speaking_until` / `composable_pipeline.py`) already handles the app side, and
  the Tier-1 witness should timestamp-correlate rather than transcribe to avoid
  confusion.
- **Safety:** `robot_play_prompt` makes the robot move/speak. Keep it opt-in per
  session and rate-limited; never wire it into an unattended tight loop without a
  stop condition.
- **Privacy:** the audio/visual witnesses record the room. Keep logs local,
  short-retention, and out of the repo (the pre-commit audio guard already blocks
  stray audio commits).

## Suggested build order

1. ✅ Tier 0 JSONL sink + `robot_get_recent_events` (smallest, highest signal).
2. ✅ Tier 1 mic witness + `robot_get_audio_activity`.
3. ✅ `robot_play_prompt` actuator → first real closed loop.
4. Tier 2 visual witness, on-demand.
