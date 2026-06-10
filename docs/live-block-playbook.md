# Live-Block Playbook

Operating contract for **autonomous live-robot sessions** ("live blocks") run by
an agent without the operator physically present. Blessed by the operator
2026-06-10; changes to abort criteria or duration caps require operator sign-off.

A *live block* is any window where the agent starts the robot app (or drives
motion via the daemon) for testing. Outside a live block the app stays stopped
(operating policy).

## Preconditions (all must pass before `systemctl start`)

1. **Log sink running** on the workstation (UDP 9477 → sink file) and the
   observer MCP responding (`robot_get_recent_events` returns cleanly).
   A dead sink silently discards all telemetry — never start without it.
2. **Daemon healthy**: `reachy-mini-daemon` active, motors enabled.
3. **Volume at the canonical level** (70 %) — check, don't assume.
4. **Denylists present** in the robot's `.env`
   (`REACHY_MINI_PLAY_EMOTION_DENYLIST`, `REACHY_MINI_DANCE_DENYLIST`).
5. **Bonk watcher armed**: `robot_watch_bonks` running for the duration of any
   phase with motion.
6. **Intended persona/backend verified** in the robot's startup settings
   *before* boot — not corrected mid-session.
7. **Artifacts directory created** on the workstation
   (`D:\logs\<block-name>\`) — every phase writes there.

## Hard limits

- **Max block duration: 45 minutes** from app start to app stop. One extension
  only if actively mid-capture, never past 60 minutes total.
- **App stop is unconditional** at block end — including on agent error,
  context exhaustion, or abort. The kill switch is:
  `ssh ricci 'sudo -n systemctl stop reachy-app-autostart'`
- **No firmware, daemon-config, or motor-limit changes** during a live block.

## Abort criteria (stop the app immediately, then diagnose)

- A **bonk** is detected (`robot_watch_bonks` event) or the witness camera
  shows the robot contacting anything.
- A **motion-smoothness spike** comparable to the pre-#519 head-slam signature
  (`robot_measure_motion` jerk attribution names a runaway move).
- **3 consecutive failed turns** (no TTS audio witnessed within 30 s of a
  trigger) — the session is wedged; restarting mid-block counts toward the
  duration cap.
- **Self-echo loop**: ≥2 consecutive `outcome=self_echo` turns that trigger
  further robot speech (cascade signature).
- **Any systemd restart of the app** that the agent did not command
  (`NRestarts` increments) — the cause must be diagnosed offline, not live.
- **Sink goes quiet** while the app claims activity (relay dead → flying blind).

## During the block

- Announce start and stop in the conversation ("robot is live" / "robot is
  stopped") so the operator can always tell the physical state at a glance.
- Phases that play audio into the mic (`robot_play_prompt`) must wait for
  quiet (`robot_get_audio_activity`) before triggering, so turn attribution
  stays clean.
- Every turn/capture writes artifacts as it goes (WAV, frames, span excerpts) —
  a crash must not lose the evidence collected so far.

## After the block

1. Stop the app; confirm `inactive` via `systemctl is-active`.
2. Verify motors returned to a safe idle pose (witness frame).
3. Generate the review artifact (report/HTML) in the block's artifacts
   directory and link it in the conversation + relevant issue(s).
4. Subjective-quality judgments (was the riff *good*?) are recorded as agent
   verdicts against the operator's rubric and queued for async operator
   review — never silently treated as final.

## Operator touchpoints

The operator is **not required** during a block. Their touchpoints are:
- one-time blessing of this playbook (and any later edits to limits/aborts),
- async review of the generated artifacts,
- the standing rubric for subjective quality calls
  (`profiles` repo / private notes; encode once, reuse).
