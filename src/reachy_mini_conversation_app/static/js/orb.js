/**
 * Conversation orb: SSE-driven state machine → CSS data-state attribute.
 * The root is a <button> - the talk view uses it as the mic toggle.
 */

import { h } from "./ui.js";
import { GLOW_BY_STATE, ORB_STATES } from "./constants.js";

/** Map a backend activity reason to a visual state; null means keep current. */
export function mapActivityToState(reason) {
  switch (reason) {
    case "user_speech_started":
    case "user_transcription_delta":
      return ORB_STATES.LISTENING;

    case "user_speech_stopped":
    case "user_transcription_completed":
    case "response_created":
    case "tool_call_received":
    case "tool_result_ready":
      return ORB_STATES.THINKING;

    case "assistant_audio_delta":
      return ORB_STATES.SPEAKING;

    case "assistant_transcript_done":
      return ORB_STATES.IDLE;

    default:
      return null;
  }
}

// The backend never emits an explicit idle event; this timeout returns the orb to idle.
const IDLE_FALLBACK_MS = 1500;

/** Build the orb DOM. Returns { root, setState, dispose }. */
export function createOrb({ initialState = ORB_STATES.IDLE, onStateChange } = {}) {
  let currentState = initialState;
  let idleTimer = null;

  const indicator = h(
    "span",
    { class: "convo-orb__indicator", "aria-hidden": "true" },
    micIcon(),
    micOffIcon(),
    spinnerIndicator(),
    barsIndicator(),
    thinkingDotsIndicator(),
    voiceWaveIcon(),
    errorIcon()
  );

  const root = h(
    "button",
    {
      type: "button",
      class: "convo-orb",
      dataset: { state: currentState },
      "aria-label": "Conversation status",
      style: { "--glow": GLOW_BY_STATE[currentState] },
    },
    h("span", { class: "convo-orb__glow", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__ring", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__ring-outer", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__core" }, indicator)
  );

  /** Update the orb to reflect a new visual state. */
  function setState(nextState) {
    if (!Object.values(ORB_STATES).includes(nextState)) return;
    if (nextState === currentState) {
      bumpIdleTimer(nextState); // refresh timer on repeated events (e.g. continued audio deltas)
      return;
    }
    currentState = nextState;
    root.dataset.state = nextState;
    root.style.setProperty("--glow", GLOW_BY_STATE[nextState]);
    bumpIdleTimer(nextState);
    onStateChange?.(nextState);
  }

  function bumpIdleTimer(state) {
    if (idleTimer != null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    const transient =
      state === ORB_STATES.LISTENING ||
      state === ORB_STATES.THINKING ||
      state === ORB_STATES.SPEAKING;
    if (!transient) return;
    idleTimer = setTimeout(() => {
      idleTimer = null;
      setState(ORB_STATES.IDLE);
    }, IDLE_FALLBACK_MS);
  }

  /** Stop any pending timer. Call before detaching the DOM node. */
  function dispose() {
    if (idleTimer != null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  }

  return { root, setState, dispose };
}

// Indicators — stacked in the same grid cell, toggled via CSS data-state rules.

function barsIndicator() {
  return h(
    "span",
    { class: "ind ind-bars" },
    h("span", { class: "bar" }),
    h("span", { class: "bar" }),
    h("span", { class: "bar" }),
    h("span", { class: "bar" }),
    h("span", { class: "bar" })
  );
}

function thinkingDotsIndicator() {
  return h(
    "span",
    { class: "ind ind-thinking" },
    h("span", { class: "dot" }),
    h("span", { class: "dot" }),
    h("span", { class: "dot" })
  );
}

function spinnerIndicator() {
  return h("span", { class: "ind ind-spinner" });
}

function voiceWaveIcon() {
  return h("span", {
    class: "ind ind-voice",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M3 10v4a1 1 0 0 0 1 1h3l5 4V5L7 9H4a1 1 0 0 0-1 1z" fill="currentColor" stroke="none"/>
        <path class="wave wave-1" d="M16 8a5 5 0 0 1 0 8"/>
        <path class="wave wave-2" d="M19 5a9 9 0 0 1 0 14"/>
      </svg>`,
  });
}

function micIcon() {
  return h("span", {
    class: "ind ind-mic",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" stroke="none"/>
        <path d="M5 10a7 7 0 0 0 14 0"/>
        <line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="8" y1="22" x2="16" y2="22"/>
      </svg>`,
  });
}

function micOffIcon() {
  return h("span", {
    class: "ind ind-mic-off",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" stroke="none"/>
        <path d="M5 10a7 7 0 0 0 14 0"/>
        <line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="8" y1="22" x2="16" y2="22"/>
        <line x1="4" y1="3" x2="20" y2="21" stroke-width="2"/>
      </svg>`,
  });
}

function errorIcon() {
  return h("span", {
    class: "ind ind-error",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9"/>
        <line x1="12" y1="8" x2="12" y2="13"/>
        <line x1="12" y1="16" x2="12" y2="16"/>
      </svg>`,
  });
}
