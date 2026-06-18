/**
 * Talk view: conversation orb driven by the SSE activity stream.
 * Audio I/O runs entirely in Python; the orb doubles as the mic toggle.
 * Robot stays live, tapping the orb only mutes or unmutes the user's mic.
 */

import { getMicState, listPersonalities, setMicMuted } from "../api.js";
import { BUILT_IN_DEFAULT_OPTION, ORB_STATES } from "../constants.js";
import { createOrb, mapActivityToState } from "../orb.js";
import { consumePendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";
import { h, prettifyProfileName } from "../ui.js";

const SSE_ENDPOINT = "/conversation_events";

const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.MUTED]: "Muted",
  [ORB_STATES.IDLE]: "Ready",
  [ORB_STATES.CONNECTING]: "Connecting",
  [ORB_STATES.LISTENING]: "Listening",
  [ORB_STATES.THINKING]: "Thinking",
  [ORB_STATES.SPEAKING]: "Speaking",
  [ORB_STATES.ERROR]: "Reconnecting",
});

export async function mountTalkView({ outlet, signal }) {
  const pending = consumePendingApply();
  const micStatePromise = getMicState();
  let muted = true;
  let togglePending = false;

  const caption = h("p", { class: "talk__caption" }, CAPTION_BY_STATE[ORB_STATES.CONNECTING]);
  const orb = createOrb({
    initialState: ORB_STATES.CONNECTING,
    onStateChange: (state) => {
      caption.textContent = CAPTION_BY_STATE[state] || "";
    },
  });
  orb.root.addEventListener("click", onMicTap);
  syncMicAria();

  const view = h(
    "section",
    { class: "view view--talk" },
    h("div", { class: "talk__orb-wrap" }, orb.root),
    caption
  );
  outlet.replaceChildren(view);

  if (pending) {
    caption.textContent = `Applying "${prettifyProfileName(pending.name)}"…`;
    try {
      await pending.promise;
    } catch (error) {
      if (signal.aborted) return;
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = `Failed to apply personality: ${error?.message || error}`;
      return;
    }
    if (signal.aborted) return;
    // SSE "ready" will flip the orb to its resting state next tick.
    caption.textContent = CAPTION_BY_STATE[ORB_STATES.CONNECTING];
  } else {
    // Deep link to /talk with no pending apply: refresh the header badge.
    fetchActivePersonality().then((name) => {
      if (signal.aborted) return;
      if (name) setPersonality(name);
    });
  }

  try {
    muted = Boolean((await micStatePromise)?.muted);
  } catch {
    // keep the muted default
  }
  if (signal.aborted) return;
  syncMicAria();

  const subscription = subscribeConversationEvents({
    // Re-sync mic state on (re)connect: another tab may have toggled it.
    onReady: async () => {
      if (!togglePending) {
        try {
          muted = Boolean((await getMicState())?.muted);
        } catch {
          // keep the last known mute state
        }
      }
      if (signal.aborted) return;
      orb.setState(restingState());
      caption.textContent = CAPTION_BY_STATE[restingState()];
      syncMicAria();
    },
    onActivity: (reason) => {
      if (muted) return;
      const next = mapActivityToState(reason);
      if (next == null) return;
      orb.setState(next);
    },
    onError: () => {
      // The subscription keeps reconnecting, show the error meanwhile.
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = CAPTION_BY_STATE[ORB_STATES.ERROR];
    },
  });

  signal.addEventListener("abort", () => {
    subscription.close();
    orb.dispose();
  });

  function restingState() {
    return muted ? ORB_STATES.MUTED : ORB_STATES.IDLE;
  }

  async function onMicTap() {
    if (togglePending) return;
    togglePending = true;
    try {
      const data = await setMicMuted(!muted);
      muted = Boolean(data?.muted);
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to toggle the microphone: ${error?.message || error}`;
      }
      return;
    } finally {
      togglePending = false;
    }
    if (signal.aborted) return;
    orb.setState(restingState());
    // setState skips unchanged states, so set the caption explicitly
    caption.textContent = CAPTION_BY_STATE[restingState()];
    syncMicAria();
  }

  function syncMicAria() {
    orb.root.setAttribute("aria-pressed", String(!muted));
    orb.root.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
  }
}

async function fetchActivePersonality() {
  try {
    const data = await listPersonalities();
    const current = data?.current;
    if (!current || current === BUILT_IN_DEFAULT_OPTION) return "default";
    return current;
  } catch {
    return null;
  }
}

const SSE_RECONNECT_MS = 2000;

function subscribeConversationEvents({ onActivity, onReady, onError } = {}) {
  if (typeof onActivity !== "function") {
    throw new TypeError("subscribeConversationEvents: onActivity is required");
  }

  let source = null;
  let retryTimer = null;
  let closed = false;

  function connect() {
    source = new EventSource(SSE_ENDPOINT);

    source.addEventListener("activity", (ev) => {
      const reason = (ev.data || "").trim();
      if (reason) onActivity(reason);
    });

    source.addEventListener("ready", () => onReady());

    source.addEventListener("error", (err) => {
      onError(err);
      // EventSource gives up on HTTP errors (e.g. 404 while the backend is
      // still registering routes); recreate it until the route exists.
      if (!closed && source.readyState === EventSource.CLOSED) {
        retryTimer = setTimeout(connect, SSE_RECONNECT_MS);
      }
    });
  }

  connect();

  return {
    close() {
      closed = true;
      if (retryTimer != null) clearTimeout(retryTimer);
      source.close();
    },
  };
}
