/** Talk view sidebar: live transcript plus the backend's per-response timings.
 * Opened from the header's conversation button; starts from the app's history
 * of this run, so a page opened mid-conversation still shows earlier turns.
 * Timings are always rendered but stay hidden unless Settings turns them on. */

import { getConversationHistory, getShowLatencies, subscribe } from "../api.js";
import { h } from "../ui.js";

const LATENCY_STAGES = Object.freeze([
  ["vad_decision_ms", "VAD end decision"],
  ["smart_status", "Smart Turn decision"],
  ["stt_ms", "Transcription"],
  ["llm_ms", "Response generation"],
  ["tts_ttfa_ms", "Voice synthesis to first audio"],
  ["hold_ms", "Hold time before response"],
]);

const STATUS_LABELS = Object.freeze({
  cancelled: "Interrupted",
  failed: "Failed",
  incomplete: "Incomplete",
});

// Within this distance of the bottom, new rows keep the list scrolled down.
const STICK_TO_BOTTOM_PX = 24;

const CLOSE_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>';

/** Build the panel, driven by the header `toggle` button. Returns { root, dispose }. */
export function createConversationPanel({ toggle }) {
  let partialRow = null;
  let lastSeq = 0;
  // Live events that arrive before the history snapshot; null once it is rendered.
  let pending = [];

  const list = h("ol", {
    class: "talk-log__list",
    role: "log",
    "data-empty": "The conversation will show up here.",
  });
  const root = h(
    "aside",
    { id: "talk-log", class: "talk-log", "aria-label": "Conversation" },
    h(
      "div",
      { class: "talk-log__header" },
      h("h2", { class: "talk-log__title" }, "Conversation"),
      h("button", {
        type: "button",
        class: "app-shell__icon-btn",
        "aria-label": "Hide conversation",
        html: CLOSE_ICON,
        onClick: () => setOpen(false),
      })
    ),
    list
  );
  const onToggle = () => setOpen(root.hidden);
  toggle.addEventListener("click", onToggle);
  toggle.hidden = false;
  setOpen(false);

  const renderers = {
    "conversation.transcript": onTranscript,
    "conversation.latency": onLatency,
  };
  const unsubscribers = Object.entries(renderers).map(([method, render]) =>
    subscribe(method, (params) => (pending ? pending.push([render, params]) : apply(render, params)))
  );
  getShowLatencies()
    .then(({ enabled }) => root.classList.toggle("shows-latencies", enabled))
    .catch((error) => console.warn("Failed to load the latency display setting", error));
  getConversationHistory()
    .then(({ events }) => {
      for (const { method, params } of events) apply(renderers[method], params);
    })
    .catch((error) => console.warn("Failed to load the conversation history", error))
    .finally(() => {
      const buffered = pending;
      pending = null;
      // Buffered partials may predate a final transcript the snapshot already holds.
      for (const [render, params] of buffered) if (params.seq != null) apply(render, params);
    });

  function apply(render, params) {
    if (params.seq != null) {
      if (params.seq <= lastSeq) return;
      lastSeq = params.seq;
    }
    render(params);
  }

  function setOpen(open) {
    root.hidden = !open;
    toggle.classList.toggle("is-active", open);
    toggle.classList.remove("has-unread");
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Hide conversation" : "Show conversation");
    if (open) list.scrollTop = list.scrollHeight;
  }

  function keepScrolledDown(mutate, { visible = true } = {}) {
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight <= STICK_TO_BOTTOM_PX;
    mutate();
    if (root.hidden) {
      if (visible) toggle.classList.add("has-unread");
    } else if (atBottom) list.scrollTop = list.scrollHeight;
  }

  function onTranscript({ role, text, final }) {
    if (!text || (role !== "user" && role !== "assistant")) return;
    keepScrolledDown(() => {
      if (role === "user" && partialRow) {
        partialRow.textContent = text;
        partialRow.classList.toggle("is-partial", !final);
        if (final) partialRow = null;
        return;
      }
      // A user partial still open when the robot answers never got a final transcript.
      if (role === "assistant") partialRow = null;
      const row = h("li", { class: ["talk-log__row", `talk-log__row--${role}`, !final && "is-partial"] }, text);
      if (role === "user" && !final) partialRow = row;
      list.append(row);
    });
  }

  function onLatency(latency) {
    const status = STATUS_LABELS[latency.status];
    const summary = `End-to-end ${formatMs(latency.e2e_ms)}${status ? ` · ${status}` : ""}`;
    keepScrolledDown(() => {
      list.append(
        h(
          "li",
          { class: "talk-log__row talk-log__row--timings" },
          h(
            "details",
            { class: "talk-log__timings" },
            h("summary", {}, summary),
            h(
              "dl",
              {},
              LATENCY_STAGES.map(([key, label]) => [
                h("dt", {}, label),
                h("dd", {}, typeof latency[key] === "string" ? latency[key] : formatMs(latency[key])),
              ])
            )
          )
        )
      );
    }, { visible: root.classList.contains("shows-latencies") });
  }

  return {
    root,
    dispose() {
      for (const unsubscribe of unsubscribers) unsubscribe();
      toggle.removeEventListener("click", onToggle);
      toggle.classList.remove("is-active", "has-unread");
      toggle.hidden = true;
    },
  };
}

function formatMs(ms) {
  return typeof ms === "number" ? `${ms} ms` : "n/a";
}
