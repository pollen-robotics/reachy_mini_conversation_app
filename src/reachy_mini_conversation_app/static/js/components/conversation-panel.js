/** Talk view sidebar: live transcript plus the backend's per-response timings.
 * Rows live only in this panel's DOM, so every mount starts empty. */

import { subscribe } from "../api.js";
import { h } from "../ui.js";

const LATENCY_STAGES = Object.freeze([
  ["stt_ms", "Transcription"],
  ["llm_ms", "Response generation"],
  ["tts_ttfa_ms", "Voice synthesis to first audio"],
]);

const STATUS_LABELS = Object.freeze({
  cancelled: "Interrupted",
  failed: "Failed",
  incomplete: "Incomplete",
});

// Within this distance of the bottom, new rows keep the list scrolled down.
const STICK_TO_BOTTOM_PX = 24;

const SVG_ATTRS =
  'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
const MESSAGE_ICON = `<svg ${SVG_ATTRS}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
const CLOSE_ICON = `<svg ${SVG_ATTRS}><path d="M18 6 6 18M6 6l12 12"/></svg>`;

/** Build the panel. Returns { root, dispose }. */
export function createConversationPanel() {
  let partialRow = null;

  const list = h("ol", {
    id: "talk-log-list",
    class: "talk-log__list",
    role: "log",
    "data-empty": "The conversation will show up here.",
  });
  const toggle = h("button", {
    type: "button",
    class: "app-shell__icon-btn talk-log__toggle",
    "aria-controls": "talk-log-list",
  });
  const title = h("h2", { class: "talk-log__title" }, "Conversation");
  const root = h(
    "aside",
    { class: "talk-log", "aria-label": "Conversation" },
    h("div", { class: "talk-log__header" }, title, toggle),
    list
  );
  setOpen(false);
  toggle.addEventListener("click", () => setOpen(list.hidden));

  const unsubscribers = [
    subscribe("conversation.transcript", onTranscript),
    subscribe("conversation.latency", onLatency),
  ];

  function setOpen(open) {
    list.hidden = !open;
    title.hidden = !open;
    root.classList.toggle("is-open", open);
    toggle.classList.remove("has-unread");
    toggle.innerHTML = open ? CLOSE_ICON : MESSAGE_ICON;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Hide conversation" : "Show conversation");
    toggle.title = open ? "Hide conversation" : "Conversation";
    if (open) list.scrollTop = list.scrollHeight;
  }

  function keepScrolledDown(mutate) {
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight <= STICK_TO_BOTTOM_PX;
    mutate();
    if (list.hidden) toggle.classList.add("has-unread");
    else if (atBottom) list.scrollTop = list.scrollHeight;
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
              LATENCY_STAGES.map(([key, label]) => [h("dt", {}, label), h("dd", {}, formatMs(latency[key]))])
            ),
            h("p", { class: "talk-log__timings-note" }, "Measured by the backend. Robot playback is not included.")
          )
        )
      );
    });
  }

  return {
    root,
    dispose() {
      for (const unsubscribe of unsubscribers) unsubscribe();
    },
  };
}

function formatMs(ms) {
  return typeof ms === "number" ? `${ms} ms` : "n/a";
}
