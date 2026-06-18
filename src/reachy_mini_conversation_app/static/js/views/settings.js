/** Settings view: backend selector, Hugging Face connection, voice, and status. */

import {
  applyVoice,
  describeError,
  getCurrentVoice,
  getStatus,
  listVoices,
  saveBackendConfig,
  untilReady,
} from "../api.js";
import { BACKENDS } from "../constants.js";
import { h } from "../ui.js";

const HF_CONNECTION_MODES = Object.freeze({
  DEPLOYED: "deployed",
  LOCAL: "local",
});

const DEFAULT_HF_HOST = "localhost";
const DEFAULT_HF_PORT = 8765;

const BACKEND_LABELS = Object.freeze({
  [BACKENDS.HUGGINGFACE]: "Hugging Face",
  [BACKENDS.OPENAI]: "OpenAI Realtime",
  [BACKENDS.GEMINI]: "Gemini Live",
});

const BACKEND_HINTS = Object.freeze({
  [BACKENDS.HUGGINGFACE]: "Choose the hosted service or a local realtime backend.",
  [BACKENDS.OPENAI]: "Bring your own OPENAI_API_KEY.",
  [BACKENDS.GEMINI]: "Bring your own GEMINI_API_KEY.",
});

const HF_MODE_HINTS = Object.freeze({
  [HF_CONNECTION_MODES.DEPLOYED]: "Uses the hosted Hugging Face backend. No API key required.",
  [HF_CONNECTION_MODES.LOCAL]: "Connects directly to the host and port below.",
});

export async function mountSettingsView({ outlet, signal }) {
  // Backends expose different voice lists, so a save re-syncs voices and status.
  const backendSection = buildBackendSection({
    onSaved: () =>
      Promise.all([
        refreshStatus({ statusSection, backendSection, signal }),
        refreshVoices({ voiceSection, signal }),
      ]),
  });
  const voiceSection = buildVoiceSection();
  const statusSection = buildStatusSection();

  const view = h(
    "section",
    { class: "view view--settings" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Settings"),
      h("p", { class: "view-subtitle" }, "Backend, credentials and voice for Reachy Mini.")
    ),
    backendSection.element,
    voiceSection.element,
    statusSection.element
  );
  outlet.replaceChildren(view);

  await Promise.all([
    refreshStatus({ statusSection, backendSection, signal }),
    refreshVoices({ voiceSection, signal }),
  ]);
}

function buildBackendSection({ onSaved } = {}) {
  const backendSelect = h(
    "select",
    { class: "settings-select", name: "backend" },
    ...Object.entries(BACKEND_LABELS).map(([value, label]) =>
      h("option", { value }, label)
    )
  );
  const apiKeyInput = h("input", {
    type: "password",
    name: "api_key",
    autocomplete: "off",
    placeholder: "sk-… or AIza…",
    class: "settings-input",
  });
  const apiKeyField = h(
    "label",
    { class: "settings-field", "data-role": "api-key-field" },
    h("span", { class: "settings-label" }, "API key"),
    apiKeyInput
  );
  const hfModeSelect = h(
    "select",
    { class: "settings-select", name: "hf_mode" },
    h("option", { value: HF_CONNECTION_MODES.DEPLOYED }, "Hosted"),
    h("option", { value: HF_CONNECTION_MODES.LOCAL }, "Local")
  );
  const hfHostInput = h("input", {
    type: "text",
    name: "hf_host",
    autocomplete: "off",
    placeholder: DEFAULT_HF_HOST,
    value: DEFAULT_HF_HOST,
    class: "settings-input",
  });
  const hfPortInput = h("input", {
    type: "number",
    name: "hf_port",
    min: "1",
    max: "65535",
    step: "1",
    inputmode: "numeric",
    value: String(DEFAULT_HF_PORT),
    class: "settings-input",
  });
  const hfModeField = h(
    "label",
    { class: "settings-field", "data-role": "hf-mode-field" },
    h("span", { class: "settings-label" }, "Hugging Face connection"),
    hfModeSelect
  );
  const hfLocalFields = h(
    "div",
    { class: "settings-field-row", "data-role": "hf-local-fields" },
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Host/IP"),
      hfHostInput
    ),
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Port"),
      hfPortInput
    )
  );
  const hint = h("p", { class: "settings-hint" }, "");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });

  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Backend"), backendSelect),
    apiKeyField,
    hfModeField,
    hfLocalFields,
    hint,
    h(
      "div",
      { class: "settings-actions" },
      h("button", { type: "submit", class: "btn btn--primary" }, "Save backend")
    ),
    status
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Backend"),
    form
  );

  function syncApiKeyVisibility() {
    const requiresKey = backendSelect.value !== BACKENDS.HUGGINGFACE;
    const isHuggingFace = backendSelect.value === BACKENDS.HUGGINGFACE;
    const isLocalHuggingFace = isHuggingFace && hfModeSelect.value === HF_CONNECTION_MODES.LOCAL;

    apiKeyField.style.display = requiresKey ? "" : "none";
    if (!requiresKey) apiKeyInput.value = "";
    apiKeyInput.disabled = !requiresKey;

    hfModeField.style.display = isHuggingFace ? "" : "none";
    hfLocalFields.style.display = isLocalHuggingFace ? "" : "none";
    hfModeSelect.disabled = !isHuggingFace;
    hfHostInput.disabled = !isLocalHuggingFace;
    hfPortInput.disabled = !isLocalHuggingFace;
    hfHostInput.required = isLocalHuggingFace;
    hfPortInput.required = isLocalHuggingFace;

    hint.textContent =
      (isHuggingFace && HF_MODE_HINTS[hfModeSelect.value]) ||
      BACKEND_HINTS[backendSelect.value] ||
      "";
  }

  backendSelect.addEventListener("change", syncApiKeyVisibility);
  hfModeSelect.addEventListener("change", syncApiKeyVisibility);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.classList.remove("is-error");
    status.textContent = "Saving…";
    try {
      const payload = {
        backend: backendSelect.value,
      };
      if (backendSelect.value !== BACKENDS.HUGGINGFACE && apiKeyInput.value) {
        payload.api_key = apiKeyInput.value;
      }
      if (backendSelect.value === BACKENDS.HUGGINGFACE) {
        payload.hf_mode = hfModeSelect.value;
        if (hfModeSelect.value === HF_CONNECTION_MODES.LOCAL) {
          payload.hf_host = hfHostInput.value.trim();
          if (hfPortInput.value) {
            payload.hf_port = Number.parseInt(hfPortInput.value, 10);
          }
        }
      }
      const result = await saveBackendConfig(payload);
      status.textContent =
        result?.message || (result?.requires_restart ? "Saved. Restart the app to apply." : "Saved.");
      await onSaved?.();
    } catch (error) {
      status.textContent = `Failed to save: ${describeError(error)}`;
      status.classList.add("is-error");
    }
  });

  syncApiKeyVisibility();

  return {
    element,
    syncFromStatus(payload) {
      const backend = payload?.backend_provider;
      if (backend && BACKEND_LABELS[backend]) {
        backendSelect.value = backend;
      }
      if (Object.values(HF_CONNECTION_MODES).includes(payload?.hf_connection_mode)) {
        hfModeSelect.value = payload.hf_connection_mode;
      }
      if (payload?.hf_direct_host) {
        hfHostInput.value = payload.hf_direct_host;
      }
      if (payload?.hf_direct_port != null) {
        hfPortInput.value = String(payload.hf_direct_port);
      }
      syncApiKeyVisibility();
    },
  };
}

function buildVoiceSection() {
  const select = h("select", { class: "settings-select", name: "voice" });
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Voice"), select),
    h(
      "div",
      { class: "settings-actions" },
      h("button", { type: "submit", class: "btn btn--primary" }, "Apply voice")
    ),
    status
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Voice"),
    form
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.classList.remove("is-error");
    if (!select.value) return;
    status.textContent = "Applying…";
    try {
      const result = await applyVoice(select.value);
      status.textContent = result?.status || "Voice applied.";
    } catch (error) {
      status.textContent = `Failed to apply: ${describeError(error)}`;
      status.classList.add("is-error");
    }
  });

  return {
    element,
    setOptions(voices, current) {
      select.replaceChildren();
      for (const v of voices) {
        const opt = h("option", { value: v }, v);
        if (v === current) opt.selected = true;
        select.appendChild(opt);
      }
    },
  };
}

function buildStatusSection() {
  const list = h("dl", { class: "settings-status-grid" });
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Current state"),
    list
  );

  return {
    element,
    render(payload) {
      list.replaceChildren();
      list.appendChild(statusRow("Active backend", payload.active_backend || "-"));
      list.appendChild(statusRow("Selected backend", payload.backend_provider || "-"));
      if (payload.backend_provider === BACKENDS.HUGGINGFACE || payload.active_backend === BACKENDS.HUGGINGFACE) {
        list.appendChild(statusRow("HF connection", formatHfMode(payload.hf_connection_mode)));
        if (payload.hf_connection_mode === HF_CONNECTION_MODES.LOCAL) {
          list.appendChild(statusRow("HF target", formatHfTarget(payload)));
        }
      }
      list.appendChild(
        statusRow("Credentials", payload.has_key ? "Ready" : "Missing", payload.has_key ? "ok" : "warn")
      );
      if (payload.requires_restart) {
        list.appendChild(
          statusRow(
            "Restart",
            "Required to apply selected backend",
            "warn"
          )
        );
      }
    },
  };
}

function statusRow(label, value, tone) {
  return h(
    "div",
    { class: ["settings-status-row", tone && `is-${tone}`] },
    h("dt", { class: "settings-status-label" }, label),
    h("dd", { class: "settings-status-value" }, value)
  );
}

function formatHfMode(mode) {
  if (mode === HF_CONNECTION_MODES.LOCAL) return "Local";
  if (mode === HF_CONNECTION_MODES.DEPLOYED) return "Hosted";
  return "-";
}

function formatHfTarget(payload) {
  const host = payload?.hf_direct_host;
  const port = payload?.hf_direct_port;
  if (!host) return "-";
  return `${host}:${port || DEFAULT_HF_PORT}`;
}

async function refreshStatus({ statusSection, backendSection, signal }) {
  try {
    const payload = await untilReady(getStatus, signal);
    if (signal.aborted) return;
    statusSection.render(payload);
    backendSection.syncFromStatus(payload);
  } catch {
    // Status panel just stays empty; not critical for the rest of the UI.
  }
}

async function refreshVoices({ voiceSection, signal }) {
  let voices = [];
  let current = "";
  try {
    voices = await untilReady(listVoices, signal);
  } catch {
    voices = [];
  }
  try {
    const data = await getCurrentVoice();
    current = data?.voice || "";
  } catch {
    current = "";
  }
  if (signal.aborted) return;
  voiceSection.setOptions(voices, current);
}
