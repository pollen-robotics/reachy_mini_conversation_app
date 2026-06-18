/** Settings view: backend selector, voice, and status. Advanced HF options stay in .env. */

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

const BACKEND_LABELS = Object.freeze({
  [BACKENDS.HUGGINGFACE]: "Hugging Face (built-in)",
  [BACKENDS.OPENAI]: "OpenAI Realtime",
  [BACKENDS.GEMINI]: "Gemini Live",
});

const BACKEND_HINTS = Object.freeze({
  [BACKENDS.HUGGINGFACE]: "Uses the bundled Hugging Face server. No API key required.",
  [BACKENDS.OPENAI]: "Bring your own OPENAI_API_KEY.",
  [BACKENDS.GEMINI]: "Bring your own GEMINI_API_KEY.",
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
  const hint = h("p", { class: "settings-hint" }, "");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });

  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Backend"), backendSelect),
    apiKeyField,
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
    apiKeyField.style.display = requiresKey ? "" : "none";
    if (!requiresKey) apiKeyInput.value = "";
    hint.textContent = BACKEND_HINTS[backendSelect.value] || "";
  }

  backendSelect.addEventListener("change", syncApiKeyVisibility);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.classList.remove("is-error");
    status.textContent = "Saving…";
    try {
      const payload = {
        backend: backendSelect.value,
        api_key: apiKeyInput.value || undefined,
      };
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
    setActiveBackend(backend) {
      if (backend && BACKEND_LABELS[backend]) {
        backendSelect.value = backend;
        syncApiKeyVisibility();
      }
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

async function refreshStatus({ statusSection, backendSection, signal }) {
  try {
    const payload = await untilReady(getStatus, signal);
    if (signal.aborted) return;
    statusSection.render(payload);
    backendSection.setActiveBackend(payload.backend_provider);
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
