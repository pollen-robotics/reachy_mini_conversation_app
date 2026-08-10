/** Settings view: Hugging Face connection, voice, and runtime status. */

import {
  applyVoice,
  describeError,
  getCompanionConfig,
  getCompanionNamespaces,
  getCurrentVoice,
  getStatus,
  listVoices,
  saveBackendConfig,
  saveCompanionConfig,
  startCompanionSetup,
  untilReady,
} from "../api.js";
import { h } from "../ui.js";
import { ACTIVE_COMPANION_SETUP_STATES } from "../constants.js";
import { buildCompanionOwnership } from "../components/companion-owner.js";

const HF_CONNECTION_MODES = Object.freeze({
  DEPLOYED: "deployed",
  LOCAL: "local",
});

const DEFAULT_HF_HOST = "localhost";
const DEFAULT_HF_PORT = 8765;

const HF_MODE_HINTS = Object.freeze({
  [HF_CONNECTION_MODES.DEPLOYED]: "Uses the hosted Hugging Face backend. No API key required.",
  [HF_CONNECTION_MODES.LOCAL]: "Connects directly to the host and port below.",
});

export async function mountSettingsView({ outlet, signal }) {
  const connectionSection = buildConnectionSection({
    onSaved: () =>
      Promise.all([
        refreshStatus({ statusSection, connectionSection, signal }),
        refreshVoices({ voiceSection, signal }),
      ]),
  });
  const voiceSection = buildVoiceSection();
  const assistantSection = buildAssistantSection({ signal });
  const statusSection = buildStatusSection();

  const view = h(
    "section",
    { class: "view view--settings" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Settings"),
      h(
        "p",
        { class: "view-subtitle" },
        "Connection, background assistant, voice, and runtime state for Reachy Mini."
      )
    ),
    connectionSection.element,
    assistantSection.element,
    voiceSection.element,
    statusSection.element
  );
  outlet.replaceChildren(view);

  await Promise.all([
    refreshStatus({ statusSection, connectionSection, signal }),
    assistantSection.load(),
    refreshVoices({ voiceSection, signal }),
  ]);
}

function buildAssistantSection({ signal }) {
  const setupDescription = h(
    "p",
    { id: "companion-setup-hint", class: "settings-hint", hidden: "hidden" },
    "Private Docker Spaces require PRO for personal accounts or Team/Enterprise for organizations. Organization access rules apply."
  );
  const namespaceSelect = h(
    "select",
    {
      class: "settings-select",
      name: "companion_namespace",
      "aria-describedby": "companion-setup-hint companion-setup-status",
      disabled: "disabled",
    },
    h("option", { value: "" }, "Loading accounts and organizations…")
  );
  const namespaceField = h(
    "label",
    { class: "settings-field", hidden: "hidden" },
    h("span", { class: "settings-label" }, "Hugging Face namespace"),
    namespaceSelect
  );
  const setupButton = h(
    "button",
    { type: "submit", class: "btn btn--primary", disabled: "disabled" },
    "Set up assistant"
  );
  const setupActions = h(
    "div",
    { class: "settings-actions", hidden: "hidden" },
    setupButton
  );
  const checkbox = h("input", {
    type: "checkbox",
    name: "companion_enabled",
    role: "switch",
    "aria-describedby": "companion-setup-hint companion-setup-status",
    disabled: "disabled",
  });
  const choice = h(
    "label",
    { class: "settings-tool-choice is-disabled", hidden: "hidden" },
    checkbox,
    h(
      "span",
      { class: "settings-tool-choice-copy" },
      h("strong", { class: "settings-tool-choice-name" }, "Use background assistant"),
      h(
        "span",
        { class: "settings-tool-choice-description" },
        "Allow Reachy to delegate longer tasks. Applies to every personality."
      )
    )
  );
  const progress = h("progress", {
    class: "settings-assistant-progress",
    hidden: "hidden",
    "aria-hidden": "true",
  });
  const status = h(
    "p",
    {
      id: "companion-setup-status",
      class: "settings-status",
      role: "status",
      "aria-live": "polite",
    },
    "Loading assistant settings…"
  );
  const ownership = buildCompanionOwnership();
  const controls = h(
    "form",
    { class: "settings-form" },
    choice,
    namespaceField,
    setupDescription,
    setupActions,
    progress,
    status,
    ownership.element
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Background assistant"),
    controls
  );
  let configured = false;
  let savedEnabled = false;
  let setupState = "idle";
  let namespacesLoaded = false;
  let changing = false;
  let polling = false;

  function syncControl() {
    const setupBusy = ACTIVE_COMPANION_SETUP_STATES.includes(setupState);
    const unavailable = changing || setupBusy || setupState === "restart_required";
    checkbox.disabled = !configured || unavailable;
    choice.classList.toggle("is-disabled", !configured || unavailable);
    namespaceSelect.disabled = configured || unavailable || !namespacesLoaded;
    setupButton.disabled = configured || unavailable || !namespacesLoaded || !namespaceSelect.value;
    controls.setAttribute("aria-busy", changing || setupBusy ? "true" : "false");
  }

  function render(payload, message = "") {
    configured = payload?.configured === true;
    savedEnabled = configured && payload?.enabled === true;
    setupState = payload?.setup?.state || "idle";
    const setupBusy = ACTIVE_COMPANION_SETUP_STATES.includes(setupState);
    checkbox.checked = savedEnabled;
    choice.hidden = !configured;
    const showSetupDescription = !configured && setupState !== "restart_required";
    namespaceField.hidden = !showSetupDescription;
    setupActions.hidden = !showSetupDescription;
    setupDescription.hidden = !showSetupDescription;
    checkbox.setAttribute(
      "aria-describedby",
      showSetupDescription
        ? "companion-setup-hint companion-setup-status"
        : "companion-setup-status"
    );
    progress.hidden = !setupBusy;
    ownership.render(payload?.setup);
    status.classList.toggle("is-error", setupState === "failed");
    status.textContent = configured
      ? message || (savedEnabled ? "Assistant ready for every personality." : "Turned off.")
      : payload?.setup?.message ||
        "Set up a private background assistant to delegate longer tasks.";
    syncControl();
  }

  function setNamespaces(namespaces) {
    const selected = namespaceSelect.value;
    const validNamespaces = Array.isArray(namespaces)
      ? namespaces.filter(
          (namespace) =>
            typeof namespace?.name === "string" &&
            ["personal", "organization"].includes(namespace?.kind)
        )
      : [];
    namespaceSelect.replaceChildren();
    if (!validNamespaces.length) {
      namespaceSelect.appendChild(h("option", { value: "" }, "No writable namespaces available"));
      namespacesLoaded = false;
      syncControl();
      return;
    }
    if (validNamespaces.length > 1) {
      namespaceSelect.appendChild(
        h("option", { value: "" }, "Choose an account or organization")
      );
    }
    for (const namespace of validNamespaces) {
      namespaceSelect.appendChild(
        h(
          "option",
          { value: namespace.name },
          `@${namespace.name} (${namespace.kind === "personal" ? "personal" : "organization"})`
        )
      );
    }
    if (validNamespaces.some((namespace) => namespace.name === selected)) {
      namespaceSelect.value = selected;
    }
    namespacesLoaded = true;
    syncControl();
  }

  async function pollSetup(payload) {
    if (polling || !ACTIVE_COMPANION_SETUP_STATES.includes(payload?.setup?.state)) return;
    polling = true;
    try {
      let current = payload;
      while (ACTIVE_COMPANION_SETUP_STATES.includes(current?.setup?.state) && !signal.aborted) {
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
        if (signal.aborted) return;
        current = await getCompanionConfig();
        render(current);
      }
    } finally {
      polling = false;
    }
  }

  namespaceSelect.addEventListener("change", syncControl);

  controls.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (changing || setupButton.disabled) return;
    changing = true;
    syncControl();
    status.classList.remove("is-error");
    status.textContent = `Checking @${namespaceSelect.value}…`;
    try {
      const payload = await startCompanionSetup(namespaceSelect.value);
      if (signal.aborted) return;
      render(payload);
      await pollSetup(payload);
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      if (!signal.aborted) {
        changing = false;
        syncControl();
      }
    }
  });

  checkbox.addEventListener("change", async () => {
    if (changing) return;
    const enabled = checkbox.checked;
    changing = true;
    syncControl();
    status.classList.remove("is-error");
    status.textContent = enabled ? "Turning on…" : "Turning off…";
    try {
      const payload = await saveCompanionConfig(enabled);
      if (signal.aborted) return;
      render(payload, payload?.message || "Background-assistant setting saved.");
      await pollSetup(payload);
    } catch (error) {
      if (signal.aborted) return;
      checkbox.checked = configured ? savedEnabled : false;
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      if (!signal.aborted) {
        changing = false;
        syncControl();
      }
    }
  });

  return {
    element,
    async load() {
      try {
        const payload = await untilReady(getCompanionConfig, signal);
        if (!signal.aborted) {
          render(payload);
          if (!configured && setupState !== "restart_required") {
            const namespacePayload = await getCompanionNamespaces();
            if (signal.aborted) return;
            setNamespaces(namespacePayload?.namespaces);
          }
          await pollSetup(payload);
        }
      } catch (error) {
        if (signal.aborted) return;
        configured = false;
        changing = false;
        checkbox.checked = false;
        choice.hidden = true;
        namespaceField.hidden = false;
        setupActions.hidden = false;
        setupDescription.hidden = false;
        status.textContent = `Could not load: ${describeError(error)}`;
        status.classList.add("is-error");
        syncControl();
      }
    },
  };
}

function buildConnectionSection({ onSaved } = {}) {
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
  const submitButton = h("button", { type: "submit", class: "btn btn--primary" }, "Save connection");

  const form = h(
    "form",
    { class: "settings-form" },
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Hugging Face connection"),
      hfModeSelect
    ),
    hfLocalFields,
    hint,
    h("div", { class: "settings-actions" }, submitButton),
    status
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Connection"),
    form
  );

  function syncLocalFields() {
    const isLocal = hfModeSelect.value === HF_CONNECTION_MODES.LOCAL;
    hfLocalFields.style.display = isLocal ? "" : "none";
    hfHostInput.disabled = !isLocal;
    hfPortInput.disabled = !isLocal;
    hfHostInput.required = isLocal;
    hfPortInput.required = isLocal;
    hint.textContent = HF_MODE_HINTS[hfModeSelect.value] || "";
  }

  hfModeSelect.addEventListener("change", syncLocalFields);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled) return;
    submitButton.disabled = true;
    hfModeSelect.disabled = true;
    hfHostInput.disabled = true;
    hfPortInput.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Saving…";
    try {
      const payload = { hf_mode: hfModeSelect.value };
      if (hfModeSelect.value === HF_CONNECTION_MODES.LOCAL) {
        payload.hf_host = hfHostInput.value.trim();
        if (hfPortInput.value) {
          payload.hf_port = Number.parseInt(hfPortInput.value, 10);
        }
      }
      const result = await saveBackendConfig(payload);
      status.textContent =
        result?.message || (result?.requires_restart ? "Saved. Restart the app to apply." : "Saved.");
      await onSaved?.();
    } catch (error) {
      status.textContent = `Failed to save: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = false;
      hfModeSelect.disabled = false;
      syncLocalFields();
      form.removeAttribute("aria-busy");
    }
  });

  syncLocalFields();

  return {
    element,
    syncFromStatus(payload) {
      if (Object.values(HF_CONNECTION_MODES).includes(payload?.hf_connection_mode)) {
        hfModeSelect.value = payload.hf_connection_mode;
      }
      if (payload?.hf_direct_host) {
        hfHostInput.value = payload.hf_direct_host;
      }
      if (payload?.hf_direct_port != null) {
        hfPortInput.value = String(payload.hf_direct_port);
      }
      syncLocalFields();
    },
  };
}

function buildVoiceSection() {
  const select = h(
    "select",
    { class: "settings-select", name: "voice", disabled: "disabled" },
    h("option", { value: "" }, "Loading voices…")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h(
    "button",
    { type: "submit", class: "btn btn--primary", disabled: "disabled" },
    "Apply voice"
  );
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Voice"), select),
    h("div", { class: "settings-actions" }, submitButton),
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
    if (submitButton.disabled || !select.value) return;
    submitButton.disabled = true;
    select.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Applying…";
    try {
      const result = await applyVoice(select.value);
      status.textContent = result?.status || "Voice applied.";
    } catch (error) {
      status.textContent = `Failed to apply: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = !select.value;
      select.disabled = !select.value;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    setOptions(voices, current) {
      select.replaceChildren();
      if (!voices.length) {
        select.appendChild(h("option", { value: "" }, "No voices available"));
        select.disabled = true;
        submitButton.disabled = true;
        status.textContent = "Voices are unavailable right now.";
        return;
      }
      for (const v of voices) {
        const opt = h("option", { value: v }, v);
        if (v === current) opt.selected = true;
        select.appendChild(opt);
      }
      select.disabled = false;
      submitButton.disabled = false;
      status.textContent = "";
    },
  };
}

function buildStatusSection() {
  const list = h(
    "dl",
    { class: "settings-status-grid" },
    statusRow("Backend", "Loading…")
  );
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
      list.appendChild(statusRow("HF connection", formatHfMode(payload.hf_connection_mode)));
      if (payload.hf_connection_mode === HF_CONNECTION_MODES.LOCAL) {
        list.appendChild(statusRow("HF target", formatHfTarget(payload)));
      }
      list.appendChild(
        statusRow(
          "Configuration",
          payload.has_hf_connection ? "Ready" : "Missing",
          payload.has_hf_connection ? "ok" : "warn"
        )
      );
      const backendState = payload.backend_connected
        ? "connected"
        : payload.backend_connection_state || "not_started";
      const backendLabels = {
        connected: "Connected",
        connecting: "Connecting…",
        disconnected: "Disconnected",
        not_started: "Not started",
        restart_required: "Restart required",
        waiting_for_config: "Waiting for configuration",
      };
      list.appendChild(
        statusRow(
          "Backend",
          backendLabels[backendState] || "Unavailable",
          backendState === "connected" ? "ok" : backendState === "not_started" ? undefined : "warn"
        )
      );
      if (payload.backend_error) {
        list.appendChild(statusRow("Backend error", payload.backend_error, "warn"));
      }
      if (payload.requires_restart) {
        list.appendChild(statusRow("Restart", "Required to apply changes", "warn"));
      }
    },
    renderUnavailable(error) {
      list.replaceChildren(statusRow("Backend", `Unavailable: ${describeError(error)}`, "warn"));
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

async function refreshStatus({ statusSection, connectionSection, signal }) {
  try {
    const payload = await untilReady(getStatus, signal);
    if (signal.aborted) return;
    statusSection.render(payload);
    connectionSection.syncFromStatus(payload);
  } catch (error) {
    if (signal.aborted) return;
    statusSection.renderUnavailable(error);
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
  if (signal.aborted) return;
  try {
    const data = await getCurrentVoice();
    current = data?.voice || "";
  } catch {
    current = "";
  }
  if (signal.aborted) return;
  voiceSection.setOptions(voices, current);
}
