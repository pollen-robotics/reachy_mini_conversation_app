const OPENAI_BACKEND = "openai";
const GEMINI_BACKEND = "gemini";
const HF_BACKEND = "huggingface";
const DEFAULT_BACKEND = HF_BACKEND;
const HF_DEFAULT_HOST = "localhost";
const HF_DEFAULT_PORT = 8765;
const BACKEND_META = {
  [OPENAI_BACKEND]: {
    label: "OpenAI Realtime",
    formTitle: "Connect OpenAI",
    inputLabel: "OpenAI API Key",
    placeholder: "sk-...",
    saveButton: "Save key",
    changeButton: "Change OpenAI key",
    readyTitle: "OpenAI Realtime ready",
    readyCopy: "OpenAI Realtime is configured. Your saved OpenAI key is ready to use.",
    formCopy: "Paste your OPENAI_API_KEY once and we will store it locally for the headless conversation loop.",
    requiredCredentialsCopy: "OpenAI Realtime requires your own OPENAI_API_KEY before you can switch.",
    note: "OpenAI Realtime requires your own OPENAI_API_KEY.",
  },
  [GEMINI_BACKEND]: {
    label: "Gemini Live",
    formTitle: "Connect Gemini Live",
    inputLabel: "GEMINI_API_KEY",
    placeholder: "AIza...",
    saveButton: "Save token",
    changeButton: "Change Gemini token",
    readyTitle: "Gemini Live ready",
    readyCopy: "Gemini Live is configured. Your saved Gemini token is ready to use.",
    formCopy: "Paste your GEMINI_API_KEY once and we will store it locally for the headless conversation loop.",
    requiredCredentialsCopy: "Gemini Live requires your own GEMINI_API_KEY before you can switch.",
    note: "OpenAI Realtime requires OPENAI_API_KEY. Gemini Live needs GEMINI_API_KEY.",
  },
  [HF_BACKEND]: {
    label: "Hugging Face",
    formTitle: "Configure Hugging Face",
    inputLabel: "",
    placeholder: "",
    saveButton: "Save connection",
    changeButton: "Edit connection",
    readyTitle: "Hugging Face ready",
    readyCopy: "Hugging Face is configured. You can jump straight to personalities.",
    formCopy: "Choose where Reachy should connect for Hugging Face.",
    requiredCredentialsCopy: "Set up the Hugging Face connection details before switching.",
    note: "Hugging Face can use the built-in server or your own local realtime websocket.",
  },
};

function backendHasCredentials(status, backend) {
  if (backend === GEMINI_BACKEND) return !!status.has_gemini_key;
  if (backend === HF_BACKEND) return !!(status.has_hf_connection ?? (status.has_hf_session_url || status.has_hf_ws_url));
  return !!status.has_openai_key;
}

function backendCanProceed(status, backend) {
  if (backend === GEMINI_BACKEND) {
    return status.can_proceed_with_gemini !== undefined
      ? !!status.can_proceed_with_gemini
      : backendHasCredentials(status, backend);
  }
  if (backend === HF_BACKEND) {
    return status.can_proceed_with_hf !== undefined
      ? !!status.can_proceed_with_hf
      : backendHasCredentials(status, backend);
  }
  return status.can_proceed_with_openai !== undefined
    ? !!status.can_proceed_with_openai
    : backendHasCredentials(status, backend);
}

function backendMeta(backend) {
  return BACKEND_META[backend] || BACKEND_META[DEFAULT_BACKEND];
}

function formatBackendNote(text) {
  return text
    .replace("GEMINI_API_KEY", "<code>GEMINI_API_KEY</code>")
    .replace("HF_REALTIME_WS_URL", "<code>HF_REALTIME_WS_URL</code>");
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function waitForStatus(timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

async function waitForPersonalityData(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  let attempts = 0;
  const deadline = Date.now() + timeoutMs;
  while (true) {
    attempts += 1;
    try {
      const url = new URL("/personalities", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}

    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend…" : "Loading…";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

async function validateKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/validate_api_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "validation_failed");
  }
  return data;
}

async function saveBackendConfig(backend, { key = "", hfMode = "", hfHost = "", hfPort = null } = {}) {
  const body = { backend, api_key: key };
  if (backend === HF_BACKEND) {
    if (hfMode) body.hf_mode = hfMode;
    if (hfHost) body.hf_host = hfHost;
    if (hfPort !== null && hfPort !== undefined) body.hf_port = hfPort;
  }
  const resp = await fetch("/backend_config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

// ---------- Personalities API ----------
async function loadPersonality(name) {
  const url = new URL("/personalities/load", window.location.origin);
  url.searchParams.set("name", name);
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, {}, 3000);
  if (!resp.ok) throw new Error("load_failed");
  return await resp.json();
}

async function savePersonality(payload) {
  // Try JSON POST first
  const saveUrl = new URL("/personalities/save", window.location.origin);
  saveUrl.searchParams.set("_", Date.now().toString());
  let resp = await fetchWithTimeout(saveUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, 5000);
  if (resp.ok) return await resp.json();

  // Fallback to form-encoded POST
  try {
    const form = new URLSearchParams();
    form.set("name", payload.name || "");
    form.set("instructions", payload.instructions || "");
    form.set("tools_text", payload.tools_text || "");
    form.set("voice", payload.voice || "");
    const url = new URL("/personalities/save_raw", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    resp = await fetchWithTimeout(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    }, 5000);
    if (resp.ok) return await resp.json();
  } catch {}

  // Fallback to GET (query params)
  try {
    const url = new URL("/personalities/save_raw", window.location.origin);
    url.searchParams.set("name", payload.name || "");
    url.searchParams.set("instructions", payload.instructions || "");
    url.searchParams.set("tools_text", payload.tools_text || "");
    url.searchParams.set("voice", payload.voice || "");
    url.searchParams.set("_", Date.now().toString());
    resp = await fetchWithTimeout(url, { method: "GET" }, 5000);
    if (resp.ok) return await resp.json();
  } catch {}

  const data = await resp.json().catch(() => ({}));
  throw new Error(data.error || "save_failed");
}

async function applyVoice(voice) {
  const url = new URL("/voices/apply", window.location.origin);
  url.searchParams.set("voice", voice || "");
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, { method: "POST" }, 5000);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "apply_voice_failed");
  }
  return await resp.json();
}

async function applyPersonality(name, { persist = false } = {}) {
  // Send as query param to avoid any body parsing issues on the server
  const url = new URL("/personalities/apply", window.location.origin);
  url.searchParams.set("name", name || "");
  if (persist) {
    url.searchParams.set("persist", "1");
  }
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, { method: "POST" }, 5000);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "apply_failed");
  }
  return await resp.json();
}

async function getVoices() {
  try {
    const url = new URL("/voices", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("voices_failed");
    return await resp.json();
  } catch (e) {
    return [];
  }
}

async function getCurrentVoice() {
  try {
    const url = new URL("/voices/current", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("current_voice_failed");
    const data = await resp.json();
    return typeof data.voice === "string" ? data.voice : "";
  } catch (e) {
    return "";
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

function setStatusMessage(el, text, tone = "") {
  el.textContent = text;
  el.className = tone ? `status ${tone}` : "status";
  el.setAttribute("role", tone === "error" ? "alert" : "status");
  el.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
  el.setAttribute("aria-atomic", "true");
}

function describeHFConfiguration(status) {
  if (status.hf_connection_mode === "local") {
    const host = status.hf_direct_host || HF_DEFAULT_HOST;
    const port = status.hf_direct_port || HF_DEFAULT_PORT;
    return `Hugging Face will connect directly to ${host}:${port}.`;
  }
  if (status.has_hf_session_url) {
    return "Hugging Face will use the built-in server.";
  }
  return "Choose the Hugging Face server or a local realtime endpoint.";
}

function isLocalHFHost(host) {
  return !host || host === "localhost" || host === "127.0.0.1";
}

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);
  const backendChip = document.getElementById("backend-chip");
  const backendNote = document.getElementById("backend-note");
  const backendStatusEl = document.getElementById("backend-status");
  const backendSaveBtn = document.getElementById("save-backend-btn");
  const backendInputs = Array.from(document.querySelectorAll('input[name="backend"]'));
  const backendCards = Array.from(document.querySelectorAll("[data-backend-card]"));
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const configuredTitle = document.getElementById("configured-title");
  const configuredCopy = document.getElementById("configured-copy");
  const personalityPanel = document.getElementById("personality-panel");
  const formTitle = document.getElementById("form-title");
  const formCopy = document.getElementById("form-copy");
  const apiKeyFields = document.getElementById("api-key-fields");
  const apiKeyLabel = document.getElementById("api-key-label");
  const saveBtn = document.getElementById("save-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const input = document.getElementById("api-key");
  const hfFields = document.getElementById("hf-fields");
  const hfMode = document.getElementById("hf-mode");
  const hfDirectFields = document.getElementById("hf-direct-fields");
  const hfHostPreset = document.getElementById("hf-host-preset");
  const hfHostCustomWrap = document.getElementById("hf-host-custom-wrap");
  const hfHostCustom = document.getElementById("hf-host-custom");
  const hfPort = document.getElementById("hf-port");
  const hfPreview = document.getElementById("hf-preview");

  // Personality elements
  const pSelect = document.getElementById("personality-select");
  const pApply = document.getElementById("apply-personality");
  const pPersist = document.getElementById("persist-personality");
  const pNew = document.getElementById("new-personality");
  const pSave = document.getElementById("save-personality");
  const pStartupLabel = document.getElementById("startup-label");
  const pName = document.getElementById("personality-name");
  const pInstr = document.getElementById("instructions-ta");
  const pTools = document.getElementById("tools-ta");
  const pStatus = document.getElementById("personality-status");
  const pVoice = document.getElementById("voice-select");
  const pApplyVoice = document.getElementById("apply-voice");
  const pAvail = document.getElementById("tools-available");

  const AUTO_WITH = {
    dance: ["stop_dance"],
    play_emotion: ["stop_emotion"],
  };
  let selectedBackend = DEFAULT_BACKEND;
  let editingCredentials = false;

  function resolveHFHost() {
    return hfHostPreset.value === "custom" ? hfHostCustom.value.trim() : HF_DEFAULT_HOST;
  }

  function updateHFControls() {
    const localMode = hfMode.value !== "deployed";
    const customHost = hfHostPreset.value === "custom";
    show(hfDirectFields, localMode);
    show(hfHostCustomWrap, localMode && customHost);

    if (!localMode) {
      setStatusMessage(hfPreview, "Hugging Face will use the built-in server.");
      return;
    }

    const host = resolveHFHost() || "<host>";
    const port = (hfPort.value || String(HF_DEFAULT_PORT)).trim();
    setStatusMessage(hfPreview, `Will save ws://${host}:${port}/v1/realtime`);
  }

  function populateHFFields(status) {
    const mode = status.hf_connection_mode
      || (status.has_hf_session_url ? "deployed" : "local");
    const existingHost = status.hf_direct_host || HF_DEFAULT_HOST;
    const existingPort = status.hf_direct_port || HF_DEFAULT_PORT;

    hfMode.value = mode;
    if (isLocalHFHost(existingHost)) {
      hfHostPreset.value = "localhost";
      hfHostCustom.value = "";
    } else {
      hfHostPreset.value = "custom";
      hfHostCustom.value = existingHost;
    }
    hfPort.value = String(existingPort);
    updateHFControls();
  }

  function setSelectedBackend(backend) {
    selectedBackend = [OPENAI_BACKEND, GEMINI_BACKEND, HF_BACKEND].includes(backend)
      ? backend
      : DEFAULT_BACKEND;
    backendInputs.forEach((radio) => {
      radio.checked = radio.value === selectedBackend;
    });
    backendCards.forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.backendCard === selectedBackend);
    });
  }

  function renderCredentialPanels(status) {
    const persistedBackend = status.backend_provider || DEFAULT_BACKEND;
    const activeBackend = status.active_backend || persistedBackend;
    const requiresRestart = !!status.requires_restart;
    const meta = backendMeta(selectedBackend);
    const canProceedWithSelectedBackend = backendCanProceed(status, selectedBackend);
    const selectedMatchesPersisted = selectedBackend === persistedBackend;
    const selectedMatchesActive = selectedBackend === activeBackend;
    const usesApiKeyForm = selectedBackend === OPENAI_BACKEND || selectedBackend === GEMINI_BACKEND;
    const usesHFForm = selectedBackend === HF_BACKEND;
    const supportsForm = usesApiKeyForm || usesHFForm;

    backendChip.textContent = selectedBackend === persistedBackend ? "Saved" : "Selected";
    backendNote.innerHTML = formatBackendNote(meta.note);

    configuredTitle.textContent = meta.readyTitle;
    configuredCopy.textContent = usesHFForm ? describeHFConfiguration(status) : meta.readyCopy;
    formTitle.textContent = meta.formTitle;
    formCopy.textContent = usesHFForm
      ? meta.formCopy
      : canProceedWithSelectedBackend
        ? meta.formCopy
        : meta.requiredCredentialsCopy;
    apiKeyLabel.textContent = meta.inputLabel;
    input.placeholder = meta.placeholder;
    saveBtn.textContent = meta.saveButton;
    changeKeyBtn.textContent = meta.changeButton;

    show(configuredPanel, canProceedWithSelectedBackend && !editingCredentials);
    show(formPanel, supportsForm && (editingCredentials || !canProceedWithSelectedBackend));
    show(apiKeyFields, usesApiKeyForm);
    show(hfFields, usesHFForm);
    if (usesHFForm) updateHFControls();
    show(changeKeyBtn, supportsForm && canProceedWithSelectedBackend && !editingCredentials);
    show(
      backendSaveBtn,
      canProceedWithSelectedBackend && !selectedMatchesPersisted && !editingCredentials,
    );
    backendSaveBtn.textContent = `Use ${meta.label}`;

    if (requiresRestart && selectedMatchesPersisted) {
      setStatusMessage(
        backendStatusEl,
        `Backend saved. Restart Reachy Mini Conversation from the dashboard or desktop app to use ${backendMeta(persistedBackend).label}.`,
        "warn",
      );
    } else if (!selectedMatchesPersisted) {
      setStatusMessage(
        backendStatusEl,
        canProceedWithSelectedBackend
          ? selectedMatchesActive && requiresRestart
            ? `Use ${meta.label} to cancel the pending backend change.`
            : `Ready to switch to ${meta.label}.`
          : meta.requiredCredentialsCopy,
        canProceedWithSelectedBackend ? "" : "warn",
      );
    } else {
      setStatusMessage(backendStatusEl, "");
    }
  }

  statusEl.textContent = "Checking configuration...";
  show(formPanel, false);
  show(configuredPanel, false);
  show(personalityPanel, false);

  const st = (await waitForStatus()) || {
    active_backend: DEFAULT_BACKEND,
    backend_provider: DEFAULT_BACKEND,
    has_key: false,
    has_openai_key: false,
    has_gemini_key: false,
    has_hf_session_url: true,
    has_hf_ws_url: false,
    has_hf_connection: true,
    hf_connection_mode: "deployed",
    hf_direct_host: HF_DEFAULT_HOST,
    hf_direct_port: HF_DEFAULT_PORT,
    can_proceed: true,
    can_proceed_with_openai: false,
    can_proceed_with_gemini: false,
    can_proceed_with_hf: true,
    requires_restart: false,
  };
  populateHFFields(st);
  setSelectedBackend(st.backend_provider || DEFAULT_BACKEND);
  statusEl.textContent = "";
  renderCredentialPanels(st);

  // Handler for "Change API key" button
  changeKeyBtn.addEventListener("click", () => {
    editingCredentials = true;
    input.value = "";
    setStatusMessage(statusEl, "");
    renderCredentialPanels(st);
  });

  // Remove error styling when user starts typing
  input.addEventListener("input", () => {
    input.classList.remove("error");
  });
  hfHostCustom.addEventListener("input", () => {
    hfHostCustom.classList.remove("error");
    updateHFControls();
  });
  hfPort.addEventListener("input", () => {
    hfPort.classList.remove("error");
    updateHFControls();
  });
  hfMode.addEventListener("change", () => {
    hfHostCustom.classList.remove("error");
    hfPort.classList.remove("error");
    updateHFControls();
  });
  hfHostPreset.addEventListener("change", () => {
    hfHostCustom.classList.remove("error");
    updateHFControls();
  });

  backendInputs.forEach((radio) => {
    radio.addEventListener("change", () => {
      editingCredentials = false;
      input.value = "";
      setSelectedBackend(radio.value);
      renderCredentialPanels(st);
    });
  });

  backendSaveBtn.addEventListener("click", async () => {
    setStatusMessage(backendStatusEl, `Saving ${backendMeta(selectedBackend).label}...`);
    try {
      const response = await saveBackendConfig(selectedBackend);
      setStatusMessage(backendStatusEl, response.message || "Saved. Reloading…", "ok");
      window.location.reload();
    } catch (e) {
      setStatusMessage(backendStatusEl, "Failed to save backend selection. Please try again.", "error");
    }
  });

  saveBtn.addEventListener("click", async () => {
    if (selectedBackend === HF_BACKEND) {
      const localMode = hfMode.value !== "deployed";
      setStatusMessage(statusEl, "Saving connection...");
      hfHostCustom.classList.remove("error");
      hfPort.classList.remove("error");

      try {
        if (localMode) {
          const host = resolveHFHost();
          const port = Number.parseInt((hfPort.value || "").trim(), 10);
          if (!host) {
            hfHostCustom.classList.add("error");
            setStatusMessage(statusEl, "Enter a valid host or IP address.", "warn");
            return;
          }
          if (!Number.isInteger(port) || port < 1 || port > 65535) {
            hfPort.classList.add("error");
            setStatusMessage(statusEl, "Enter a valid port between 1 and 65535.", "warn");
            return;
          }

          await saveBackendConfig(selectedBackend, {
            hfMode: "local",
            hfHost: host,
            hfPort: port,
          });
        } else {
          await saveBackendConfig(selectedBackend, {
            hfMode: "deployed",
          });
        }
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        if (e.message === "missing_hf_session_url") {
          setStatusMessage(
            statusEl,
            "The built-in Hugging Face server URL is unavailable. Restart the app and try again.",
            "error",
          );
        } else if (e.message === "empty_hf_host" || e.message === "invalid_hf_host") {
          hfHostCustom.classList.add("error");
          setStatusMessage(statusEl, "Enter a valid host or IP address.", "error");
        } else if (e.message === "invalid_hf_port") {
          hfPort.classList.add("error");
          setStatusMessage(statusEl, "Enter a valid port between 1 and 65535.", "error");
        } else {
          setStatusMessage(statusEl, "Failed to save the Hugging Face connection.", "error");
        }
      }
      return;
    }

    const key = input.value.trim();
    if (!key) {
      setStatusMessage(statusEl, "Please enter a valid key.", "warn");
      input.classList.add("error");
      return;
    }
    setStatusMessage(statusEl, selectedBackend === GEMINI_BACKEND ? "Saving token..." : "Validating API key...");
    input.classList.remove("error");
    try {
      if (selectedBackend === OPENAI_BACKEND) {
        const validation = await validateKey(key);
        if (!validation.valid) {
          setStatusMessage(statusEl, "Invalid API key. Please check your key and try again.", "error");
          input.classList.add("error");
          return;
        }
        setStatusMessage(statusEl, "Key valid! Saving...", "ok");
      } else {
        setStatusMessage(statusEl, "Saving Gemini token...", "ok");
      }
      await saveBackendConfig(selectedBackend, { key });
      setStatusMessage(statusEl, "Saved. Reloading…", "ok");
      window.location.reload();
    } catch (e) {
      input.classList.add("error");
      if (selectedBackend === OPENAI_BACKEND && e.message === "invalid_api_key") {
        setStatusMessage(statusEl, "Invalid API key. Please check your key and try again.", "error");
      } else {
        setStatusMessage(
          statusEl,
          selectedBackend === GEMINI_BACKEND
            ? "Failed to save Gemini token. Please try again."
            : "Failed to validate/save key. Please try again.",
          "error",
        );
      }
    }
  });

  if (!(st.can_proceed ?? backendCanProceed(st, st.backend_provider || DEFAULT_BACKEND)) || st.requires_restart) {
    show(loading, false);
    return;
  }

  // Wait until backend routes are ready before rendering personalities UI
  const list = (await waitForPersonalityData()) || { choices: [] };
  setStatusMessage(statusEl, "");
  show(formPanel, false);
  if (!list.choices.length) {
    setStatusMessage(statusEl, "Personality endpoints not ready yet. Retry shortly.", "warn");
    show(loading, false);
    return;
  }

  // Initialize personalities UI
  try {
    const choices = Array.isArray(list.choices) ? list.choices : [];
    const DEFAULT_OPTION = choices[0] || "(built-in default)";
    const startupChoice = choices.includes(list.startup) ? list.startup : DEFAULT_OPTION;
    const currentChoice = choices.includes(list.current) ? list.current : startupChoice;

    function setStartupLabel(name) {
      const display = name && name !== DEFAULT_OPTION ? name : "Built-in default";
      pStartupLabel.textContent = `Launch on start: ${display}`;
    }

    // Populate select
    pSelect.innerHTML = "";
    for (const n of choices) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      pSelect.appendChild(opt);
    }
    if (choices.length) {
      const preferred = choices.includes(startupChoice) ? startupChoice : currentChoice;
      pSelect.value = preferred;
    }
    const voices = await getVoices();
    let currentVoice = await getCurrentVoice();
    pVoice.innerHTML = "";
    if (voices.length) {
      for (const v of voices) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        pVoice.appendChild(opt);
      }
    } else {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Backend default (recommended)";
      pVoice.appendChild(opt);
    }
    setStartupLabel(startupChoice);

    function renderToolCheckboxes(available, enabled) {
      pAvail.innerHTML = "";
      const enabledSet = new Set(enabled);
      for (const t of available) {
        const wrap = document.createElement("div");
        wrap.className = "chk";
        const id = `tool-${t}`;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.id = id;
        cb.value = t;
        cb.checked = enabledSet.has(t);
        const lab = document.createElement("label");
        lab.htmlFor = id;
        lab.textContent = t;
        wrap.appendChild(cb);
        wrap.appendChild(lab);
        pAvail.appendChild(wrap);
      }
    }

    function getSelectedTools() {
      const selected = new Set();
      pAvail.querySelectorAll('input[type="checkbox"]').forEach((el) => {
        if (el.checked) selected.add(el.value);
      });
      // Auto-include dependencies
      for (const [main, deps] of Object.entries(AUTO_WITH)) {
        if (selected.has(main)) {
          for (const d of deps) selected.add(d);
        }
      }
      return Array.from(selected);
    }

    function syncToolsTextarea() {
      const selected = getSelectedTools();
      const comments = pTools.value
        .split("\n")
        .filter((ln) => ln.trim().startsWith("#"));
      const body = selected.join("\n");
      pTools.value = (comments.join("\n") + (comments.length ? "\n" : "") + body).trim() + "\n";
    }

    pAvail.addEventListener("change", (ev) => {
      const target = ev.target;
      if (!(target instanceof HTMLInputElement) || target.type !== "checkbox") return;
      const name = target.value;
      if (AUTO_WITH[name]) {
        for (const dep of AUTO_WITH[name]) {
          const depEl = pAvail.querySelector(`input[value="${dep}"]`);
          if (depEl) depEl.checked = target.checked || depEl.checked;
        }
      }
      syncToolsTextarea();
    });

    async function loadSelected() {
      const selected = pSelect.value;
      const data = await loadPersonality(selected);
      pInstr.value = data.instructions || "";
      pTools.value = data.tools_text || "";
      const fallbackVoice = pVoice.options[0]?.value || "";
      const loadedVoice = voices.includes(data.voice) ? data.voice : fallbackVoice;
      const activeVoice = voices.includes(currentVoice) ? currentVoice : loadedVoice;
      pVoice.value = data.uses_default_voice ? activeVoice : loadedVoice;
      // Available tools as checkboxes
      renderToolCheckboxes(data.available_tools, data.enabled_tools);
      // Default name field to last segment of selection
      const idx = selected.lastIndexOf("/");
      pName.value = idx >= 0 ? selected.slice(idx + 1) : "";
      setStatusMessage(pStatus, `Loaded ${selected}`);
    }

    pSelect.addEventListener("change", loadSelected);
    await loadSelected();
    if (!voices.length) {
      setStatusMessage(pStatus, "Voices unavailable. The backend default voice will be used.", "warn");
    }
    show(personalityPanel, true);

    pApplyVoice.addEventListener("click", async () => {
      const voice = pVoice.value;
      if (!voice) return;
      setStatusMessage(pStatus, "Applying voice...");
      try {
        const res = await applyVoice(voice);
        currentVoice = voice;
        pVoice.value = voice;
        setStatusMessage(pStatus, res.status || `Voice changed to ${voice}.`, "ok");
      } catch (e) {
        setStatusMessage(pStatus, `Failed to apply voice${e.message ? ": " + e.message : ""}`, "error");
      }
    });

    pApply.addEventListener("click", async () => {
      setStatusMessage(pStatus, "Applying...");
      try {
        const res = await applyPersonality(pSelect.value);
        currentVoice = await getCurrentVoice();
        if (res.startup) setStartupLabel(res.startup);
        setStatusMessage(pStatus, res.status || "Applied.", "ok");
      } catch (e) {
        setStatusMessage(pStatus, `Failed to apply${e.message ? ": " + e.message : ""}`, "error");
      }
    });

    pPersist.addEventListener("click", async () => {
      setStatusMessage(pStatus, "Saving for startup...");
      try {
        const res = await applyPersonality(pSelect.value, { persist: true });
        currentVoice = await getCurrentVoice();
        if (res.startup) setStartupLabel(res.startup);
        setStatusMessage(pStatus, res.status || "Saved for startup.", "ok");
      } catch (e) {
        setStatusMessage(pStatus, `Failed to persist${e.message ? ": " + e.message : ""}`, "error");
      }
    });

    pNew.addEventListener("click", () => {
      pName.value = "";
      pInstr.value = "# Write your instructions here\n# e.g., Keep responses concise and friendly.";
      pTools.value = "# tools enabled for this profile\n";
      // Keep available tools list, clear selection
      pAvail.querySelectorAll('input[type="checkbox"]').forEach((el) => {
        el.checked = false;
      });
      pVoice.value = pVoice.options[0]?.value || "";
      setStatusMessage(pStatus, "Fill fields and click Save.");
    });

    pSave.addEventListener("click", async () => {
      const name = (pName.value || "").trim();
      if (!name) {
        setStatusMessage(pStatus, "Enter a valid name.", "warn");
        return;
      }
      setStatusMessage(pStatus, "Saving...");
      try {
        // Ensure tools.txt reflects checkbox selection and auto-includes
        syncToolsTextarea();
        const res = await savePersonality({
          name,
          instructions: pInstr.value || "",
          tools_text: pTools.value || "",
          voice: pVoice.value || pVoice.options[0]?.value || "",
        });
        // Refresh select choices
        pSelect.innerHTML = "";
        for (const n of res.choices) {
          const opt = document.createElement("option");
          opt.value = n;
          opt.textContent = n;
          if (n === res.value) opt.selected = true;
          pSelect.appendChild(opt);
        }
        setStatusMessage(pStatus, "Saved.", "ok");
        // Auto-apply
        try { await applyPersonality(pSelect.value); } catch {}
      } catch (e) {
        setStatusMessage(pStatus, "Failed to save.", "error");
      }
    });
  } catch (e) {
    setStatusMessage(statusEl, "UI failed to load. Please refresh.", "warn");
  } finally {
    // Hide loading when initial setup is done (regardless of key presence)
    show(loading, false);
  }
}

window.addEventListener("DOMContentLoaded", init);
