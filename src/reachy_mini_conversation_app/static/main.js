async function fetchStatus() {
  try {
    const url = new URL("/status", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 2000);
    if (!resp.ok) throw new Error("status error");
    return await resp.json();
  } catch (e) {
    return {
      active_backend: "openai",
      backend_provider: "openai",
      has_key: false,
      has_openai_key: false,
      has_gemini_key: false,
      requires_restart: false,
      error: true,
    };
  }
}

const OPENAI_BACKEND = "openai";
const GEMINI_BACKEND = "gemini";
const BACKEND_META = {
  [OPENAI_BACKEND]: {
    label: "OpenAI Realtime",
    formTitle: "Connect OpenAI",
    inputLabel: "OpenAI API Key",
    placeholder: "sk-...",
    saveButton: "Save key",
    changeButton: "Change OpenAI key",
    readyTitle: "OpenAI Realtime ready",
    readyCopy: "OpenAI Realtime is configured. You can jump straight to personalities.",
    formCopy: "OpenAI Realtime uses the distributed key when available. Paste your own key if you want an override or need a fallback.",
    requiredCredentialsCopy: "OpenAI Realtime usually uses the distributed key. If it is unavailable here, paste your own OpenAI key to continue.",
    note: "OpenAI Realtime uses the distributed OpenAI key. You can still paste your own key if you want to override it.",
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
    note: "OpenAI Realtime uses the distributed OpenAI key. Gemini Live needs your own GEMINI_API_KEY.",
  },
};

function backendHasCredentials(status, backend) {
  return backend === GEMINI_BACKEND ? !!status.has_gemini_key : !!status.has_openai_key;
}

function backendCanProceed(status, backend) {
  if (backend === GEMINI_BACKEND) {
    return status.can_proceed_with_gemini !== undefined
      ? !!status.can_proceed_with_gemini
      : backendHasCredentials(status, backend);
  }
  return status.can_proceed_with_openai !== undefined
    ? !!status.can_proceed_with_openai
    : backendHasCredentials(status, backend);
}

function backendMeta(backend) {
  return BACKEND_META[backend] || BACKEND_META[OPENAI_BACKEND];
}

function formatBackendNote(text) {
  return text.replace("GEMINI_API_KEY", "<code>GEMINI_API_KEY</code>");
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

async function saveBackendConfig(backend, key = "") {
  const body = { backend, api_key: key };
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
async function getPersonalities() {
  const url = new URL("/personalities", window.location.origin);
  url.searchParams.set("_", Date.now().toString());
  const resp = await fetchWithTimeout(url, {}, 2000);
  if (!resp.ok) throw new Error("list_failed");
  return await resp.json();
}

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

const VOICE_FALLBACK = [];

async function getVoices() {
  try {
    const url = new URL("/voices", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("voices_failed");
    return await resp.json();
  } catch (e) {
    return VOICE_FALLBACK;
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
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
  const apiKeyLabel = document.getElementById("api-key-label");
  const saveBtn = document.getElementById("save-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const input = document.getElementById("api-key");

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
  const pAvail = document.getElementById("tools-available");

  const AUTO_WITH = {
    dance: ["stop_dance"],
    play_emotion: ["stop_emotion"],
  };
  let selectedBackend = OPENAI_BACKEND;
  let editingCredentials = false;

  function setSelectedBackend(backend) {
    selectedBackend = backend === GEMINI_BACKEND ? GEMINI_BACKEND : OPENAI_BACKEND;
    backendInputs.forEach((radio) => {
      radio.checked = radio.value === selectedBackend;
    });
    backendCards.forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.backendCard === selectedBackend);
    });
  }

  function renderCredentialPanels(status) {
    const persistedBackend = status.backend_provider || OPENAI_BACKEND;
    const activeBackend = status.active_backend || persistedBackend;
    const requiresRestart = !!status.requires_restart;
    const meta = backendMeta(selectedBackend);
    const canProceedWithSelectedBackend = backendCanProceed(status, selectedBackend);
    const selectedMatchesPersisted = selectedBackend === persistedBackend;
    const selectedMatchesActive = selectedBackend === activeBackend;

    backendChip.textContent = selectedBackend === persistedBackend ? "Saved" : "Selected";
    backendNote.innerHTML = formatBackendNote(meta.note);

    configuredTitle.textContent = meta.readyTitle;
    configuredCopy.textContent = meta.readyCopy;
    formTitle.textContent = meta.formTitle;
    formCopy.textContent = canProceedWithSelectedBackend ? meta.formCopy : meta.requiredCredentialsCopy;
    apiKeyLabel.textContent = meta.inputLabel;
    input.placeholder = meta.placeholder;
    saveBtn.textContent = meta.saveButton;
    changeKeyBtn.textContent = meta.changeButton;

    show(configuredPanel, canProceedWithSelectedBackend && !editingCredentials);
    show(formPanel, editingCredentials || !canProceedWithSelectedBackend);
    show(
      backendSaveBtn,
      canProceedWithSelectedBackend && !selectedMatchesPersisted,
    );
    backendSaveBtn.textContent = `Use ${meta.label}`;

    if (requiresRestart && selectedMatchesPersisted) {
      backendStatusEl.textContent = `Backend saved. Restart Reachy Mini Conversation from the dashboard or desktop app to use ${backendMeta(persistedBackend).label}.`;
      backendStatusEl.className = "status warn";
    } else if (!selectedMatchesPersisted) {
      backendStatusEl.textContent = canProceedWithSelectedBackend
        ? selectedMatchesActive && requiresRestart
          ? `Use ${meta.label} to cancel the pending backend change.`
          : `Ready to switch to ${meta.label}.`
        : meta.requiredCredentialsCopy;
      backendStatusEl.className = canProceedWithSelectedBackend ? "status" : "status warn";
    } else {
      backendStatusEl.textContent = "";
      backendStatusEl.className = "status";
    }
  }

  statusEl.textContent = "Checking configuration...";
  show(formPanel, false);
  show(configuredPanel, false);
  show(personalityPanel, false);

  const st = (await waitForStatus()) || {
    active_backend: OPENAI_BACKEND,
    backend_provider: OPENAI_BACKEND,
    has_key: false,
    has_openai_key: false,
    has_gemini_key: false,
    can_proceed: false,
    can_proceed_with_openai: false,
    can_proceed_with_gemini: false,
    requires_restart: false,
  };
  setSelectedBackend(st.backend_provider || OPENAI_BACKEND);
  statusEl.textContent = "";
  renderCredentialPanels(st);

  // Handler for "Change API key" button
  changeKeyBtn.addEventListener("click", () => {
    editingCredentials = true;
    input.value = "";
    statusEl.textContent = "";
    statusEl.className = "status";
    renderCredentialPanels(st);
  });

  // Remove error styling when user starts typing
  input.addEventListener("input", () => {
    input.classList.remove("error");
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
    backendStatusEl.textContent = `Saving ${backendMeta(selectedBackend).label}...`;
    backendStatusEl.className = "status";
    try {
      const response = await saveBackendConfig(selectedBackend);
      backendStatusEl.textContent = response.message || "Saved. Reloading…";
      backendStatusEl.className = "status ok";
      window.location.reload();
    } catch (e) {
      backendStatusEl.textContent = "Failed to save backend selection. Please try again.";
      backendStatusEl.className = "status error";
    }
  });

  saveBtn.addEventListener("click", async () => {
    const key = input.value.trim();
    if (!key) {
      statusEl.textContent = "Please enter a valid key.";
      statusEl.className = "status warn";
      input.classList.add("error");
      return;
    }
    statusEl.textContent = selectedBackend === GEMINI_BACKEND ? "Saving token..." : "Validating API key...";
    statusEl.className = "status";
    input.classList.remove("error");
    try {
      if (selectedBackend === OPENAI_BACKEND) {
        const validation = await validateKey(key);
        if (!validation.valid) {
          statusEl.textContent = "Invalid API key. Please check your key and try again.";
          statusEl.className = "status error";
          input.classList.add("error");
          return;
        }
        statusEl.textContent = "Key valid! Saving...";
      } else {
        statusEl.textContent = "Saving Gemini token...";
      }
      statusEl.className = "status ok";
      await saveBackendConfig(selectedBackend, key);
      statusEl.textContent = "Saved. Reloading…";
      statusEl.className = "status ok";
      window.location.reload();
    } catch (e) {
      input.classList.add("error");
      if (selectedBackend === OPENAI_BACKEND && e.message === "invalid_api_key") {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
      } else {
        statusEl.textContent = selectedBackend === GEMINI_BACKEND
          ? "Failed to save Gemini token. Please try again."
          : "Failed to validate/save key. Please try again.";
      }
      statusEl.className = "status error";
    }
  });

  if (!(st.can_proceed ?? backendCanProceed(st, st.backend_provider || OPENAI_BACKEND)) || st.requires_restart) {
    show(loading, false);
    return;
  }

  // Wait until backend routes are ready before rendering personalities UI
  const list = (await waitForPersonalityData()) || { choices: [] };
  statusEl.textContent = "";
  show(formPanel, false);
  if (!list.choices.length) {
    statusEl.textContent = "Personality endpoints not ready yet. Retry shortly.";
    statusEl.className = "status warn";
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
    pVoice.innerHTML = "";
    for (const v of voices) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
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

    function attachToolHandlers() {
      pAvail.addEventListener("change", (ev) => {
        const target = ev.target;
        if (!(target instanceof HTMLInputElement) || target.type !== "checkbox") return;
        const name = target.value;
        // If a main tool toggled, propagate to deps
        if (AUTO_WITH[name]) {
          for (const dep of AUTO_WITH[name]) {
            const depEl = pAvail.querySelector(`input[value="${dep}"]`);
            if (depEl) depEl.checked = target.checked || depEl.checked;
          }
        }
        syncToolsTextarea();
      });
    }

    async function loadSelected() {
      const selected = pSelect.value;
      const data = await loadPersonality(selected);
      pInstr.value = data.instructions || "";
      pTools.value = data.tools_text || "";
      const fallbackVoice = pVoice.options[0]?.value || "";
      pVoice.value = voices.includes(data.voice) ? data.voice : fallbackVoice;
      // Available tools as checkboxes
      renderToolCheckboxes(data.available_tools, data.enabled_tools);
      attachToolHandlers();
      // Default name field to last segment of selection
      const idx = selected.lastIndexOf("/");
      pName.value = idx >= 0 ? selected.slice(idx + 1) : "";
      pStatus.textContent = `Loaded ${selected}`;
      pStatus.className = "status";
    }

    pSelect.addEventListener("change", loadSelected);
    await loadSelected();
    show(personalityPanel, true);

    // pAvail change handler registered in attachToolHandlers()

    pApply.addEventListener("click", async () => {
      pStatus.textContent = "Applying...";
      pStatus.className = "status";
      try {
        const res = await applyPersonality(pSelect.value);
        if (res.startup) setStartupLabel(res.startup);
        pStatus.textContent = res.status || "Applied.";
        pStatus.className = "status ok";
      } catch (e) {
        pStatus.textContent = `Failed to apply${e.message ? ": " + e.message : ""}`;
        pStatus.className = "status error";
      }
    });

    pPersist.addEventListener("click", async () => {
      pStatus.textContent = "Saving for startup...";
      pStatus.className = "status";
      try {
        const res = await applyPersonality(pSelect.value, { persist: true });
        if (res.startup) setStartupLabel(res.startup);
        pStatus.textContent = res.status || "Saved for startup.";
        pStatus.className = "status ok";
      } catch (e) {
        pStatus.textContent = `Failed to persist${e.message ? ": " + e.message : ""}`;
        pStatus.className = "status error";
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
      pStatus.textContent = "Fill fields and click Save.";
      pStatus.className = "status";
    });

    pSave.addEventListener("click", async () => {
      const name = (pName.value || "").trim();
      if (!name) {
        pStatus.textContent = "Enter a valid name.";
        pStatus.className = "status warn";
        return;
      }
      pStatus.textContent = "Saving...";
      pStatus.className = "status";
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
        pStatus.textContent = "Saved.";
        pStatus.className = "status ok";
        // Auto-apply
        try { await applyPersonality(pSelect.value); } catch {}
      } catch (e) {
        pStatus.textContent = "Failed to save.";
        pStatus.className = "status error";
      }
    });
  } catch (e) {
    statusEl.textContent = "UI failed to load. Please refresh.";
    statusEl.className = "status warn";
  } finally {
    // Hide loading when initial setup is done (regardless of key presence)
    show(loading, false);
  }
}

window.addEventListener("DOMContentLoaded", init);
