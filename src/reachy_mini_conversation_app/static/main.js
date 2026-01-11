async function fetchStatus() {
  try {
    const url = new URL("/status", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 2000);
    if (!resp.ok) throw new Error("status error");
    return await resp.json();
  } catch (e) {
    return { has_key: false, error: true };
  }
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

// ---------- Configuration API ----------
async function getConfig() {
  try {
    const url = new URL("/config", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return { variables: [] };
    return await resp.json();
  } catch (e) {
    return { variables: [] };
  }
}

async function setConfigValue(key, value) {
  const resp = await fetch(`/config/${encodeURIComponent(key)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value, persist: true }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

async function reloadConfig() {
  const resp = await fetch("/config/reload", {
    method: "POST",
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "reload_failed");
  }
  return await resp.json();
}

async function getProfileConfig(profileName) {
  try {
    const url = new URL(`/config/profile/${encodeURIComponent(profileName)}`, window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return { variables: [] };
    return await resp.json();
  } catch (e) {
    return { variables: [] };
  }
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
    form.set("voice", payload.voice || "cedar");
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
    url.searchParams.set("voice", payload.voice || "cedar");
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

async function getVoices() {
  try {
    const url = new URL("/voices", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("voices_failed");
    return await resp.json();
  } catch (e) {
    return ["cedar"];
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);
  const personalityPanel = document.getElementById("personality-panel");

  // Config panel elements
  const configPanel = document.getElementById("config-panel");
  const configVarsContainer = document.getElementById("config-vars-container");
  const saveConfigBtn = document.getElementById("save-config-btn");
  const reloadConfigBtn = document.getElementById("reload-config-btn");
  const configStatus = document.getElementById("config-status");

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

  // Profile requirements elements
  const pReqSection = document.getElementById("profile-requirements");
  const pReqSummary = document.getElementById("profile-req-summary");
  const pEnvVars = document.getElementById("profile-env-vars");

  const AUTO_WITH = {
    dance: ["stop_dance"],
    play_emotion: ["stop_emotion"],
  };

  show(personalityPanel, false);
  show(configPanel, true);

  // Store current config values for saving
  let configInputs = {};

  function renderConfigVars(variables) {
    configVarsContainer.innerHTML = "";
    configInputs = {};

    for (const v of variables) {
      const row = document.createElement("div");
      row.className = "row";

      const label = document.createElement("label");
      label.htmlFor = `config-${v.key}`;
      label.textContent = v.key;
      label.title = v.description;

      const inputWrapper = document.createElement("div");
      inputWrapper.className = "input-wrapper";

      const inp = document.createElement("input");
      inp.id = `config-${v.key}`;
      inp.type = v.is_secret ? "password" : "text";
      inp.autocomplete = "off";
      inp.dataset.key = v.key;
      inp.dataset.isSecret = v.is_secret ? "true" : "false";

      if (v.is_set) {
        if (v.is_secret) {
          inp.placeholder = v.value || "••••••••";
        } else {
          inp.value = v.value || "";
        }
      } else {
        inp.placeholder = v.description;
      }

      // Status indicator
      const statusIndicator = document.createElement("span");
      statusIndicator.className = v.is_set ? "indicator indicator-set" : "indicator indicator-unset";
      statusIndicator.title = v.is_set ? "Configured" : "Not set";

      inputWrapper.appendChild(inp);
      inputWrapper.appendChild(statusIndicator);

      row.appendChild(label);
      row.appendChild(inputWrapper);
      configVarsContainer.appendChild(row);

      configInputs[v.key] = inp;
    }
  }

  try {
    const configData = await getConfig();
    renderConfigVars(configData.variables || []);
  } catch (e) {
    configStatus.textContent = "Failed to load configuration.";
    configStatus.className = "status error";
  }

  saveConfigBtn.addEventListener("click", async () => {
    configStatus.textContent = "Saving...";
    configStatus.className = "status";

    let savedCount = 0;
    let errorCount = 0;

    for (const [key, inp] of Object.entries(configInputs)) {
      const value = inp.value.trim();
      // Only save if user entered a value (don't overwrite with empty for secrets)
      if (value) {
        try {
          await setConfigValue(key, value);
          savedCount++;
          // Clear the input and update placeholder for secrets
          if (inp.dataset.isSecret === "true") {
            inp.value = "";
            inp.placeholder = value.slice(0, 4) + "..." + value.slice(-4);
          }
          // Update indicator
          const indicator = inp.parentElement.querySelector(".indicator");
          if (indicator) {
            indicator.className = "indicator indicator-set";
            indicator.title = "Configured";
          }
        } catch (e) {
          errorCount++;
        }
      }
    }

    if (errorCount > 0) {
      configStatus.textContent = `Saved ${savedCount}, ${errorCount} failed.`;
      configStatus.className = "status warn";
    } else if (savedCount > 0) {
      configStatus.textContent = `Saved ${savedCount} value(s).`;
      configStatus.className = "status ok";
    } else {
      configStatus.textContent = "No changes to save.";
      configStatus.className = "status";
    }
  });

  reloadConfigBtn.addEventListener("click", async () => {
    configStatus.textContent = "Reloading...";
    configStatus.className = "status";
    try {
      const result = await reloadConfig();
      renderConfigVars(result.variables || []);
      configStatus.textContent = "Configuration reloaded.";
      configStatus.className = "status ok";
    } catch (e) {
      configStatus.textContent = "Failed to reload configuration.";
      configStatus.className = "status error";
    }
  });

  // Wait until backend routes are ready before rendering personalities UI
  const list = (await waitForPersonalityData()) || { choices: [] };
  if (!list.choices.length) {
    configStatus.textContent = "Personality endpoints not ready yet. Retry shortly.";
    configStatus.className = "status warn";
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

    async function renderProfileEnvVars(profileName) {
      // Fetch and display environment variables required by this profile
      const profileConfig = await getProfileConfig(profileName);
      const variables = profileConfig.variables || [];

      // Filter to show only profile-specific vars (beyond base vars like OPENAI_API_KEY)
      // We show all vars but highlight tool-specific ones
      if (variables.length === 0) {
        show(pReqSection, false);
        return;
      }

      pEnvVars.innerHTML = "";
      let configured = 0;
      let missing = 0;
      let missingRequired = 0;

      for (const v of variables) {
        const tag = document.createElement("span");
        const isConfigured = v.is_set;
        const isMissing = !isConfigured;
        const isRequired = v.required !== false;

        if (isConfigured) {
          configured++;
          tag.className = "env-var-tag configured";
        } else if (isRequired) {
          missing++;
          missingRequired++;
          tag.className = "env-var-tag missing required";
        } else {
          missing++;
          tag.className = "env-var-tag missing";
        }

        tag.title = v.description || v.key;

        const dot = document.createElement("span");
        dot.className = "dot";

        const varName = document.createElement("span");
        varName.className = "var-name";
        varName.textContent = v.key;

        tag.appendChild(dot);
        tag.appendChild(varName);

        if (v.description && v.description.length <= 30) {
          const desc = document.createElement("span");
          desc.className = "var-desc";
          desc.textContent = v.description;
          tag.appendChild(desc);
        }

        pEnvVars.appendChild(tag);
      }

      // Update summary chip
      if (missingRequired > 0) {
        pReqSummary.textContent = `${missingRequired} required missing`;
        pReqSummary.className = "chip chip-warn";
      } else if (missing > 0) {
        pReqSummary.textContent = `${missing} optional missing`;
        pReqSummary.className = "chip";
      } else {
        pReqSummary.textContent = `All ${configured} configured`;
        pReqSummary.className = "chip chip-ok";
      }

      show(pReqSection, true);
    }

    async function loadSelected() {
      const selected = pSelect.value;
      const data = await loadPersonality(selected);
      pInstr.value = data.instructions || "";
      pTools.value = data.tools_text || "";
      pVoice.value = data.voice || "cedar";
      // Available tools as checkboxes
      renderToolCheckboxes(data.available_tools, data.enabled_tools);
      attachToolHandlers();
      // Default name field to last segment of selection
      const idx = selected.lastIndexOf("/");
      pName.value = idx >= 0 ? selected.slice(idx + 1) : "";
      pStatus.textContent = `Loaded ${selected}`;
      pStatus.className = "status";

      // Load and display profile environment requirements
      await renderProfileEnvVars(selected);
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
      pVoice.value = "cedar";
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
          voice: pVoice.value || "cedar",
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
    configStatus.textContent = "UI failed to load. Please refresh.";
    configStatus.className = "status warn";
  } finally {
    // Hide loading when initial setup is done (regardless of key presence)
    show(loading, false);
  }
}

window.addEventListener("DOMContentLoaded", init);
