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

async function saveKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/openai_api_key", {
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

async function requestEphemeralKey(orderNumber) {
  const body = { order_number: orderNumber };
  const resp = await fetch("/request_ephemeral_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "request_failed");
  }
  return data;
}

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);
  const statusEl = document.getElementById("status");
  const requestStatusEl = document.getElementById("request-status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const personalityPanel = document.getElementById("personality-panel");
  const saveBtn = document.getElementById("save-btn");
  const requestKeyBtn = document.getElementById("request-key-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const input = document.getElementById("api-key");
  const orderInput = document.getElementById("order-number");
  const sessionInfo = document.getElementById("session-info");
  const sessionsRemaining = document.getElementById("sessions-remaining");
  const keyStatus = document.getElementById("key-status");
  const configuredSessionInfo = document.getElementById("configured-session-info");
  const configuredSessionsRemaining = document.getElementById("configured-sessions-remaining");
  const configuredKeyStatus = document.getElementById("configured-key-status");

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

  statusEl.textContent = "Checking configuration...";
  show(formPanel, false);
  show(configuredPanel, false);
  show(personalityPanel, false);

  const st = (await waitForStatus()) || { has_key: false };
  if (st.has_key) {
    statusEl.textContent = "";
    show(configuredPanel, true);

    // Check if we have session info from a recent ephemeral key request
    try {
      const storedInfo = localStorage.getItem('session_info');
      if (storedInfo) {
        const sessionData = JSON.parse(storedInfo);
        // Only show if it's an ephemeral key and less than 1 hour old
        const hoursSinceStored = (Date.now() - sessionData.timestamp) / (1000 * 60 * 60);
        if (sessionData.is_ephemeral && hoursSinceStored < 1) {
          show(configuredSessionInfo, true);

          // Update sessions remaining with color coding
          const remaining = sessionData.keys_remaining;
          configuredSessionsRemaining.textContent = `${remaining} of ${sessionData.keys_total || '?'}`;
          configuredSessionsRemaining.className = "info-value";
          if (remaining <= 5) {
            configuredSessionsRemaining.className = "info-value critical";
          } else if (remaining <= 20) {
            configuredSessionsRemaining.className = "info-value warning";
          }

          // Update key status
          if (sessionData.is_reused) {
            configuredKeyStatus.textContent = "Reusing recent key";
            configuredKeyStatus.className = "info-value";
          } else {
            configuredKeyStatus.textContent = "Key created";
            configuredKeyStatus.className = "info-value";
          }
        } else {
          // Clear old or non-ephemeral data
          localStorage.removeItem('session_info');
        }
      }
    } catch (e) {
      // Ignore errors reading from localStorage
    }
  }

  // Handler for "Change API key" button
  changeKeyBtn.addEventListener("click", () => {
    show(configuredPanel, false);
    show(formPanel, true);
    input.value = "";
    statusEl.textContent = "";
    statusEl.className = "status";
  });

  // Remove error styling when user starts typing
  input.addEventListener("input", () => {
    input.classList.remove("error");
  });

  orderInput.addEventListener("input", () => {
    orderInput.classList.remove("error");
  });

  requestKeyBtn.addEventListener("click", async () => {
    const orderNumber = orderInput.value.trim();
    if (!orderNumber) {
      requestStatusEl.textContent = "Please enter your order number.";
      requestStatusEl.className = "status warn";
      orderInput.classList.add("error");
      show(sessionInfo, false);
      return;
    }
    requestStatusEl.textContent = "Requesting access...";
    requestStatusEl.className = "status";
    orderInput.classList.remove("error");
    show(sessionInfo, false);
    try {
      const result = await requestEphemeralKey(orderNumber);
      if (result.success) {
        // Build success message with session details
        let message = "Access granted! ";

        if (result.is_reused) {
          message += "Reusing your recent key. ";
        } else {
          message += "New key created. ";
        }

        message += "Starting service...";

        requestStatusEl.textContent = message;
        requestStatusEl.className = "status ok";

        // Display session info if available
        if (result.keys_remaining !== null && result.keys_remaining !== undefined) {
          show(sessionInfo, true);

          // Update sessions remaining with color coding
          const remaining = result.keys_remaining;
          sessionsRemaining.textContent = `${remaining} of ${result.keys_total || '?'}`;
          sessionsRemaining.className = "info-value";
          if (remaining <= 5) {
            sessionsRemaining.className = "info-value critical";
          } else if (remaining <= 20) {
            sessionsRemaining.className = "info-value warning";
          }

          // Update key status
          if (result.is_reused) {
            keyStatus.textContent = "Reusing recent key";
            keyStatus.className = "info-value";
          } else {
            keyStatus.textContent = "New key created";
            keyStatus.className = "info-value";
          }

          // Store session info in localStorage for display after reload
          try {
            localStorage.setItem('session_info', JSON.stringify({
              keys_remaining: result.keys_remaining,
              keys_total: result.keys_total,
              keys_used: result.keys_used,
              is_reused: result.is_reused,
              timestamp: Date.now(),
              is_ephemeral: true  // Mark this as an ephemeral key
            }));
          } catch (e) {
            // localStorage might not be available, ignore
          }
        }

        // Wait a moment for the user to see the info before reloading
        await sleep(2000);
        // Reload to start the service with the ephemeral key in memory
        window.location.reload();
      } else {
        show(sessionInfo, false);
        requestStatusEl.textContent = result.error || "Failed to get access. Please check your order number.";
        requestStatusEl.className = "status error";
        orderInput.classList.add("error");
      }
    } catch (e) {
      show(sessionInfo, false);
      orderInput.classList.add("error");
      requestStatusEl.textContent = e.message || "Failed to request access. Please try again.";
      requestStatusEl.className = "status error";
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
    statusEl.textContent = "Validating API key...";
    statusEl.className = "status";
    input.classList.remove("error");
    try {
      // First validate the key
      const validation = await validateKey(key);
      if (!validation.valid) {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
        statusEl.className = "status error";
        input.classList.add("error");
        return;
      }

      // If valid, save it
      statusEl.textContent = "Key valid! Saving...";
      statusEl.className = "status ok";
      await saveKey(key);

      // Clear any ephemeral session info since user is using their own key
      try {
        localStorage.removeItem('session_info');
      } catch (e) {
        // Ignore if localStorage isn't available
      }

      statusEl.textContent = "Saved. Reloading…";
      statusEl.className = "status ok";
      window.location.reload();
    } catch (e) {
      input.classList.add("error");
      if (e.message === "invalid_api_key") {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
      } else {
        statusEl.textContent = "Failed to validate/save key. Please try again.";
      }
      statusEl.className = "status error";
    }
  });

  if (!st.has_key) {
    statusEl.textContent = "";
    show(formPanel, true);
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
      pVoice.value = data.voice || "cedar";
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
    statusEl.textContent = "UI failed to load. Please refresh.";
    statusEl.className = "status warn";
  } finally {
    // Hide loading when initial setup is done (regardless of key presence)
    show(loading, false);
  }
}

window.addEventListener("DOMContentLoaded", init);
