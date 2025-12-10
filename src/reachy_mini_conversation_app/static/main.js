async function fetchStatus() {
  try {
    const resp = await fetch("/status");
    if (!resp.ok) throw new Error("status error");
    return await resp.json();
  } catch (e) {
    return { has_key: false, error: true };
  }
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
  const resp = await fetch("/personalities");
  if (!resp.ok) throw new Error("list_failed");
  return await resp.json();
}

async function loadPersonality(name) {
  const url = new URL("/personalities/load", window.location.origin);
  url.searchParams.set("name", name);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error("load_failed");
  return await resp.json();
}

async function savePersonality(payload) {
  // Try JSON POST first
  let resp = await fetch("/personalities/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (resp.ok) return await resp.json();

  // Fallback to form-encoded POST
  try {
    const form = new URLSearchParams();
    form.set("name", payload.name || "");
    form.set("instructions", payload.instructions || "");
    form.set("tools_text", payload.tools_text || "");
    form.set("voice", payload.voice || "cedar");
    resp = await fetch("/personalities/save_raw", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    });
    if (resp.ok) return await resp.json();
  } catch {}

  // Fallback to GET (query params)
  try {
    const url = new URL("/personalities/save_raw", window.location.origin);
    url.searchParams.set("name", payload.name || "");
    url.searchParams.set("instructions", payload.instructions || "");
    url.searchParams.set("tools_text", payload.tools_text || "");
    url.searchParams.set("voice", payload.voice || "cedar");
    resp = await fetch(url, { method: "GET" });
    if (resp.ok) return await resp.json();
  } catch {}

  const data = await resp.json().catch(() => ({}));
  throw new Error(data.error || "save_failed");
}

async function applyPersonality(name) {
  // Send as query param to avoid any body parsing issues on the server
  const url = new URL("/personalities/apply", window.location.origin);
  url.searchParams.set("name", name || "");
  const resp = await fetch(url, { method: "POST" });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "apply_failed");
  }
  return await resp.json();
}

async function getVoices() {
  try {
    const resp = await fetch("/voices");
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
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const personalityPanel = document.getElementById("personality-panel");
  const saveBtn = document.getElementById("save-btn");
  const input = document.getElementById("api-key");

  // Personality elements
  const pSelect = document.getElementById("personality-select");
  const pApply = document.getElementById("apply-personality");
  const pNew = document.getElementById("new-personality");
  const pSave = document.getElementById("save-personality");
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

  const st = await fetchStatus();
  if (st.has_key) {
    statusEl.textContent = "";
    show(configuredPanel, true);
  }

  statusEl.textContent = "";
  // Only show the API key form if key is missing
  show(formPanel, !st.has_key);

  saveBtn.addEventListener("click", async () => {
    const key = input.value.trim();
    if (!key) {
      statusEl.textContent = "Please enter a valid key.";
      statusEl.className = "status warn";
      return;
    }
    statusEl.textContent = "Saving...";
    statusEl.className = "status";
    try {
      await saveKey(key);
      statusEl.textContent = "Saved. The app will start automatically.";
      statusEl.className = "status ok";
      // Optimistically switch to configured state
      show(formPanel, false);
      show(configuredPanel, true);
    } catch (e) {
      statusEl.textContent = "Failed to save key.";
      statusEl.className = "status error";
    }
  });

  // Initialize personalities UI
  try {
    const list = await getPersonalities();
    // Populate select
    pSelect.innerHTML = "";
    for (const n of list.choices) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      pSelect.appendChild(opt);
    }
    const voices = await getVoices();
    pVoice.innerHTML = "";
    for (const v of voices) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      pVoice.appendChild(opt);
    }

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
        pStatus.textContent = res.status || "Applied.";
        pStatus.className = "status ok";
      } catch (e) {
        pStatus.textContent = `Failed to apply${e.message ? ": " + e.message : ""}`;
        pStatus.className = "status error";
      }
    });

    pNew.addEventListener("click", () => {
      pName.value = "";
      pInstr.value = "# Write your instructions here\n# e.g., Keep responses concise and friendly.";
      pTools.value = "# tools enabled for this profile\n";
      // Keep available tools list, clear selection
      for (const opt of pAvail.options) opt.selected = false;
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
    // If endpoints are not available, silently skip personality UI
  }
}

window.addEventListener("DOMContentLoaded", init);
