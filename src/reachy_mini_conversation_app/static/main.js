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

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

async function init() {
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const saveBtn = document.getElementById("save-btn");
  const input = document.getElementById("api-key");

  statusEl.textContent = "Checking configuration...";
  show(formPanel, false);
  show(configuredPanel, false);

  const st = await fetchStatus();
  if (st.has_key) {
    statusEl.textContent = "";
    show(configuredPanel, true);
    return;
  }

  statusEl.textContent = "";
  show(formPanel, true);

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
}

window.addEventListener("DOMContentLoaded", init);

