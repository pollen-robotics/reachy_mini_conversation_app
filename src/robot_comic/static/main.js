const OPENAI_BACKEND = "openai";
const GEMINI_BACKEND = "gemini";
const HF_BACKEND = "huggingface";
const LOCAL_STT_BACKEND = "local_stt";
const GEMINI_TTS_OUTPUT = "gemini_tts";
const CHATTERBOX_OUTPUT = "chatterbox";
const ELEVENLABS_OUTPUT = "elevenlabs";
const LLAMA_ELEVENLABS_TTS_OUTPUT = "llama_elevenlabs_tts";
const DEFAULT_BACKEND = HF_BACKEND;
const HF_DEFAULT_HOST = "localhost";
const HF_DEFAULT_PORT = 8765;

// Phase 4f canonical dial values shared with src/robot_comic/config.py.
const PIPELINE_MODE_COMPOSABLE = "composable";
const PIPELINE_MODE_OPENAI_REALTIME = "openai_realtime";
const PIPELINE_MODE_GEMINI_LIVE = "gemini_live";
const PIPELINE_MODE_HF_REALTIME = "hf_realtime";
const AUDIO_INPUT_MOONSHINE = "moonshine";
const AUDIO_INPUT_OPENAI_REALTIME = "openai_realtime_input";
const AUDIO_INPUT_GEMINI_LIVE = "gemini_live_input";
const AUDIO_INPUT_HF = "hf_input";
const AUDIO_OUTPUT_CHATTERBOX = "chatterbox";
const AUDIO_OUTPUT_GEMINI_TTS = "gemini_tts";
const AUDIO_OUTPUT_ELEVENLABS = "elevenlabs";
const AUDIO_OUTPUT_XTTS = "xtts";
const XTTS_OUTPUT = "xtts";
const AUDIO_OUTPUT_OPENAI_REALTIME = "openai_realtime_output";
const AUDIO_OUTPUT_GEMINI_LIVE = "gemini_live_output";
const AUDIO_OUTPUT_HF = "hf_output";

// Map the family-radio value (huggingface / openai / gemini / local_stt) to
// the bundled pipeline_mode that the server expects.  ``local_stt`` is the
// only composable family — its TTS row carries the output-backend choice
// separately.
const FAMILY_TO_PIPELINE_MODE = {
  [OPENAI_BACKEND]: PIPELINE_MODE_OPENAI_REALTIME,
  [GEMINI_BACKEND]: PIPELINE_MODE_GEMINI_LIVE,
  [HF_BACKEND]: PIPELINE_MODE_HF_REALTIME,
  [LOCAL_STT_BACKEND]: PIPELINE_MODE_COMPOSABLE,
};

// Inverse of FAMILY_TO_PIPELINE_MODE plus the composable-output fallback
// used to drive the family radio from the server's status payload.
function pipelineModeToFamily(pipelineMode, audioOutputBackend) {
  if (pipelineMode === PIPELINE_MODE_OPENAI_REALTIME) return OPENAI_BACKEND;
  if (pipelineMode === PIPELINE_MODE_GEMINI_LIVE) return GEMINI_BACKEND;
  if (pipelineMode === PIPELINE_MODE_HF_REALTIME) return HF_BACKEND;
  return LOCAL_STT_BACKEND;
}

// Translate the historical "response backend" radio value (the row inside the
// local_stt 3-column picker) into a canonical AUDIO_OUTPUT_* identifier.
function legacyResponseToAudioOutput(legacy) {
  switch (legacy) {
    case CHATTERBOX_OUTPUT: return AUDIO_OUTPUT_CHATTERBOX;
    case ELEVENLABS_OUTPUT: return AUDIO_OUTPUT_ELEVENLABS;
    case LLAMA_ELEVENLABS_TTS_OUTPUT: return AUDIO_OUTPUT_ELEVENLABS;
    case GEMINI_TTS_OUTPUT: return AUDIO_OUTPUT_GEMINI_TTS;
    case XTTS_OUTPUT: return AUDIO_OUTPUT_XTTS;
    case HF_BACKEND: return AUDIO_OUTPUT_HF;
    case OPENAI_BACKEND:
    default: return AUDIO_OUTPUT_OPENAI_REALTIME;
  }
}

// Inverse of legacyResponseToAudioOutput. The ``llmBackend`` argument
// disambiguates the two LLM variants of the elevenlabs audio output.
function audioOutputToLegacyResponse(audioOutputBackend, llmBackend) {
  if (audioOutputBackend === AUDIO_OUTPUT_CHATTERBOX) return CHATTERBOX_OUTPUT;
  if (audioOutputBackend === AUDIO_OUTPUT_ELEVENLABS) {
    return llmBackend === LLM_BACKEND_LLAMA ? LLAMA_ELEVENLABS_TTS_OUTPUT : ELEVENLABS_OUTPUT;
  }
  if (audioOutputBackend === AUDIO_OUTPUT_GEMINI_TTS) return GEMINI_TTS_OUTPUT;
  if (audioOutputBackend === AUDIO_OUTPUT_XTTS) return XTTS_OUTPUT;
  if (audioOutputBackend === AUDIO_OUTPUT_HF) return HF_BACKEND;
  return OPENAI_BACKEND;
}

// LLM axis values for the 3-column pipeline picker
const LLM_BACKEND_LLAMA = "llama";
const LLM_BACKEND_GEMINI = "gemini";

/**
 * Maps (llm_axis|output_axis) to the legacy single-string backend value used
 * by the local-STT response radio (legacy display value) and the LLM backend env var.
 *
 * Bundled handlers (openai / huggingface) ignore the LLM column — they are
 * self-contained. Undefined entries are unsupported combinations.
 */
const PIPELINE_TO_BACKEND = {
  [`${LLM_BACKEND_LLAMA}|${CHATTERBOX_OUTPUT}`]:  { output: CHATTERBOX_OUTPUT,        llm_backend: LLM_BACKEND_LLAMA },
  [`${LLM_BACKEND_LLAMA}|${ELEVENLABS_OUTPUT}`]:  { output: LLAMA_ELEVENLABS_TTS_OUTPUT, llm_backend: LLM_BACKEND_LLAMA },
  [`${LLM_BACKEND_LLAMA}|${GEMINI_TTS_OUTPUT}`]:  { output: GEMINI_TTS_OUTPUT,         llm_backend: LLM_BACKEND_LLAMA },
  [`${LLM_BACKEND_LLAMA}|${XTTS_OUTPUT}`]:        { output: XTTS_OUTPUT,               llm_backend: LLM_BACKEND_LLAMA },
  [`${LLM_BACKEND_LLAMA}|${OPENAI_BACKEND}`]:     { output: OPENAI_BACKEND,            llm_backend: LLM_BACKEND_LLAMA },
  [`${LLM_BACKEND_LLAMA}|${HF_BACKEND}`]:         { output: HF_BACKEND,                llm_backend: LLM_BACKEND_LLAMA },
  [`${LLM_BACKEND_GEMINI}|${CHATTERBOX_OUTPUT}`]: { output: CHATTERBOX_OUTPUT,         llm_backend: LLM_BACKEND_GEMINI },
  [`${LLM_BACKEND_GEMINI}|${ELEVENLABS_OUTPUT}`]: { output: ELEVENLABS_OUTPUT,         llm_backend: LLM_BACKEND_GEMINI },
  [`${LLM_BACKEND_GEMINI}|${XTTS_OUTPUT}`]:       { output: XTTS_OUTPUT,               llm_backend: LLM_BACKEND_GEMINI },
  // gemini_tts + gemini LLM is redundant (uses Gemini for both — handled as llama path)
  // openai/hf realtime do not support a separate LLM backend
};

/**
 * Output values that are only supported with llama.cpp (not Gemini text).
 * These cells are disabled when Gemini text LLM is selected.
 */
const GEMINI_LLM_UNSUPPORTED_OUTPUTS = new Set([GEMINI_TTS_OUTPUT, OPENAI_BACKEND, HF_BACKEND]);

/**
 * Bundled output backends: selecting them auto-selects the matching LLM slot
 * (and vice versa) — the LLM and TTS are inseparable for these handlers.
 */
const BUNDLED_OUTPUT_TO_LLM = {
  [OPENAI_BACKEND]: OPENAI_BACKEND,
  [HF_BACKEND]: HF_BACKEND,
};
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
  [LOCAL_STT_BACKEND]: {
    label: "Local STT",
    formTitle: "Configure Local STT",
    inputLabel: "OPENAI_API_KEY",
    placeholder: "sk-...",
    saveButton: "Save local STT",
    changeButton: "Edit local STT",
    readyTitle: "Local STT ready",
    readyCopy: "Moonshine will transcribe speech on-device, then the selected output backend will generate the spoken response.",
    formCopy: "Choose a local Moonshine model and a separate voice output backend.",
    requiredCredentialsCopy: "Local STT needs credentials or connection details for the selected output backend.",
    note: "Local STT keeps speech recognition on-device, then sends text to the selected voice output backend.",
  },
};
const JOURNEY_META = {
  [HF_BACKEND]: {
    inputLabel: "Hugging Face realtime audio",
    inputCopy: "Speech streams to the Hugging Face voice backend.",
    brainLabel: "Hugging Face response model",
    brainCopy: "The configured endpoint handles speech understanding, tools, and replies.",
    outputLabel: "Hugging Face voice",
    outputCopy: "Audio returns from the built-in server or your direct endpoint.",
  },
  [OPENAI_BACKEND]: {
    inputLabel: "OpenAI Realtime audio",
    inputCopy: "Microphone audio streams directly to OpenAI.",
    brainLabel: "OpenAI Realtime",
    brainCopy: "Realtime handles understanding, personality, tools, and response timing.",
    outputLabel: "OpenAI voice",
    outputCopy: "Speech comes back through the selected OpenAI voice.",
  },
  [GEMINI_BACKEND]: {
    inputLabel: "Gemini Live audio",
    inputCopy: "Microphone audio streams directly to Gemini Live.",
    brainLabel: "Gemini Live",
    brainCopy: "Gemini handles understanding, personality, tools, and response timing.",
    outputLabel: "Gemini Live voice",
    outputCopy: "Speech comes back through the selected Gemini voice.",
  },
  [LOCAL_STT_BACKEND]: {
    inputLabel: "Moonshine STT",
    inputCopy: "Speech recognition runs locally on the robot.",
    brainLabel: "Text response backend",
    brainCopy: "Robot Comic sends transcripts to the selected output backend.",
    outputLabel: "OpenAI voice",
    outputCopy: "Choose OpenAI or Hugging Face today; Gemini Flash 3.1 TTS is reserved as a future route.",
  },
};

function backendHasCredentials(status, backend) {
  if (backend === GEMINI_BACKEND) return !!status.has_gemini_key;
  if (backend === HF_BACKEND) return !!(status.has_hf_connection ?? (status.has_hf_session_url || status.has_hf_ws_url));
  if (backend === LOCAL_STT_BACKEND) return !!status.has_local_stt_key;
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
  if (backend === LOCAL_STT_BACKEND) {
    return status.can_proceed_with_local_stt !== undefined
      ? !!status.can_proceed_with_local_stt
      : backendHasCredentials(status, backend);
  }
  return status.can_proceed_with_openai !== undefined
    ? !!status.can_proceed_with_openai
    : backendHasCredentials(status, backend);
}

function backendMeta(backend) {
  return BACKEND_META[backend] || BACKEND_META[DEFAULT_BACKEND];
}

function journeyMeta(backend, outputBackend = OPENAI_BACKEND) {
  const meta = { ...(JOURNEY_META[backend] || JOURNEY_META[DEFAULT_BACKEND]) };
  if (backend === LOCAL_STT_BACKEND) {
    if (outputBackend === HF_BACKEND) {
      meta.brainLabel = "Hugging Face response backend";
      meta.outputLabel = "Hugging Face voice";
      meta.outputCopy = "Speech comes back through the configured Hugging Face endpoint.";
    } else if (outputBackend === GEMINI_TTS_OUTPUT) {
      meta.brainLabel = "Gemini Flash response backend";
      meta.outputLabel = "Gemini Flash 3.1 TTS";
      meta.outputCopy = "Speech comes back through Gemini 3.1 Flash TTS with the Algenib voice.";
    } else if (outputBackend === CHATTERBOX_OUTPUT) {
      meta.brainLabel = "Ollama LLM (local)";
      meta.outputLabel = "Chatterbox TTS";
      meta.outputCopy = "Text goes to the local Ollama model, then to Chatterbox voice-clone TTS.";
    } else if (outputBackend === ELEVENLABS_OUTPUT) {
      meta.brainLabel = "Gemini Flash response backend";
      meta.outputLabel = "ElevenLabs TTS";
      meta.outputCopy = "Text goes to Gemini Flash, then to the selected ElevenLabs voice.";
    } else if (outputBackend === LLAMA_ELEVENLABS_TTS_OUTPUT) {
      meta.brainLabel = "llama.cpp (local LLM)";
      meta.outputLabel = "ElevenLabs TTS";
      meta.outputCopy = "Text goes to the local llama.cpp server, then to the selected ElevenLabs voice.";
    } else if (outputBackend === XTTS_OUTPUT) {
      meta.brainLabel = "llama.cpp (local LLM)";
      meta.outputLabel = "xtts (LAN)";
      meta.outputCopy = "Text goes to the local LLM, then to the LAN xtts-v2 voice-clone TTS.";
    } else {
      meta.brainLabel = "OpenAI response backend";
      meta.outputLabel = "OpenAI voice";
      meta.outputCopy = "Speech comes back through OpenAI text-in, audio-out realtime.";
    }
  }
  return meta;
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
  // ``backend`` here is the *family* radio value (huggingface / openai /
  // gemini / local_stt).  Translate it into the Phase 4f-canonical pipeline
  // dials before sending.
  const pipelineMode = FAMILY_TO_PIPELINE_MODE[backend] || PIPELINE_MODE_COMPOSABLE;
  const body = { pipeline_mode: pipelineMode, api_key: key };
  if (backend === HF_BACKEND) {
    if (hfMode) body.hf_mode = hfMode;
    if (hfHost) body.hf_host = hfHost;
    if (hfPort !== null && hfPort !== undefined) body.hf_port = hfPort;
  }
  if (backend === LOCAL_STT_BACKEND) {
    const languageEl = document.getElementById("local-stt-language");
    const cacheEl = document.getElementById("local-stt-cache");
    const responseEl = document.getElementById("local-stt-response");
    const llmBackendEl = document.getElementById("local-stt-llm-backend");
    const modelEl = document.getElementById("local-stt-model");
    const updateEl = document.getElementById("local-stt-update");
    body.local_stt_language = (languageEl?.value || "en").trim();
    body.local_stt_cache_dir = (cacheEl?.value || "./cache/moonshine_voice").trim();
    body.audio_input_backend = AUDIO_INPUT_MOONSHINE;
    body.audio_output_backend = legacyResponseToAudioOutput(responseEl?.value || OPENAI_BACKEND);
    body.llm_backend = (llmBackendEl?.value || LLM_BACKEND_LLAMA).trim();
    body.local_stt_model = modelEl?.value || "tiny_streaming";
    const updateInterval = Number.parseFloat((updateEl?.value || "0.35").trim());
    if (Number.isFinite(updateInterval)) body.local_stt_update_interval = updateInterval;
    if (responseEl?.value === CHATTERBOX_OUTPUT) {
      const urlEl = document.getElementById("chatterbox-url");
      const voiceEl = document.getElementById("chatterbox-voice");
      if (urlEl?.value.trim()) body.chatterbox_url = urlEl.value.trim();
      if (voiceEl?.value.trim()) body.chatterbox_voice = voiceEl.value.trim();
    }
    if (
      responseEl?.value === ELEVENLABS_OUTPUT
      || responseEl?.value === LLAMA_ELEVENLABS_TTS_OUTPUT
    ) {
      const keyEl = document.getElementById("elevenlabs-key");
      const voiceEl = document.getElementById("elevenlabs-voice");
      if (keyEl?.value.trim()) body.elevenlabs_api_key = keyEl.value.trim();
      if (voiceEl?.value.trim()) body.elevenlabs_voice = voiceEl.value.trim();
    }
    if (responseEl?.value === XTTS_OUTPUT) {
      const voiceEl = document.getElementById("xtts-voice");
      if (voiceEl?.value.trim()) body.xtts_voice = voiceEl.value.trim();
    }
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

async function clearCrowdHistory() {
  const resp = await fetchWithTimeout("/crowd_history/clear", { method: "POST" }, 5000);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "clear_failed");
  }
  return data;
}

async function getPausePhrases() {
  try {
    const url = new URL("/pause_phrases", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) throw new Error("pause_phrases_failed");
    return await resp.json();
  } catch (e) {
    return null;
  }
}

function parsePhraseTextarea(value) {
  if (typeof value !== "string") return [];
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function savePausePhrases({ stop, resume, shutdown, switch: switchPhrases }) {
  const resp = await fetchWithTimeout(
    "/pause_phrases",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stop: stop.length ? stop : null,
        resume: resume.length ? resume : null,
        shutdown: shutdown.length ? shutdown : null,
        switch: switchPhrases.length ? switchPhrases : null,
      }),
    },
    5000,
  );
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "save_failed");
  }
  return data;
}

async function restartApp() {
  const resp = await fetchWithTimeout("/admin/restart", { method: "POST" }, 5000);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "restart_failed");
  }
  return data;
}

async function getMovementSpeed() {
  try {
    const url = new URL("/movement_speed", window.location.origin);
    url.searchParams.set("_", Date.now().toString());
    const resp = await fetchWithTimeout(url, {}, 3000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function setMovementSpeed(value) {
  const resp = await fetchWithTimeout(
    "/movement_speed",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    },
    3000,
  );
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "set_failed");
  }
  return data;
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
  const journeyInputLabel = document.getElementById("journey-input-label");
  const journeyInputCopy = document.getElementById("journey-input-copy");
  const journeyBrainLabel = document.getElementById("journey-brain-label");
  const journeyBrainCopy = document.getElementById("journey-brain-copy");
  const journeyOutputLabel = document.getElementById("journey-output-label");
  const journeyOutputCopy = document.getElementById("journey-output-copy");
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const configuredTitle = document.getElementById("configured-title");
  const configuredCopy = document.getElementById("configured-copy");
  const crowdHistoryChip = document.getElementById("crowd-history-chip");
  const crowdHistoryPath = document.getElementById("crowd-history-path");
  const crowdHistoryStatus = document.getElementById("crowd-history-status");
  const clearCrowdHistoryBtn = document.getElementById("clear-crowd-history");
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
  const localSttFields = document.getElementById("local-stt-fields");
  const localSttLanguage = document.getElementById("local-stt-language");
  const localSttCache = document.getElementById("local-stt-cache");
  const localSttResponse = document.getElementById("local-stt-response");
  const localSttModel = document.getElementById("local-stt-model");
  const localSttUpdate = document.getElementById("local-stt-update");
  const localSttOutputInputs = Array.from(document.querySelectorAll('input[name="local-stt-output"]'));
  const localSttOutputCards = Array.from(document.querySelectorAll("[data-output-card]"));

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

  function populateLocalSTTFields(status) {
    localSttLanguage.value = status.local_stt_language || "en";
    localSttCache.value = status.local_stt_cache_dir || "./cache/moonshine_voice";
    // Phase 4f: derive the legacy response-radio value from the canonical
    // audio_output_backend + llm_backend pair the server now exposes.
    localSttResponse.value = audioOutputToLegacyResponse(
      status.audio_output_backend || AUDIO_OUTPUT_OPENAI_REALTIME,
      status.llm_backend || LLM_BACKEND_LLAMA,
    );
    localSttModel.innerHTML = "";
    const choices = Array.isArray(status.local_stt_model_choices) && status.local_stt_model_choices.length
      ? status.local_stt_model_choices
      : ["tiny_streaming", "small_streaming"];
    for (const choice of choices) {
      const opt = document.createElement("option");
      opt.value = choice;
      opt.textContent = choice === "small_streaming" ? "Small streaming" : "Tiny streaming";
      localSttModel.appendChild(opt);
    }
    localSttModel.value = choices.includes(status.local_stt_model) ? status.local_stt_model : choices[0];
    localSttUpdate.value = String(status.local_stt_update_interval || 0.35);
    const chatterboxUrl = document.getElementById("chatterbox-url");
    const chatterboxVoice = document.getElementById("chatterbox-voice");
    if (chatterboxUrl) chatterboxUrl.value = status.chatterbox_url || "http://astralplane.lan:8004";
    if (chatterboxVoice) chatterboxVoice.value = status.chatterbox_voice || "don_rickles";
    elevenlabsSavedVoice = status.elevenlabs_voice || "";
    // Restore LLM backend axis from server status (added by #245)
    const llmBackendEl = document.getElementById("local-stt-llm-backend");
    const restoredLlm = status.llm_backend || LLM_BACKEND_LLAMA;
    if (llmBackendEl) llmBackendEl.value = restoredLlm;
    setSelectedLocalSTTOutput(localSttResponse.value || OPENAI_BACKEND);
  }

  let elevenlabsVoicesLoaded = false;
  let elevenlabsSavedVoice = "";
  async function populateElevenLabsVoices() {
    const select = document.getElementById("elevenlabs-voice");
    if (!select) return;
    if (elevenlabsVoicesLoaded) {
      if (elevenlabsSavedVoice) {
        const match = Array.from(select.options).find((o) => o.value === elevenlabsSavedVoice);
        if (match) select.value = elevenlabsSavedVoice;
      }
      return;
    }
    elevenlabsVoicesLoaded = true;
    select.innerHTML = "";
    const loading = document.createElement("option");
    loading.value = "";
    loading.textContent = "(loading voices…)";
    select.appendChild(loading);
    try {
      const url = new URL("/elevenlabs/voices", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 5000);
      if (!resp.ok) throw new Error("voices_failed");
      const data = await resp.json();
      const voices = Array.isArray(data?.voices) ? data.voices : [];
      select.innerHTML = "";
      if (!voices.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "(no voices available)";
        select.appendChild(opt);
        return;
      }
      for (const v of voices) {
        const opt = document.createElement("option");
        const name = (v && typeof v.name === "string") ? v.name : "";
        if (!name) continue;
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
      if (elevenlabsSavedVoice) {
        const match = Array.from(select.options).find((o) => o.value === elevenlabsSavedVoice);
        if (match) select.value = elevenlabsSavedVoice;
      }
    } catch (e) {
      select.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(unable to load voices)";
      select.appendChild(opt);
      // Allow retry on next selection click.
      elevenlabsVoicesLoaded = false;
    }
  }

  let xttsVoicesLoaded = false;
  async function populateXttsVoices() {
    const select = document.getElementById("xtts-voice");
    if (!select) return;
    if (xttsVoicesLoaded) return;
    xttsVoicesLoaded = true;
    select.innerHTML = "";
    const loading = document.createElement("option");
    loading.value = "";
    loading.textContent = "(loading voices…)";
    select.appendChild(loading);
    try {
      const url = new URL("/api/xtts/voices", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 5000);
      if (!resp.ok) throw new Error("voices_failed");
      const data = await resp.json();
      const voices = Array.isArray(data?.voices) ? data.voices : [];
      select.innerHTML = "";
      if (!voices.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "(no voices available)";
        select.appendChild(opt);
        return;
      }
      for (const v of voices) {
        const name = (typeof v === "string") ? v : (v?.name ?? "");
        if (!name) continue;
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
    } catch (e) {
      select.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(unable to load voices)";
      select.appendChild(opt);
      // Allow retry on next selection click.
      xttsVoicesLoaded = false;
    }
  }

  function renderCrowdHistory(status) {
    const count = Number.parseInt(status.crowd_history_count ?? 0, 10) || 0;
    crowdHistoryChip.textContent = `${count} stored`;
    crowdHistoryPath.textContent = status.crowd_history_dir
      ? `Local path: ${status.crowd_history_dir}`
      : "Local path unavailable.";
    clearCrowdHistoryBtn.disabled = count === 0;
  }

  function setSelectedBackend(backend) {
    selectedBackend = [OPENAI_BACKEND, GEMINI_BACKEND, HF_BACKEND, LOCAL_STT_BACKEND].includes(backend)
      ? backend
      : DEFAULT_BACKEND;
    backendInputs.forEach((radio) => {
      radio.checked = radio.value === selectedBackend;
    });
    backendCards.forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.backendCard === selectedBackend);
    });
  }

  /**
   * Map a legacy response-backend string to the (llm, output) pair used by
   * the 3-column picker. The hidden select and llm-backend input stay as the
   * source of truth for the server save path.
   */
  function legacyBackendToAxes(backend) {
    switch (backend) {
      case LLAMA_ELEVENLABS_TTS_OUTPUT:
        return { llm: LLM_BACKEND_LLAMA, output: ELEVENLABS_OUTPUT };
      case GEMINI_TTS_OUTPUT:
        // gemini_tts: historically uses Gemini for LLM too — keep llama slot
        // consistent since we write llm_backend=llama on save for this path.
        return { llm: LLM_BACKEND_LLAMA, output: GEMINI_TTS_OUTPUT };
      case CHATTERBOX_OUTPUT:
        return { llm: LLM_BACKEND_LLAMA, output: CHATTERBOX_OUTPUT };
      case XTTS_OUTPUT:
        return { llm: LLM_BACKEND_LLAMA, output: XTTS_OUTPUT };
      case ELEVENLABS_OUTPUT:
        return { llm: LLM_BACKEND_GEMINI, output: ELEVENLABS_OUTPUT };
      case HF_BACKEND:
        return { llm: HF_BACKEND, output: HF_BACKEND };
      case OPENAI_BACKEND:
      default:
        return { llm: OPENAI_BACKEND, output: OPENAI_BACKEND };
    }
  }

  /**
   * Synchronise the 3-column pipeline UI state given the current output radio
   * value and the llm-backend hidden input.  Also updates the legacy hidden
   * select and the secondary credential panels.
   *
   * @param {string} outputVal  - value of the output (TTS) column radio
   * @param {string} llmVal     - value of the LLM column radio
   */
  /**
   * Resolve the legacy single-string backend value from (llm, output) axes.
   * This is what gets stored in the hidden select and sent to /backend_config.
   */
  function resolveLegacyBackend(llmVal, outputVal) {
    const key = `${llmVal}|${outputVal}`;
    const entry = PIPELINE_TO_BACKEND[key];
    if (entry) return entry.output;
    // Bundled: openai/hf output strings are already the legacy values
    return outputVal;
  }

  function syncPipelineColumns(outputVal, llmVal) {
    // 1. Resolve legacy string and update hidden inputs (server save path)
    const legacyBackend = resolveLegacyBackend(llmVal, outputVal);
    localSttResponse.value = legacyBackend;
    const llmBackendInput = document.getElementById("local-stt-llm-backend");
    if (llmBackendInput) llmBackendInput.value = llmVal;

    // 2. Update output column radio checks and cell highlight
    localSttOutputInputs.forEach((radio) => {
      radio.checked = radio.value === outputVal;
    });
    const outputCells = Array.from(document.querySelectorAll("[data-output-cell]"));
    outputCells.forEach((cell) => {
      cell.classList.toggle("is-selected", cell.dataset.outputCell === outputVal);
    });
    // Legacy output-card highlight (back-compat with any code reading data-output-card)
    localSttOutputCards.forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.outputCard === outputVal);
    });

    // 3. Update LLM column radio checks and cell highlight
    const llmRadios = Array.from(document.querySelectorAll('input[name="pipeline-llm"]'));
    llmRadios.forEach((radio) => {
      radio.checked = radio.value === llmVal;
    });
    const llmCells = Array.from(document.querySelectorAll("[data-llm-cell]"));
    llmCells.forEach((cell) => {
      cell.classList.toggle("is-selected", cell.dataset.llmCell === llmVal);
    });

    // 4. Disable unsupported combos based on LLM selection
    const isBundledLlm = (llmVal === OPENAI_BACKEND || llmVal === HF_BACKEND);
    outputCells.forEach((cell) => {
      const outKey = cell.dataset.outputCell;
      if (isBundledLlm) {
        // Bundled: only the matching output is enabled
        const supported = outKey === llmVal;
        cell.classList.toggle("disabled", !supported);
        const radio = cell.querySelector("input[type=radio]");
        if (radio) radio.disabled = !supported;
      } else {
        // llama / gemini: check support table
        const key = `${llmVal}|${outKey}`;
        const supported = key in PIPELINE_TO_BACKEND;
        cell.classList.toggle("disabled", !supported);
        const radio = cell.querySelector("input[type=radio]");
        if (radio) radio.disabled = !supported;
      }
    });

    // 5. Secondary fields (chatterbox, elevenlabs, xtts)
    // Use the resolved legacy backend since that's what determines UI extras
    const chatterboxFields = document.getElementById("chatterbox-fields");
    if (chatterboxFields) chatterboxFields.style.display = legacyBackend === CHATTERBOX_OUTPUT ? "" : "none";
    const elevenlabsFields = document.getElementById("elevenlabs-fields");
    const usesElevenLabs = legacyBackend === ELEVENLABS_OUTPUT || legacyBackend === LLAMA_ELEVENLABS_TTS_OUTPUT;
    if (elevenlabsFields) elevenlabsFields.style.display = usesElevenLabs ? "" : "none";
    if (usesElevenLabs) {
      populateElevenLabsVoices();
    }
    const xttsFields = document.getElementById("xtts-fields");
    if (xttsFields) xttsFields.style.display = legacyBackend === XTTS_OUTPUT ? "" : "none";
    if (legacyBackend === XTTS_OUTPUT) {
      populateXttsVoices();
    }
  }

  function setSelectedLocalSTTOutput(outputBackend) {
    // Map legacy backend string → axis values, then drive the 3-column UI.
    const axes = legacyBackendToAxes(outputBackend);
    // Resolve the actual output radio value (some legacy strings need mapping)
    const outputRadioVal = (
      outputBackend === LLAMA_ELEVENLABS_TTS_OUTPUT ? ELEVENLABS_OUTPUT : outputBackend
    );
    syncPipelineColumns(outputRadioVal, axes.llm);
  }

  function renderJourneyMap() {
    const meta = journeyMeta(selectedBackend, localSttResponse.value || OPENAI_BACKEND);
    journeyInputLabel.textContent = meta.inputLabel;
    journeyInputCopy.textContent = meta.inputCopy;
    journeyBrainLabel.textContent = meta.brainLabel;
    journeyBrainCopy.textContent = meta.brainCopy;
    journeyOutputLabel.textContent = meta.outputLabel;
    journeyOutputCopy.textContent = meta.outputCopy;
  }

  function renderCredentialPanels(status) {
    const persistedBackend = pipelineModeToFamily(
      status.pipeline_mode || PIPELINE_MODE_HF_REALTIME,
      status.audio_output_backend,
    );
    const activeBackend = status.active_backend || persistedBackend;
    const requiresRestart = !!status.requires_restart;
    const meta = backendMeta(selectedBackend);
    const selectedMatchesPersisted = selectedBackend === persistedBackend;
    const selectedMatchesActive = selectedBackend === activeBackend;
    const localSttUsesHF = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === HF_BACKEND;
    const localSttUsesGeminiTTS = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === GEMINI_TTS_OUTPUT;
    const localSttUsesChatterbox = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === CHATTERBOX_OUTPUT;
    const localSttUsesXtts = selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === XTTS_OUTPUT;
    const localSttUsesElevenLabs = selectedBackend === LOCAL_STT_BACKEND && (
      localSttResponse.value === ELEVENLABS_OUTPUT
      || localSttResponse.value === LLAMA_ELEVENLABS_TTS_OUTPUT
    );
    const localSttUsesOpenAI = selectedBackend === LOCAL_STT_BACKEND && !localSttUsesHF && !localSttUsesGeminiTTS && !localSttUsesChatterbox && !localSttUsesXtts && !localSttUsesElevenLabs;
    const canProceedWithSelectedBackend = localSttUsesHF
      ? backendCanProceed(status, HF_BACKEND)
      : localSttUsesGeminiTTS
        ? backendCanProceed(status, GEMINI_BACKEND)
        : localSttUsesChatterbox
          ? !!(status.can_proceed_with_chatterbox ?? true)
          : localSttUsesXtts
            ? true
            : localSttUsesElevenLabs
              ? !!status.has_elevenlabs_key
              : localSttUsesOpenAI
                ? backendCanProceed(status, OPENAI_BACKEND)
                : backendCanProceed(status, selectedBackend);
    const usesApiKeyForm = selectedBackend === OPENAI_BACKEND || selectedBackend === GEMINI_BACKEND || localSttUsesOpenAI || localSttUsesGeminiTTS;
    const usesHFForm = selectedBackend === HF_BACKEND || localSttUsesHF;
    const usesLocalSTTForm = selectedBackend === LOCAL_STT_BACKEND;
    const supportsForm = usesApiKeyForm || usesHFForm || usesLocalSTTForm;

    renderJourneyMap();
    backendChip.textContent = selectedBackend === persistedBackend ? "Saved" : "Selected";
    backendNote.innerHTML = formatBackendNote(meta.note);

    configuredTitle.textContent = meta.readyTitle;
    configuredCopy.textContent = usesHFForm && selectedBackend !== LOCAL_STT_BACKEND
      ? describeHFConfiguration(status)
      : meta.readyCopy;
    formTitle.textContent = meta.formTitle;
    formCopy.textContent = usesHFForm && selectedBackend !== LOCAL_STT_BACKEND
      ? meta.formCopy
      : canProceedWithSelectedBackend
        ? meta.formCopy
        : meta.requiredCredentialsCopy;
    apiKeyLabel.textContent = localSttUsesGeminiTTS ? "GEMINI_API_KEY" : meta.inputLabel;
    input.placeholder = localSttUsesGeminiTTS ? "AIza..." : meta.placeholder;
    saveBtn.textContent = meta.saveButton;
    changeKeyBtn.textContent = meta.changeButton;

    show(configuredPanel, canProceedWithSelectedBackend && !editingCredentials);
    show(formPanel, supportsForm && (editingCredentials || !canProceedWithSelectedBackend));
    show(apiKeyFields, usesApiKeyForm);
    show(localSttFields, usesLocalSTTForm);
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
        `Backend saved. Restart Robot Comic from the dashboard or desktop app to use ${backendMeta(persistedBackend).label}.`,
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
    pipeline_mode: PIPELINE_MODE_HF_REALTIME,
    audio_input_backend: AUDIO_INPUT_HF,
    audio_output_backend: AUDIO_OUTPUT_HF,
    has_key: false,
    has_openai_key: false,
    has_gemini_key: false,
    has_hf_session_url: false,
    has_hf_ws_url: false,
    has_hf_connection: false,
    hf_connection_mode: "local",
    hf_direct_host: HF_DEFAULT_HOST,
    hf_direct_port: HF_DEFAULT_PORT,
    can_proceed: false,
    can_proceed_with_openai: false,
    can_proceed_with_gemini: false,
    can_proceed_with_hf: false,
    has_local_stt_key: false,
    can_proceed_with_local_stt: false,
    local_stt_language: "en",
    local_stt_cache_dir: "./cache/moonshine_voice",
    local_stt_model: "tiny_streaming",
    local_stt_update_interval: 0.35,
    local_stt_model_choices: ["tiny_streaming", "small_streaming"],
    requires_restart: false,
    crowd_history_dir: "",
    crowd_history_count: 0,
    crowd_history_latest: null,
  };
  populateHFFields(st);
  populateLocalSTTFields(st);
  renderCrowdHistory(st);
  setSelectedBackend(pipelineModeToFamily(st.pipeline_mode, st.audio_output_backend));
  statusEl.textContent = "";
  renderCredentialPanels(st);

  clearCrowdHistoryBtn.addEventListener("click", async () => {
    setStatusMessage(crowdHistoryStatus, "Clearing crowd history...");
    clearCrowdHistoryBtn.disabled = true;
    try {
      const data = await clearCrowdHistory();
      renderCrowdHistory(data);
      const plural = data.removed === 1 ? "" : "s";
      setStatusMessage(crowdHistoryStatus, `Cleared ${data.removed || 0} session file${plural}.`, "ok");
    } catch (e) {
      clearCrowdHistoryBtn.disabled = false;
      setStatusMessage(crowdHistoryStatus, "Failed to clear crowd history.", "error");
    }
  });

  // Pause phrases admin section
  const pauseStopEl = document.getElementById("pause-phrase-stop");
  const pauseResumeEl = document.getElementById("pause-phrase-resume");
  const pauseShutdownEl = document.getElementById("pause-phrase-shutdown");
  const pauseSwitchEl = document.getElementById("pause-phrase-switch");
  const pausePhrasesStatus = document.getElementById("pause-phrases-status");
  const savePausePhrasesBtn = document.getElementById("save-pause-phrases");
  const resetPausePhrasesBtn = document.getElementById("reset-pause-phrases");

  function fillPauseField(textarea, savedList, effectiveList) {
    if (!textarea) return;
    if (Array.isArray(savedList) && savedList.length > 0) {
      textarea.value = savedList.join("\n");
      textarea.placeholder = (effectiveList || []).join(", ");
    } else {
      textarea.value = "";
      textarea.placeholder = (effectiveList || []).join(", ");
    }
  }

  function applyPausePhrasePayload(payload) {
    if (!payload) return;
    const saved = payload.saved || {};
    const effective = payload.effective || {};
    fillPauseField(pauseStopEl, saved.stop, effective.stop);
    fillPauseField(pauseResumeEl, saved.resume, effective.resume);
    fillPauseField(pauseShutdownEl, saved.shutdown, effective.shutdown);
    fillPauseField(pauseSwitchEl, saved.switch, effective.switch);
  }

  (async () => {
    const data = await getPausePhrases();
    if (data && data.ok) {
      applyPausePhrasePayload(data);
    } else {
      setStatusMessage(pausePhrasesStatus, "Could not load saved phrases.", "error");
    }
  })();

  if (savePausePhrasesBtn) {
    savePausePhrasesBtn.addEventListener("click", async () => {
      setStatusMessage(pausePhrasesStatus, "Saving phrases…");
      savePausePhrasesBtn.disabled = true;
      try {
        const data = await savePausePhrases({
          stop: parsePhraseTextarea(pauseStopEl.value),
          resume: parsePhraseTextarea(pauseResumeEl.value),
          shutdown: parsePhraseTextarea(pauseShutdownEl.value),
          switch: parsePhraseTextarea(pauseSwitchEl.value),
        });
        applyPausePhrasePayload(data);
        const message = data.applied_live
          ? "Saved and applied to running session."
          : "Saved. Restart Robot Comic to apply.";
        setStatusMessage(pausePhrasesStatus, message, "ok");
      } catch (e) {
        setStatusMessage(pausePhrasesStatus, "Failed to save phrases.", "error");
      } finally {
        savePausePhrasesBtn.disabled = false;
      }
    });
  }

  if (resetPausePhrasesBtn) {
    resetPausePhrasesBtn.addEventListener("click", async () => {
      setStatusMessage(pausePhrasesStatus, "Resetting to defaults…");
      resetPausePhrasesBtn.disabled = true;
      try {
        pauseStopEl.value = "";
        pauseResumeEl.value = "";
        pauseShutdownEl.value = "";
        pauseSwitchEl.value = "";
        const data = await savePausePhrases({ stop: [], resume: [], shutdown: [], switch: [] });
        applyPausePhrasePayload(data);
        const message = data.applied_live
          ? "Reset to defaults and applied to running session."
          : "Reset to defaults. Restart Robot Comic to apply.";
        setStatusMessage(pausePhrasesStatus, message, "ok");
      } catch (e) {
        setStatusMessage(pausePhrasesStatus, "Failed to reset phrases.", "error");
      } finally {
        resetPausePhrasesBtn.disabled = false;
      }
    });
  }

  // Movement speed slider
  const speedSlider = document.getElementById("movement-speed-slider");
  const speedChip = document.getElementById("movement-speed-chip");
  const speedStatus = document.getElementById("movement-speed-status");

  function renderSpeedChip(value) {
    if (speedChip) speedChip.textContent = Number(value).toFixed(2) + "×";
  }

  (async () => {
    const data = await getMovementSpeed();
    if (!data || !data.ok) {
      if (speedSlider) speedSlider.disabled = true;
      setStatusMessage(speedStatus, "Movement manager unavailable.", "error");
      if (speedChip) speedChip.textContent = "off";
      return;
    }
    if (speedSlider) {
      speedSlider.min = String(data.min);
      speedSlider.max = String(data.max);
      speedSlider.step = String(data.step);
      speedSlider.value = String(data.value);
    }
    renderSpeedChip(data.value);
  })();

  if (speedSlider) {
    let debounceTimer = null;
    let lastSentValue = null;
    const commit = async () => {
      const value = parseFloat(speedSlider.value);
      if (Number.isNaN(value) || value === lastSentValue) return;
      lastSentValue = value;
      try {
        const data = await setMovementSpeed(value);
        if (data && data.ok) {
          renderSpeedChip(data.value);
          setStatusMessage(speedStatus, "");
        }
      } catch (e) {
        setStatusMessage(speedStatus, "Failed to update speed.", "error");
      }
    };
    speedSlider.addEventListener("input", () => {
      renderSpeedChip(speedSlider.value);
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(commit, 150);
    });
    speedSlider.addEventListener("change", () => {
      if (debounceTimer) clearTimeout(debounceTimer);
      commit();
    });
  }

  // Restart Comic admin button
  const restartBtn = document.getElementById("restart-app");
  const restartStatus = document.getElementById("restart-status");
  if (restartBtn) {
    restartBtn.addEventListener("click", async () => {
      if (!window.confirm("Restart Robot Comic now? The app will shut down gracefully and the autostart service should relaunch it.")) {
        return;
      }
      setStatusMessage(restartStatus, "Requesting restart…");
      restartBtn.disabled = true;
      try {
        const data = await restartApp();
        setStatusMessage(restartStatus, data.message || "Restart requested.", "ok");
      } catch (e) {
        restartBtn.disabled = false;
        setStatusMessage(restartStatus, "Restart hook unavailable.", "error");
      }
    });
  }

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
  localSttLanguage.addEventListener("input", () => localSttLanguage.classList.remove("error"));
  localSttCache.addEventListener("input", () => localSttCache.classList.remove("error"));
  localSttUpdate.addEventListener("input", () => localSttUpdate.classList.remove("error"));
  localSttResponse.addEventListener("change", () => {
    editingCredentials = false;
    setStatusMessage(statusEl, "");
    setSelectedLocalSTTOutput(localSttResponse.value);
    renderCredentialPanels(st);
  });

  // Output (TTS) column radio listeners
  localSttOutputInputs.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.disabled) return;
      // Resolve current LLM value from the LLM column
      const llmBackendEl = document.getElementById("local-stt-llm-backend");
      const currentLlm = llmBackendEl?.value || LLM_BACKEND_LLAMA;
      // For bundled outputs, auto-select the matching LLM slot
      const bundledLlm = BUNDLED_OUTPUT_TO_LLM[radio.value];
      const llmVal = bundledLlm || currentLlm;
      syncPipelineColumns(radio.value, llmVal);
      setStatusMessage(statusEl, "");
      renderCredentialPanels(st);
    });
  });

  // LLM column radio listeners
  Array.from(document.querySelectorAll('input[name="pipeline-llm"]')).forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.disabled) return;
      // Resolve current output value from the output column
      const currentOutput = localSttResponse.value || OPENAI_BACKEND;
      // Map legacy string to output radio value if needed
      const outputRadioVal = currentOutput === LLAMA_ELEVENLABS_TTS_OUTPUT
        ? ELEVENLABS_OUTPUT
        : currentOutput;
      // For bundled LLM slots, auto-select the matching output
      if (radio.value === OPENAI_BACKEND || radio.value === HF_BACKEND) {
        syncPipelineColumns(radio.value, radio.value);
      } else {
        // If the current output is unsupported by the new LLM, fall back to chatterbox
        const key = `${radio.value}|${outputRadioVal}`;
        const supportedOutput = (key in PIPELINE_TO_BACKEND) ? outputRadioVal : CHATTERBOX_OUTPUT;
        syncPipelineColumns(supportedOutput, radio.value);
      }
      setStatusMessage(statusEl, "");
      renderCredentialPanels(st);
    });
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
    if (selectedBackend === HF_BACKEND || (selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === HF_BACKEND)) {
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
      if (selectedBackend === HF_BACKEND) return;
      setStatusMessage(statusEl, "Saved. Reloading…", "ok");
      window.location.reload();
      return;
    }

    if (selectedBackend === LOCAL_STT_BACKEND) {
      const language = (localSttLanguage.value || "").trim();
      const cacheDir = (localSttCache.value || "").trim();
      const updateInterval = Number.parseFloat((localSttUpdate.value || "").trim());
      localSttLanguage.classList.remove("error");
      localSttCache.classList.remove("error");
      localSttUpdate.classList.remove("error");
      if (!language || language.includes("/") || language.includes("\\")) {
        localSttLanguage.classList.add("error");
        setStatusMessage(statusEl, "Enter a valid language code, such as en.", "warn");
        return;
      }
      if (!cacheDir) {
        localSttCache.classList.add("error");
        setStatusMessage(statusEl, "Enter a writable model cache path.", "warn");
        return;
      }
      if (!Number.isFinite(updateInterval) || updateInterval < 0.1 || updateInterval > 2.0) {
        localSttUpdate.classList.add("error");
        setStatusMessage(statusEl, "Use an update interval from 0.1 to 2.0 seconds.", "warn");
        return;
      }
    }

    // Chatterbox needs no API key — save directly
    if (selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === CHATTERBOX_OUTPUT) {
      setStatusMessage(statusEl, "Saving Chatterbox config...");
      try {
        await saveBackendConfig(selectedBackend, {});
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        setStatusMessage(statusEl, "Failed to save Chatterbox config. Please try again.", "error");
      }
      return;
    }

    // xtts (LAN) needs no API key — save directly
    if (selectedBackend === LOCAL_STT_BACKEND && localSttResponse.value === XTTS_OUTPUT) {
      setStatusMessage(statusEl, "Saving xtts config...");
      try {
        await saveBackendConfig(selectedBackend, {});
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        setStatusMessage(statusEl, "Failed to save xtts config. Please try again.", "error");
      }
      return;
    }

    // ElevenLabs reads its API key + voice from dedicated fields, so don't fall
    // through to the main `#api-key` validation path.
    if (
      selectedBackend === LOCAL_STT_BACKEND
      && (localSttResponse.value === ELEVENLABS_OUTPUT || localSttResponse.value === LLAMA_ELEVENLABS_TTS_OUTPUT)
    ) {
      const elevenlabsKeyEl = document.getElementById("elevenlabs-key");
      const hasSavedKey = !!st.has_elevenlabs_key;
      const enteredKey = (elevenlabsKeyEl?.value || "").trim();
      if (!hasSavedKey && !enteredKey) {
        setStatusMessage(statusEl, "Enter your ELEVENLABS_API_KEY.", "warn");
        elevenlabsKeyEl?.classList.add("error");
        return;
      }
      elevenlabsKeyEl?.classList.remove("error");
      setStatusMessage(statusEl, "Saving ElevenLabs config...");
      try {
        await saveBackendConfig(selectedBackend, {});
        setStatusMessage(statusEl, "Saved. Reloading…", "ok");
        window.location.reload();
      } catch (e) {
        if (e.message === "empty_key") {
          setStatusMessage(statusEl, "Enter your ELEVENLABS_API_KEY.", "warn");
          elevenlabsKeyEl?.classList.add("error");
        } else {
          setStatusMessage(statusEl, "Failed to save ElevenLabs config. Please try again.", "error");
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
    const needsOpenAIValidation = (selectedBackend === OPENAI_BACKEND) ||
      (selectedBackend === LOCAL_STT_BACKEND && !localSttUsesGeminiTTS);
    setStatusMessage(statusEl, needsOpenAIValidation ? "Validating API key..." : "Saving token...");
    input.classList.remove("error");
    try {
      if (needsOpenAIValidation) {
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
      if (needsOpenAIValidation && e.message === "invalid_api_key") {
        setStatusMessage(statusEl, "Invalid API key. Please check your key and try again.", "error");
      } else {
        setStatusMessage(
          statusEl,
          localSttUsesGeminiTTS || selectedBackend === GEMINI_BACKEND
            ? "Failed to save Gemini token. Please try again."
            : "Failed to validate/save key. Please try again.",
          "error",
        );
      }
    }
  });

  if (
    !(st.can_proceed ?? backendCanProceed(st, pipelineModeToFamily(st.pipeline_mode, st.audio_output_backend)))
    || st.requires_restart
  ) {
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

    // ElevenLabs voice catalog table (#304)
    const catalogTbody = document.querySelector("#voice-catalog-table tbody");
    const catalogStatus = document.getElementById("voice-catalog-status");
    const catalogReload = document.getElementById("reload-voice-catalog");
    async function loadVoiceCatalog() {
      if (!catalogTbody) return;
      catalogStatus.textContent = "Loading...";
      catalogTbody.innerHTML = "";
      try {
        const resp = await fetch("/api/voices/catalog");
        if (resp.status === 503) {
          catalogStatus.textContent = "ElevenLabs not configured.";
          return;
        }
        if (!resp.ok) {
          catalogStatus.textContent = `Failed (${resp.status})`;
          return;
        }
        const data = await resp.json();
        const voices = Array.isArray(data.voices) ? data.voices : [];
        for (const v of voices) {
          const tr = document.createElement("tr");
          const nameTd = document.createElement("td");
          nameTd.textContent = v.name || "";
          const idTd = document.createElement("td");
          idTd.textContent = v.voice_id || "";
          idTd.classList.add("mono");
          const catTd = document.createElement("td");
          catTd.textContent = v.category || "";
          tr.appendChild(nameTd);
          tr.appendChild(idTd);
          tr.appendChild(catTd);
          catalogTbody.appendChild(tr);
        }
        catalogStatus.textContent = `${voices.length} voice${voices.length === 1 ? "" : "s"} cached.`;
      } catch (e) {
        catalogStatus.textContent = `Error: ${e.message || e}`;
      }
    }
    if (catalogReload) catalogReload.addEventListener("click", loadVoiceCatalog);
    loadVoiceCatalog();

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

// ── Battery indicator ─────────────────────────────────────────────────────────

function _batteryIcon(source, percent) {
  // Simple unicode battery icon chosen by level.
  if (source === "sim" || source === "unknown" || percent === null || percent === undefined) {
    return "🔋"; // 🔋 generic
  }
  if (percent < 20) return "🪫"; // 🪫 empty
  if (percent < 50) return "🔋"; // 🔋 mid
  return "🔋"; // 🔋 full
}

function _applyBatteryUI(data) {
  const el = document.getElementById("battery-indicator");
  if (!el) return;

  const source = data.source || "unknown";

  // Hide the indicator entirely in sim mode or if source is unknown
  if (source === "sim" || source === "unknown") {
    el.hidden = true;
    return;
  }

  const percent = data.percent;
  const charging = data.charging;

  // Remove all level classes
  el.classList.remove("battery--full", "battery--med", "battery--low");

  let levelClass = "battery--full";
  if (percent !== null && percent !== undefined) {
    if (percent < 20) levelClass = "battery--low";
    else if (percent < 50) levelClass = "battery--med";
  }
  el.classList.add(levelClass);

  const icon = charging ? "⚡" : _batteryIcon(source, percent);
  const label =
    percent !== null && percent !== undefined
      ? `${icon} ${percent}%${charging ? " ⚡" : ""}`
      : `${icon} --`;
  el.textContent = label;
  el.hidden = false;
}

let _batteryPollTimer = null;

async function pollBattery() {
  try {
    const resp = await fetchWithTimeout("/api/battery", {}, 4000);
    if (resp.ok) {
      const data = await resp.json();
      _applyBatteryUI(data);
    }
  } catch (_) {
    // Silently ignore — battery is informational only
  }
  _batteryPollTimer = setTimeout(pollBattery, 30000);
}

window.addEventListener("DOMContentLoaded", init);
window.addEventListener("DOMContentLoaded", () => { pollBattery(); });
