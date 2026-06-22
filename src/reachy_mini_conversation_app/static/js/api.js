/** HTTP client for all calls to the settings backend. */

const DEFAULT_TIMEOUT_MS = 8000;

class HttpError extends Error {
  constructor(status, body, message) {
    super(message || `HTTP ${status}`);
    this.body = body;
  }
}

/** fetch with timeout and JSON decoding; throws HttpError on non-2xx. */
async function request(method, url, { body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method,
      signal: controller.signal,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await response.text();
    let json = null;
    if (text) {
      try {
        json = JSON.parse(text);
      } catch {
        json = { raw: text };
      }
    }
    if (!response.ok) {
      throw new HttpError(response.status, json, json?.error || response.statusText);
    }
    return json;
  } finally {
    clearTimeout(timer);
  }
}

const STARTUP_POLL_MS = 2000;
const STARTUP_DEADLINE_MS = 90000;

/** Retry a request while the backend is still registering its routes at startup. */
export async function untilReady(requestFn, signal, onRetry) {
  const deadline = Date.now() + STARTUP_DEADLINE_MS;
  let notified = false;
  for (;;) {
    try {
      return await requestFn();
    } catch (error) {
      if (signal.aborted || Date.now() >= deadline) throw error;
      if (!notified) {
        notified = true;
        onRetry?.();
      }
    }
    await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_MS));
    if (signal.aborted) throw new Error("view unmounted");
  }
}

export const getStatus = () => request("GET", "/status");

export const saveBackendConfig = (payload) =>
  request("POST", "/backend_config", { body: payload });

export const listPersonalities = () => request("GET", "/personalities");
export const loadPersonality = (name) =>
  request("GET", `/personalities/load?name=${encodeURIComponent(name)}`);
export const savePersonality = (payload) =>
  request("POST", "/personalities/save", { body: payload });
export const applyPersonality = (name, { persist = false } = {}) =>
  request("POST", "/personalities/apply", { body: { name, persist } });

export const getMicState = () => request("GET", "/mic");
export const setMicMuted = (muted) => request("POST", "/mic", { body: { muted } });

export const listVoices = () => request("GET", "/voices");
export const getCurrentVoice = () => request("GET", "/voices/current");
export const applyVoice = (voice) =>
  request("POST", "/voices/apply", { body: { voice } });

/** Backend error codes that need friendlier copy than the raw code. */
const ERROR_MESSAGES = Object.freeze({
  invalid_backend: "Unknown backend selected.",
  empty_key: "An API key is required for this backend.",
  empty_hf_host: "Enter a Hugging Face host.",
  invalid_hf_host: "That Hugging Face host doesn't look right.",
  invalid_hf_port: "That Hugging Face port doesn't look right.",
  invalid_hf_mode: "Unknown Hugging Face mode.",
  missing_hf_session_url: "Couldn't reach the Hugging Face Space. Check it's running.",
  invalid_name: "Enter a valid profile name.",
  missing_voice: "Choose a voice first.",
  profile_locked: "Profile switching is locked by the administrator.",
  loop_unavailable: "Reachy is still starting up. Try again in a moment.",
});

/** Map a thrown error to user-facing copy, falling back to its raw message. */
export function describeError(error) {
  const code = error?.body?.error;
  return ERROR_MESSAGES[code] || error?.message || String(error);
}
