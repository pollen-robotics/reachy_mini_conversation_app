/** JSON-RPC-over-WebSocket client for the settings backend (/rpc).
 *
 * One control surface: the same JSON-RPC the daemon relays to remote WebRTC
 * clients. Exposes rpcCall() + subscribe() and reimplements the settings API on
 * top of them, preserving the exported signatures the views rely on. The error
 * shape (`error.body.error` = stable reason) is kept so describeError() and the
 * views work unchanged. `backend_config` is the one holdout still on REST.
 */

const DEFAULT_TIMEOUT_MS = 8000;

const RPC_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/rpc`;

class RpcError extends Error {
  constructor(message, reason) {
    super(message || reason || "rpc error");
    // Views branch on error.body.error (the stable reason); keep that contract.
    this.body = { error: reason };
    this.reason = reason;
  }
}

let socket = null;
let connecting = null;
let rpcCounter = 0;
const pending = new Map(); // id -> { resolve, reject, timer }
const subscribers = new Map(); // method -> Set<cb>

/** Open (or reuse) the shared /rpc socket. Resolves once OPEN, rejects on fail. */
function connect() {
  if (socket && socket.readyState === WebSocket.OPEN) return Promise.resolve();
  if (connecting) return connecting;
  connecting = new Promise((resolve, reject) => {
    let opened = false;
    const ws = new WebSocket(RPC_URL);
    socket = ws;
    ws.onopen = () => {
      opened = true;
      connecting = null;
      resolve();
    };
    ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data));
    ws.onclose = () => {
      socket = null;
      connecting = null;
      for (const p of pending.values()) {
        clearTimeout(p.timer);
        p.reject(new RpcError("connection closed", "disconnected"));
      }
      pending.clear();
      if (!opened) reject(new RpcError("cannot reach /rpc", "disconnected"));
      // Keep the event stream alive across drops while anyone is listening.
      else if (subscribers.size > 0) setTimeout(() => connect().catch(() => {}), 1000);
    };
  });
  return connecting;
}

function handleMessage(msg) {
  if (msg.id != null && ("result" in msg || "error" in msg)) {
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    clearTimeout(p.timer);
    if (msg.error) p.reject(new RpcError(msg.error.message, msg.error.data?.reason));
    else p.resolve(msg.result);
    return;
  }
  if (typeof msg.method === "string") {
    const cbs = subscribers.get(msg.method);
    if (cbs) for (const cb of cbs) {
      try {
        cb(msg.params || {});
      } catch (e) {
        console.error(`subscribe(${msg.method}) callback threw:`, e);
      }
    }
  }
}

/** Call a JSON-RPC method and await its result. Rejects with RpcError. */
export async function rpcCall(method, params = {}, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  await connect();
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    throw new RpcError("not connected", "disconnected");
  }
  const id = `ui-${++rpcCounter}`;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new RpcError(`timed out: ${method}`, "timeout"));
    }, timeoutMs);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
  });
}

/** Subscribe to a one-way notification (event). Returns an unsubscribe fn. */
export function subscribe(method, cb) {
  let set = subscribers.get(method);
  if (!set) {
    set = new Set();
    subscribers.set(method, set);
  }
  set.add(cb);
  connect().catch(() => {});
  return () => {
    subscribers.get(method)?.delete(cb);
  };
}

const STARTUP_POLL_MS = 2000;
const STARTUP_DEADLINE_MS = 90000;

/** Retry a request while the backend is still coming up at startup. */
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

export const getStatus = () => rpcCall("conversation.status");

export const listPersonalities = () => rpcCall("personalities.list");
export const loadPersonality = (name) => rpcCall("personalities.load", { name });
export const savePersonality = (payload) => rpcCall("personalities.save", payload);
export const applyPersonality = (name, { persist = false } = {}) =>
  rpcCall("personalities.apply", { name, persist });
export const deletePersonality = (name) => rpcCall("personalities.delete", { name });

export const getMicState = () => rpcCall("conversation.mic", {});
export const setMicMuted = (muted) => rpcCall("conversation.mic", { muted });

export const listVoices = () => rpcCall("voices.list");
export const getCurrentVoice = () => rpcCall("voices.current");
export const applyVoice = (voice) => rpcCall("voices.apply", { voice });

export const saveBackendConfig = (payload) => rpcCall("backend.config", payload);

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
  profile_in_use: "This personality is active or set to load at startup. Switch to another one first.",
  not_deletable: "This personality can't be deleted.",
  loop_unavailable: "Reachy is still starting up. Try again in a moment.",
});

/** Map a thrown error to user-facing copy, falling back to its raw message. */
export function describeError(error) {
  const code = error?.body?.error;
  return ERROR_MESSAGES[code] || error?.message || String(error);
}
