/**
 * WebRTC + SSE conversation hook.
 *
 * Audio flows through a native WebRTC connection (fastrtc), giving us
 * browser-level echo cancellation for free. Control messages (transcripts,
 * tool events, state) arrive via a Server-Sent Events stream.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { WebRTCSession } from "../audio/webrtcSession";
import { voiceEventBus } from "../voice/eventBus";
import type { AppSettings } from "../config/settings";
import type { useChat } from "./useChat";

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

const DEBUG = import.meta.env.DEV;

function getBaseUrl(settings: AppSettings): string {
  if (settings.daemonUrl) return settings.daemonUrl.replace(/\/$/, "");
  return `${location.protocol}//${location.host}`;
}

function getEventsUrl(settings: AppSettings, webrtcId: string): string {
  const base = getBaseUrl(settings);
  return `${base}/api/events?webrtc_id=${encodeURIComponent(webrtcId)}`;
}

let _idCounter = 0;
function generateWebrtcId(): string {
  return `webrtc-${Date.now()}-${++_idCounter}`;
}

export function useConversation(
  settings: AppSettings,
  chat: ReturnType<typeof useChat>,
) {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  const sessionRef = useRef<WebRTCSession | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const cancelledRef = useRef(false);
  const pausedRef = useRef(false);

  const chatRef = useRef(chat);
  chatRef.current = chat;

  const connect = useCallback(async () => {
    if (DEBUG) console.log("[useConversation] connect()");
    if (sessionRef.current) return;

    cancelledRef.current = false;
    setStatus("connecting");
    setError(null);
    chatRef.current.clear();

    try {
      const webrtcId = generateWebrtcId();
      const baseUrl = getBaseUrl(settings);

      // Establish WebRTC for audio
      const session = new WebRTCSession(webrtcId);
      await session.connect(baseUrl);
      sessionRef.current = session;

      if (cancelledRef.current) {
        session.disconnect();
        return;
      }

      // SSE for control messages
      const eventsUrl = getEventsUrl(settings, webrtcId);
      if (DEBUG) console.log("[useConversation] SSE connecting to", eventsUrl);

      const es = new EventSource(eventsUrl);
      eventSourceRef.current = es;

      es.onmessage = (e: MessageEvent) => {
        if (pausedRef.current) return;
        try {
          const msg = JSON.parse(e.data);
          handleServerMessage(msg);
        } catch {
          /* ignore malformed JSON */
        }
      };

      es.onerror = () => {
        if (DEBUG) console.log("[useConversation] SSE error/closed");
        // EventSource auto-reconnects; only treat as fatal if we're disconnecting
        if (cancelledRef.current) {
          cleanup();
          setStatus("disconnected");
        }
      };

      setStatus("connected");
    } catch (err) {
      if (cancelledRef.current) return;
      if (DEBUG) console.error("[useConversation] connect error:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
      cleanup();
    }
  }, [settings.daemonUrl]);

  function handleServerMessage(msg: Record<string, unknown>) {
    const type = msg.type as string;

    switch (type) {
      case "state": {
        const state = msg.state as string;
        if (state === "interrupt") {
          voiceEventBus.emit("tts:stop", {});
        } else if (state === "response_done") {
          voiceEventBus.emit("response:done", {});
        } else if (state === "vad_start") {
          chatRef.current.reserveUserMessage();
          voiceEventBus.emit("vad:start", {});
        } else if (state === "vad_end") {
          voiceEventBus.emit("vad:end", {});
        }
        break;
      }

      case "transcript": {
        const role = msg.role as string;
        const content = (msg.content as string) ?? "";
        if (role === "user") {
          chatRef.current.handleUserTranscript(content, true);
        } else if (role === "user_partial") {
          chatRef.current.handleUserTranscript(content, false);
        } else if (role === "assistant") {
          chatRef.current.handleAssistantTranscript(content, true);
        }
        break;
      }

      case "tool": {
        const title = (msg.title as string) ?? "";
        const content = (msg.content as string) ?? "";
        const toolStatus = (msg.status as "running" | "done") ?? "done";
        const callId = (msg.call_id as string) ?? undefined;
        chatRef.current.addToolMessage(title, content, toolStatus, callId);
        break;
      }

      case "image": {
        const dataUrl = msg.data as string;
        if (dataUrl) chatRef.current.attachImageToLastTool(dataUrl);
        break;
      }

      case "error": {
        const content = (msg.content as string) ?? "Unknown error";
        chatRef.current.addErrorMessage(content);
        break;
      }

      default:
        if (DEBUG) console.log("[useConversation] unknown message type:", type);
    }
  }

  function cleanup() {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    sessionRef.current?.disconnect();
    sessionRef.current = null;
    setIsMuted(false);
    setIsPaused(false);
  }

  const disconnect = useCallback(() => {
    cancelledRef.current = true;
    cleanup();
    setStatus("disconnected");
  }, []);

  const toggleMute = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;
    const wasMuted = session.isMutedState();
    session.setMuted(!wasMuted);
    setIsMuted(!wasMuted);
  }, []);

  const togglePause = useCallback(() => {
    setIsPaused((prev) => {
      const next = !prev;
      pausedRef.current = next;
      const session = sessionRef.current;
      if (next) {
        session?.setMuted(true);
        setIsMuted(true);
        voiceEventBus.emit("tts:stop", {});
      } else {
        session?.setMuted(false);
        setIsMuted(false);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const onBeforeUnload = () => { cleanup(); };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      cancelledRef.current = true;
      cleanup();
    };
  }, []);

  const getLocalStream = useCallback(
    (): MediaStream | null => sessionRef.current?.getLocalStream() ?? null,
    [],
  );

  return {
    status,
    error,
    isMuted,
    isPaused,
    connect,
    disconnect,
    toggleMute,
    togglePause,
    getLocalStream,
  };
}
