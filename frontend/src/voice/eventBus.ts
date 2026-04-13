export interface VoiceEvents {
  "tts:start": Record<string, never>;
  "tts:done": Record<string, never>;
  "tts:stop": Record<string, never>;
  "response:done": Record<string, never>;
  "stt:partial": { text: string };
  "stt:complete": { text: string };
  "vad:start": Record<string, never>;
  "vad:end": Record<string, never>;
  "tool:call": { name: string; args: Record<string, unknown> };
  "tool:result": { name: string; result: unknown };
  "gate:muted": Record<string, never>;
  "gate:unmuted": Record<string, never>;
  "bargein:trigger": Record<string, never>;
  "bot:audio_active": Record<string, never>;
  "bot:audio_silent": Record<string, never>;
}

type EventName = keyof VoiceEvents;
type Listener<K extends EventName> = (payload: VoiceEvents[K]) => void;

export class VoiceEventBus {
  private listeners = new Map<string, Set<Function>>();

  on<K extends EventName>(event: K, fn: Listener<K>): () => void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(fn);
    return () => this.listeners.get(event)?.delete(fn);
  }

  off<K extends EventName>(event: K, fn: Listener<K>): void {
    this.listeners.get(event)?.delete(fn);
  }

  emit<K extends EventName>(event: K, payload: VoiceEvents[K]): void {
    this.listeners.get(event)?.forEach((fn) => fn(payload));
  }

  removeAll(): void {
    this.listeners.clear();
  }
}

export const voiceEventBus = new VoiceEventBus();
