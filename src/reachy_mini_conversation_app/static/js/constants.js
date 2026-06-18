/** Shared constants for the modern web UI. Keep backend strings in sync with config.py. */

export const BACKENDS = Object.freeze({
  HUGGINGFACE: "huggingface",
  OPENAI: "openai",
  GEMINI: "gemini",
});

export const BUILT_IN_DEFAULT_OPTION = "(built-in default)";

export const AVATAR_BY_PROFILE = Object.freeze({
  bored_teenager: "bored-teenager.svg",
  captain_circuit: "captain-circuit.svg",
  chess_coach: "chess-coach.svg",
  cosmic_kitchen: "cosmic-kitchen.svg",
  default: "default.svg",
  hype_bot: "hype-bot.svg",
  mad_scientist_assistant: "mad-scientist.svg",
  mars_rover: "mars-rover.svg",
  nature_documentarian: "nature-doc.svg",
  noir_detective: "noir-detective.svg",
  sorry_bro: "sorry-bro.svg",
  time_traveler: "time-traveler.svg",
  victorian_butler: "victorian-butler.svg",
});

export const ORB_STATES = Object.freeze({
  MUTED: "muted",
  IDLE: "idle",
  CONNECTING: "connecting",
  LISTENING: "listening",
  THINKING: "thinking",
  SPEAKING: "speaking",
  ERROR: "error",
});

export const GLOW_BY_STATE = Object.freeze({
  [ORB_STATES.MUTED]: "#94a3b8",      // paused, waiting for the user
  [ORB_STATES.IDLE]: "#34d399",       // ready / breathing
  [ORB_STATES.CONNECTING]: "#facc15", // negotiating
  [ORB_STATES.LISTENING]: "#22d3ee",  // user speaks
  [ORB_STATES.THINKING]: "#f59e0b",   // model composes
  [ORB_STATES.SPEAKING]: "#8b7dff",   // model speaks
  [ORB_STATES.ERROR]: "#ff6a75",
});

export const ROUTES = Object.freeze({
  TALK: "#/",
  PERSONALITIES: "#/personalities",
  SETTINGS: "#/settings",
});

export function avatarFor(profileName) {
  const file = AVATAR_BY_PROFILE[profileName] || AVATAR_BY_PROFILE.default;
  return `/static/avatars/${file}`;
}
