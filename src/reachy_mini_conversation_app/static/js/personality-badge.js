/**
 * Header personality badge: avatar + "Personality" label + active profile name.
 * Lives in the app shell (index.html) so it persists across view changes.
 */

import { avatarFor } from "./constants.js";
import { prettifyProfileName } from "./ui.js";

let rootEl = null;
let nameEl = null;
let avatarImg = null;

/** Bind setters to the static markup. Safe to call multiple times. */
export function mountPersonalityBadge(headerRoot = document) {
  const next = headerRoot.querySelector('[data-component="personality-badge"]');
  if (!next) return;
  rootEl = next;
  nameEl = next.querySelector(".app-shell__personality-name");
  avatarImg = next.querySelector(".app-shell__personality-avatar img");
}

/** Update the badge content. Pass a falsy name to keep the previous value. */
export function setPersonality(rawName) {
  if (!rootEl || !nameEl || !avatarImg) return;
  if (!rawName) return;
  const cleanName = String(rawName).replace(/^user_personalities\//, "");
  nameEl.textContent = prettifyProfileName(rawName);
  avatarImg.src = avatarFor(cleanName);
}

export function showPersonalityBadge() {
  if (rootEl) rootEl.hidden = false;
}

export function hidePersonalityBadge() {
  if (rootEl) rootEl.hidden = true;
}
