/**
 * Header personality badge: avatar + "Personality" label + active profile name.
 * Lives in the app shell (index.html) so it persists across view changes.
 */

import { avatarFor } from "./constants.js";
import { prettifyProfileName } from "./ui.js";

let rootEl = null;
let rowEl = null;
let nameEl = null;
let avatarImg = null;

const DEBUG_PREFIX = "[personality-badge]";

/** Bind setters to the static markup. Safe to call multiple times. */
export function mountPersonalityBadge(headerRoot = document) {
  const next = headerRoot.querySelector('[data-component="personality-badge"]');
  if (!next) return;
  rootEl = next;
  rowEl = next.closest('[data-component="personality-row"]');
  nameEl = next.querySelector(".app-shell__personality-name");
  avatarImg = next.querySelector(".app-shell__personality-avatar img");
  console.info(DEBUG_PREFIX, "mounted", JSON.stringify({
    hasBadge: Boolean(rootEl),
    hasRow: Boolean(rowEl),
    hasDefaultAction: Boolean(headerRoot.querySelector('[data-component="default-personality-action"]')),
  }));
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
  if (rowEl) rowEl.hidden = false;
  else if (rootEl) rootEl.hidden = false;
  console.info(DEBUG_PREFIX, "show", JSON.stringify({
    rowHidden: rowEl?.hidden,
    badgeHidden: rootEl?.hidden,
  }));
}

export function hidePersonalityBadge() {
  if (rowEl) rowEl.hidden = true;
  else if (rootEl) rootEl.hidden = true;
  console.info(DEBUG_PREFIX, "hide", JSON.stringify({
    rowHidden: rowEl?.hidden,
    badgeHidden: rootEl?.hidden,
  }));
}
