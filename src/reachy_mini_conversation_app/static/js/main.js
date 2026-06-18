/** Bootstrap: wire the shell header to the hash router and mount views on demand. */

import { ROUTES } from "./constants.js";
import { createRouter } from "./router.js";
import {
  hidePersonalityBadge,
  mountPersonalityBadge,
  showPersonalityBadge,
} from "./personality-badge.js";
import { $ } from "./ui.js";
import { mountHomeView } from "./views/home.js";
import { mountTalkView } from "./views/talk.js";
import { mountSettingsView } from "./views/settings.js";

const SETTINGS_RETURN_KEY = "settings-return-route";

/** Survives reloads while on /settings (sessionStorage may be unavailable when embedded). */
function readSettingsReturn() {
  try {
    return sessionStorage.getItem(SETTINGS_RETURN_KEY) === ROUTES.PERSONALITIES ? ROUTES.PERSONALITIES : ROUTES.TALK;
  } catch {
    return ROUTES.TALK;
  }
}

function storeSettingsReturn(route) {
  try {
    sessionStorage.setItem(SETTINGS_RETURN_KEY, route);
  } catch {
    // in-memory fallback is enough
  }
}

/** Honor the desktop app's ?theme=dark|light, overriding prefers-color-scheme. */
function applyEmbeddedTheme() {
  const theme = new URLSearchParams(window.location.search).get("theme");
  if (theme !== "dark" && theme !== "light") return;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

function boot() {
  applyEmbeddedTheme();

  const outlet = $("#view-outlet");
  if (!outlet) {
    console.error("#view-outlet missing from index.html");
    return;
  }

  const router = createRouter(
    {
      [ROUTES.TALK]: (ctx) => mountTalkView(ctx),
      [ROUTES.PERSONALITIES]: (ctx) => mountHomeView({ ...ctx, navigate: router.navigate }),
      [ROUTES.SETTINGS]: (ctx) => mountSettingsView(ctx),
    },
    { fallback: ROUTES.TALK, outlet }
  );

  // Settings is an overlay: closing it returns to where it was opened
  let settingsReturn = readSettingsReturn();
  const gear = $('[data-action="open-settings"]');
  if (gear) {
    gear.addEventListener("click", () => {
      if (window.location.hash === ROUTES.SETTINGS) {
        router.navigate(settingsReturn);
      } else {
        settingsReturn = window.location.hash === ROUTES.PERSONALITIES ? ROUTES.PERSONALITIES : ROUTES.TALK;
        storeSettingsReturn(settingsReturn);
        router.navigate(ROUTES.SETTINGS);
      }
    });
  }

  const brand = $('[data-action="go-home"]');
  if (brand) {
    brand.addEventListener("click", (event) => {
      event.preventDefault();
      router.navigate(ROUTES.TALK);
    });
  }

  const personalityBadge = $('[data-action="open-personalities"]');
  if (personalityBadge) {
    personalityBadge.addEventListener("click", () => router.navigate(ROUTES.PERSONALITIES));
  }

  const back = $('[data-action="go-back"]');
  if (back) {
    back.addEventListener("click", () => {
      router.navigate(window.location.hash === ROUTES.SETTINGS ? settingsReturn : ROUTES.TALK);
    });
  }

  mountPersonalityBadge(document);

  function syncHeaderForRoute() {
    const route = window.location.hash || ROUTES.TALK;
    if (gear) {
      const onSettings = route === ROUTES.SETTINGS;
      gear.classList.toggle("is-active", onSettings);
      gear.setAttribute("aria-label", onSettings ? "Close settings" : "Open settings");
    }
    if (back) {
      back.hidden = route === ROUTES.TALK;
      const toPersonalities = route === ROUTES.SETTINGS && settingsReturn === ROUTES.PERSONALITIES;
      back.setAttribute("aria-label", toPersonalities ? "Back to personalities" : "Back to conversation");
    }
    if (route === ROUTES.TALK) showPersonalityBadge();
    else hidePersonalityBadge();
  }
  window.addEventListener("hashchange", syncHeaderForRoute);
  syncHeaderForRoute();

  router.start();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
