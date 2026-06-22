/** Home view: grid of personality cards. Select one to apply it and navigate to Talk. */

import {
  applyPersonality,
  describeError,
  listPersonalities,
  loadPersonality,
  savePersonality,
  untilReady,
} from "../api.js";
import {
  AVATAR_BY_PROFILE,
  BUILT_IN_DEFAULT_OPTION,
  ROUTES,
  avatarFor,
} from "../constants.js";
import { $, h, prettifyProfileName } from "../ui.js";
import { openCustomProfileModal } from "../components/profile-modal.js";
import { setPendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";

export async function mountHomeView({ outlet, signal, navigate }) {
  const view = h(
    "section",
    { class: "view view--home" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Choose a personality"),
      h(
        "p",
        { class: "view-subtitle" },
        "Pick how Reachy Mini should think and talk. Tap a card to start a conversation."
      )
    ),
    h("div", { class: "personality-grid", role: "list" }, h("p", { class: "muted" }, "Loading…"))
  );
  outlet.replaceChildren(view);

  const grid = $(".personality-grid", view);
  const status = h("p", { class: "view-status", role: "status", "aria-live": "polite" });
  view.appendChild(status);

  let personalities;
  try {
    personalities = await untilReady(listPersonalities, signal, () => {
      grid.replaceChildren();
      grid.appendChild(h("p", { class: "muted" }, "Waiting for Reachy to finish starting…"));
    });
  } catch (error) {
    if (signal.aborted) return;
    grid.replaceChildren();
    grid.appendChild(renderError("Could not list personalities", error));
    return;
  }
  if (signal.aborted) return;

  const choices = (personalities?.choices || []).filter((name) => name !== BUILT_IN_DEFAULT_OPTION);
  const current = personalities?.current;
  const lockedTo = personalities?.locked ? personalities.locked_to : null;

  grid.replaceChildren();
  for (const name of choices) {
    grid.appendChild(
      buildPersonalityCard({
        name,
        isActive: name === current,
        disabled: Boolean(lockedTo) && name !== lockedTo,
        onSelect: () => handleSelection(name),
      })
    );
  }
  grid.appendChild(buildCustomCard({ onClick: handleCustomClick }));

  if (lockedTo) {
    status.textContent = `Profile locked to "${lockedTo}" by REACHY_MINI_LOCKED_PROFILE; switching is disabled.`;
    status.classList.add("is-warning");
  }

  function handleSelection(name) {
    // Optimistic header update so the badge already reads the chosen
    // personality while the apply request is still in flight.
    setPersonality(name);
    // Fire-and-forget: hand the apply promise to talk.js so the user sees
    // the orb in CONNECTING immediately instead of waiting on home.
    setPendingApply({ name, promise: applyPersonality(name, { persist: false }) });
    navigate(ROUTES.TALK);
  }

  async function handleCustomClick() {
    // Load the available tool palette so the modal can offer a checklist.
    let defaults;
    try {
      defaults = await loadPersonality(BUILT_IN_DEFAULT_OPTION);
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Could not load tools: ${describeError(error)}`;
      status.classList.add("is-error");
      return;
    }
    if (signal.aborted) return;

    const created = await openCustomProfileModal({
      availableTools: defaults?.available_tools || [],
      signal,
    });
    if (!created || signal.aborted) return;
    status.classList.remove("is-warning", "is-error");
    status.textContent = `Saving "${created.name}"…`;
    let newName;
    try {
      const saveResult = await savePersonality({
        name: created.name,
        instructions: created.instructions,
        tools_text: created.tools.join("\n"),
        voice: "", // falls back to backend default; user can change in Settings
      });
      if (signal.aborted) return;
      newName = saveResult?.value || created.name;
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Failed to create profile: ${describeError(error)}`;
      status.classList.add("is-error");
      return;
    }
    setPersonality(newName);
    setPendingApply({ name: newName, promise: applyPersonality(newName, { persist: false }) });
    navigate(ROUTES.TALK);
  }
}

function buildPersonalityCard({ name, isActive, disabled, onSelect }) {
  const hasAvatar = Object.prototype.hasOwnProperty.call(AVATAR_BY_PROFILE, stripUserPrefix(name));
  return h(
    "button",
    {
      type: "button",
      class: ["personality-card", isActive && "is-active", disabled && "is-disabled"],
      role: "listitem",
      disabled: disabled ? "disabled" : null,
      "aria-pressed": isActive ? "true" : "false",
      "aria-label": `Use personality ${prettifyProfileName(name)}`,
      onClick: disabled ? undefined : onSelect,
    },
    h(
      "span",
      { class: "personality-card__avatar" },
      h("img", {
        src: avatarFor(stripUserPrefix(name)),
        alt: "",
        loading: "lazy",
        "aria-hidden": "true", // card label already names the personality

        class: !hasAvatar ? "personality-card__avatar--fallback" : null,
      })
    ),
    h("span", { class: "personality-card__name" }, prettifyProfileName(name)),
    isActive && checkBadge()
  );
}

function checkBadge() {
  const badge = h("span", { class: "personality-card__badge", "aria-hidden": "true" });
  badge.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>`;
  return badge;
}

function buildCustomCard({ onClick }) {
  return h(
    "button",
    {
      type: "button",
      class: "personality-card personality-card--custom",
      role: "listitem",
      "aria-label": "Create a custom personality",
      onClick,
    },
    h("span", { class: "personality-card__plus", "aria-hidden": "true" }, "+"),
    h("span", { class: "personality-card__name" }, "Custom"),
    h("span", { class: "personality-card__hint" }, "Write your own prompt")
  );
}

function stripUserPrefix(name) {
  return name.replace(/^user_personalities\//, "");
}

function renderError(label, error) {
  return h(
    "div",
    { class: "view-error" },
    h("p", null, label),
    h("p", { class: "muted small" }, describeError(error))
  );
}
