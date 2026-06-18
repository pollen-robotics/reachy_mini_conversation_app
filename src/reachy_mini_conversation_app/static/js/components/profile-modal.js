/** Modal to collect name + instructions for a new custom personality. Returns { name, instructions } or null. */

import { h } from "../ui.js";

const NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

/**
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<{ name: string, instructions: string }|null>}
 */
export function openCustomProfileModal({ signal } = {}) {
  return new Promise((resolve) => {
    const overlay = buildOverlay();
    const dialog = buildDialog();
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Focus the first text input on next paint so the user can type immediately.
    requestAnimationFrame(() => dialog.querySelector("input")?.focus());

    function close(value) {
      cleanup();
      resolve(value);
    }

    function onKeydown(event) {
      if (event.key === "Escape") {
        close(null);
        return;
      }
      if (event.key === "Tab") {
        const focusable = Array.from(
          dialog.querySelectorAll('button, input, textarea, select, [tabindex]:not([tabindex="-1"])')
        ).filter((el) => !el.disabled);
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey) {
          if (document.activeElement === first) {
            event.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            event.preventDefault();
            first.focus();
          }
        }
      }
    }

    function onAbort() {
      close(null);
    }

    function cleanup() {
      window.removeEventListener("keydown", onKeydown);
      signal?.removeEventListener("abort", onAbort);
      overlay.remove();
    }

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) close(null);
    });

    window.addEventListener("keydown", onKeydown);
    signal?.addEventListener("abort", onAbort);

    dialog.querySelector("[data-action='cancel']").addEventListener("click", () => close(null));

    const errorBox = dialog.querySelector(".modal__error");
    dialog.querySelectorAll("input, textarea").forEach((field) => {
      field.addEventListener("input", () => errorBox.classList.remove("is-visible"));
    });

    dialog.querySelector("form").addEventListener("submit", (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);
      const name = String(formData.get("name") || "").trim();
      const instructions = String(formData.get("instructions") || "").trim();

      if (!name) return showError(errorBox, "Please pick a name.");
      if (!NAME_PATTERN.test(name)) {
        return showError(errorBox, "Use only letters, numbers, dashes or underscores.");
      }
      if (!instructions) return showError(errorBox, "Please write some instructions.");

      close({ name, instructions });
    });
  });
}

function buildOverlay() {
  return h("div", {
    class: "modal-overlay",
    role: "presentation",
  });
}

function buildDialog() {
  return h(
    "div",
    {
      class: "modal",
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": "custom-profile-title",
    },
    h(
      "header",
      { class: "modal__header" },
      h("h2", { id: "custom-profile-title", class: "modal__title" }, "Create a custom personality"),
      h(
        "p",
        { class: "modal__subtitle" },
        "Define how Reachy should behave. The full standard tool set (dance, emotions, head tracking, ...) will be enabled by default."
      )
    ),
    h(
      "form",
      { class: "modal__form" },
      h(
        "label",
        { class: "modal__field" },
        h("span", { class: "modal__label" }, "Name"),
        h("input", {
          type: "text",
          name: "name",
          required: "required",
          autocomplete: "off",
          spellcheck: "false",
          placeholder: "e.g. zen_master",
          pattern: "[a-zA-Z0-9_-]+",
          class: "modal__input",
        })
      ),
      h(
        "label",
        { class: "modal__field" },
        h("span", { class: "modal__label" }, "Instructions"),
        h("textarea", {
          name: "instructions",
          required: "required",
          rows: "8",
          placeholder:
            "You are a calm, slow-speaking zen guide. Pause between sentences. Encourage the user to breathe.",
          class: "modal__textarea",
        })
      ),
      h("p", { class: "modal__error", role: "alert", "aria-live": "polite" }),
      h(
        "div",
        { class: "modal__actions" },
        h("button", { type: "button", class: "btn btn--ghost", "data-action": "cancel" }, "Cancel"),
        h("button", { type: "submit", class: "btn btn--primary" }, "Create & start")
      )
    )
  );
}

function showError(errorBox, message) {
  errorBox.textContent = message;
  errorBox.classList.add("is-visible");
}
