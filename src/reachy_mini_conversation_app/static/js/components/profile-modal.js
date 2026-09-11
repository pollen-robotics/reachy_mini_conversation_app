/** Modal to create or edit a personality. */

import { h, prettifyProfileName } from "../ui.js";

const NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

/**
 * @param {{
 *   mode?: "create" | "edit",
 *   initial?: { name?: string, instructions?: string, greeting?: string },
 *   signal?: AbortSignal,
 * }} [options]
 * @returns {Promise<{ name: string, instructions: string, greeting: string }|null>}
 */
export function openProfileModal({ mode = "create", initial = {}, signal } = {}) {
  const isEdit = mode === "edit";

  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(null);
      return;
    }

    const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overlay = buildOverlay();
    const dialog = buildDialog({ isEdit, initial });
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Focus the first editable field on next paint (the name in create mode, the textarea in edit).
    requestAnimationFrame(() => {
      const target = isEdit ? dialog.querySelector("textarea") : dialog.querySelector("input");
      target?.focus();
    });

    let settled = false;
    function close(value) {
      if (settled) return;
      settled = true;
      cleanup();
      if (returnFocus?.isConnected) returnFocus.focus();
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
    signal?.addEventListener("abort", onAbort, { once: true });

    dialog.querySelector("[data-action='cancel']").addEventListener("click", () => close(null));

    const errorBox = dialog.querySelector(".modal__error");
    dialog.querySelectorAll("input, textarea").forEach((field) => {
      field.addEventListener("input", () => errorBox.classList.remove("is-visible"));
    });

    dialog.querySelector("form").addEventListener("submit", (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);
      // The name is locked in edit mode (renaming would mean a new profile dir), so keep the original.
      const name = isEdit ? String(initial.name || "") : String(formData.get("name") || "").trim();
      const instructions = String(formData.get("instructions") || "").trim();
      const greeting = String(formData.get("greeting") || "").trim();

      if (!isEdit) {
        if (!name) return showError(errorBox, "Please pick a name.");
        if (!NAME_PATTERN.test(name)) {
          return showError(errorBox, "Use only letters, numbers, dashes or underscores.");
        }
      }
      if (!instructions) return showError(errorBox, "Please write some instructions.");

      close({ name, instructions, greeting });
    });
  });
}

function buildOverlay() {
  return h("div", {
    class: "modal-overlay",
    role: "presentation",
  });
}

function buildDialog({ isEdit, initial }) {
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
      h(
        "h2",
        { id: "custom-profile-title", class: "modal__title" },
        isEdit ? `Edit ${prettifyProfileName(initial.name || "personality")}` : "Create a custom personality"
      ),
      h(
        "p",
        { class: "modal__subtitle" },
        "Define how Reachy should behave and greet people."
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
          required: isEdit ? null : "required",
          readonly: isEdit ? "readonly" : null,
          autocomplete: "off",
          spellcheck: "false",
          placeholder: "e.g. zen_master",
          pattern: "[a-zA-Z0-9_-]+",
          value: isEdit ? initial.name || "" : null,
          class: ["modal__input", isEdit && "is-readonly"],
        })
      ),
      h(
        "label",
        { class: "modal__field" },
        h("span", { class: "modal__label" }, "Instructions"),
        h(
          "textarea",
          {
            name: "instructions",
            required: "required",
            rows: "8",
            placeholder:
              "You are a calm, slow-speaking zen guide. Pause between sentences. Encourage the user to breathe.",
            class: "modal__textarea",
          },
          initial.instructions || ""
        )
      ),
      h(
        "label",
        { class: "modal__field" },
        h("span", { class: "modal__label" }, "Startup greeting prompt"),
        h(
          "textarea",
          {
            name: "greeting",
            rows: "3",
            placeholder: "Start the conversation with a short greeting in character.",
            class: "modal__textarea",
          },
          initial.greeting || ""
        )
      ),
      h("p", { class: "modal__error", role: "alert", "aria-live": "polite" }),
      h(
        "div",
        { class: "modal__actions" },
        h("button", { type: "button", class: "btn btn--ghost", "data-action": "cancel" }, "Cancel"),
        h("button", { type: "submit", class: "btn btn--primary" }, isEdit ? "Save changes" : "Create & start")
      )
    )
  );
}

function showError(errorBox, message) {
  errorBox.textContent = message;
  errorBox.classList.add("is-visible");
}
