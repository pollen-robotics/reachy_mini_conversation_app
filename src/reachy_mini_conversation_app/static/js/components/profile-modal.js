/** Modal to create or edit a personality (name + instructions + tools). Returns { name, instructions, tools } or null. */

import { h } from "../ui.js";

const NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

/**
 * @param {{
 *   mode?: "create" | "edit",
 *   initial?: { name?: string, instructions?: string, enabledTools?: string[] },
 *   availableTools?: string[],
 *   signal?: AbortSignal,
 * }} [options]
 * @returns {Promise<{ name: string, instructions: string, tools: string[] }|null>}
 */
export function openProfileModal({ mode = "create", initial = {}, availableTools = [], signal } = {}) {
  const isEdit = mode === "edit";
  const enabledTools = initial.enabledTools || [];
  // Create starts from the available palette with everything enabled. Edit shows the union of
  // available and currently-enabled tools (so tools that aren't importable modules, e.g. a profile's
  // own files, still appear pre-checked) and never silently drops an enabled tool on save.
  const enabledSet = new Set(enabledTools);
  const toolChoices = isEdit
    ? [...new Set([...availableTools, ...enabledTools])].sort()
    : [...availableTools].sort();
  const isToolChecked = isEdit ? (tool) => enabledSet.has(tool) : () => true;

  return new Promise((resolve) => {
    const overlay = buildOverlay();
    const dialog = buildDialog({ isEdit, initial, toolChoices, isToolChecked });
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Focus the first editable field on next paint (the name in create mode, the textarea in edit).
    requestAnimationFrame(() => {
      const target = isEdit ? dialog.querySelector("textarea") : dialog.querySelector("input");
      target?.focus();
    });

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
      // The name is locked in edit mode (renaming would mean a new profile dir), so keep the original.
      const name = isEdit ? String(initial.name || "") : String(formData.get("name") || "").trim();
      const instructions = String(formData.get("instructions") || "").trim();

      if (!isEdit) {
        if (!name) return showError(errorBox, "Please pick a name.");
        if (!NAME_PATTERN.test(name)) {
          return showError(errorBox, "Use only letters, numbers, dashes or underscores.");
        }
      }
      if (!instructions) return showError(errorBox, "Please write some instructions.");

      const tools = Array.from(dialog.querySelectorAll('input[name="tool"]:checked')).map((el) => el.value);
      close({ name, instructions, tools });
    });
  });
}

function buildOverlay() {
  return h("div", {
    class: "modal-overlay",
    role: "presentation",
  });
}

function buildDialog({ isEdit, initial, toolChoices, isToolChecked }) {
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
        isEdit ? `Edit ${initial.name || "personality"}` : "Create a custom personality"
      ),
      h(
        "p",
        { class: "modal__subtitle" },
        "Define how Reachy should behave and which tools it can use."
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
      buildToolsField(toolChoices, isToolChecked),
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

/** Render the tool checklist; isToolChecked decides each box's initial state. */
function buildToolsField(toolChoices, isToolChecked) {
  return h(
    "fieldset",
    { class: "modal__field modal__tools" },
    h("legend", { class: "modal__label" }, "Tools"),
    h(
      "div",
      { class: "modal__tools-grid" },
      ...toolChoices.map((tool) =>
        h(
          "label",
          { class: "modal__tool" },
          h("input", { type: "checkbox", name: "tool", value: tool, checked: isToolChecked(tool) ? "checked" : null }),
          h("span", null, tool)
        )
      )
    )
  );
}

function showError(errorBox, message) {
  errorBox.textContent = message;
  errorBox.classList.add("is-visible");
}
