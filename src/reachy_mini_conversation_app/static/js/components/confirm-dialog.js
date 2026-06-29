/** In-app confirmation dialog. Replaces window.confirm, which embedded app hosts
 * (the Reachy Mini control's webview / sandboxed iframes) silently suppress.
 * Resolves true if confirmed, false otherwise. */

import { h } from "../ui.js";

export function confirmDialog({
  title = "Are you sure?",
  message = "",
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
} = {}) {
  return new Promise((resolve) => {
    const overlay = h("div", { class: "modal-overlay", role: "presentation" });
    const cancelBtn = h("button", { type: "button", class: "btn btn--ghost" }, cancelLabel);
    const confirmBtn = h(
      "button",
      { type: "button", class: ["btn", danger ? "btn--danger" : "btn--primary"] },
      confirmLabel
    );
    const dialog = h(
      "div",
      { class: "modal modal--confirm", role: "dialog", "aria-modal": "true", "aria-labelledby": "confirm-title" },
      h("h2", { id: "confirm-title", class: "modal__title" }, title),
      message ? h("p", { class: "modal__subtitle" }, message) : null,
      h("div", { class: "modal__actions" }, cancelBtn, confirmBtn)
    );
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    function close(value) {
      window.removeEventListener("keydown", onKeydown);
      overlay.remove();
      resolve(value);
    }
    function onKeydown(event) {
      // Only handle Escape globally; let the focused button handle Enter/click
      // natively, so pressing Enter on Cancel cancels rather than confirms.
      if (event.key === "Escape") close(false);
    }
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) close(false);
    });
    cancelBtn.addEventListener("click", () => close(false));
    confirmBtn.addEventListener("click", () => close(true));
    window.addEventListener("keydown", onKeydown);
    requestAnimationFrame(() => confirmBtn.focus());
  });
}
