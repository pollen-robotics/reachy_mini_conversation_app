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
  signal,
} = {}) {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(false);
      return;
    }

    const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
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

    let settled = false;
    function close(value) {
      if (settled) return;
      settled = true;
      window.removeEventListener("keydown", onKeydown);
      signal?.removeEventListener("abort", onAbort);
      overlay.remove();
      if (returnFocus?.isConnected) returnFocus.focus();
      resolve(value);
    }

    function onAbort() {
      close(false);
    }

    function onKeydown(event) {
      if (event.key === "Escape") {
        close(false);
        return;
      }
      if (event.key !== "Tab") return;
      if (event.shiftKey && document.activeElement === cancelBtn) {
        event.preventDefault();
        confirmBtn.focus();
      } else if (!event.shiftKey && document.activeElement === confirmBtn) {
        event.preventDefault();
        cancelBtn.focus();
      }
    }
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) close(false);
    });
    cancelBtn.addEventListener("click", () => close(false));
    confirmBtn.addEventListener("click", () => close(true));
    window.addEventListener("keydown", onKeydown);
    signal?.addEventListener("abort", onAbort, { once: true });
    requestAnimationFrame(() => (danger ? cancelBtn : confirmBtn).focus());
  });
}
