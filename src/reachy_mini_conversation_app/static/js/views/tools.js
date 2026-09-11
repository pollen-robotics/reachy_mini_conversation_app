/** Tools view: per-personality access and Hugging Face Tool Spaces. */

import {
  addToolSpace,
  describeError,
  listToolSpaces,
  removeToolSpace,
} from "../api.js";
import { ROUTES } from "../constants.js";
import { h } from "../ui.js";
import { confirmDialog } from "../components/confirm-dialog.js";
import { buildProfileToolsSection } from "../components/profile-tools.js";

export async function mountToolsView({ outlet, signal, searchParams, setLeaveGuard, replaceRoute }) {
  const fromPersonalities = searchParams.get("from") === "personalities";
  const profileToolsSection = buildProfileToolsSection({
    signal,
    initialProfile: searchParams.get("profile"),
    onProfileChanged(profile) {
      const params = new URLSearchParams({ profile });
      if (fromPersonalities) params.set("from", "personalities");
      replaceRoute(`${ROUTES.TOOLS}?${params}`);
    },
  });
  setLeaveGuard({
    shouldBlock: profileToolsSection.hasUnsavedChanges,
    confirm: profileToolsSection.confirmDiscard,
  });
  const toolSpacesSection = buildToolSpacesSection({
    signal,
    onBeforeChange: profileToolsSection.confirmDiscard,
    onChanged: profileToolsSection.refresh,
  });
  const view = h(
    "section",
    { class: "view view--tools" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Tools"),
      h(
        "p",
        { class: "view-subtitle" },
        "Choose tool access by personality and manage Tool Spaces."
      )
    ),
    profileToolsSection.element,
    toolSpacesSection.element
  );
  outlet.replaceChildren(view);

  await Promise.all([profileToolsSection.refresh(), toolSpacesSection.refresh()]);
}

function buildToolSpacesSection({ signal, onBeforeChange, onChanged } = {}) {
  const slugInput = h("input", {
    id: "tool-space-slug",
    type: "text",
    name: "slug",
    required: "required",
    autocomplete: "off",
    autocapitalize: "none",
    spellcheck: "false",
    enterkeyhint: "go",
    placeholder: "owner/space-name",
    class: "settings-input",
  });
  const addButton = h("button", { type: "submit", class: "btn btn--primary" }, "Add Space");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const list = h(
    "div",
    { class: "settings-tool-spaces", role: "list", "aria-live": "polite" },
    h("p", { class: "settings-hint" }, "Loading installed Spaces…")
  );
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-label", for: "tool-space-slug" }, "Space"),
    h("div", { class: "settings-tool-space-controls" }, slugInput, addButton),
    h(
      "p",
      { class: "settings-hint" },
      "Install an MCP-compatible Hugging Face Space, then choose access per personality above."
    ),
    status
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Tool Spaces"),
    form,
    h("h3", { class: "settings-list-title" }, "Installed"),
    list
  );
  let busy = false;
  let editable = true;

  function setBusy(nextBusy, addLabel = "Add Space") {
    busy = nextBusy;
    form.toggleAttribute("aria-busy", nextBusy);
    list.toggleAttribute("aria-busy", nextBusy);
    slugInput.disabled = nextBusy || !editable;
    addButton.disabled = nextBusy || !editable;
    addButton.textContent = nextBusy ? addLabel : "Add Space";
    list.querySelectorAll("button").forEach((button) => {
      button.disabled = nextBusy || !editable;
    });
  }

  function render(payload) {
    editable = payload?.editable !== false;
    const spaces = Array.isArray(payload?.spaces) ? payload.spaces : [];
    list.replaceChildren();
    if (!spaces.length) {
      list.appendChild(h("p", { class: "settings-hint" }, "No Tool Spaces installed."));
      setBusy(busy);
      if (!editable) status.textContent = "Tool Space editing is locked by the administrator.";
      return;
    }

    for (const space of spaces) {
      const toolCount = Number(space.tool_count) || 0;
      const details = [`${toolCount} ${toolCount === 1 ? "tool" : "tools"}`];
      if (space.private) details.push("Private");
      const removeButton = h(
        "button",
        {
          type: "button",
          class: "btn btn--ghost",
          "aria-label": `Remove Tool Space ${space.slug}`,
          disabled: busy || !editable ? "disabled" : null,
        },
        "Remove"
      );
      removeButton.addEventListener("click", async () => {
        if (onBeforeChange && !(await onBeforeChange())) return;
        const confirmed = await confirmDialog({
          title: "Remove Tool Space?",
          message: `Removing “${space.slug}” disables its tools in every personality.`,
          confirmLabel: "Remove",
          danger: true,
          signal,
        });
        if (!confirmed || signal?.aborted) return;

        status.classList.remove("is-error");
        status.textContent = `Removing “${space.slug}”…`;
        setBusy(true);
        try {
          const result = await removeToolSpace(space.slug);
          if (signal?.aborted) return;
          render(result);
          status.textContent = result?.message || `Removed “${space.slug}”.`;
          await onChanged?.();
        } catch (error) {
          if (signal?.aborted) return;
          status.textContent = `Failed to remove: ${describeError(error)}`;
          status.classList.add("is-error");
        } finally {
          if (!signal?.aborted) setBusy(false);
        }
      });
      list.appendChild(
        h(
          "div",
          { class: "settings-tool-space", role: "listitem" },
          h(
            "div",
            { class: "settings-tool-space-summary" },
            h("strong", { class: "settings-tool-space-name" }, space.slug),
            h("span", { class: "settings-tool-space-meta" }, details.join(" · "))
          ),
          removeButton
        )
      );
    }
    setBusy(busy);
    if (!editable) status.textContent = "Tool Space editing is locked by the administrator.";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (addButton.disabled) return;
    const slug = slugInput.value.trim();
    if (!slug) return;
    if (onBeforeChange && !(await onBeforeChange())) return;

    status.classList.remove("is-error");
    status.textContent = `Checking “${slug}”…`;
    setBusy(true, "Checking Space…");
    try {
      const result = await addToolSpace(slug);
      if (signal?.aborted) return;
      slugInput.value = "";
      render(result);
      status.textContent = result?.message || `Added “${slug}”.`;
      await onChanged?.();
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Failed to add: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  });

  return {
    element,
    async refresh() {
      try {
        const payload = await listToolSpaces();
        if (signal?.aborted) return;
        render(payload);
      } catch (error) {
        if (signal?.aborted) return;
        list.replaceChildren(
          h("p", { class: "settings-status is-error" }, `Could not load Tool Spaces: ${describeError(error)}`)
        );
      }
    },
  };
}
