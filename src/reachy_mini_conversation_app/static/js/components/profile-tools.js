/** Per-personality tool access controls. */

import {
  describeError,
  getProfileTools,
  resetProfileTools,
  saveProfileTools,
} from "../api.js";
import { h, prettifyProfileName, prettifyToolName } from "../ui.js";
import { confirmDialog } from "./confirm-dialog.js";

export function buildProfileToolsSection({ signal, initialProfile = null, onProfileChanged } = {}) {
  const profileSelect = h("select", {
    class: "settings-select",
    name: "profile_tools_profile",
    disabled: "disabled",
    "aria-label": "Personality to configure",
  });
  const summary = h("div", { class: "settings-toolset-summary", "aria-live": "polite" });
  const toolGroups = h(
    "div",
    { class: "settings-tool-groups", "aria-live": "polite" },
    h("p", { class: "settings-hint" }, "Loading tool access…")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const resetButton = h(
    "button",
    { type: "button", class: "btn btn--ghost", disabled: "disabled" },
    "Restore defaults"
  );
  const saveButton = h(
    "button",
    { type: "button", class: "btn btn--primary", disabled: "disabled" },
    "Save tool access"
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Tool access"),
    h(
      "p",
      { class: "settings-hint settings-section-intro" },
      "Choose exactly what each personality can use. Installed Tool Spaces stay off until selected."
    ),
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Configure for"),
      profileSelect
    ),
    summary,
    toolGroups,
    h("div", { class: "settings-actions settings-toolset-actions" }, resetButton, saveButton),
    status
  );

  let currentPayload = null;
  let initialEnabledTools = new Set();
  let dirty = false;
  let busy = false;

  function selectedToolIds() {
    return Array.from(toolGroups.querySelectorAll('input[type="checkbox"]:checked')).map(
      (checkbox) => checkbox.value
    );
  }

  function selectionChanged() {
    const selectedTools = selectedToolIds();
    return (
      selectedTools.length !== initialEnabledTools.size ||
      selectedTools.some((toolId) => !initialEnabledTools.has(toolId))
    );
  }

  function syncActions() {
    const editable = currentPayload?.editable !== false;
    profileSelect.disabled = busy || !currentPayload?.profiles?.length;
    toolGroups.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
      const disabled = busy || !editable;
      checkbox.disabled = disabled;
      checkbox.closest(".settings-tool-choice")?.classList.toggle("is-disabled", disabled);
    });
    resetButton.disabled = busy || !editable || !currentPayload?.overridden;
    saveButton.disabled = busy || !editable || !dirty;
  }

  function setBusy(nextBusy) {
    busy = nextBusy;
    element.toggleAttribute("aria-busy", nextBusy);
    syncActions();
  }

  function setDirty(nextDirty) {
    dirty = nextDirty;
    syncActions();
  }

  function populateProfiles(payload) {
    profileSelect.replaceChildren();
    for (const profile of payload.profiles || []) {
      const label = `${prettifyProfileName(profile.id)}${profile.active ? " · Active" : ""}`;
      profileSelect.appendChild(
        h("option", { value: profile.id, selected: profile.id === payload.profile ? "selected" : null }, label)
      );
    }
  }

  function renderSummary(payload) {
    const enabledCount = payload.enabled_tools?.length || 0;
    const pills = [];
    if (payload.is_active) pills.push(h("span", { class: "settings-pill is-active" }, "Active"));
    if (payload.editable === false) pills.push(h("span", { class: "settings-pill" }, "Locked"));
    summary.replaceChildren(
      h(
        "div",
        { class: "settings-toolset-summary-copy" },
        h("strong", null, `${enabledCount} ${enabledCount === 1 ? "tool" : "tools"} enabled`),
        h(
          "span",
          { class: "settings-toolset-mode" },
          payload.overridden ? "Customized for this personality" : "Using profile defaults"
        )
      ),
      ...pills
    );
  }

  function renderToolGroups(payload) {
    const enabled = new Set(payload.enabled_tools || []);
    const groups = new Map();
    for (const tool of payload.available_tools || []) {
      const key = tool.kind === "tool_space" ? `space:${tool.source}` : tool.kind;
      if (!groups.has(key)) {
        const sourceName = prettifyProfileName(tool.source.split("/").at(-1));
        let title = `Tool Space · ${sourceName}`;
        if (tool.kind === "shared") title = "Built-in tools";
        if (tool.kind === "external") title = "External tools";
        groups.set(key, {
          title,
          tools: [],
        });
      }
      groups.get(key).tools.push(tool);
    }

    const unavailable = payload.unavailable_enabled_tools || [];
    if (unavailable.length) {
      groups.set("unavailable", {
        title: "Unavailable selections",
        tools: unavailable.map((toolId) => ({
          id: toolId,
          kind: "unavailable",
          source: "Unavailable",
          description: "Its source is not installed or the tool is no longer exposed.",
        })),
      });
    }

    toolGroups.replaceChildren();
    if (!groups.size) {
      toolGroups.appendChild(h("p", { class: "settings-hint" }, "No configurable tools are available."));
      return;
    }

    for (const group of groups.values()) {
      const fieldset = h(
        "fieldset",
        { class: "settings-tool-group" },
        h(
          "legend",
          { class: "settings-tool-group-title" },
          group.title,
          h("span", { class: "settings-tool-group-count" }, String(group.tools.length))
        ),
        h(
          "div",
          { class: "settings-tool-grid" },
          ...group.tools.map((tool) => toolChoice(tool, enabled.has(tool.id)))
        )
      );
      fieldset.addEventListener("change", () => {
        const changed = selectionChanged();
        status.textContent = changed ? "Unsaved changes" : "";
        status.classList.remove("is-error");
        setDirty(changed);
      });
      toolGroups.appendChild(fieldset);
    }
  }

  function render(payload) {
    currentPayload = payload;
    initialEnabledTools = new Set(payload.enabled_tools || []);
    populateProfiles(payload);
    renderSummary(payload);
    renderToolGroups(payload);
    status.classList.remove("is-error");
    status.textContent = payload.editable === false ? "Tool editing is locked by the administrator." : "";
    setDirty(false);
  }

  async function load(profile = null) {
    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Loading tools…";
    try {
      const payload = await getProfileTools(profile);
      if (signal?.aborted) return;
      render(payload);
      onProfileChanged?.(payload.profile);
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not load tool access: ${describeError(error)}`;
      status.classList.add("is-error");
      if (currentPayload) {
        profileSelect.value = currentPayload.profile;
      } else {
        toolGroups.replaceChildren();
      }
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  }

  async function confirmDiscard() {
    if (!dirty) return true;
    return confirmDialog({
      title: "Discard unsaved tool changes?",
      message: "Your selections for this personality have not been saved.",
      confirmLabel: "Discard changes",
      signal,
    });
  }

  profileSelect.addEventListener("change", async () => {
    const selectedProfile = profileSelect.value;
    if (!(await confirmDiscard())) {
      profileSelect.value = currentPayload?.profile || "";
      return;
    }
    await load(selectedProfile);
  });

  saveButton.addEventListener("click", async () => {
    if (saveButton.disabled || !currentPayload) return;
    const enabledTools = selectedToolIds();
    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Saving tool access…";
    try {
      const payload = await saveProfileTools(currentPayload.profile, enabledTools);
      if (signal?.aborted) return;
      render(payload);
      status.textContent = payload?.message || "Tool access saved.";
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not save tool access: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  });

  resetButton.addEventListener("click", async () => {
    if (resetButton.disabled || !currentPayload) return;
    const confirmed = await confirmDialog({
      title: "Restore profile defaults?",
      message: `This replaces the custom tool selection for “${prettifyProfileName(currentPayload.profile)}”.`,
      confirmLabel: "Restore defaults",
      signal,
    });
    if (!confirmed || signal?.aborted) return;

    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Restoring defaults…";
    try {
      const payload = await resetProfileTools(currentPayload.profile);
      if (signal?.aborted) return;
      render(payload);
      status.textContent = payload?.message || "Profile defaults restored.";
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not restore defaults: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  });

  return {
    element,
    confirmDiscard,
    hasUnsavedChanges() {
      return dirty;
    },
    async refresh() {
      await load(currentPayload?.profile || initialProfile);
    },
  };
}

function toolChoice(tool, checked) {
  const unavailable = tool.kind === "unavailable";
  const input = h("input", {
    type: "checkbox",
    value: tool.id,
    checked: checked ? "checked" : null,
  });
  return h(
    "label",
    {
      class: ["settings-tool-choice", unavailable && "is-unavailable"],
    },
    input,
    h(
      "span",
      { class: "settings-tool-choice-copy" },
      h("strong", { class: "settings-tool-choice-name" }, prettifyToolName(tool.id)),
      tool.kind === "unavailable" ? h("code", { class: "settings-tool-choice-id" }, tool.id) : null,
      tool.description ? h("span", { class: "settings-tool-choice-description" }, tool.description) : null
    )
  );
}
