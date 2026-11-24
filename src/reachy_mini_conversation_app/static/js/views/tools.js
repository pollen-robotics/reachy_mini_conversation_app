/** Tools view: per-personality access and Hugging Face Tool Spaces. */

import {
  addToolSpace,
  describeError,
  listToolSpaces,
  removeToolSpace,
  searchToolSpaces,
  getToolSpaceStatuses,
} from "../api.js";
import { ROUTES } from "../constants.js";
import { h } from "../ui.js";
import { confirmDialog } from "../components/confirm-dialog.js";
import { buildProfileToolsSection } from "../components/profile-tools.js";

// Tool Spaces are community apps that can go offline at any time; the badge shows what the Hub reports now.
const SPACE_STATUS = {
  online: { label: "Online", tone: "is-active" },
  space_asleep: { label: "Asleep", tone: null },
  space_starting: { label: "Starting", tone: "is-warn" },
  space_paused: { label: "Paused", tone: "is-danger" },
  space_broken: { label: "Broken", tone: "is-danger" },
  no_access: { label: "No access", tone: "is-danger" },
  unknown: { label: "Unknown", tone: null },
};

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

function textInput(id, placeholder, enterkeyhint) {
  return h("input", {
    id,
    type: "text",
    required: "required",
    autocomplete: "off",
    autocapitalize: "none",
    spellcheck: "false",
    enterkeyhint,
    placeholder,
    class: "settings-input",
  });
}

function buildToolSpacesSection({ signal, onBeforeChange, onChanged } = {}) {
  const searchInput = textInput("tool-space-search", "What should it do? Weather, text to speech…", "search");
  const searchButton = h("button", { type: "submit", class: "btn btn--primary" }, "Search");
  const searchResults = h("div", { class: "settings-tool-spaces", role: "list", "aria-live": "polite" });
  const searchForm = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-label", for: "tool-space-search" }, "Find a Space"),
    h("div", { class: "settings-tool-space-controls" }, searchInput, searchButton),
    searchResults
  );

  const slugInput = textInput("tool-space-slug", "owner/space-name or its Hugging Face link", "go");
  const addButton = h("button", { type: "submit", class: "btn btn--ghost" }, "Add Space");
  const addForm = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-label", for: "tool-space-slug" }, "Or add one by name or link"),
    h("div", { class: "settings-tool-space-controls" }, slugInput, addButton)
  );

  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const list = h(
    "div",
    { class: "settings-tool-spaces", role: "list", "aria-live": "polite" },
    h("p", { class: "settings-hint" }, "Loading installed Spaces…")
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Tool Spaces"),
    h(
      "p",
      { class: "settings-hint" },
      "Tool Spaces are apps on Hugging Face that give Reachy new tools, such as search or weather. " +
        "Community Spaces can go offline at any time: each one shows its current status. " +
        "After adding a Space, choose which personalities can use its tools above."
    ),
    searchForm,
    addForm,
    status,
    h("h3", { class: "settings-list-title" }, "Installed"),
    list
  );
  let busy = false;
  let editable = true;
  let statusBadges = new Map();
  let statusRequest = 0;

  function setBusy(nextBusy, addLabel = "Add Space") {
    busy = nextBusy;
    for (const container of [addForm, list, searchResults]) container.toggleAttribute("aria-busy", nextBusy);
    slugInput.disabled = nextBusy || !editable;
    addButton.disabled = nextBusy || !editable;
    addButton.textContent = nextBusy ? addLabel : "Add Space";
    for (const button of [...list.querySelectorAll("button"), ...searchResults.querySelectorAll("button")]) {
      button.disabled = nextBusy || !editable || button.dataset.installed === "true";
    }
  }

  function showStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle("is-error", isError);
  }

  function renderStatus(slug, availability) {
    const badge = statusBadges.get(slug);
    if (!badge) return;
    const { label, tone } = SPACE_STATUS[availability?.state] || SPACE_STATUS.unknown;
    badge.pill.className = ["settings-pill", tone].filter(Boolean).join(" ");
    badge.pill.textContent = label;
    badge.pill.title = availability?.message || "";
    // Spell out anything the user has to act on; a running or sleeping Space needs nothing.
    const needsAttention = tone === "is-danger" || tone === "is-warn";
    badge.note.textContent = needsAttention ? availability.message : "";
    badge.note.hidden = !needsAttention;
  }

  async function refreshStatuses() {
    const request = ++statusRequest;
    try {
      const payload = await getToolSpaceStatuses();
      if (signal?.aborted || request !== statusRequest) return;
      for (const slug of statusBadges.keys()) renderStatus(slug, payload?.statuses?.[slug]);
    } catch (error) {
      if (signal?.aborted || request !== statusRequest) return;
      for (const slug of statusBadges.keys()) {
        renderStatus(slug, { state: "unknown", message: `Could not check: ${describeError(error)}` });
      }
    }
  }

  function syncSearchButtons(installedSlugs) {
    for (const button of searchResults.querySelectorAll("button[data-slug]")) {
      const installed = installedSlugs.has(button.dataset.slug);
      button.dataset.installed = installed ? "true" : "false";
      button.textContent = installed ? "Installed" : "Add";
      button.disabled = busy || !editable || installed;
    }
  }

  function render(payload) {
    editable = payload?.editable !== false;
    const spaces = Array.isArray(payload?.spaces) ? payload.spaces : [];
    syncSearchButtons(new Set(spaces.map((space) => space.slug)));
    list.replaceChildren();
    statusBadges = new Map();
    if (!spaces.length) {
      list.appendChild(h("p", { class: "settings-hint" }, "No Tool Spaces installed."));
      setBusy(busy);
      if (!editable) showStatus("Tool Space editing is locked by the administrator.");
      return;
    }

    for (const space of spaces) {
      const toolCount = Number(space.tool_count) || 0;
      const details = [`${toolCount} ${toolCount === 1 ? "tool" : "tools"}`];
      if (space.private) details.push("Private");
      const pill = h("span", { class: "settings-pill" }, "Checking…");
      const note = h("span", { class: "settings-tool-space-meta", hidden: "hidden" });
      statusBadges.set(space.slug, { pill, note });
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
      removeButton.addEventListener("click", () => removeSpace(space.slug));
      list.appendChild(
        h(
          "div",
          { class: "settings-tool-space", role: "listitem" },
          h(
            "div",
            { class: "settings-tool-space-summary" },
            h(
              "div",
              { class: "settings-tool-space-heading" },
              h("strong", { class: "settings-tool-space-name" }, space.slug),
              pill
            ),
            h("span", { class: "settings-tool-space-meta" }, details.join(" · ")),
            note
          ),
          removeButton
        )
      );
    }
    setBusy(busy);
    if (!editable) showStatus("Tool Space editing is locked by the administrator.");
    refreshStatuses();
  }

  async function installSpace(slug) {
    if (onBeforeChange && !(await onBeforeChange())) return false;
    showStatus(`Checking “${slug}”…`);
    setBusy(true, "Checking Space…");
    try {
      const result = await addToolSpace(slug);
      if (signal?.aborted) return false;
      render(result);
      showStatus(result?.message || `Added “${slug}”.`);
      await onChanged?.();
      return true;
    } catch (error) {
      if (!signal?.aborted) showStatus(`Failed to add: ${describeError(error)}`, true);
      return false;
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  }

  async function removeSpace(slug) {
    if (onBeforeChange && !(await onBeforeChange())) return;
    const confirmed = await confirmDialog({
      title: "Remove Tool Space?",
      message: `Removing “${slug}” disables its tools in every personality.`,
      confirmLabel: "Remove",
      danger: true,
      signal,
    });
    if (!confirmed || signal?.aborted) return;

    showStatus(`Removing “${slug}”…`);
    setBusy(true);
    try {
      const result = await removeToolSpace(slug);
      if (signal?.aborted) return;
      render(result);
      showStatus(result?.message || `Removed “${slug}”.`);
      await onChanged?.();
    } catch (error) {
      if (!signal?.aborted) showStatus(`Failed to remove: ${describeError(error)}`, true);
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  }

  function renderSearchResults(results) {
    searchResults.replaceChildren();
    if (!results.length) {
      searchResults.appendChild(
        h("p", { class: "settings-hint" }, "No running MCP Spaces match. Try other words.")
      );
      return;
    }
    for (const result of results) {
      const likes = Number(result.likes) || 0;
      const addResultButton = h(
        "button",
        {
          type: "button",
          class: "btn btn--ghost",
          "aria-label": `Add Tool Space ${result.slug}`,
          dataset: { slug: result.slug, installed: result.installed ? "true" : "false" },
          disabled: busy || !editable || result.installed ? "disabled" : null,
        },
        result.installed ? "Installed" : "Add"
      );
      addResultButton.addEventListener("click", () => installSpace(result.slug));
      searchResults.appendChild(
        h(
          "div",
          { class: "settings-tool-space", role: "listitem" },
          h(
            "div",
            { class: "settings-tool-space-summary" },
            h("strong", { class: "settings-tool-space-name" }, `${result.emoji || ""} ${result.title}`.trim()),
            result.description ? h("span", { class: "settings-tool-space-meta" }, result.description) : null,
            h("span", { class: "settings-tool-space-meta" }, `${result.slug} · ${likes} ${likes === 1 ? "like" : "likes"}`)
          ),
          addResultButton
        )
      );
    }
  }

  searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = searchInput.value.trim();
    if (!query || searchButton.disabled) return;
    searchButton.disabled = true;
    searchResults.replaceChildren(h("p", { class: "settings-hint" }, "Searching Hugging Face…"));
    try {
      const payload = await searchToolSpaces(query);
      if (signal?.aborted) return;
      renderSearchResults(Array.isArray(payload?.results) ? payload.results : []);
    } catch (error) {
      if (signal?.aborted) return;
      searchResults.replaceChildren(
        h("p", { class: "settings-status is-error" }, `Search failed: ${describeError(error)}`)
      );
    } finally {
      if (!signal?.aborted) searchButton.disabled = false;
    }
  });

  addForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (addButton.disabled) return;
    const slug = slugInput.value.trim();
    if (!slug) return;
    if (await installSpace(slug)) slugInput.value = "";
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
