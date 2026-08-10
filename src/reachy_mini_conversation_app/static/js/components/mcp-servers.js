/** Custom MCP servers section: add, remove, and set each server's access token. */

import {
  addMcpServer,
  describeError,
  listMcpServers,
  removeMcpServer,
  saveMcpServerToken,
} from "../api.js";
import { h } from "../ui.js";
import { confirmDialog } from "./confirm-dialog.js";

export function buildMcpServersSection({ signal, onBeforeChange, onChanged } = {}) {
  const aliasInput = h("input", {
    id: "mcp-server-alias",
    type: "text",
    name: "alias",
    required: "required",
    autocomplete: "off",
    autocapitalize: "none",
    spellcheck: "false",
    placeholder: "my_server",
    class: "settings-input",
  });
  const urlInput = h("input", {
    id: "mcp-server-url",
    type: "url",
    name: "url",
    required: "required",
    autocomplete: "off",
    autocapitalize: "none",
    spellcheck: "false",
    enterkeyhint: "go",
    placeholder: "https://example.com/mcp",
    class: "settings-input",
  });
  const tokenEnvInput = h("input", {
    id: "mcp-server-token-env",
    type: "text",
    name: "token_env",
    autocomplete: "off",
    autocapitalize: "none",
    spellcheck: "false",
    placeholder: "MY_SERVER_TOKEN (optional)",
    class: "settings-input",
  });
  const addButton = h("button", { type: "submit", class: "btn btn--primary" }, "Add server");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const list = h(
    "div",
    { class: "settings-tool-spaces", role: "list", "aria-live": "polite" },
    h("p", { class: "settings-hint" }, "Loading configured servers…")
  );
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-label", for: "mcp-server-alias" }, "Alias"),
    aliasInput,
    h("label", { class: "settings-label", for: "mcp-server-url" }, "MCP endpoint"),
    h("div", { class: "settings-tool-space-controls" }, urlInput, addButton),
    h("label", { class: "settings-label", for: "mcp-server-token-env" }, "Token variable"),
    tokenEnvInput,
    h(
      "p",
      { class: "settings-hint" },
      "Connect any MCP server over HTTPS. The alias namespaces its tools, so tools arrive as " +
        "alias__tool. If the server needs a token, name the environment variable to read it from — " +
        "the token itself is stored separately, never in the server list."
    ),
    status
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "MCP servers"),
    form,
    h("h3", { class: "settings-list-title" }, "Configured"),
    list
  );

  let busy = false;
  let editable = true;
  // Tokens typed but not yet saved, carried across every re-render so a refresh
  // does not clobber a token the user is in the middle of typing.
  const pendingTokens = new Map();
  // Skip a re-render when the server list is unchanged, for the same reason.
  let renderedSignature = null;

  function harvestPendingTokens() {
    list.querySelectorAll("input[data-alias]").forEach((input) => {
      if (input.value) pendingTokens.set(input.dataset.alias, input.value);
      else pendingTokens.delete(input.dataset.alias);
    });
  }

  function setBusy(nextBusy, addLabel = "Add server") {
    busy = nextBusy;
    form.toggleAttribute("aria-busy", nextBusy);
    list.toggleAttribute("aria-busy", nextBusy);
    for (const input of [aliasInput, urlInput, tokenEnvInput]) {
      input.disabled = nextBusy || !editable;
    }
    addButton.disabled = nextBusy || !editable;
    addButton.textContent = nextBusy ? addLabel : "Add server";
    list.querySelectorAll("button").forEach((button) => {
      button.disabled = nextBusy || !editable;
    });
  }

  function buildTokenControls(server) {
    if (!server.token_env) return null;
    const input = h("input", {
      type: "password",
      class: "settings-input",
      autocomplete: "off",
      spellcheck: "false",
      dataset: { alias: server.alias },
      "aria-label": `Access token for ${server.alias}`,
      placeholder: server.token_set ? "Token stored — enter a new one to replace" : `token for ${server.token_env}`,
    });
    input.value = pendingTokens.get(server.alias) || "";
    const saveButton = h(
      "button",
      { type: "button", class: "btn btn--ghost", disabled: busy || !editable ? "disabled" : null },
      server.token_set ? "Replace token" : "Save token"
    );
    saveButton.addEventListener("click", async () => {
      const token = (input.value || "").trim();
      if (!token) {
        status.classList.add("is-error");
        status.textContent = `Enter a token for ${server.alias}.`;
        return;
      }
      status.classList.remove("is-error");
      status.textContent = `Saving token for ${server.alias}…`;
      setBusy(true, "Saving token…");
      try {
        const result = await saveMcpServerToken(server.alias, token);
        if (signal?.aborted) return;
        pendingTokens.delete(server.alias);
        input.value = "";
        renderedSignature = null;
        render(result);
        status.textContent = result?.message || `Saved the token for ${server.alias}.`;
        await onChanged?.();
      } catch (error) {
        if (signal?.aborted) return;
        status.textContent = `Could not save the token for ${server.alias}: ${describeError(error)}`;
        status.classList.add("is-error");
      } finally {
        if (!signal?.aborted) setBusy(false);
      }
    });
    return h("div", { class: "settings-tool-space-controls" }, input, saveButton);
  }

  function buildRemoveButton(server) {
    const removeButton = h(
      "button",
      {
        type: "button",
        class: "btn btn--ghost",
        "aria-label": `Remove MCP server ${server.alias}`,
        disabled: busy || !editable ? "disabled" : null,
      },
      "Remove"
    );
    removeButton.addEventListener("click", async () => {
      if (onBeforeChange && !(await onBeforeChange())) return;
      const confirmed = await confirmDialog({
        title: "Remove MCP server?",
        message: `Removing “${server.alias}” disables its tools in every personality.`,
        confirmLabel: "Remove",
        danger: true,
        signal,
      });
      if (!confirmed || signal?.aborted) return;

      status.classList.remove("is-error");
      status.textContent = `Removing “${server.alias}”…`;
      setBusy(true, "Removing…");
      try {
        const result = await removeMcpServer(server.alias);
        if (signal?.aborted) return;
        pendingTokens.delete(server.alias);
        renderedSignature = null;
        render(result);
        status.textContent = result?.message || `Removed “${server.alias}”.`;
        await onChanged?.();
      } catch (error) {
        if (signal?.aborted) return;
        status.textContent = `Failed to remove: ${describeError(error)}`;
        status.classList.add("is-error");
      } finally {
        if (!signal?.aborted) setBusy(false);
      }
    });
    return removeButton;
  }

  function render(payload) {
    editable = payload?.editable !== false;
    const servers = Array.isArray(payload?.servers) ? payload.servers : [];
    const nextSignature = JSON.stringify(
      servers.map((server) => [server.alias, server.url, server.tool_count, server.token_env, !!server.token_set])
    );
    if (nextSignature === renderedSignature) {
      setBusy(busy);
      return;
    }
    harvestPendingTokens();
    renderedSignature = nextSignature;

    list.replaceChildren();
    if (!servers.length) {
      list.appendChild(h("p", { class: "settings-hint" }, "No MCP servers configured."));
      setBusy(busy);
      if (!editable) status.textContent = "MCP server editing is locked by the administrator.";
      return;
    }

    for (const server of servers) {
      const toolCount = Number(server.tool_count) || 0;
      const details = [server.url, `${toolCount} ${toolCount === 1 ? "tool" : "tools"}`];
      if (server.token_env) details.push(server.token_set ? "Token set" : "Token needed");
      list.appendChild(
        h(
          "div",
          { class: "settings-tool-space", role: "listitem" },
          h(
            "div",
            { class: "settings-tool-space-summary" },
            h("strong", { class: "settings-tool-space-name" }, server.alias),
            h("span", { class: "settings-tool-space-meta" }, details.join(" · ")),
            buildTokenControls(server)
          ),
          buildRemoveButton(server)
        )
      );
    }
    setBusy(busy);
    if (!editable) status.textContent = "MCP server editing is locked by the administrator.";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (addButton.disabled) return;
    const alias = aliasInput.value.trim();
    const url = urlInput.value.trim();
    if (!alias || !url) return;
    if (onBeforeChange && !(await onBeforeChange())) return;

    status.classList.remove("is-error");
    status.textContent = `Connecting to “${alias}”…`;
    setBusy(true, "Connecting…");
    try {
      const tokenEnv = tokenEnvInput.value.trim();
      const result = await addMcpServer(tokenEnv ? { alias, url, token_env: tokenEnv } : { alias, url });
      if (signal?.aborted) return;
      aliasInput.value = "";
      urlInput.value = "";
      tokenEnvInput.value = "";
      renderedSignature = null;
      render(result);
      status.textContent = result?.message || `Added “${alias}”.`;
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
        const payload = await listMcpServers();
        if (signal?.aborted) return;
        render(payload);
      } catch (error) {
        if (signal?.aborted) return;
        renderedSignature = null;
        list.replaceChildren(
          h("p", { class: "settings-status is-error" }, `Could not load MCP servers: ${describeError(error)}`)
        );
      }
    },
  };
}
