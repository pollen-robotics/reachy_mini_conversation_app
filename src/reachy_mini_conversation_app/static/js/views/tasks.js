/** View for durable companion tasks and results. */

import {
  cancelCompanionTask,
  describeError,
  getCompanionTaskResult,
  listCompanionTasks,
} from "../api.js";
import { h } from "../ui.js";
import { ACTIVE_COMPANION_SETUP_STATES } from "../constants.js";
import { confirmDialog } from "../components/confirm-dialog.js";
import { buildCompanionOwnership } from "../components/companion-owner.js";

const STATUS_LABELS = Object.freeze({
  queued: "Queued",
  running: "Running",
  input_required: "Needs input",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
});

const TASK_GROUPS = Object.freeze([
  { title: "Needs input", statuses: ["input_required"] },
  { title: "In progress", statuses: ["queued", "running"] },
  { title: "Recent", statuses: ["completed", "failed", "cancelled"] },
]);
const ACTIVE_TASK_STATUSES = new Set(["queued", "running", "input_required"]);
const AUTO_REFRESH_INTERVAL_MS = 5000;

export async function mountTasksView({ outlet, signal, openSettings }) {
  const ownership = buildCompanionOwnership();
  const refreshButton = h(
    "button",
    { type: "button", class: "btn btn--ghost tasks-refresh" },
    "Refresh"
  );
  const status = h("p", {
    class: "tasks-status",
    role: "status",
    "aria-live": "polite",
  });
  const list = h(
    "div",
    { class: "tasks-list", "aria-live": "polite" },
    h("p", { class: "muted" }, "Loading tasks…")
  );
  const view = h(
    "section",
    { class: "view view--tasks" },
    h(
      "header",
      { class: "view-header tasks-header" },
      h(
        "div",
        {},
        h("h1", { class: "view-title" }, "Tasks"),
        h("p", { class: "view-subtitle" }, "Recent work delegated to the background assistant.")
      ),
      refreshButton
    ),
    ownership.element,
    status,
    list
  );
  outlet.replaceChildren(view);
  let hasRendered = false;
  let loading = false;
  let stopping = false;
  let autoRefreshTimer = null;
  let latestTasks = [];
  let latestSetupState = "idle";
  let lastPayloadFingerprint = "";

  function clearAutoRefresh() {
    if (autoRefreshTimer === null) return;
    window.clearTimeout(autoRefreshTimer);
    autoRefreshTimer = null;
  }

  function scheduleAutoRefresh() {
    if (
      signal.aborted ||
      autoRefreshTimer !== null ||
      (!latestTasks.some((task) => ACTIVE_TASK_STATUSES.has(task.status)) &&
        !ACTIVE_COMPANION_SETUP_STATES.includes(latestSetupState))
    ) {
      return;
    }
    autoRefreshTimer = window.setTimeout(() => {
      autoRefreshTimer = null;
      void refresh({ automatic: true });
    }, AUTO_REFRESH_INTERVAL_MS);
  }

  async function refresh({ automatic = false } = {}) {
    if (loading || stopping) return;
    clearAutoRefresh();
    loading = true;
    refreshButton.disabled = true;
    if (!automatic) refreshButton.textContent = "Refreshing…";
    const stopButtons = list.querySelectorAll(".task-stop");
    for (const button of stopButtons) button.disabled = true;
    try {
      const payload = await listCompanionTasks();
      if (signal.aborted) return;
      ownership.render(payload?.setup);
      const tasks = Array.isArray(payload?.tasks) ? payload.tasks : [];
      latestTasks = tasks;
      latestSetupState = payload?.setup?.state || "idle";
      status.classList.remove("is-error");
      status.textContent =
        payload?.enabled === false
          ? "The background assistant is turned off. Existing tasks and briefs remain available."
          : "";
      const payloadFingerprint = JSON.stringify(payload);
      if (automatic && payloadFingerprint === lastPayloadFingerprint) {
        return;
      }
      list.replaceChildren();
      hasRendered = true;
      if (payload?.configured === false) {
        const setupState = payload?.setup?.state || "idle";
        const setupHeadings = {
          failed: "Assistant setup needs attention",
          provisioning: "Assistant setup in progress",
          restart_required: "Restart required",
          verifying: "Assistant setup in progress",
        };
        const setupChildren = [
          h("strong", {}, setupHeadings[setupState] || "Background assistant not configured"),
          h(
            "span",
            {},
            payload?.setup?.message || "Set up your private background assistant to see tasks here."
          ),
        ];
        if (setupState !== "restart_required") {
          const settingsButton = h(
            "button",
            { type: "button", class: "btn btn--ghost" },
            "Open assistant settings"
          );
          settingsButton.addEventListener("click", () => openSettings?.());
          setupChildren.push(settingsButton);
        }
        list.appendChild(
          h("div", { class: "tasks-empty" }, setupChildren)
        );
        lastPayloadFingerprint = payloadFingerprint;
        return;
      }
      if (!tasks.length) {
        list.appendChild(
          h(
            "div",
            { class: "tasks-empty" },
            h("strong", {}, "No delegated tasks yet"),
            h("span", {}, "Ask Reachy to hand off a longer task, then refresh this view.")
          )
        );
        lastPayloadFingerprint = payloadFingerprint;
        return;
      }

      for (const group of TASK_GROUPS) {
        const groupTasks = tasks.filter((task) => group.statuses.includes(task.status));
        if (!groupTasks.length) continue;
        const groupId = `tasks-${group.title.toLowerCase().replaceAll(" ", "-")}`;
        list.appendChild(
          h(
            "section",
            { class: "tasks-group", "aria-labelledby": groupId },
            h("h2", { class: "tasks-group__title", id: groupId }, group.title),
            h(
              "div",
              { class: "tasks-group__list" },
              groupTasks.map((task) => buildTaskCard(task, signal, stopTask))
            )
          )
        );
      }
      lastPayloadFingerprint = payloadFingerprint;
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Could not load tasks: ${describeError(error)}`;
      status.classList.add("is-error");
      if (!hasRendered) {
        list.replaceChildren(h("p", { class: "muted" }, "Tasks are unavailable."));
      }
    } finally {
      if (!signal.aborted) {
        loading = false;
        refreshButton.disabled = false;
        refreshButton.textContent = "Refresh";
        for (const button of stopButtons) {
          if (button.isConnected) button.disabled = false;
        }
        scheduleAutoRefresh();
      }
    }
  }

  async function stopTask(taskId, button) {
    if (loading || stopping || signal.aborted) return;
    clearAutoRefresh();
    const confirmed = await confirmDialog({
      title: "Stop this task?",
      message: "Its current work will be cancelled. You can start a new task anytime.",
      confirmLabel: "Stop task",
      danger: true,
      signal,
    });
    if (!confirmed || loading || stopping || signal.aborted) {
      scheduleAutoRefresh();
      return;
    }
    stopping = true;
    refreshButton.disabled = true;
    const originalLabel = button.getAttribute("aria-label") || "Stop task";
    button.disabled = true;
    button.textContent = "Stopping…";
    button.setAttribute("aria-label", originalLabel.replace("Stop task", "Stopping task"));
    status.classList.remove("is-error");
    status.textContent = "Stopping task…";
    let message = "Task stopped.";
    let failed = false;
    try {
      const payload = await cancelCompanionTask(taskId);
      if (signal.aborted) return;
      if (payload?.status !== "cancelled") {
        message = "The task finished before it could be stopped.";
      }
    } catch (error) {
      if (signal.aborted) return;
      message = `Could not stop task: ${describeError(error)}`;
      failed = true;
    } finally {
      stopping = false;
      if (signal.aborted) return;
      await refresh();
      if (!signal.aborted) {
        status.textContent = message;
        status.classList.toggle("is-error", failed);
        refreshButton.focus();
      }
    }
  }

  signal.addEventListener("abort", clearAutoRefresh, { once: true });
  refreshButton.addEventListener("click", () => void refresh());
  await refresh();
}

function buildTaskCard(task, signal, onStop) {
  const taskId = typeof task.task_id === "string" ? task.task_id : "";
  const status = typeof task.status === "string" ? task.status : "unknown";
  const summary = task.summary || "Background task";
  const timestamp = task.updated_at || task.created_at;
  const parsedTime = timestamp ? new Date(timestamp) : null;
  const timeLabel =
    parsedTime && !Number.isNaN(parsedTime.getTime())
      ? parsedTime.toLocaleString([], { dateStyle: "medium", timeStyle: "short" })
      : "Time unavailable";
  const body = [];

  if (status === "input_required" && task.question?.text) {
    const question = [h("p", { class: "task-question__text" }, task.question.text)];
    if (Array.isArray(task.question.options) && task.question.options.length) {
      question.push(
        h(
          "ul",
          { class: "task-question__options" },
          task.question.options.map((option) => h("li", {}, option))
        )
      );
    }
    question.push(h("p", { class: "muted small" }, "Answer through Reachy."));
    body.push(h("div", { class: "task-question" }, question));
  }
  if (status === "failed" && task.error) {
    body.push(h("p", { class: "task-error" }, task.error));
  }
  if (status === "completed" && task.result_available && taskId) {
    body.push(buildResultDisclosure(taskId, signal));
  }
  const controls = [
    h(
      "span",
      { class: ["task-status", `task-status--${status.replaceAll("_", "-")}`] },
      STATUS_LABELS[status] || "Unknown"
    ),
  ];
  if (taskId && ACTIVE_TASK_STATUSES.has(status)) {
    const stopButton = h(
      "button",
      {
        type: "button",
        class: "btn btn--ghost task-stop",
        "aria-label": `Stop task: ${summary}`,
      },
      "Stop"
    );
    stopButton.addEventListener("click", () => onStop(taskId, stopButton));
    controls.push(stopButton);
  }

  return h(
    "article",
    { class: "task-card" },
    h(
      "div",
      { class: "task-card__header" },
      h(
        "div",
        { class: "task-card__summary" },
        h("h3", {}, summary),
        h(
          "p",
          { class: "task-card__meta" },
          parsedTime ? h("time", { datetime: parsedTime.toISOString() }, timeLabel) : timeLabel
        )
      ),
      h("div", { class: "task-card__controls" }, controls)
    ),
    body
  );
}

function buildResultDisclosure(taskId, signal) {
  const summary = h("summary", { class: "task-result__summary" }, "Open brief");
  const content = h("div", { class: "task-result__content" });
  const details = h("details", { class: "task-result" }, summary, content);
  let loaded = false;
  let loading = false;

  details.addEventListener("toggle", async () => {
    if (!details.open || loaded || loading || signal.aborted) return;
    loading = true;
    content.replaceChildren(h("p", { class: "muted small" }, "Loading brief…"));
    try {
      const result = await getCompanionTaskResult(taskId);
      if (signal.aborted) return;
      const markdown = typeof result.markdown === "string" ? result.markdown : "";
      const downloadButton = h(
        "button",
        { type: "button", class: "btn btn--ghost task-result__download" },
        "Download brief"
      );
      downloadButton.addEventListener("click", () => {
        const downloadUrl = URL.createObjectURL(
          new Blob([markdown], { type: "text/markdown;charset=utf-8" })
        );
        const link = h("a", { href: downloadUrl, download: "brief.md" });
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
      });
      content.replaceChildren(
        downloadButton,
        h("pre", { class: "task-result__markdown" }, markdown)
      );
      loaded = true;
    } catch (error) {
      if (signal.aborted) return;
      content.replaceChildren(
        h("p", { class: "task-error" }, `Could not open brief: ${describeError(error)}`)
      );
    } finally {
      loading = false;
    }
  });

  return details;
}
