/** Accessory view: the NFC reader, and what the accessory on it carries. */

import {
  describeError,
  eraseRfidTag,
  getRfidStatus,
  linkRfidTag,
  listPersonalities,
  openAddOnStore,
  subscribe,
  untilReady,
} from "../api.js";
import { h, prettifyProfileName } from "../ui.js";
import { confirmDialog } from "../components/confirm-dialog.js";

const NO_READER_ACCESSORY = Object.freeze({ state: "unavailable", personality: null, content: null });

const LINK_LABEL = "Link accessory";
const ERASE_LABEL = "Erase content";

/** States that mean the tag holds something, so there is something to erase. */
function carriesContent(state) {
  return state === "known" || state === "unknown";
}

const ACCESSORY_COPY = Object.freeze({
  none: {
    title: "No accessory",
    hint: "Place an accessory on Reachy Mini's head to see what it carries.",
  },
  blank: {
    title: "Blank accessory",
    hint: "This accessory carries nothing yet. Link it to a personality below, or ask Reachy Mini to invent one for it.",
  },
  known: {
    title: "Linked accessory",
    hint: "Reachy Mini takes on this personality while the accessory is on its head.",
  },
  unknown: {
    title: "Unrecognized accessory",
    hint: "This accessory carries something this robot cannot use — a personality it does not have, or a tag written elsewhere.",
  },
});

export async function mountAccessoryView({ outlet, signal }) {
  const readerStatus = h("p", { class: "settings-hint", "data-role": "reader" }, "Checking the reader…");
  const accessoryCard = h("div", { class: "accessory-card", "aria-live": "polite" });
  const readerTitle = h("h2", { class: "settings-section-title" }, "On the reader");
  const addOnButton = h(
    "button",
    { type: "button", class: "btn btn--ghost accessory-requirement__link" },
    "Get the NFC add-on"
  );
  const addOnAddress = h("p", { class: "settings-hint accessory-requirement__address", hidden: "hidden" });
  // The control app's webview ignores window.open, so the app opens the store
  // itself; with no browser on its host, the address is shown instead.
  addOnButton.addEventListener("click", async () => {
    let store = null;
    let failure = null;
    try {
      store = await openAddOnStore();
    } catch (error) {
      failure = describeError(error);
      console.warn("Could not open the add-on store:", error);
    }
    if (store?.opened) return;
    if (store?.url && window.open(store.url, "_blank", "noopener")) return;
    addOnAddress.replaceChildren(
      store?.url
        ? "Open this address in a browser: "
        : `The store could not be opened: ${failure || "the app gave no address"}.`
    );
    if (store?.url) addOnAddress.appendChild(h("code", null, store.url));
    addOnAddress.hidden = false;
  });
  // Shown in the reader panel's place when there is no reader to report on.
  const requirementPanel = [
    h("h2", { class: "settings-section-title" }, "What does the accessory feature require?"),
    h(
      "p",
      { class: "settings-hint" },
      "Accessories are read by the Reachy Mini NFC add-on. Fit it to give a personality to a hat or "
        + "any other object, and Reachy Mini takes that personality on as soon as the object is placed "
        + "on its head."
    ),
    addOnButton,
    addOnAddress,
  ];
  // Shown instead when the reader was switched off: the add-on is most likely
  // fitted already, so the store link would only be noise.
  const turnedOffPanel = [
    h("h2", { class: "settings-section-title" }, "NFC reader turned off"),
    h(
      "p",
      { class: "settings-hint" },
      "The NFC reader is switched off, so Reachy Mini cannot read accessories. "
        + "Turn it back on in the NFC Reader settings of the Reachy Mini Control app."
    ),
  ];
  const readerSection = h("section", { class: "settings-section" }, readerTitle, readerStatus, accessoryCard);
  const personalitySelect = h("select", {
    class: "settings-select",
    "aria-label": "Personality to link",
    disabled: "disabled",
  });
  const linkButton = h("button", { type: "button", class: "btn btn--primary", disabled: "disabled" }, LINK_LABEL);
  // Rendered inside the tag card, next to what it erases.
  const eraseButton = h("button", { type: "button", class: "btn btn--ghost", disabled: "disabled" }, ERASE_LABEL);
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });

  const linkSection = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Link a personality"),
    h(
      "p",
      { class: "settings-hint settings-section-intro" },
      "The accessory carries the personality itself, so it means the same thing on any Reachy Mini. "
        + "Linking again replaces whatever it carries."
    ),
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Personality"),
      personalitySelect
    ),
    h("div", { class: "settings-actions" }, linkButton),
    status
  );

  const view = h(
    "section",
    { class: "view view--accessory" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Accessory"),
      h(
        "p",
        { class: "view-subtitle" },
        "See what the accessory on Reachy Mini's head carries, and choose the personality it should apply."
      )
    ),
    readerSection,
    linkSection
  );
  outlet.replaceChildren(view);

  let latest = null;
  let busy = false;

  function setBusy(nextBusy, labels = {}) {
    busy = nextBusy;
    view.toggleAttribute("aria-busy", nextBusy);
    linkButton.disabled = nextBusy || !canLink();
    eraseButton.disabled = nextBusy || !canErase();
    personalitySelect.disabled = nextBusy || !personalitySelect.options.length;
    linkButton.textContent = (nextBusy && labels.link) || LINK_LABEL;
    eraseButton.textContent = (nextBusy && labels.erase) || ERASE_LABEL;
  }

  // Only a tag state means a tag is on a reader that answers, so neither action
  // needs to check the connection on top.
  function canLink() {
    const state = latest?.accessory?.state;
    return (state === "blank" || state === "known" || state === "unknown") && personalitySelect.value !== "";
  }

  function canErase() {
    return carriesContent(latest?.accessory?.state);
  }

  function renderAccessory(accessory) {
    const copy = ACCESSORY_COPY[accessory.state] || ACCESSORY_COPY.none;
    accessoryCard.dataset.state = accessory.state;
    accessoryCard.replaceChildren(
      h(
        "div",
        { class: "accessory-card__copy" },
        h("strong", null, copy.title),
        accessory.personality
          ? h("span", { class: "accessory-card__name" }, prettifyProfileName(accessory.personality))
          : null,
        h("span", { class: "settings-hint" }, copy.hint),
        tagDetails(accessory),
        carriesContent(accessory.state)
          ? h("div", { class: "accessory-card__actions" }, eraseButton)
          : null
      )
    );
  }

  /** What the tag literally holds. Useful when a tag is not recognized. */
  function tagDetails(accessory) {
    if (!carriesContent(accessory.state)) return null;
    return h(
      "dl",
      { class: "accessory-card__details" },
      detailRow("Personality", accessory.personality ? prettifyProfileName(accessory.personality) : "—"),
      detailRow("Written on the tag", accessory.content || "—")
    );
  }

  function detailRow(label, value) {
    return h(
      "div",
      { class: "accessory-card__detail" },
      h("dt", null, label),
      h("dd", null, value)
    );
  }

  function render(payload) {
    latest = payload;
    const accessory = payload?.accessory || NO_READER_ACCESSORY;
    const hasReader = accessory.state !== "unavailable";
    // Reporting on a reader that is not there would read as an empty one, so
    // the panel says what the feature needs instead.
    if (hasReader) {
      readerSection.replaceChildren(readerTitle, readerStatus, accessoryCard);
      readerStatus.textContent = payload.port ? `Reader connected on ${payload.port}.` : "Reader connected.";
      renderAccessory(accessory);
    } else if (payload?.enabled === false) {
      readerSection.replaceChildren(...turnedOffPanel);
    } else {
      readerSection.replaceChildren(...requirementPanel);
    }
    linkSection.hidden = !hasReader;
    // A tag arriving or leaving changes what the buttons can do.
    setBusy(busy);
  }

  let choices;
  try {
    choices = await untilReady(listPersonalities, signal, () => {
      readerStatus.textContent = "Waiting for Reachy to finish starting…";
    });
  } catch (error) {
    if (signal.aborted) return;
    status.textContent = describeError(error);
    status.classList.add("is-error");
    return;
  }
  if (signal.aborted) return;

  for (const name of choices?.choices || []) {
    personalitySelect.appendChild(h("option", { value: name }, prettifyProfileName(name)));
  }
  if (!personalitySelect.options.length) {
    personalitySelect.appendChild(h("option", { value: "" }, "No personality to link"));
  }

  try {
    render(await getRfidStatus());
  } catch (error) {
    if (!signal.aborted) {
      status.textContent = describeError(error);
      status.classList.add("is-error");
    }
  }

  const unsubscribe = subscribe("rfid.tag", (payload) => {
    if (!signal.aborted) render(payload);
  });
  signal.addEventListener("abort", unsubscribe, { once: true });

  personalitySelect.addEventListener("change", () => setBusy(busy));

  linkButton.addEventListener("click", async () => {
    if (busy || !canLink()) return;
    const personality = personalitySelect.value;
    status.classList.remove("is-error");
    status.textContent = "";
    setBusy(true, { link: "Linking…" });
    try {
      const result = await linkRfidTag(personality);
      status.textContent = result?.written
        ? `Accessory linked to ${prettifyProfileName(personality)}.`
        : `This accessory already carries ${prettifyProfileName(personality)}.`;
      render(await getRfidStatus());
    } catch (error) {
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      setBusy(false);
    }
  });

  eraseButton.addEventListener("click", async () => {
    if (busy || !canErase()) return;
    const confirmed = await confirmDialog({
      title: "Erase this accessory?",
      message: "It will carry nothing afterwards, and Reachy Mini will stop changing personality for it.",
      confirmLabel: "Erase",
      danger: true,
      signal,
    });
    if (!confirmed) return;
    status.classList.remove("is-error");
    status.textContent = "";
    setBusy(true, { erase: "Erasing…" });
    try {
      await eraseRfidTag(false);
      status.textContent = "Accessory erased.";
      render(await getRfidStatus());
    } catch (error) {
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      setBusy(false);
    }
  });
}
