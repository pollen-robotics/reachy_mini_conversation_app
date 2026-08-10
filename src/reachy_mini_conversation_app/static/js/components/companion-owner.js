/** Render links to the exact managed assistant resources. */

import { h } from "../ui.js";

export function buildCompanionOwnership() {
  const namespaceLabel = h("strong");
  const spaceLink = h("a", { target: "_blank", rel: "noopener noreferrer" }, "Space");
  const bucketLink = h("a", { target: "_blank", rel: "noopener noreferrer" }, "Storage");
  const element = h(
    "p",
    { class: "companion-ownership", hidden: "hidden" },
    "Hugging Face namespace ",
    namespaceLabel,
    h("span", { "aria-hidden": "true" }, " · "),
    spaceLink,
    h("span", { "aria-hidden": "true" }, " · "),
    bucketLink
  );

  return {
    element,
    render(setup) {
      const namespace = typeof setup?.namespace === "string" ? setup.namespace : "";
      if (!namespace || typeof setup?.space_url !== "string" || typeof setup?.bucket_url !== "string") {
        element.hidden = true;
        return;
      }
      let spaceUrl;
      let bucketUrl;
      try {
        spaceUrl = new URL(setup?.space_url);
        bucketUrl = new URL(setup?.bucket_url);
      } catch {
        element.hidden = true;
        return;
      }
      const valid =
        spaceUrl.origin === "https://huggingface.co" &&
        !spaceUrl.username &&
        !spaceUrl.password &&
        spaceUrl.pathname.startsWith("/spaces/") &&
        !spaceUrl.search &&
        !spaceUrl.hash &&
        bucketUrl.origin === "https://huggingface.co" &&
        !bucketUrl.username &&
        !bucketUrl.password &&
        bucketUrl.pathname.startsWith("/buckets/") &&
        !bucketUrl.search &&
        !bucketUrl.hash;
      element.hidden = !valid;
      if (!valid) return;
      namespaceLabel.textContent = `@${namespace}`;
      spaceLink.href = spaceUrl.href;
      bucketLink.href = bucketUrl.href;
    },
  };
}
