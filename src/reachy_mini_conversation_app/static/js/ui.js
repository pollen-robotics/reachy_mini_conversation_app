/** Tiny DOM helpers: h(tag, attrs, ...children), $, prettifyProfileName. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") {
      el.className = Array.isArray(value) ? value.filter(Boolean).join(" ") : String(value);
    } else if (key === "style" && typeof value === "object") {
      Object.assign(el.style, value);
    } else if (key === "dataset" && typeof value === "object") {
      for (const [dk, dv] of Object.entries(value)) {
        if (dv != null) el.dataset[dk] = String(dv);
      }
    } else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "html") {
      // Escape hatch for trusted HTML (e.g. inline SVG icons we author).
      el.innerHTML = String(value);
    } else {
      el.setAttribute(key, String(value));
    }
  }
  appendChildren(el, children);
  return el;
}

function appendChildren(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false || child === true) continue;
    if (child instanceof Node) {
      parent.appendChild(child);
    } else {
      parent.appendChild(document.createTextNode(String(child)));
    }
  }
}

export function $(selector, root = document) {
  return root.querySelector(selector);
}

export function prettifyProfileName(name) {
  const stripped = name.replace(/^user_personalities\//, "");
  return stripped
    .split(/[_-]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
