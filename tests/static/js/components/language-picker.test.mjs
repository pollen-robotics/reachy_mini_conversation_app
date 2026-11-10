import { test } from "node:test";
import assert from "node:assert/strict";

class Element extends EventTarget {
  constructor() {
    super();
    this.children = [];
    this.hidden = false;
  }
  setAttribute(name, value) {
    this[name] = value;
  }
  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
  }
  replaceChildren(...children) {
    this.children = children;
  }
}

globalThis.Node = Element;
globalThis.document = Object.assign(new EventTarget(), {
  createElement: () => new Element(),
  createTextNode: (textContent) => Object.assign(new Element(), { textContent })
});
globalThis.location = { protocol: "http:", host: "localhost" };
let languageRequests = 0;
class WebSocket {
  static OPEN = 1;
  readyState = WebSocket.OPEN;
  constructor() {
    queueMicrotask(() => this.onopen());
  }
  send(request) {
    const { id, method } = JSON.parse(request);
    assert.equal(method, "language.get");
    languageRequests += 1;
    const reply =
      languageRequests === 1
        ? {
            id,
            error: {
              message: "Still starting",
              data: { reason: "loop_unavailable" }
            }
          }
        : {
            id,
            result: {
              language: "en",
              profile: "default",
              profile_language: null,
              choices: [
                {
                  code: "en",
                  name: "English",
                  native_name: "English",
                  flag: "🇬🇧"
                }
              ]
            }
          };
    queueMicrotask(() => this.onmessage({ data: JSON.stringify(reply) }));
  }
}
globalThis.WebSocket = WebSocket;
const { createLanguagePicker } =
  await import("../../../../src/reachy_mini_conversation_app/static/js/components/language-picker.js");

test("language picker becomes visible when the backend becomes ready", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const controller = new AbortController();
  const root = new Element();
  root.hidden = true;
  const picker = createLanguagePicker({ root, signal: controller.signal });
  const refresh = picker.refresh();
  await new Promise(setImmediate);
  assert.equal(root.hidden, true);
  context.mock.timers.tick(2000);
  await refresh;
  assert.equal(languageRequests, 2);
  assert.equal(root.hidden, false);
  assert.match(root.children[0]["aria-label"], /Conversation language: English/);
  controller.abort();
});

test("leaving the view during startup keeps the picker hidden", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  context.mock.method(console, "warn", () => {});
  languageRequests = 0;
  const controller = new AbortController();
  const root = new Element();
  root.hidden = true;
  const picker = createLanguagePicker({ root, signal: controller.signal });
  const refresh = picker.refresh();
  await new Promise(setImmediate);
  controller.abort();
  context.mock.timers.tick(2000);
  await refresh;
  assert.equal(languageRequests, 1);
  assert.equal(root.hidden, true);
  assert.deepEqual(root.children, []);
});
