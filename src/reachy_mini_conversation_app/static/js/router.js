/**
 * Minimal hash router. Handlers receive a ViewContext { outlet, signal, route };
 * signal is aborted when the view is replaced so cleanups (timers, SSE) can use it.
 */

export function createRouter(routes, { fallback = "#/", outlet } = {}) {
  if (!outlet) throw new Error("createRouter: outlet is required");
  let currentController = null;
  let lastRoute = null;

  function resolve() {
    const raw = window.location.hash || fallback;
    return Object.prototype.hasOwnProperty.call(routes, raw) ? raw : fallback;
  }

  function dispatch() {
    const route = resolve();
    if (route === lastRoute) return;

    // Tear down the previous view first.
    if (currentController) {
      currentController.abort();
      currentController = null;
    }
    outlet.replaceChildren();

    lastRoute = route;
    currentController = new AbortController();
    /** @type {ViewContext} */
    const ctx = { outlet, signal: currentController.signal, route };
    try {
      routes[route](ctx);
    } catch (error) {
      console.error("Route handler failed for", route, error);
      outlet.replaceChildren(renderRouteError(route, error));
    }
  }

  function renderRouteError(route, error) {
    const div = document.createElement("div");
    div.className = "route-error";
    div.textContent = `Failed to render ${route}: ${error?.message || error}`;
    return div;
  }

  return {
    start() {
      window.addEventListener("hashchange", dispatch);
      const target = resolve();
      if (window.location.hash !== target) {
        window.location.replace(target);
      }
      dispatch();
    },
    /** Navigate and dispatch synchronously; the resulting hashchange is a no-op via lastRoute. */
    navigate(route) {
      if (window.location.hash === route) return;
      window.location.hash = route;
      dispatch();
    },
  };
}
