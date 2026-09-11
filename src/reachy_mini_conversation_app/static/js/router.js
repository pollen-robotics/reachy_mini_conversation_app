/**
 * Minimal hash router. Handlers receive
 * { outlet, signal, searchParams, setLeaveGuard, replaceRoute }.
 */

export function createRouter(routes, { fallback = "#/", outlet, onRouteChange } = {}) {
  if (!outlet) throw new Error("createRouter: outlet is required");
  let currentController = null;
  let currentRoute = null;
  let leaveGuard = null;
  let pendingTransition = Promise.resolve();

  function resolve(route = window.location.hash || fallback) {
    const routeName = route.split("?")[0];
    return Object.prototype.hasOwnProperty.call(routes, routeName) ? route : fallback;
  }

  function renderRouteError(route, error) {
    const div = document.createElement("div");
    div.className = "route-error";
    div.textContent = `Failed to render ${route}: ${error?.message || error}`;
    return div;
  }

  function mount(route) {
    leaveGuard = null;
    currentController?.abort();
    outlet.replaceChildren();

    currentRoute = route;
    currentController = new AbortController();
    const controller = currentController;
    const routeName = route.split("?")[0];
    const queryStart = route.indexOf("?");
    const searchParams = new URLSearchParams(queryStart === -1 ? "" : route.slice(queryStart + 1));
    const context = {
      outlet,
      signal: controller.signal,
      searchParams,
      setLeaveGuard(guard) {
        if (!controller.signal.aborted) leaveGuard = guard;
      },
      replaceRoute(nextRoute) {
        if (controller.signal.aborted) return;
        const resolvedRoute = resolve(nextRoute);
        if (resolvedRoute.split("?")[0] !== routeName) {
          throw new Error("replaceRoute cannot change views");
        }
        currentRoute = resolvedRoute;
        window.history.replaceState(null, "", resolvedRoute);
        onRouteChange?.(resolvedRoute);
      },
    };
    try {
      Promise.resolve(routes[routeName](context)).catch((error) => {
        if (context.signal.aborted) return;
        console.error("Route handler failed for", route, error);
        outlet.replaceChildren(renderRouteError(route, error));
      });
    } catch (error) {
      console.error("Route handler failed for", route, error);
      outlet.replaceChildren(renderRouteError(route, error));
    }
    onRouteChange?.(route);
  }

  async function transitionTo(route, updateHash) {
    const nextRoute = resolve(route);
    if (nextRoute === currentRoute) {
      if (currentRoute && window.location.hash !== currentRoute) {
        window.history.replaceState(null, "", currentRoute);
      }
      return true;
    }

    if (leaveGuard?.shouldBlock() && !(await leaveGuard.confirm())) {
      if (currentRoute && window.location.hash !== currentRoute) {
        window.location.hash = currentRoute;
      }
      return false;
    }

    if (updateHash && window.location.hash !== nextRoute) {
      window.location.hash = nextRoute;
    } else if (window.location.hash !== nextRoute) {
      window.history.replaceState(null, "", nextRoute);
    }
    mount(nextRoute);
    return true;
  }

  function enqueueTransition(route, updateHash = false) {
    const transition = pendingTransition.then(() => transitionTo(route, updateHash));
    pendingTransition = transition.catch((error) => {
      console.error("Route transition failed", error);
    });
    return transition;
  }

  function onBeforeUnload(event) {
    if (!leaveGuard?.shouldBlock()) return;
    event.preventDefault();
    event.returnValue = "";
  }

  return {
    start() {
      window.addEventListener("hashchange", () => enqueueTransition(window.location.hash));
      window.addEventListener("beforeunload", onBeforeUnload);
      const target = resolve();
      if (window.location.hash !== target) {
        window.history.replaceState(null, "", target);
      }
      void enqueueTransition(target);
    },
    navigate(route) {
      return enqueueTransition(route, true);
    },
    currentRoute() {
      return currentRoute;
    },
  };
}
