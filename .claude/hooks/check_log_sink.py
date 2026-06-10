"""SessionStart hook: report whether the robot event-relay log sink is up.

The robot fire-and-forgets telemetry spans over UDP (``ROBOT_EVENT_RELAY``);
``robot-comic-logsink`` on the dev workstation must be listening or the spans
silently vanish (that is the relay's graceful-by-design contract). This hook
surfaces the sink's status into session context so the agent nudges the
operator — or offers to start it — before live-robot work begins.

Detection is dependency-free and cross-platform: if binding the sink's UDP
port succeeds, nothing is listening (NOT RUNNING); if the bind is refused,
the sink (or something) holds the port (RUNNING). File freshness of the sink
output adds "is data actually flowing" signal.

Quiet by design on machines where the relay setup doesn't exist (e.g. the
robot itself): no sink file configured and default port free → single line,
no noise. Never exits non-zero — a hook must not block a session.
"""

from __future__ import annotations
import os
import sys
import time
import socket


PORT = int(os.getenv("ROBOT_LOGSINK_PORT", "9477"))
SINK_FILE = os.getenv("ROBOT_LOGSINK_FILE", r"D:\logs\ricci_events.log" if os.name == "nt" else "")


def port_in_use(port: int) -> bool:
    """Return True when something already listens on the UDP port (the sink)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Match the sink's exact bind (0.0.0.0, no SO_REUSEADDR): if it is
        # up, this raises; if the bind succeeds, the port was free.
        probe.bind(("0.0.0.0", port))
        return False
    except OSError:
        return True
    finally:
        probe.close()


def main() -> int:
    """Print one status line for the session context."""
    running = port_in_use(PORT)
    if running:
        freshness = ""
        if SINK_FILE and os.path.exists(SINK_FILE):
            age_s = time.time() - os.path.getmtime(SINK_FILE)
            freshness = f"; {os.path.basename(SINK_FILE)} updated {int(age_s)}s ago"
        print(f"[log sink] RUNNING on udp/{PORT}{freshness}")
    else:
        cmd = f"robot-comic-logsink --port {PORT}" + (f" --file {SINK_FILE}" if SINK_FILE else "")
        # ASCII-only: hook stdout may pass through a cp1252 console.
        print(
            f"[log sink] NOT RUNNING (udp/{PORT} free) - robot telemetry relay is being dropped. "
            f"Nudge the operator to start it, or offer to run it in the background yourself: {cmd}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a status hook must never block the session
        print(f"[log sink] status check failed: {exc}")
        sys.exit(0)
