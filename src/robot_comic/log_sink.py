"""Workstation-side UDP log sink for the robot's event relay.

The robot fire-and-forgets one UDP datagram per telemetry span when
``ROBOT_EVENT_RELAY=host:port`` is set (see ``telemetry.UdpEventExporter``).
This sink listens on that port and appends each datagram as an
``RCSPAN``-prefixed line, so the captured file is directly consumable by:

  * ``robot-comic-monitor --file <sink file>`` — live turn table off-robot
  * the MCP observer — a local event history that survives the robot's
    journald vacuuming and reboots

Usage:
    robot-comic-logsink                                  # 0.0.0.0:9477 → ./robot_comic_remote_events.log
    robot-comic-logsink --port 9477 --file D:/logs/ricci_events.log
"""

from __future__ import annotations
import os
import sys
import json
import socket
import logging
import argparse
import threading


logger = logging.getLogger(__name__)

DEFAULT_PORT = 9477
DEFAULT_FILE = "robot_comic_remote_events.log"
DEFAULT_MAX_BYTES = 50_000_000

_PREFIX = "RCSPAN "


class SinkWriter:
    """Appends relay datagrams to a rolling RCSPAN-prefixed line file."""

    def __init__(self, path: str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        """Write lines to ``path``, rotating to ``<path>.1`` past ``max_bytes``."""
        self._path = path
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self.received = 0
        self.dropped = 0

    def write_datagram(self, payload: bytes) -> bool:
        """Append one datagram as an RCSPAN line; returns False if dropped."""
        try:
            text = payload.decode("utf-8").strip()
            json.loads(text)  # must be a valid JSON span line
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.dropped += 1
            return False
        with self._lock:
            try:
                self._rotate_if_needed()
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(_PREFIX + text + "\n")
            except OSError:
                logger.warning("log sink: write failed (%s)", self._path, exc_info=True)
                self.dropped += 1
                return False
        self.received += 1
        return True

    def _rotate_if_needed(self) -> None:
        try:
            if os.path.getsize(self._path) < self._max_bytes:
                return
        except OSError:
            return  # file doesn't exist yet — nothing to rotate
        try:
            os.replace(self._path, self._path + ".1")
        except OSError:
            logger.warning("log sink: rotation failed (%s)", self._path, exc_info=True)


def serve(port: int, path: str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    """Bind the UDP port and append datagrams until interrupted."""
    writer = SinkWriter(path, max_bytes=max_bytes)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"robot-comic-logsink: listening on udp/{port} -> {path}", flush=True)
    try:
        while True:
            payload, _addr = sock.recvfrom(65535)
            writer.write_datagram(payload)
    except KeyboardInterrupt:
        print(
            f"robot-comic-logsink: stopped ({writer.received} lines written, {writer.dropped} dropped)",
            flush=True,
        )
    finally:
        sock.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="UDP log sink for the robot_comic event relay")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port to listen on (default: %(default)s)")
    parser.add_argument("--file", default=DEFAULT_FILE, help="output line file (default: %(default)s)")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="rotate the output file to .1 past this size (default: %(default)s)",
    )
    args = parser.parse_args()
    try:
        serve(args.port, args.file, max_bytes=args.max_bytes)
    except OSError as exc:
        print(f"robot-comic-logsink: cannot bind udp/{args.port}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
