"""HTTP client for the Reachy Mini daemon's NFC reader API (``/api/nfc``).

The daemon owns the serial link to the reader; this app never touches the port.
Failures come back as stable codes, which :func:`describe_write_error` turns
into a line for a user.
"""

from __future__ import annotations
import queue
import logging
import threading
from typing import Any, Optional
from dataclasses import dataclass

import requests


logger = logging.getLogger(__name__)

# Stable error codes returned by the daemon's write and erase endpoints.
WRITE_ERROR_MESSAGES = {
    "NO_TAG": "No tag on the reader",
    "TOO_LONG": "Code too long for this tag's capacity",
    "LOCKED": "Tag is locked and can no longer be written",
    "UNKNOWN_TAG": "Unrecognised tag model",
    "WRITE_REFUSED": "Tag refused the write",
    "WRITE_ERROR": "Write failed partway through",
    "COLLISION": "Several tags on the reader — present only one",
    "NOT_CONNECTED": "NFC reader not connected",
    "LINK_LOST": "Lost the link to the NFC reader",
    "TIMEOUT": "Tag was not presented in time",
    "DRIVER_MISSING": "NFC driver not installed on the robot",
}

# The daemon's status error when no add-on is plugged in: the normal state, not a fault.
NO_BOARD_ERROR = "no NFC reader board found"


def describe_write_error(code: str) -> str:
    """Turn a write/erase error code into a line for a user; unknown codes are shown as-is."""
    return WRITE_ERROR_MESSAGES.get(code, code or "Unknown error")


@dataclass
class NfcTagSnapshot:
    """Point-in-time state of the NFC reader."""

    present: bool
    content: Optional[str]  # text content; None if blank or no tag
    blank: bool  # tag present but no content written
    # False while a tag is on the reader but its content could not be read: it
    # is then neither blank nor known, and says nothing about the accessory.
    readable: bool = True
    uid: Optional[str] = None


class NfcDaemonClient:
    """HTTP client wrapping the daemon's /api/nfc routes."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0) -> None:
        """Build a client for the daemon's NFC endpoints."""
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._write_results: queue.SimpleQueue[tuple[bool, str]] = queue.SimpleQueue()
        # Pooled connection for the frequent reads only: writes run on their own
        # threads, and requests.Session is not documented as thread-safe.
        self._session = requests.Session()

    def get_tag(self) -> NfcTagSnapshot:
        """Return the current tag state (never raises; returns absent on error)."""
        try:
            r = self._session.get(f"{self.base}/api/nfc/tag", timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
            return NfcTagSnapshot(
                present=bool(d.get("present")),
                content=d.get("content") or None,
                blank=bool(d.get("blank")),
                # Older daemons omit it; an error then means the read failed.
                readable=bool(d.get("readable", d.get("error") is None)),
                uid=d.get("uid") or None,
            )
        except Exception as exc:
            logger.debug("NFC get_tag error: %s", exc)
            return NfcTagSnapshot(present=False, content=None, blank=False)

    def get_status(self) -> dict[str, Any]:
        """Return the daemon's NFC reader status (never raises; returns disconnected on error)."""
        try:
            r = self._session.get(f"{self.base}/api/nfc/status", timeout=self.timeout)
            r.raise_for_status()
            status: dict[str, Any] = r.json()
            return status
        except Exception as exc:
            logger.debug("NFC get_status error: %s", exc)
            return {"connected": False, "driver_available": False, "error": str(exc)}

    def driver_available(self) -> bool:
        """Whether the robot has the NFC driver installed; without it the reader is disabled."""
        return bool(self.get_status().get("driver_available"))

    def _post(self, path: str, payload: dict[str, Any], ok_code: str, timeout: float) -> tuple[bool, str]:
        """POST to a write-like endpoint and normalise the outcome to (success, code)."""
        try:
            r = requests.post(f"{self.base}/api/nfc/{path}", json=payload, timeout=timeout)
            if r.status_code == 503:
                # The reader itself is unavailable: disabled or link down.
                detail = r.json().get("detail", "unavailable")
                return False, "NOT_CONNECTED" if "connect" in detail.lower() else detail
            if r.status_code == 422:
                # The daemon's validator rejects text over its length limit.
                return False, "TOO_LONG"
            r.raise_for_status()
            d = r.json()
            if d.get("success"):
                return True, ok_code
            return False, d.get("error") or "WRITE_ERROR"
        except Exception as exc:
            logger.warning("NFC %s error: %s", path, exc)
            return False, "LINK_LOST"

    def write_tag(self, code: str, timeout: float = 12.0) -> tuple[bool, str]:
        """Write ``code`` onto the tag on the reader; returns (True, "WRITE_OK") or (False, error code)."""
        return self._post("write", {"text": code}, "WRITE_OK", timeout)

    def write_tag_in_background(self, code: str) -> None:
        """Start writing ``code``; the (success, code) result comes from drain_write_results()."""

        def _worker() -> None:
            self._write_results.put(self.write_tag(code))

        threading.Thread(target=_worker, daemon=True).start()

    def drain_write_results(self) -> list[tuple[bool, str]]:
        """Drain and return all finished background write results (non-blocking)."""
        results: list[tuple[bool, str]] = []
        try:
            while True:
                results.append(self._write_results.get_nowait())
        except queue.Empty:
            pass
        return results

    def erase_tag(self, full: bool = False, timeout: float = 30.0) -> tuple[bool, str]:
        """Make the tag on the reader blank again.

        ``full`` also zeroes the whole user memory, needed to remove a non-NDEF
        payload; it writes page by page, hence the wider timeout.
        """
        return self._post("erase", {"full": full}, "ERASE_OK", timeout)

    def close(self) -> None:
        """Close the pooled connection used by the read endpoints."""
        self._session.close()
