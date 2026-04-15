"""Run the optional YOLO head tracker in a dedicated subprocess."""

from __future__ import annotations
import os
import sys
import time
import queue
import atexit
import pickle
import struct
import logging
import threading
import subprocess
import importlib.util
from typing import IO, TypeGuard
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.vision.head_tracking import HeadTracker, HeadTrackerResult


logger = logging.getLogger(__name__)

_PROCESS_START_TIMEOUT = 20.0
_REQUEST_TIMEOUT = 0.5
_SHUTDOWN_TIMEOUT = 2.0
_HEADER_STRUCT = struct.Struct("!I")


def _build_tracker_backend() -> HeadTracker:
    """Instantiate a concrete head-tracker backend."""
    from reachy_mini_conversation_app.vision.head_tracking.yolo import YoloHeadTracker

    return YoloHeadTracker()


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    """Read exactly `size` bytes or raise EOFError."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("Unexpected EOF while reading tracker message")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_message(stream: IO[bytes], payload: object) -> None:
    """Serialize and write a single message."""
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER_STRUCT.pack(len(data)))
    stream.write(data)
    stream.flush()


def _receive_message(stream: IO[bytes]) -> object:
    """Read and deserialize a single message."""
    header = _read_exact(stream, _HEADER_STRUCT.size)
    (size,) = _HEADER_STRUCT.unpack(header)
    data = _read_exact(stream, size)
    return pickle.loads(data)


def _reader_loop(stream: IO[bytes], messages: queue.Queue[tuple[str, object | None]]) -> None:
    """Read responses from the child process in the background."""
    try:
        while True:
            payload = _receive_message(stream)
            messages.put(("message", payload))
    except EOFError:
        messages.put(("eof", None))
    except Exception as exc:
        messages.put(("error", repr(exc)))


def _worker_main() -> int:
    """Run the tracker worker protocol."""
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    try:
        tracker = _build_tracker_backend()
        _send_message(protocol_out, ("ready", None))
    except Exception as exc:
        _send_message(protocol_out, ("error", repr(exc)))
        return 1

    while True:
        try:
            message = _receive_message(sys.stdin.buffer)
        except (EOFError, KeyboardInterrupt):
            return 0

        if not isinstance(message, tuple) or not message or not isinstance(message[0], str):
            _send_message(protocol_out, ("error", -1, f"Invalid command: {message!r}"))
            continue

        command = message[0]
        if command == "close":
            return 0
        if command != "frame" or len(message) != 3 or not isinstance(message[1], int):
            _send_message(protocol_out, ("error", -1, f"Unknown command: {message!r}"))
            continue

        request_id = message[1]
        payload = message[2]
        try:
            result = tracker.get_head_position(payload)
            _send_message(protocol_out, ("result", request_id, result))
        except Exception as exc:
            _send_message(protocol_out, ("error", request_id, repr(exc)))


def _is_tracker_result(payload: object) -> TypeGuard[HeadTrackerResult]:
    """Return whether the payload matches a tracker result."""
    if not isinstance(payload, tuple) or len(payload) != 2:
        return False

    eye_center, roll = payload
    if eye_center is not None and not isinstance(eye_center, np.ndarray):
        return False
    if roll is not None and not isinstance(roll, (float, np.floating)):
        return False
    return True


class YoloHeadTrackerProcess:
    """Proxy that runs the optional YOLO head tracker out of process."""

    def __init__(self, *, request_timeout: float = _REQUEST_TIMEOUT) -> None:
        """Start the child process and wait until the tracker is ready."""
        self.request_timeout = request_timeout
        self._closed = False
        self._send_lock = threading.Lock()
        self._messages: queue.Queue[tuple[str, object | None]] = queue.Queue()
        self._next_request_id = 0
        self._timed_out_request_id: int | None = None
        self._tracker_name = "yolo"

        module_path = "reachy_mini_conversation_app.vision.head_tracking.yolo_process"
        env = os.environ.copy()
        package_spec = importlib.util.find_spec("reachy_mini_conversation_app")
        package_locations = None if package_spec is None else package_spec.submodule_search_locations
        if not package_locations:
            raise RuntimeError("Unable to determine the reachy_mini_conversation_app package path")
        project_src = Path(next(iter(package_locations))).resolve().parent
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(project_src)
            if not existing_pythonpath
            else f"{project_src}{os.pathsep}{existing_pythonpath}"
        )
        env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            [sys.executable, "-m", module_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            env=env,
            start_new_session=True,
        )

        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("Failed to create pipes for yolo head tracker process")

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._reader = threading.Thread(
            target=_reader_loop,
            args=(self._stdout, self._messages),
            daemon=True,
            name="yolo-head-tracker-reader",
        )
        self._reader.start()
        atexit.register(self.close)

        message = self._wait_for_message(_PROCESS_START_TIMEOUT)
        if not (isinstance(message, tuple) and len(message) == 2 and isinstance(message[0], str)):
            self.close()
            raise RuntimeError(f"yolo head tracker returned an invalid startup message: {message!r}")

        status, payload = message
        if status != "ready":
            self.close()
            raise RuntimeError(f"Failed to initialize yolo head tracker: {payload}")

    def _wait_for_message(self, timeout: float | None = None) -> object | None:
        """Return the next child-process message payload, or None if no queued payload is available."""
        try:
            if timeout is None:
                event, payload = self._messages.get_nowait()
            else:
                event, payload = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            if timeout is None:
                return None
            raise TimeoutError(f"{self._tracker_name} head tracker timed out after {timeout:.2f}s") from exc

        if event == "message":
            return payload
        if event == "eof":
            raise RuntimeError(f"{self._tracker_name} head tracker exited unexpectedly")
        raise RuntimeError(f"{self._tracker_name} head tracker reader failed: {payload}")

    def _unpack_response(self, message: object) -> tuple[str, int, object]:
        """Validate a tracker response and return its fields."""
        if not (
            isinstance(message, tuple)
            and len(message) == 3
            and isinstance(message[0], str)
            and isinstance(message[1], int)
        ):
            raise RuntimeError(f"{self._tracker_name} head tracker returned an invalid response: {message!r}")

        status, message_request_id, payload = message
        return status, message_request_id, payload

    def _wait_for_response(self, request_id: int, timeout: float) -> tuple[str, object]:
        """Wait for the response matching the requested frame."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self._tracker_name} head tracker timed out after {timeout:.2f}s")

            wait_started = time.monotonic()
            message = self._wait_for_message(remaining)
            status, message_request_id, payload = self._unpack_response(message)
            if message_request_id < request_id:
                deadline += time.monotonic() - wait_started
                logger.debug(
                    "Discarding stale reply from %s head tracker: expected request %s, got %s",
                    self._tracker_name,
                    request_id,
                    message_request_id,
                )
                continue

            if message_request_id > request_id:
                raise RuntimeError(
                    f"{self._tracker_name} head tracker returned out-of-order response "
                    f"{message_request_id} while waiting for {request_id}"
                )

            return status, payload

    def _drain_timed_out_reply(self) -> bool:
        """Return whether the last timed-out request has been consumed."""
        if self._timed_out_request_id is None:
            return True

        while True:
            message = self._wait_for_message()
            if message is None:
                return False

            status, message_request_id, payload = self._unpack_response(message)
            if message_request_id < self._timed_out_request_id:
                logger.debug(
                    "Discarding stale reply from %s head tracker: expected request %s, got %s",
                    self._tracker_name,
                    self._timed_out_request_id,
                    message_request_id,
                )
                continue

            if message_request_id > self._timed_out_request_id:
                raise RuntimeError(
                    f"{self._tracker_name} head tracker returned out-of-order response "
                    f"{message_request_id} while waiting for {self._timed_out_request_id}"
                )

            logger.debug(
                "Discarding delayed %s reply from %s head tracker for request %s: %r",
                status,
                self._tracker_name,
                message_request_id,
                payload,
            )
            self._timed_out_request_id = None
            return True

    def get_head_position(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32] | None, float | None]:
        """Return the detected head position from the child process."""
        if self._closed:
            return None, None

        if self._process.poll() is not None:
            logger.error("Head tracker process for %s is not alive", self._tracker_name)
            return None, None

        request_id: int | None = None
        try:
            with self._send_lock:
                # Avoid writing another frame while the child is still finishing a timed-out one.
                if not self._drain_timed_out_reply():
                    return None, None

                request_id = self._next_request_id
                self._next_request_id += 1
                _send_message(self._stdin, ("frame", request_id, frame))
                status, payload = self._wait_for_response(request_id, self.request_timeout)
        except TimeoutError as exc:
            if request_id is not None:
                self._timed_out_request_id = request_id
            logger.warning("Head tracker %s communication timed out: %s", self._tracker_name, exc)
            return None, None
        except Exception as exc:
            logger.error("Head tracker %s communication failed: %s", self._tracker_name, exc)
            return None, None

        if status == "result":
            if _is_tracker_result(payload):
                eye_center, roll = payload
                return eye_center, None if roll is None else float(roll)

            logger.error("%s head tracker returned an invalid result: %r", self._tracker_name, payload)
            return None, None

        logger.error("%s head tracker failed to process frame: %s", self._tracker_name, payload)
        return None, None

    def close(self) -> None:
        """Stop the child process."""
        if self._closed:
            return

        self._closed = True

        try:
            if self._process.poll() is None:
                with self._send_lock:
                    _send_message(self._stdin, ("close", None))
        except Exception:
            pass

        try:
            self._stdin.close()
        except Exception:
            pass

        try:
            self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("Force-terminating %s head tracker process", self._tracker_name)
            self._process.terminate()
            try:
                self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Force-killing %s head tracker process", self._tracker_name)
                self._process.kill()
                self._process.wait()

        try:
            self._stdout.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """Best-effort shutdown when the proxy is garbage-collected."""
        self.close()


def main() -> int:
    """CLI entrypoint for the head-tracker worker process."""
    if len(sys.argv) != 1:
        print(
            "usage: python -m reachy_mini_conversation_app.vision.head_tracking.yolo_process",
            file=sys.stderr,
        )
        return 2
    return _worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
