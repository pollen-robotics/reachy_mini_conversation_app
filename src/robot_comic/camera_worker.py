"""Camera worker thread with frame buffering and optional head tracking."""

from __future__ import annotations
import os
import time
import logging
import threading
from typing import TYPE_CHECKING, List, Tuple

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation
from robot_comic.vision.head_tracking import HeadTracker


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# How long (seconds) to wait between head-tracker inference calls when no face
# has been seen recently.  At 25 Hz the tracker fires on every camera frame
# even when the room is empty; dropping to 5 Hz (0.2 s) cuts idle MediaPipe
# CPU substantially without affecting tracking responsiveness — the first
# "face present" result immediately restores full-rate inference.
# Override via REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S.
_DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S = 0.2


def _head_tracker_idle_poll_interval_s() -> float:
    """Return the head-tracker inference interval when no face is visible.

    When ``get_head_position`` returns None for the most recent frame the
    worker switches to idle-rate polling so MediaPipe does not burn CPU
    scanning an empty room at 25 Hz.  The first frame where a face is found
    immediately resets the last-call timestamp so the next iteration runs at
    full speed again.

    ``REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S`` overrides the default.
    Values below 0.04 s (one camera frame) are clamped upward — going faster
    than the camera provides no benefit and risks busy-spinning.
    """
    raw = os.environ.get("REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S")
    if raw is None or not raw.strip():
        return _DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid REACHY_MINI_HEAD_TRACKER_IDLE_POLL_INTERVAL_S=%r; using default %.2f",
            raw,
            _DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S,
        )
        return _DEFAULT_HEAD_TRACKER_IDLE_POLL_INTERVAL_S
    return max(0.04, value)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and optional head tracking."""

    def __init__(self, reachy_mini: ReachyMini, head_tracker: HeadTracker | None = None) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Guards lazy-start so the first concurrent caller wins the race and
        # subsequent callers are cheap (see ensure_started()).
        self._start_lock = threading.Lock()

        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # x, y, z, roll, pitch, yaw
        self.face_tracking_lock = threading.Lock()

        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 2.0
        self.interpolation_duration = 1.0

        self.previous_head_tracking_state = self.is_head_tracking_enabled

        # Idle-rate gating: track the last time we ran get_head_position so we
        # can throttle inference to _head_tracker_idle_poll_interval_s() when
        # no face has been seen.  None means "never called" → fire immediately.
        self._last_tracker_call_time: float | None = None
        # Track whether the previous tracker call found a face so we know
        # whether to use idle-rate or full-rate for the next call.
        self._last_tracker_found_face: bool = False

        # Rate-limiting state for the except handler: track last logged message,
        # when it was last logged, and how many times it was suppressed.
        self._last_error_msg: str | None = None
        self._last_error_time: float = 0.0
        self._suppressed_error_count: int = 0
        self._error_suppress_window: float = 60.0  # seconds

        # Diagnostic state: count consecutive None frames so we can log
        # pipeline/appsink/source introspection when the camera stalls.
        self._consecutive_none_frames: int = 0
        self._last_diag_log_time: float = 0.0
        self._diag_log_interval: float = 30.0  # seconds between periodic re-logs

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Get the latest frame (thread-safe).

        Lazily starts the underlying capture thread on first call. The first
        camera-using tool call therefore pays the gstreamer pipeline cost
        (~5 s) instead of paying it during boot. Subsequent callers reuse the
        same worker.
        """
        # Trigger lazy start before any frame read so a hot path doesn't
        # silently return None forever when boot skipped the explicit start().
        self.ensure_started()
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable/disable head tracking.

        Enabling head tracking lazy-starts the camera worker; disabling does
        not start a worker that has not already been started.
        """
        self.is_head_tracking_enabled = enabled
        logger.info(f"Head tracking {'enabled' if enabled else 'disabled'}")
        if enabled:
            self.ensure_started()

    def start(self) -> None:
        """Start the camera worker loop in a thread.

        Idempotent: if a worker thread is already running this is a no-op so
        callers (including lazy ``get_latest_frame()`` consumers and an
        explicit boot start) cannot accidentally spawn duplicate capture
        loops.
        """
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self.working_loop, daemon=True)
            self._thread.start()
            logger.debug("Camera worker started")

    def ensure_started(self) -> bool:
        """Lazy-start the worker; return True if this call started the thread.

        The first camera-touching tool call (``camera``, ``greet``, ``roast``,
        or a head-tracking enable) triggers this so the ~5 s gstreamer
        pipeline cost shifts off the boot critical path. Concurrent callers
        are serialised on ``_start_lock`` so exactly one thread is spawned.
        """
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self.working_loop, daemon=True)
            self._thread.start()
            logger.info("Camera worker lazy-started on first use")
            return True

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        head_tracker_close = getattr(self.head_tracker, "close", None)
        if callable(head_tracker_close):
            head_tracker_close()

        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Run the camera worker loop."""
        # Lazy import: numpy + scipy are heavy (~0.5-1 s); deferred until the
        # background thread actually starts so READY=1 fires first.
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        logger.debug("Starting camera working loop")

        neutral_pose = np.eye(4)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                frame = self.reachy_mini.media.get_frame()

                if frame is None:
                    self._consecutive_none_frames += 1
                    now = time.monotonic()
                    _thresholds = {5, 25, 125}
                    _at_threshold = self._consecutive_none_frames in _thresholds
                    _interval_elapsed = (now - self._last_diag_log_time) >= self._diag_log_interval
                    if _at_threshold or (_interval_elapsed and self._consecutive_none_frames > 0):
                        self._last_diag_log_time = now
                        # --- pipeline state ---
                        pipeline_state = "unavailable"
                        try:
                            pipeline = self.reachy_mini.media.camera.pipeline
                            _ret, _state, _pending = pipeline.get_state(timeout=0)
                            pipeline_state = f"({_ret},{_state},{_pending})"
                        except Exception:  # noqa: BLE001
                            pass
                        # --- appsink buffer count ---
                        appsink_info = "unavailable"
                        try:
                            _appsink = getattr(self.reachy_mini.media.camera, "_appsink", None)
                            if _appsink is not None:
                                _lvl = _appsink.get_property("current-level-buffers")
                                appsink_info = f"current-level-buffers={_lvl}"
                            else:
                                appsink_info = "not-found"
                        except Exception:  # noqa: BLE001
                            pass
                        # --- source element state ---
                        source_state = "unavailable"
                        try:
                            pipeline = self.reachy_mini.media.camera.pipeline
                            for _elem in pipeline.iterate_elements():
                                _name = _elem.get_name()
                                if "unixfdsrc" in _name or "v4l2src" in _name or "src" in _name.lower():
                                    _r, _s, _p = _elem.get_state(timeout=0)
                                    source_state = f"{_name}:({_r},{_s},{_p})"
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                        logger.info(
                            "camera_worker: frame=None streak=%d pipeline_state=%s appsink=%s source=%s",
                            self._consecutive_none_frames,
                            pipeline_state,
                            appsink_info,
                            source_state,
                        )
                elif frame is not None:
                    if self._consecutive_none_frames > 0:
                        logger.info(
                            "camera_worker: Camera recovered after %d consecutive None frames",
                            self._consecutive_none_frames,
                        )
                        self._consecutive_none_frames = 0
                    # Keep the latest frame available for tools and UI consumers
                    with self.frame_lock:
                        self.latest_frame = frame

                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        # Reuse the face-lost interpolation path to return smoothly to neutral
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled and self.head_tracker is not None:
                        # Idle-rate gate: when the previous call saw no face,
                        # throttle inference to _head_tracker_idle_poll_interval_s()
                        # so MediaPipe does not burn CPU scanning an empty room at
                        # 25 Hz.  As soon as a face is found we reset
                        # _last_tracker_call_time to None so the *next* iteration
                        # fires immediately (i.e. no inter-frame gap while tracking
                        # is active).  The interpolation-back-to-neutral path is
                        # entirely independent of get_head_position — it runs from
                        # last_face_detected_time and elapsed wall-clock time below,
                        # so skipping the tracker call here does not break it.
                        idle_interval = _head_tracker_idle_poll_interval_s()
                        run_tracker = (
                            self._last_tracker_found_face
                            or self._last_tracker_call_time is None
                            or (current_time - self._last_tracker_call_time) >= idle_interval
                        )

                        eye_center = None
                        if run_tracker:
                            self._last_tracker_call_time = current_time
                            eye_center, _ = self.head_tracker.get_head_position(frame)
                            self._last_tracker_found_face = eye_center is not None

                        if eye_center is not None:
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

                            # The tracker returns normalized coordinates in [-1, 1]
                            h, w, _ = frame.shape
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [
                                max(0.0, min(float(w), eye_center_norm[0] * w)),
                                max(0.0, min(float(h), eye_center_norm[1] * h)),
                            ]

                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            translation = target_pose[:3, 3]
                            rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

                            # The camera FOV is tighter than the motion model expects
                            translation *= 0.6
                            rotation *= 0.6

                            # Tracker safe-envelope clamp (#308 hypothesis 3): the
                            # continuous tracker can produce targets that, after
                            # composition with breathing/wobble, push the daemon's
                            # IK into "Collision detected or head pose not
                            # achievable!" territory. Clamping the rotation offset
                            # here cuts those warnings off at the source.
                            from robot_comic.motion_safety import clamp_tracker_rotation_offsets  # noqa: PLC0415

                            roll_c, pitch_c, yaw_c = clamp_tracker_rotation_offsets(
                                (float(rotation[0]), float(rotation[1]), float(rotation[2]))
                            )

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    roll_c,
                                    pitch_c,
                                    yaw_c,
                                ]

                        elif self.last_face_detected_time is None or self.last_face_detected_time == current_time:
                            pass

                    if self.last_face_detected_time is not None:
                        time_since_face_lost = current_time - self.last_face_detected_time

                        if time_since_face_lost >= self.face_lost_delay:
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[3:]
                                    # Interpolate from the current tracking pose back to neutral
                                    pose_matrix = np.eye(4, dtype=np.float32)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler(
                                        "xyz",
                                        current_rotation_euler,
                                    ).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            elapsed_interpolation = current_time - self.interpolation_start_time
                            t = min(1.0, elapsed_interpolation / self.interpolation_duration)

                            interpolated_pose = linear_pose_interpolation(
                                self.interpolation_start_pose,
                                neutral_pose,
                                t,
                            )

                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ]

                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None

                time.sleep(0.04)

            except Exception as e:
                now = time.monotonic()
                msg = str(e)
                same_error = msg == self._last_error_msg
                window_elapsed = (now - self._last_error_time) >= self._error_suppress_window

                if not same_error or window_elapsed:
                    if same_error and self._suppressed_error_count > 0:
                        # Emit suppression summary before the fresh log
                        logger.error(
                            "Camera worker error suppressed %d times in last %.0fs: %s",
                            self._suppressed_error_count,
                            self._error_suppress_window,
                            self._last_error_msg,
                        )
                    # Log full traceback on the first occurrence (or after window)
                    logger.exception("Camera worker error: %s", msg)
                    self._last_error_msg = msg
                    self._last_error_time = now
                    self._suppressed_error_count = 0
                else:
                    self._suppressed_error_count += 1

                time.sleep(0.1)

        logger.debug("Camera worker thread exited")
