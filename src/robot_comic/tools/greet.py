"""Greet tool — face presence scan, head sweep, and returning-visitor identity matching."""

from __future__ import annotations
import os
import json
import time
import asyncio
import difflib
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from robot_comic import config as _config
from robot_comic.tools.move_head import MoveHead
from robot_comic.tools.core_tools import Tool, ToolDependencies
from robot_comic.tools.crowd_work import resolve_session_dir
from robot_comic.tools.name_validation import validate_name_or_warn


logger = logging.getLogger(__name__)

SESSION_WINDOW_DAYS = 30
MATCH_THRESHOLD = 0.75
SWEEP_POSITIONS = ["left", "up", "right", "front"]

# How long (seconds) to wait for the face tracker to latch before giving up and
# sweeping.  Overrideable via REACHY_MINI_GREET_SCAN_WAIT_S.
_DEFAULT_SCAN_WAIT_S = 1.5
# How often (seconds) to re-run MediaPipe inference during the face-scan poll
# loop when no face has been detected yet.  Keeping this above the camera
# worker's 40 ms frame period (25 Hz) avoids saturating the CPU with redundant
# inference calls when the room is empty.  Overrideable via
# REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S.
_DEFAULT_SCAN_POLL_INTERVAL_S = 0.3
_SCAN_POLL_INTERVAL_S = _DEFAULT_SCAN_POLL_INTERVAL_S  # kept for backward-compat with tests

# Default MediaPipe face-detection confidence threshold (#269). 0.3 is lower
# than MediaPipe's 0.5 default to compensate for the Reachy Mini camera lens /
# lighting envelope where 0.5 was missing square-on faces. Tunable per-unit
# via REACHY_MINI_FACE_DETECTION_CONFIDENCE.
_DEFAULT_FACE_DETECTION_CONFIDENCE = 0.3

# MediaPipe is deferred to first use — ``import mediapipe`` costs ~1.2s on
# the Pi and is the dominant chunk of greet.py's ~3s boot import (#323
# lever 2). The first ``_scan`` call pays the cost; subsequent calls reuse
# the cached detector module reference. ``MP_AVAILABLE`` stays a module-level
# attribute (initialized to ``None`` = "not yet checked") so tests that
# ``monkeypatch.setattr(greet_mod, "MP_AVAILABLE", True/False)`` keep working.
_mp_face_detection: Any = None
MP_AVAILABLE: Optional[bool] = None


def _face_detection_confidence() -> float:
    """Return the MediaPipe min_detection_confidence threshold, env-tunable.

    REACHY_MINI_FACE_DETECTION_CONFIDENCE overrides the default. Values are
    clamped to [0.0, 1.0]; non-numeric values fall back to the default.
    """
    raw = os.environ.get("REACHY_MINI_FACE_DETECTION_CONFIDENCE")
    if raw is None or not raw.strip():
        return _DEFAULT_FACE_DETECTION_CONFIDENCE
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid REACHY_MINI_FACE_DETECTION_CONFIDENCE=%r; using default %.2f",
            raw,
            _DEFAULT_FACE_DETECTION_CONFIDENCE,
        )
        return _DEFAULT_FACE_DETECTION_CONFIDENCE
    return max(0.0, min(1.0, value))


def _scan_poll_interval_s() -> float:
    """Return the MediaPipe inference poll interval for the face-scan loop.

    A higher value reduces CPU load when the room is empty: instead of
    running inference on every camera frame we only fire it every
    ``_DEFAULT_SCAN_POLL_INTERVAL_S`` seconds (default 0.3 s ≈ 3 Hz).
    Override via ``REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S``.  Values below
    0.05 s are clamped upward to avoid unintentional busy-spin.
    """
    raw = os.environ.get("REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S")
    if raw is None or not raw.strip():
        return _DEFAULT_SCAN_POLL_INTERVAL_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S=%r; using default %.2f",
            raw,
            _DEFAULT_SCAN_POLL_INTERVAL_S,
        )
        return _DEFAULT_SCAN_POLL_INTERVAL_S
    return max(0.05, value)


def _check_mp_available() -> bool:
    """Return whether MediaPipe is importable, lazy-loading on first call.

    Tests that monkey-patch ``MP_AVAILABLE`` directly short-circuit the lazy
    load: this function returns the patched value without re-importing.
    """
    global _mp_face_detection, MP_AVAILABLE
    if MP_AVAILABLE is not None:
        return bool(MP_AVAILABLE)
    try:
        import mediapipe as mp  # noqa: PLC0415 — deferred from boot
    except ImportError:
        _mp_face_detection = None
        MP_AVAILABLE = False
        return False
    _mp_face_detection = mp.solutions.face_detection
    MP_AVAILABLE = True
    logger.info(
        "MediaPipe face detection ready; min_detection_confidence=%.2f "
        "(override via REACHY_MINI_FACE_DETECTION_CONFIDENCE)",
        _face_detection_confidence(),
    )
    return True


def _detect_face(frame: Any) -> bool:
    """Return True if MediaPipe detects at least one face in the BGR frame."""
    if not _check_mp_available() or _mp_face_detection is None:
        return False
    rgb = frame[..., ::-1].copy()
    with _mp_face_detection.FaceDetection(min_detection_confidence=_face_detection_confidence()) as detector:
        results = detector.process(rgb)
        return bool(results.detections)


def _detect_face_with_scores(frame: Any) -> Tuple[bool, List[float]]:
    """Return (face_detected, confidence_scores) for diagnostic logging.

    Mirrors `_detect_face` but exposes the per-detection confidence scores so
    callers can log how MediaPipe scored the frame even when it ultimately
    returned no detections (empty list).
    """
    if not _check_mp_available() or _mp_face_detection is None:
        return False, []
    rgb = frame[..., ::-1].copy()
    with _mp_face_detection.FaceDetection(min_detection_confidence=_face_detection_confidence()) as detector:
        results = detector.process(rgb)
        detections = results.detections or []
        scores: List[float] = []
        for det in detections:
            # `det.score` is a RepeatedScalarContainer of floats; take the first.
            try:
                scores.append(float(det.score[0]))
            except (IndexError, TypeError, AttributeError):
                continue
        return bool(detections), scores


def _frame_stats(frame: Any) -> Tuple[Tuple[int, ...], str, float, float, float]:
    """Compute (shape, dtype, mean, min, max) for a numpy frame; safe for empty."""
    try:
        shape = tuple(frame.shape)
        dtype = str(frame.dtype)
        mean = float(np.mean(frame))
        fmin = float(np.min(frame))
        fmax = float(np.max(frame))
    except Exception:
        shape = ()
        dtype = "unknown"
        mean = fmin = fmax = float("nan")
    return shape, dtype, mean, fmin, fmax


def _build_callbacks(state: Dict[str, Any]) -> List[str]:
    """Build callback hints from session state — mirrors CrowdWork._build_callbacks."""
    hints: List[str] = []
    name = state.get("name")
    job = state.get("job")
    hometown = state.get("hometown")
    details: List[str] = state.get("details") or []

    identity = [(k, v) for k, v in [("name", name), ("job", job), ("hometown", hometown)] if v]
    if len(identity) >= 2:
        label = " + ".join(k for k, v in identity)
        values = ", ".join(v for _, v in identity)
        hints.append(f"{label}: {values}")

    if job and details:
        hints.append(f"job + visual: {job} + {details[0]}")

    already_in_hints = {details[0]} if (job and details) else set()
    for detail in details[:2]:
        if detail not in already_in_hints:
            hints.append(f"detail callback: {detail}")
            already_in_hints.add(detail)

    return hints[:3]


def _fuzzy_match(
    query: str,
    candidates: List[Tuple[str, Path]],
) -> Tuple[Optional[str], Optional[Path], float]:
    """Return (best_name, best_path, score) for the highest match >= MATCH_THRESHOLD, else (None, None, score).

    Tries both the full query and the first word so that "Tony Anzelmo" still
    matches a stored name of "Tony" (full-string ratio would be ~0.5; first-word
    ratio is 1.0).
    """
    best_name: Optional[str] = None
    best_path: Optional[Path] = None
    best_score = 0.0
    q_full = query.lower().strip()
    q_tokens = q_full.split()
    for name, path in candidates:
        n = name.lower().strip()
        n_first = n.split()[0] if n else n
        scores = [difflib.SequenceMatcher(None, q_full, n).ratio()]
        for tok in q_tokens:
            scores.append(difflib.SequenceMatcher(None, tok, n).ratio())
            scores.append(difflib.SequenceMatcher(None, tok, n_first).ratio())
        score = max(scores)
        if score > best_score:
            best_score, best_name, best_path = score, name, path
    if best_score >= MATCH_THRESHOLD:
        return best_name, best_path, best_score
    return None, None, best_score


def _load_candidates(session_dir: Path) -> List[Tuple[str, Path]]:
    """Return (name, path) pairs from session files modified within SESSION_WINDOW_DAYS."""
    if not session_dir.exists():
        return []
    cutoff = datetime.now() - timedelta(days=SESSION_WINDOW_DAYS)
    candidates: List[Tuple[str, Path]] = []
    for path in session_dir.glob("session_*.json"):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name")
            if name and isinstance(name, str):
                candidates.append((name, path))
        except Exception:
            logger.warning("Skipping unreadable session file %s", path)
    return candidates


class Greet(Tool):
    """Detect face presence and identify returning visitors by name."""

    name = "greet"
    description = (
        "Two actions. "
        "action='scan': detect whether a face is present; executes a slow head sweep if not found immediately. "
        "action='identify': attempt to identify the visitor. "
        "When face recognition is enabled, the camera frame is checked first — on a face match, returns "
        "returning=true with the recalled name and profile. "
        "If the face is not recognised, returns returning=false with a pending_embedding flag so the robot "
        "knows to ask the visitor's name and pass it to crowd_work update. "
        "When face recognition is disabled or unavailable, falls back to fuzzy name matching against stored "
        "sessions from the last 30 days: on match returns returning=true with profile and callback hints; "
        "on no match returns returning=false and name_received=<name>. "
        "The 'name' parameter is required when face recognition is disabled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "identify"],
                "description": (
                    "scan: detect face presence and sweep if needed. "
                    "identify: identify the visitor by face (if enabled) or spoken name."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "The name spoken by the person. "
                    "Optional when face recognition is enabled (face is checked first). "
                    "Required for action='identify' when face recognition is disabled."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self, session_dir: Optional[Path] = None) -> None:
        """Initialise with an optional session directory override (for testing).

        When no override is provided, the directory resolves to
        ~/.robot_comic/.comedy_sessions/ on first call.
        """
        self._explicit_session_dir: Optional[Path] = session_dir
        self._session_dir: Optional[Path] = session_dir

    def _ensure_session_dir(self, deps: ToolDependencies) -> None:
        if self._explicit_session_dir is not None:
            return
        if self._session_dir is None:
            self._session_dir = resolve_session_dir()

    async def _scan(self, deps: ToolDependencies) -> Dict[str, Any]:
        if deps.camera_worker is None:
            return {"error": "Camera not available"}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            logger.debug("greet._scan: camera_worker.get_latest_frame() returned None on initial guard")
            return {"error": "No frame available"}

        if not _check_mp_available():
            return {"face_detected": True, "note": "MediaPipe unavailable, assuming face present"}

        # Track the latest stats so the no_subject summary line can name them.
        last_shape: Tuple[int, ...] = ()
        last_dtype: str = "unknown"
        last_mean: float = float("nan")
        last_min: float = float("nan")
        last_max: float = float("nan")
        last_detection_count: int = 0
        last_highest_confidence: float = 0.0

        # Poll for up to REACHY_MINI_GREET_SCAN_WAIT_S seconds before sweeping.
        # On fresh startup the camera / tracker needs a moment to latch onto a
        # face even when one is already in frame; a single sample (or 3 rapid
        # retries) is not enough to bridge that initialisation gap.
        #
        # The poll interval caps how often we run MediaPipe inference.  Running
        # inference every camera frame (40 ms / 25 Hz) burns ~5% CPU on a Pi 5
        # when the room is empty and the scan window is several seconds long.
        # The default 0.3 s cap (≈ 3 Hz) is aggressive enough to detect a face
        # within one interaction window while keeping inference-induced CPU load
        # negligible.  Override via REACHY_MINI_GREET_SCAN_POLL_INTERVAL_S.
        scan_wait_s = float(os.environ.get("REACHY_MINI_GREET_SCAN_WAIT_S", _DEFAULT_SCAN_WAIT_S))
        poll_interval_s = _scan_poll_interval_s()
        deadline = time.monotonic() + scan_wait_s
        while True:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.debug("greet._scan: camera_worker.get_latest_frame() returned None during poll")
            else:
                last_shape, last_dtype, last_mean, last_min, last_max = _frame_stats(frame)
                logger.debug(
                    "greet._scan: frame shape=%s dtype=%s mean=%.2f min=%.2f max=%.2f",
                    last_shape,
                    last_dtype,
                    last_mean,
                    last_min,
                    last_max,
                )
                detected, scores = _detect_face_with_scores(frame)
                last_detection_count = len(scores)
                last_highest_confidence = max(scores) if scores else 0.0
                logger.debug(
                    "greet._scan: mediapipe detections count=%d scores=%s",
                    last_detection_count,
                    [round(s, 3) for s in scores],
                )
                if detected:
                    return {"face_detected": True}
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_interval_s)

        # Kill-switch: when this is set, skip the head sweep entirely and just
        # report no_subject. Used to keep the chassis safe while the discrete
        # 4-step sweep is observed to swing the head hard enough into the
        # cowling on at least one unit (tracked in #264). Default off so this
        # is opt-in via env until the proper velocity-clamped path lands.
        if os.environ.get("REACHY_MINI_GREET_SWEEP_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.info(
                "greet: sweep disabled by REACHY_MINI_GREET_SWEEP_DISABLED; "
                "returning terminal no_subject (do_not_retry)"
            )
            self._log_no_subject_summary(
                last_shape,
                last_mean,
                last_detection_count,
                last_highest_confidence,
                deps,
            )
            # Terminal return — the outcome is fixed by configuration, not a
            # transient miss. Hardware validation 2026-05-16 caught the
            # Gemini-Don-Rickles persona looping ``greet action=scan`` four
            # times in one turn (~5s each) because the bare
            # ``{"no_subject": True}`` looked retryable. The added fields
            # (``sweep_disabled``, ``retry_hint``, ``note``) make the
            # result self-describing for any LLM backend — current Gemini,
            # future Llama / Hermes — without per-persona prompt tuning.
            # The legacy ``no_subject: True`` key is preserved so existing
            # persona prompts that read it keep working.
            return {
                "no_subject": True,
                "sweep_disabled": True,
                "retry_hint": "do_not_retry",
                "note": (
                    "Head sweep is disabled by configuration "
                    "(REACHY_MINI_GREET_SWEEP_DISABLED). The scan only "
                    "checked the head-on view and saw nobody. Calling "
                    "greet again this turn will produce the same result "
                    "— do not retry. Improvise as if the room is empty."
                ),
            }

        for direction in SWEEP_POSITIONS:
            move_result = await MoveHead()(deps, direction=direction)
            if "error" in move_result:
                logger.warning("move_head failed during sweep (%s): %s", direction, move_result["error"])
            await asyncio.sleep(deps.motion_duration_s + 0.2)
            sweep_frame = deps.camera_worker.get_latest_frame()
            if sweep_frame is None:
                logger.debug(
                    "greet._scan: camera_worker.get_latest_frame() returned None after sweep step %s",
                    direction,
                )
                continue
            last_shape, last_dtype, last_mean, last_min, last_max = _frame_stats(sweep_frame)
            logger.debug(
                "greet._scan: sweep[%s] frame shape=%s dtype=%s mean=%.2f min=%.2f max=%.2f",
                direction,
                last_shape,
                last_dtype,
                last_mean,
                last_min,
                last_max,
            )
            detected, scores = _detect_face_with_scores(sweep_frame)
            last_detection_count = len(scores)
            last_highest_confidence = max(scores) if scores else 0.0
            logger.debug(
                "greet._scan: sweep[%s] mediapipe detections count=%d scores=%s",
                direction,
                last_detection_count,
                [round(s, 3) for s in scores],
            )
            if detected:
                return {"face_detected": True}

        self._log_no_subject_summary(
            last_shape,
            last_mean,
            last_detection_count,
            last_highest_confidence,
            deps,
        )
        return {"no_subject": True}

    def _log_no_subject_summary(
        self,
        frame_shape: Tuple[int, ...],
        frame_mean: float,
        detection_count: int,
        highest_confidence: float,
        deps: ToolDependencies,
    ) -> None:
        """Emit the no_subject INFO summary and optionally dump a diag frame.

        When REACHY_MINI_GREET_DIAG=1 (or true/yes) we attempt a one-shot dump
        of the latest frame to ``~/.robot_comic/diag/greet_no_subject_<ts>.png``
        so the operator can eyeball what the camera was actually seeing. The
        dump is best-effort: any error is swallowed with a warning so we never
        break the scan path.
        """
        logger.info(
            "greet._scan: no_subject frame_shape=%s mean=%.2f detection_count=%d highest_confidence=%.3f",
            frame_shape,
            frame_mean,
            detection_count,
            highest_confidence,
        )
        diag_flag = os.environ.get("REACHY_MINI_GREET_DIAG", "").lower()
        if diag_flag not in ("1", "true", "yes"):
            return
        try:
            if deps.camera_worker is None:
                return
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                return
            diag_dir = Path.home() / ".robot_comic" / "diag"
            diag_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = diag_dir / f"greet_no_subject_{ts}.npy"
            # Use np.save (always available) — avoids a hard cv2 dependency.
            np.save(out_path, frame)
            logger.info("greet._scan: dumped diag frame to %s", out_path)
        except Exception as exc:  # pragma: no cover - best-effort dump
            logger.warning("greet._scan: failed to dump diag frame: %s", exc)

    async def _identify_by_face(self, deps: ToolDependencies) -> Dict[str, Any] | None:
        """Attempt face-embedding based identification.

        Returns a result dict on success (match or no-match-with-embedding), or
        ``None`` when face recognition is disabled / unavailable / no frame.
        """
        if not _config.config.FACE_RECOGNITION_ENABLED:
            return None
        if deps.face_embedder is None or deps.face_db is None:
            return None

        frame = deps.camera_worker.get_latest_frame() if deps.camera_worker else None
        if frame is None:
            logger.debug("greet._identify_by_face: no frame available, skipping face path")
            return None

        embedding: "np.ndarray[Any, np.dtype[Any]] | None" = deps.face_embedder.embed(frame)
        if embedding is None:
            logger.debug("greet._identify_by_face: no face detected in frame")
            return {"returning": False, "name": None, "pending_embedding": True}

        match = deps.face_db.match(embedding)
        if match is not None:
            matched_name: str = match.get("name", "")
            logger.info("greet._identify_by_face: recognised %r", matched_name)
            return {
                "returning": True,
                "name": matched_name,
                "last_seen": match.get("last_seen"),
                "session_count": match.get("session_count"),
                "face_match": True,
                "load_instruction": "Call crowd_work action=update with name before proceeding.",
            }

        # Face detected but not in DB — return embedding as list for crowd_work persistence.
        logger.debug("greet._identify_by_face: face detected but no DB match")
        return {
            "returning": False,
            "name": None,
            "pending_embedding": embedding.tolist(),
        }

    async def _identify(self, deps: ToolDependencies, name: str) -> Dict[str, Any]:
        assert self._session_dir is not None
        candidates = _load_candidates(self._session_dir)
        if not candidates:
            return {"returning": False, "name_received": name}

        matched_name, matched_path, _score = _fuzzy_match(name, candidates)
        if matched_name is None or matched_path is None:
            return {"returning": False, "name_received": name}

        try:
            state = json.loads(matched_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read matched session %s", matched_path)
            return {"returning": False, "name_received": name}

        last_seen = state.get("last_updated") or str(datetime.fromtimestamp(matched_path.stat().st_mtime).isoformat())
        profile = {
            "job": state.get("job"),
            "hometown": state.get("hometown"),
            "details": state.get("details", []),
        }
        callbacks = _build_callbacks(state)

        return {
            "returning": True,
            "name": matched_name,
            "last_seen": last_seen,
            "profile": profile,
            "callbacks": callbacks,
            "load_instruction": "Call crowd_work action=update with this profile before proceeding.",
        }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch to scan or identify."""
        self._ensure_session_dir(deps)
        action = kwargs.get("action")
        logger.info("Tool call: greet action=%s", action)

        if action == "scan":
            return await self._scan(deps)

        if action == "identify":
            # Face-recognition path: try to identify the visitor from their face
            # before falling back to spoken-name fuzzy matching.
            face_result = await self._identify_by_face(deps)
            if face_result is not None:
                return face_result

            # Name-based fallback (also the only path when face recognition is off).
            name = (kwargs.get("name") or "").strip()
            if not name:
                return {
                    "error": (
                        "name parameter is required for identify. "
                        "The person has not spoken their name yet. "
                        "Greet them and ask for their name first — do NOT invent or guess a name."
                    )
                }
            # Hallucination guard (#287): the LLM occasionally invents a name
            # the user never spoke.  Reject any name that does not appear
            # (word-boundary, case-insensitive) in the recent user transcripts
            # and tell the LLM to ask for the name instead of persisting a
            # fabricated one.
            if not validate_name_or_warn(
                name,
                deps.recent_user_transcripts,
                tool_name="greet.identify",
            ):
                return {
                    "returning": False,
                    "name_received": None,
                    "needs_name": True,
                    "note": (
                        "The provided name was not heard in recent user speech. "
                        "Ask the visitor for their name before calling identify again."
                    ),
                }
            return await self._identify(deps, name)

        return {"error": f"Unknown action {action!r}. Use 'scan' or 'identify'."}
