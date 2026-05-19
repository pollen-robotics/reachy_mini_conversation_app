"""FastAPI application for the reachy-stt service.

Lifecycle
---------
The model is **not** loaded at process start.  Loading is triggered by the
first call to ``POST /preload`` or ``POST /transcribe``.  PR-B will add a
systemd ``ExecStartPost`` that conditionally hits ``/preload`` depending on
whether on-device STT is the active backend.

Thread safety
-------------
Model loading is slow (~10–20 s).  We run it in :func:`asyncio.to_thread` so
the event loop stays responsive and ``GET /healthz`` keeps answering while the
load is in progress.  A :class:`asyncio.Lock` serialises concurrent load
requests so the model is only constructed once even when ``/preload`` and
``/transcribe`` race.
"""

from __future__ import annotations
import os
import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-configurable defaults (mirror faster_whisper_stt_adapter.py read sites)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = os.getenv("REACHY_MINI_FASTER_WHISPER_MODEL", "base.en")
_DEFAULT_COMPUTE_TYPE = os.getenv("REACHY_MINI_FASTER_WHISPER_COMPUTE_TYPE", "int8")

# ---------------------------------------------------------------------------
# Module-level model state
# ---------------------------------------------------------------------------
_model: Any = None  # faster_whisper.WhisperModel instance, or None
_model_lock: asyncio.Lock | None = None  # created lazily inside an event loop


def _get_lock() -> asyncio.Lock:
    """Return the singleton asyncio.Lock, creating it on first call."""
    global _model_lock
    if _model_lock is None:
        _model_lock = asyncio.Lock()
    return _model_lock


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="reachy-stt", version="0.1.0")


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Return liveness status and whether the WhisperModel is in memory."""
    return JSONResponse({"status": "ok", "model_loaded": _model is not None})


# ---------------------------------------------------------------------------
# /preload
# ---------------------------------------------------------------------------


class _PreloadRequest(BaseModel):
    model: str | None = None
    compute_type: str | None = None


@app.post("/preload")
async def preload(body: _PreloadRequest | None = None) -> JSONResponse:
    """Load the WhisperModel into memory if not already loaded.

    Accepts an optional JSON body ``{"model": "...", "compute_type": "..."}``.
    Defaults are pulled from the process environment
    (``REACHY_MINI_FASTER_WHISPER_MODEL`` / ``REACHY_MINI_FASTER_WHISPER_COMPUTE_TYPE``).

    Idempotent: if the model is already loaded this returns immediately.
    """
    model_name = (body.model if body else None) or _DEFAULT_MODEL
    compute_type = (body.compute_type if body else None) or _DEFAULT_COMPUTE_TYPE
    await _ensure_model_loaded(model_name, compute_type)
    return JSONResponse({"loaded": True})


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------


@app.post("/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    """Transcribe raw int16 PCM audio @ 16 kHz mono.

    Request body: raw bytes (``Content-Type: application/octet-stream``).
    The payload must be int16 little-endian PCM sampled at 16 kHz, mono.
    An empty body returns a 400 error.

    Returns ``{"text": str, "segments": [{"text": str, "start": float, "end": float}, ...]}``.

    If the model is not yet loaded, loads it on demand using the env-default
    model name / compute type.
    """
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body — expected int16 PCM @ 16 kHz mono.")

    # Load on demand.
    await _ensure_model_loaded(_DEFAULT_MODEL, _DEFAULT_COMPUTE_TYPE)

    try:
        result = await asyncio.to_thread(_transcribe_sync, audio_bytes)
    except Exception as exc:  # pragma: no cover — surface unexpected errors
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}") from exc

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _ensure_model_loaded(model_name: str, compute_type: str) -> None:
    """Load the WhisperModel if not already loaded.

    Serialised by ``_get_lock()`` so concurrent callers do not race.
    Runs the blocking load in a thread so the event loop stays live.
    """
    global _model
    if _model is not None:
        return
    async with _get_lock():
        # Re-check under lock — another coroutine may have loaded while we waited.
        if _model is not None:
            return
        logger.info("Loading faster-whisper model=%s compute_type=%s", model_name, compute_type)
        try:
            await asyncio.to_thread(_load_model_sync, model_name, compute_type)
        except Exception as exc:
            logger.error("Failed to load faster-whisper model: %s", exc)
            raise HTTPException(status_code=503, detail=f"Model load failed: {exc}") from exc


def _load_model_sync(model_name: str, compute_type: str) -> None:
    """Load the WhisperModel synchronously — runs inside :func:`asyncio.to_thread`."""
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install with: uv pip install -e .[faster_whisper_stt]"
        ) from exc

    _model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    logger.info("faster-whisper model loaded: %s / %s", model_name, compute_type)


def _transcribe_sync(audio_bytes: bytes) -> dict[str, Any]:
    """Run transcription synchronously — called inside :func:`asyncio.to_thread`."""
    import numpy as np

    assert _model is not None, "_transcribe_sync called before model is loaded"

    # Decode int16 PCM bytes to float32 in [-1, 1].
    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    segments_iter, _info = _model.transcribe(audio_f32, language="en")

    segments_out: list[dict[str, Any]] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = getattr(seg, "text", "") or ""
        text = text.strip()
        if text:
            parts.append(text)
        segments_out.append(
            {
                "text": text,
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
            }
        )

    return {"text": " ".join(parts).strip(), "segments": segments_out}


# ---------------------------------------------------------------------------
# Teardown helper (used by tests to reset module state between runs)
# ---------------------------------------------------------------------------


def _reset_state_for_tests() -> None:
    """Reset module-level model state.  Call from test teardown only."""
    global _model, _model_lock
    _model = None
    _model_lock = None
