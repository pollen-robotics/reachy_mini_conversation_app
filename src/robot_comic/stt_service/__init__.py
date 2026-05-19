"""reachy-stt service package.

Exposes a FastAPI application that keeps a faster-whisper WhisperModel in
memory and serves three endpoints over a Unix-domain socket (Linux/macOS) or
a TCP port (Windows dev-only):

- ``GET  /healthz``   — liveness + model-loaded flag.
- ``POST /preload``   — warm-load the model in a background thread.
- ``POST /transcribe``— accept raw int16 PCM @ 16 kHz mono, return transcript.

The process entrypoint is ``__main__.py``; run with::

    python -m robot_comic.stt_service --uds /run/reachy-stt.sock

See :mod:`robot_comic.stt_service.server` for the FastAPI app object.
"""

from robot_comic.stt_service.server import app


__all__ = ["app"]
