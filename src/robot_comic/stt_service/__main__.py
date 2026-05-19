"""CLI entrypoint for the reachy-stt service.

Usage::

    python -m robot_comic.stt_service --uds /run/reachy-stt.sock
    python -m robot_comic.stt_service --uds /run/reachy-stt.sock --model base.en --compute-type int8

On Windows the ``--uds`` flag is not supported (Unix-domain sockets require
Linux or macOS).  The script will print a clear error and exit.  For local
Windows development, bind to a TCP port instead by using uvicorn directly::

    uvicorn robot_comic.stt_service.server:app --host 127.0.0.1 --port 8021
"""

from __future__ import annotations
import os
import sys
import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robot_comic.stt_service",
        description="reachy-stt — persistent faster-whisper STT service on a Unix-domain socket.",
    )
    parser.add_argument(
        "--uds",
        required=True,
        metavar="PATH",
        help="Path to the Unix-domain socket file (e.g. /run/reachy-stt.sock).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("REACHY_MINI_FASTER_WHISPER_MODEL", "base.en"),
        metavar="NAME",
        help="faster-whisper model name (default: base.en, overridden by REACHY_MINI_FASTER_WHISPER_MODEL).",
    )
    parser.add_argument(
        "--compute-type",
        default=os.getenv("REACHY_MINI_FASTER_WHISPER_COMPUTE_TYPE", "int8"),
        metavar="TYPE",
        help="CTranslate2 compute type (default: int8, overridden by REACHY_MINI_FASTER_WHISPER_COMPUTE_TYPE).",
    )
    return parser


def main() -> None:
    """Parse CLI args and start the uvicorn server on a Unix-domain socket."""
    parser = _build_parser()
    args = parser.parse_args()

    if sys.platform == "win32":
        parser.error(
            "Unix-domain sockets (--uds) are not supported on Windows. "
            "The production target is Linux (Raspberry Pi). "
            "For local Windows dev, run uvicorn directly on a TCP port: "
            "uvicorn robot_comic.stt_service.server:app --host 127.0.0.1 --port 8021"
        )

    # Expose the chosen model/compute-type via env so server.py defaults pick them up.
    os.environ["REACHY_MINI_FASTER_WHISPER_MODEL"] = args.model
    os.environ["REACHY_MINI_FASTER_WHISPER_COMPUTE_TYPE"] = args.compute_type

    import uvicorn

    from robot_comic.stt_service.server import app

    uvicorn.run(app, uds=args.uds, log_level="info")


if __name__ == "__main__":
    main()
