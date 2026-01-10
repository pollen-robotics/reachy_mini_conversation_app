import logging
import argparse
import warnings
from typing import Any, Tuple, Optional

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker


def parse_args() -> Tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe", None],
        default=None,
        help="Choose head tracker (default: None)",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of gpt-realtime vision",
    )
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name/prefix for Zenoh topics (must match daemon's --robot-name). Only needed for development with multiple robots.",
    )
    return parser.parse_known_args()


def handle_vision_stuff(args: argparse.Namespace, current_robot: ReachyMini) -> Tuple[CameraWorker | None, Any, Any]:
    """Initialize camera, head tracker, camera worker, and vision manager.

    By default, vision is handled by gpt-realtime model when camera tool is used.
    If --local-vision flag is used, a local vision model will process images periodically.
    """
    camera_worker = None
    head_tracker = None
    vision_manager = None

    if not args.no_camera:
        # Initialize head tracker if specified
        if args.head_tracker is not None:
            if args.head_tracker == "yolo":
                from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

                head_tracker = HeadTracker()
            elif args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import HeadTracker  # type: ignore[no-redef]

                head_tracker = HeadTracker()

        # Initialize camera worker
        camera_worker = CameraWorker(current_robot, head_tracker)

        # Initialize vision manager only if local vision is requested
        if args.local_vision:
            try:
                from reachy_mini_conversation_app.vision.processors import initialize_vision_manager

                vision_manager = initialize_vision_manager(camera_worker)
            except ImportError as e:
                raise ImportError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e
        else:
            logging.getLogger(__name__).info(
                "Using gpt-realtime for vision (default). Use --local-vision for local processing.",
            )

    return camera_worker, head_tracker, vision_manager


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger

def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(
            f"  2. Daemon must be started with: --robot-name '{robot_name}'"
        )
    else:
        logger.error(
            "  2. If daemon uses --robot-name, add the same flag here: "
            "--robot-name <name>"
        )

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
