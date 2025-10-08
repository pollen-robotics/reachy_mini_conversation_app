import logging
import argparse
import warnings

from reachy_mini_conversation_demo.camera_worker import CameraWorker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation Demo")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe", None],
        default=None,
        help="Choose head tracker (default: None)",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    return parser.parse_args()


def handle_vision_stuff(args, current_robot):
    """Initialize camera, head tracker and camera worker."""
    logger = logging.getLogger(__name__)
    camera_worker = None
    head_tracker = None
    vision_manager = None
    if not args.no_camera:
        # Head tracking is disabled in simulation mode
        if args.head_tracker is not None and not args.sim:
            if args.head_tracker == "yolo":
                from reachy_mini_conversation_demo.vision.yolo_head_tracker import (
                    HeadTracker,
                )

                head_tracker = HeadTracker()

            elif args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import HeadTracker

                head_tracker = HeadTracker()
        elif args.head_tracker is not None and args.sim:
            logger.warning(
                f"Head tracking (--head-tracker {args.head_tracker}) is disabled in simulation mode (--sim)"
            )

        camera_worker = CameraWorker(current_robot, head_tracker, use_sim=args.sim)

    return camera_worker, head_tracker, vision_manager


def setup_logger(debug):
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
