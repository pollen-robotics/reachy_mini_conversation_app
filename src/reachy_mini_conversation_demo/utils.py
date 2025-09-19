import argparse  # noqa: D100
import asyncio
import logging
import warnings
from threading import Thread

from reachy_mini.utils.camera import find_camera

from reachy_mini_conversation_demo.camera_worker import CameraWorker

logger = logging.getLogger(__name__)


def print_registered_loggers():
    """Print registered logger namespaces to stdout for troubleshooting."""
    manager = logging.root.manager
    registered = sorted(name for name in manager.loggerDict if name)
    if registered:
        print("[Logging] Registered loggers: " + ", ".join(registered))
    else:
        print("[Logging] Registered loggers: (none detected)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation Demo")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe", None],
        default=None,
        help="Choose head tracker (default: mediapipe)",
    )
    parser.add_argument(
        "--no-camera", default=False, action="store_true", help="Disable camera usage"
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Run in headless mode"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def handle_vision_stuff(args, current_robot):
    """Initialize camera, head tracker and camera worker."""
    camera = None
    camera_worker = None
    head_tracker = None
    vision_manager = None
    if not args.no_camera:
        camera = find_camera()

        if args.head_tracker is not None:
            if args.head_tracker == "yolo":
                from reachy_mini_conversation_demo.vision.yolo_head_tracker import (
                    HeadTracker,
                )

        if camera is None:
            msg = "[Vision] No camera detected (find_camera returned None); vision features disabled."
            print(msg)
            logger.warning(msg)
        else:
            camera_ready = True
            try:
                if hasattr(camera, "isOpened") and not camera.isOpened():
                    camera_ready = False
                    msg = (
                        "[Vision] Camera handle is closed or unavailable; vision features disabled."
                    )
                    print(msg)
                    logger.warning(msg)
                else:
                    ret, _ = camera.read()
                    if not ret:
                        msg = (
                            "[Vision] Camera opened but initial frame read failed; will retry in worker."
                        )
                        print(msg)
                        logger.warning(msg)
            except Exception as exc:
                camera_ready = False
                msg = f"[Vision] Camera test failed ({exc}); vision features disabled."
                print(msg)
                logger.warning(msg)

            if not camera_ready and hasattr(camera, "release"):
                try:
                    camera.release()
                except Exception:
                    pass
                camera = None

        if camera_ready:
            if args.head_tracker is not None:
                if args.head_tracker == "yolo":
                    from reachy_mini_conversation_demo.vision.yolo_head_tracker import (
                        HeadTracker,
                    )

                    head_tracker = HeadTracker()

                elif args.head_tracker == "mediapipe":
                    from reachy_mini_toolbox.vision import HeadTracker

                    head_tracker = HeadTracker()

            camera_worker = CameraWorker(camera, current_robot, head_tracker)
        else:
            camera = None

    return camera, camera_worker, head_tracker, vision_manager


class AioTaskThread:
    """Runs a single coroutine in its own thread and event loop."""

    def __init__(self, coro_fn, *args, **kwargs):
        """Coro_fn will be called as: await coro_fn(*args, _stop_async, **kwargs)."""
        self.coro_fn = coro_fn
        self.args = args
        self.kwargs = kwargs
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run, daemon=True)
        self._stop_async: asyncio.Event | None = None

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._stop_async = asyncio.Event()

        async def runner():
            await self.coro_fn(*self.args, self._stop_async, **self.kwargs)

        try:
            self.loop.run_until_complete(runner())
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def start(self):
        """Start the thread and its event loop."""
        self.thread.start()

    def request_stop(self):
        """Request the coroutine to stop by setting the _stop_async event."""
        if self._stop_async is not None:
            self.loop.call_soon_threadsafe(self._stop_async.set)

    def join(self):
        """Wait for the thread to finish."""
        self.thread.join()


def setup_logger(debug):
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # print_registered_loggers()

    app_logger = logging.getLogger("reachy_mini_conversation_demo")
    app_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    app_handler = logging.StreamHandler()
    app_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s"
        )
    )
    app_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    app_logger.handlers.clear()
    app_logger.addHandler(app_handler)
    app_logger.propagate = False

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

<<<<<<< HEAD
    third_party_levels = {
        "aiortc": logging.INFO,
        "fastrtc": logging.INFO,
        "aioice": logging.INFO,
        "websockets": logging.INFO,
        "openai": logging.INFO,
        "httpcore": logging.INFO,
    }
    for name, level in third_party_levels.items():
        logging.getLogger(name).setLevel(level)
=======
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
>>>>>>> develop
    return logger
