import logging
import argparse
import warnings
from typing import Any, Tuple, Optional
from importlib.metadata import PackageNotFoundError, version

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
        help="[Optional] Robot name to target during discovery. Useful when multiple robots are available on the same network.",
    )
    return parser.parse_known_args()


def apply_runtime_compatibility_fixes() -> None:
    """Apply runtime compatibility fixes for specific SDK versions.

    Keep this in the main app module tree (not a dedicated patch module) so the
    workaround lifecycle is explicit in production code.
    """
    _patch_reachy_mini_1_5_0_webrtc_audio()


def _patch_reachy_mini_1_5_0_webrtc_audio() -> None:
    """Patch Reachy Mini 1.5.0 WebRTC outbound audio chain on desktop clients."""
    logger = logging.getLogger(__name__)

    try:
        sdk_version = version("reachy-mini")
    except PackageNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.debug("Could not resolve reachy-mini version: %s", exc)
        return

    if sdk_version != "1.5.0":
        return

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("WebRTC audio compatibility fix not applied: %s", exc)
        return

    if getattr(GstWebRTCClient, "_conversation_app_audio_chain_patched", False):
        return

    def _make_unique_element(target_bin: Any, factory_name: str, base_name: str) -> Any | None:
        for idx in range(100):
            suffix = "" if idx == 0 else f"_{idx}"
            name = f"{base_name}{suffix}"
            if target_bin.get_by_name(name) is not None:
                continue
            element = Gst.ElementFactory.make(factory_name, name)
            if element is not None:
                return element
        return None

    def _bin_add_with_parent_check(target_bin: Any, element: Any) -> bool:
        """Add element to bin and verify via parent check.

        On macOS GObject introspection bindings, ``bin.add`` may return ``None``
        even when insertion succeeds.
        """
        target_bin.add(element)
        return element.get_parent() == target_bin

    def _bin_remove_if_parent(target_bin: Any, element: Any) -> None:
        if element.get_parent() == target_bin:
            target_bin.remove(element)

    def _patched_setup_audio_send_chain(self: Any) -> None:
        if self._audio_send_ready:
            return
        self._audio_send_ready = True

        self.logger.info("Setting up audio send chain...")
        if self._webrtcbin is None:
            self.logger.error("webrtcbin not found, cannot set up audio send chain")
            self._audio_send_ready = False
            return

        webrtcbin_parent = self._webrtcbin.get_parent()

        sink_pad = None
        pt = 96
        for pad in self._iterate_gst(self._webrtcbin.iterate_sink_pads()):
            if pad.is_linked():
                continue
            caps = pad.query_caps(None)
            if caps and caps.get_size() > 0:
                structure = caps.get_structure(0)
                encoding = structure.get_string("encoding-name")
                if encoding and encoding.upper() == "OPUS":
                    sink_pad = pad
                    ok, payload = structure.get_int("payload")
                    if ok:
                        pt = payload
                    self.logger.info("Found audio sink pad: %s, pt=%s", pad.get_name(), pt)
                    break

        if sink_pad is None:
            self.logger.error("No OPUS sink pad found on webrtcbin, audio send disabled")
            self._audio_send_ready = False
            return

        target_bins = []
        for candidate in (webrtcbin_parent, self._webrtcsrc, self._pipeline_record):
            if candidate is None:
                continue
            if any(candidate is known for known in target_bins):
                continue
            target_bins.append(candidate)

        for target_bin in target_bins:
            appsrc = _make_unique_element(target_bin, "appsrc", "reachymini_send_appsrc")
            audioconvert = _make_unique_element(target_bin, "audioconvert", "reachymini_send_audioconvert")
            audioresample = _make_unique_element(target_bin, "audioresample", "reachymini_send_audioresample")
            opusenc = _make_unique_element(target_bin, "opusenc", "reachymini_send_opusenc")
            rtpopuspay = _make_unique_element(target_bin, "rtpopuspay", "reachymini_send_rtpopuspay")

            if not all((appsrc, audioconvert, audioresample, opusenc, rtpopuspay)):
                self.logger.error("Failed to create one or more audio send elements")
                continue

            appsrc.set_property("format", Gst.Format.TIME)
            appsrc.set_property("is-live", True)
            caps = Gst.Caps.from_string(
                f"audio/x-raw,format=F32LE,channels={self.CHANNELS},rate={self.SAMPLE_RATE},layout=interleaved"
            )
            appsrc.set_property("caps", caps)
            opusenc.set_property("audio-type", "restricted-lowdelay")
            opusenc.set_property("frame-size", 10)
            rtpopuspay.set_property("pt", pt)

            elements = (appsrc, audioconvert, audioresample, opusenc, rtpopuspay)
            added = []

            add_ok = True
            for element in elements:
                if not _bin_add_with_parent_check(target_bin, element):
                    self.logger.error(
                        "Failed to add %s to %s",
                        element.get_name(),
                        target_bin.get_name(),
                    )
                    add_ok = False
                    break
                added.append(element)

            if not add_ok:
                for element in reversed(added):
                    _bin_remove_if_parent(target_bin, element)
                continue

            if (
                not appsrc.link(audioconvert)
                or not audioconvert.link(audioresample)
                or not audioresample.link(opusenc)
                or not opusenc.link(rtpopuspay)
            ):
                self.logger.error("Failed to link WebRTC audio send elements")
                for element in reversed(added):
                    _bin_remove_if_parent(target_bin, element)
                continue

            src_pad = rtpopuspay.get_static_pad("src")
            link_result = src_pad.link_full(sink_pad, Gst.PadLinkCheck.NOTHING)
            if link_result != Gst.PadLinkReturn.OK:
                self.logger.error(
                    "Failed to link rtpopuspay to webrtcbin (%s) using bin %s",
                    link_result,
                    target_bin.get_name(),
                )
                for element in reversed(added):
                    _bin_remove_if_parent(target_bin, element)
                continue

            for element in elements:
                element.sync_state_with_parent()

            self._appsrc = appsrc
            self._audio_send_ready = True
            self._audio_send_setup_tries = 0
            self._audio_send_drop_count = 0
            self.logger.info(
                "Audio send chain ready (bidirectional audio enabled) via bin=%s",
                target_bin.get_name(),
            )
            return

        self._audio_send_ready = False
        self.logger.error("Audio send chain setup failed for all candidate bins")

    original_push_audio_sample = GstWebRTCClient.push_audio_sample

    def _patched_push_audio_sample(self: Any, data: Any) -> None:
        if self._appsrc is None:
            tries = int(getattr(self, "_audio_send_setup_tries", 0)) + 1
            self._audio_send_setup_tries = tries
            if tries <= 5 or tries % 100 == 0:
                self.logger.warning(
                    "WebRTC appsrc not ready, retrying audio send chain setup (attempt %d)",
                    tries,
                )
            try:
                self._setup_audio_send_chain()
            except Exception as exc:
                self.logger.error("Audio send-chain retry crashed: %s", exc)

        if self._appsrc is None:
            dropped = int(getattr(self, "_audio_send_drop_count", 0)) + 1
            self._audio_send_drop_count = dropped
            if dropped <= 5 or dropped % 200 == 0:
                self.logger.warning(
                    "Dropping outbound audio frame (appsrc unavailable, dropped=%d)",
                    dropped,
                )
            return

        try:
            import numpy as np

            if data.ndim == 1 and self.CHANNELS > 1:
                data = np.repeat(data[:, np.newaxis], self.CHANNELS, axis=1)
            elif data.ndim == 2 and data.shape[1] == 1 and self.CHANNELS > 1:
                data = np.repeat(data, self.CHANNELS, axis=1)
            elif data.ndim == 2 and data.shape[1] > self.CHANNELS:
                data = data[:, : self.CHANNELS]

            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
        except Exception:
            pass

        original_push_audio_sample(self, data)

    GstWebRTCClient._setup_audio_send_chain = _patched_setup_audio_send_chain
    GstWebRTCClient.push_audio_sample = _patched_push_audio_sample
    GstWebRTCClient._conversation_app_audio_chain_patched = True
    logger.info("Applied Reachy Mini 1.5.0 WebRTC audio compatibility fix")


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
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
