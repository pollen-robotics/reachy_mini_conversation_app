"""Entrypoint for Robot Comic."""

# ---------------------------------------------------------------------------
# Welcome WAV early-dispatch
# ---------------------------------------------------------------------------
# Operators need the welcome WAV audible within ~1s of `systemctl start`. The
# heavy non-stdlib imports below (fastrtc, google.genai, transformers, ...)
# take 5-15s on the Pi 5, so the previous warmup path in handler.start_up was
# audibly late. This block runs BEFORE any non-stdlib import so the WAV is
# dispatched immediately, in parallel with the rest of Python loading.
#
# The reachy_mini daemon runs persistently and owns the dmix-backed ALSA sink
# `plug:reachymini_audio_sink`, so subprocess-aplay-ing into it is safe from
# the app's first millisecond — no need to wait for any service to come up.
import os  # noqa: E402 — stdlib-only, must precede the early-play block
import sys  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402


def _play_welcome_early() -> None:
    """Fire-and-forget welcome WAV dispatch before heavy Python imports.

    Best-effort: never raises, never blocks. Sets
    ``REACHY_MINI_EARLY_WELCOME_PLAYED=1`` on successful dispatch so the later
    ``play_warmup_wav`` call in ``run()`` knows to skip and we don't double up.
    Honours ``REACHY_MINI_SKIP_EARLY_WELCOME=1`` to disable (sim/test use).
    """
    if os.environ.get("REACHY_MINI_SKIP_EARLY_WELCOME") == "1":
        return
    # Asset discovery mirrors ``warmup_audio.default_warmup_wav_path``:
    # ``src/robot_comic/main.py`` -> ``src/robot_comic`` -> ``src`` -> repo
    # root, then ``assets/welcome/``. Prefer the split intro from PR #311
    # if present, fall back to legacy ``welcome.wav``.
    assets = Path(__file__).resolve().parents[2] / "assets" / "welcome"
    wav: Path | None = None
    for candidate in ("welcome_intro.wav", "welcome.wav"):
        p = assets / candidate
        if p.is_file():
            wav = p
            break
    if wav is None:
        return  # nothing to play — fail silent at boot

    device = os.environ.get("REACHY_MINI_ALSA_DEVICE", "plug:reachymini_audio_sink")
    cmd = ["aplay", "-D", device, "-q", str(wav)]
    import time as _time  # noqa: PLC0415 — keep stdlib-only at this point

    started_at = _time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except FileNotFoundError:
        # ``aplay`` not on PATH (dev workstation without alsa-utils, Windows,
        # macOS). The full warmup_audio pipeline runs later and will pick the
        # right player for the host. Skip silently here.
        return
    except OSError:
        # Any other spawn failure (permissions, etc) — skip silently so boot
        # is never blocked by a welcome-WAV problem.
        return

    # Tell the in-process warmup_audio path to skip its own dispatch so the
    # operator doesn't hear the WAV start, get cut by handler audio capture,
    # and then start over again.
    os.environ["REACHY_MINI_EARLY_WELCOME_PLAYED"] = "1"

    # Fire ``welcome.wav.played`` for the dispatch row on the monitor
    # boot-timeline (#301 / #337). The in-process ``play_warmup_wav`` skips
    # when our env flag is set, so the early path is solely responsible for
    # this event. The emit is queued by ``telemetry`` until ``init()`` runs
    # later in ``run()`` (#337), so calling it before instrumentation is up
    # is safe and the row still lands on the timeline.
    try:
        from robot_comic import telemetry as _telemetry  # noqa: PLC0415

        _telemetry.emit_supporting_event(
            "welcome.wav.played",
            dur_ms=(_time.monotonic() - started_at) * 1000,
        )
    except Exception:
        # Telemetry must never break boot — drop the dispatch span if anything
        # in the wiring throws (import error, sys.path quirk, etc).
        pass

    # Fire a ``welcome.wav.completed`` span when aplay actually exits (#324).
    # ``warmup_audio`` is stdlib-only at module top, so importing it here
    # doesn't dent the early-welcome budget; the helper itself defers the
    # ``telemetry`` import until the wait thread emits.
    try:
        from robot_comic.warmup_audio import _wait_and_emit_completion  # noqa: PLC0415

        _wait_and_emit_completion(proc, cmd, started_at=started_at)
    except Exception:
        # Telemetry must never break boot — drop completion span if anything
        # in the wiring throws (import error, sys.path quirk, etc).
        pass


_play_welcome_early()

# Import the startup stopwatch first so STARTUP_T0 captures the earliest
# possible Python time. Any later "Startup: +X.XXs ..." log — including
# handler-side first-event hooks — shares this origin.
import warnings  # noqa: E402

from robot_comic import startup_timer  # noqa: F401,E402  (side-effect: captures t0)


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)

import time  # noqa: E402
import signal  # noqa: E402
import asyncio  # noqa: E402
import argparse  # noqa: E402
import threading  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from reachy_mini import ReachyMini, ReachyMiniApp
from robot_comic.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    get_requested_head_tracker,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for Robot Comic."""
    app = RobotComic()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


# Exit code used to signal the systemd unit that the app exited because the
# admin UI requested a restart. Paired with ``RestartForceExitStatus=75`` in
# the unit file so systemd relaunches us. A normal clean exit (status 0)
# continues to mean "stay down".
ADMIN_RESTART_EXIT_CODE: int = 75


def _install_shutdown_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    stop_event: Optional[threading.Event],
) -> None:
    """Install SIGTERM/SIGINT handlers that trigger graceful shutdown via *stop_event*.

    On Unix the handlers are registered with the asyncio event loop
    (``loop.add_signal_handler``) so they are delivered safely inside the loop
    without re-entrancy issues.  On Windows ``loop.add_signal_handler`` is not
    available; we fall back to ``signal.signal()`` with a thread-safe wrapper
    that schedules ``stop_event.set()`` onto the loop.

    The function is extracted so tests can exercise it without spinning up the
    full ``run()`` stack.
    """
    import logging as _logging

    _handler_logger = _logging.getLogger(__name__)

    def _on_signal(signum: int) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        _handler_logger.info("Received %s — initiating graceful shutdown", name)
        if stop_event is not None:
            stop_event.set()
        else:
            # No stop_event wired — escalate to KeyboardInterrupt so the
            # existing finally-block path still runs.
            raise KeyboardInterrupt

    if sys.platform != "win32":
        # Unix path: use the loop-safe add_signal_handler API.
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _on_signal, sig.value)
            except (NotImplementedError, RuntimeError):
                # add_signal_handler may raise if we're not in the main thread
                # or on an unsupported loop implementation.
                pass
    else:
        # Windows path: signal.signal() is limited to the main thread; use a
        # thread-safe wrapper that calls stop_event.set() and schedules a
        # loop wake-up so the running coroutine sees the event promptly.
        def _windows_handler(signum: int, _frame: Any) -> None:
            _on_signal(signum)
            # Wake the loop so stop_event is noticed without waiting for I/O.
            try:
                loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _windows_handler)
            except (ValueError, OSError):
                # signal.signal() only works in the main thread; skip silently.
                pass


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[Any] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run Robot Comic."""
    # Install signal handlers as early as possible so a SIGTERM during cold-boot
    # still triggers the graceful shutdown path (movement_manager.stop(),
    # robot.goto_sleep(), etc.) rather than escalating to SIGKILL.
    _install_shutdown_signal_handlers(asyncio.get_event_loop(), app_stop_event)

    # Putting these dependencies here makes the dashboard faster to load when Robot Comic is installed.
    from robot_comic.moves import MovementManager
    from robot_comic.config import (
        config,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
    from robot_comic.handler_factory import HandlerFactory
    from robot_comic.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)

    # Set by the admin restart endpoint to distinguish a "restart me" exit from a
    # plain "stop". Checked after the graceful-shutdown path to decide between
    # exit 0 and exit ADMIN_RESTART_EXIT_CODE.
    restart_requested_event = threading.Event()

    try:
        import subprocess

        _git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        _git_hash = "unknown"
    logger.info("Starting Robot Comic (git: %s)", _git_hash)
    startup_settings = StartupSettings()

    if instance_path is not None:
        try:
            from dotenv import load_dotenv

            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
        except Exception as e:
            logger.warning("Failed to load instance configuration: %s", e)

        try:
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except Exception as e:
            logger.warning("Failed to load startup settings: %s", e)

    from robot_comic import telemetry as _telemetry

    _telemetry.init()

    # Surface "app.startup" on the monitor boot-timeline lane (#301): time
    # from Python process import (STARTUP_T0) to telemetry initialisation.
    try:
        _telemetry.emit_supporting_event(
            "app.startup",
            dur_ms=startup_timer.since_startup() * 1000,
        )
    except Exception:
        # Telemetry must never block startup; swallow any export failure.
        pass

    # Log the active pipeline_mode + audio backends. HF-realtime keeps the
    # connection-mode detail; everything else also logs the model name.
    if config.PIPELINE_MODE == "hf_realtime":
        logger.info(
            "Configured pipeline mode: %s (%s), connection mode: %s",
            config.PIPELINE_MODE,
            get_backend_label(),
            get_hf_connection_selection().mode,
        )
    else:
        logger.info(
            "Configured pipeline mode: %s (%s), audio: %s → %s, model: %s",
            config.PIPELINE_MODE,
            get_backend_label(),
            config.AUDIO_INPUT_BACKEND,
            config.AUDIO_OUTPUT_BACKEND,
            config.MODEL_NAME,
        )

    from robot_comic.pause import PauseController
    from robot_comic.startup_timer import log_checkpoint

    log_checkpoint("import pause", logger)
    from robot_comic.console import LocalStream

    log_checkpoint("import console", logger)
    from robot_comic.pause_settings import read_pause_settings

    log_checkpoint("import pause_settings", logger)
    from robot_comic.tools.core_tools import ToolDependencies

    log_checkpoint("import core_tools", logger)
    from robot_comic.audio.head_wobbler import HeadWobbler

    log_checkpoint("import head_wobbler", logger)

    if args.no_camera and get_requested_head_tracker(args) is not None:
        logger.warning("Head tracking disabled: --no-camera flag is set. Remove --no-camera to enable head tracking.")

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    log_checkpoint("robot available", logger)

    # Fire warmup audio as early as possible on headless on-robot startup, in
    # parallel with the rest of the stack loading. The handler will take over
    # the speakers once it spins up; the warmup subprocess being cut off
    # mid-playback is expected, not a bug.
    if not args.sim and config.WARMUP_WAV_ENABLED:
        try:
            from robot_comic.warmup_audio import play_warmup_wav

            play_warmup_wav(config.WARMUP_WAV_PATH)
            # Checkpoint is emitted inside play_warmup_wav so it fires only on
            # successful Popen (dispatched) or graceful skip (skipped), making
            # the +Xs delta to first TTS audio frame meaningful.
        except Exception as e:
            logger.warning("Warmup WAV playback failed: %s", e)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.sim:
        logger.info("Simulation mode detected. Automatically enabling --sim.")
        args.sim = True

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(args, robot)
    except CameraVisionInitializationError as e:
        logger.error("Failed to initialize camera/vision: %s", e)
        sys.exit(1)
    log_checkpoint("camera/vision init", logger)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )
    if startup_settings.movement_speed is not None:
        movement_manager.set_speed_factor(startup_settings.movement_speed)
        logger.info(
            "Applied persisted movement_speed=%.2f from startup_settings.json",
            startup_settings.movement_speed,
        )
    log_checkpoint("movement manager", logger)

    # Pin antennas to neutral so the motor controller's PID has an explicit
    # target to hold during the idle window between init and the first
    # conversation turn. Without this, the antennas exhibit small-amplitude
    # PID hunting (±0.005 rad ≈ ±0.3°) visible to the operator; holding the
    # antenna physically converges the loop, so does any explicit set_target.
    # Caught live 2026-05-19 — the controller drifts a few encoder ticks
    # when no target is asserted. The movement manager's 60 Hz loop hasn't
    # started yet (start() is called at the end of this function), so this
    # single set_target sticks until the loop takes over.
    try:
        idle_head = robot.get_current_head_pose()
        # Preserve the current body yaw instead of forcing 0.0 so a restart
        # does not snap the body forward if the robot is already turned.
        # get_current_joint_positions() returns ([body_yaw, ...6 stewart], [ant1, ant2]);
        # the first element of the head list is the body yaw joint (rad).
        try:
            _head_joints, _ = robot.get_current_joint_positions()
            idle_body_yaw: float = float(_head_joints[0])
        except Exception:
            idle_body_yaw = 0.0
        robot.set_target(head=idle_head, antennas=[0.0, 0.0], body_yaw=idle_body_yaw)
        logger.info("Pinned idle antennas to neutral [0.0, 0.0] body_yaw=%.4f", idle_body_yaw)
    except Exception as idle_exc:
        logger.warning("Could not pin idle antennas to neutral: %s", idle_exc)

    head_wobbler = HeadWobbler(
        set_speech_offsets=movement_manager.set_speech_offsets,
        speed_factor_getter=lambda: movement_manager.speed_factor,
    )

    def _request_shutdown_from_pause() -> None:
        logger.info("Pause controller requested shutdown")
        if app_stop_event is not None:
            app_stop_event.set()
        else:
            logger.warning("No app_stop_event available; cannot trigger graceful shutdown")

    pause_settings = read_pause_settings(instance_path)
    pause_controller = PauseController(
        clear_move_queue=movement_manager.clear_move_queue,
        on_shutdown=_request_shutdown_from_pause,
        on_pause_state_changed=movement_manager.set_paused,
        stop_phrases=pause_settings.resolved_stop(),
        resume_phrases=pause_settings.resolved_resume(),
        shutdown_phrases=pause_settings.resolved_shutdown(),
        switch_phrases=pause_settings.resolved_switch(),
    )
    log_checkpoint("pause controller ready", logger)

    # Face recognition pipeline — instantiated here so tools receive fully-wired deps.
    face_db_instance: Any = None
    face_embedder_instance: Any = None
    if config.FACE_RECOGNITION_ENABLED:
        try:
            from robot_comic.vision.face_db import FaceDatabase
            from robot_comic.vision.face_recognition_embedder import FaceRecognitionEmbedder

            face_db_instance = FaceDatabase()
            face_embedder_instance = FaceRecognitionEmbedder()
            logger.info("Face recognition enabled: FaceRecognitionEmbedder + FaceDatabase ready")
        except ImportError as _fr_exc:
            from robot_comic.vision.face_embedder import StubFaceEmbedder

            face_embedder_instance = StubFaceEmbedder()
            logger.warning(
                "REACHY_MINI_FACE_RECOGNITION_ENABLED=1 but face_recognition library is missing "
                "(%s). Falling back to StubFaceEmbedder — face matching will not work. "
                "Install with: pip install -e '.[face_recognition]'",
                _fr_exc,
            )
        except Exception as _fr_exc:
            logger.warning("Face recognition init failed (%s); feature disabled.", _fr_exc)
    log_checkpoint("face recognition init", logger)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_processor=vision_processor,
        head_wobbler=head_wobbler,
        pause_controller=pause_controller,
        instance_path=Path(instance_path) if instance_path is not None else None,
        face_db=face_db_instance,
        face_embedder=face_embedder_instance,
    )
    log_checkpoint("tool deps ready", logger)

    logger.info(
        "Using audio backends: input=%r output=%r pipeline_mode=%r",
        config.AUDIO_INPUT_BACKEND,
        config.AUDIO_OUTPUT_BACKEND,
        config.PIPELINE_MODE,
    )
    handler = HandlerFactory.build(
        config.AUDIO_INPUT_BACKEND,
        config.AUDIO_OUTPUT_BACKEND,
        deps,
        pipeline_mode=config.PIPELINE_MODE,
        sim_mode=args.sim,
        instance_path=instance_path,
        startup_voice=startup_settings.voice,
    )
    log_checkpoint("handler init", logger)

    # ---------------------------------------------------------------------------
    # WebSocket Pi ↔ laptop channel (opt-in via REACHY_MINI_WS_ENABLED).
    # ---------------------------------------------------------------------------
    _ws_endpoint: Any = None  # WsClient (Pi) or WsServer (laptop)
    _ws_loop: asyncio.AbstractEventLoop | None = None

    if config.WS_ENABLED:
        from robot_comic.ws_client import WsClient
        from robot_comic.ws_server import WsServer

        # Heuristic: if LLAMA_CPP_URL points to localhost we're on the laptop.
        _is_laptop = config.LLAMA_CPP_URL.startswith("http://localhost") or config.LLAMA_CPP_URL.startswith(
            "http://127."
        )

        _ws_loop = asyncio.new_event_loop()

        def _run_ws_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _ws_thread = threading.Thread(target=_run_ws_loop, args=(_ws_loop,), daemon=True, name="ws-io")
        _ws_thread.start()

        if _is_laptop:
            _ws_server = WsServer(port=config.WS_PORT)

            def _laptop_on_message(msg: Any, _conn: Any) -> None:
                from robot_comic.ws_protocol import MsgType

                if msg.type in (MsgType.PI_STATUS, MsgType.PI_EVENT):
                    logger.info("ws_server: received %s: %s", msg.type, msg.payload)
                elif msg.type == MsgType.LAPTOP_COMMAND:
                    action = msg.payload.get("action") or msg.payload.get("command", "")
                    if action == "restart":
                        logger.info(
                            "ws_server: restart command received — triggering exit %d", ADMIN_RESTART_EXIT_CODE
                        )
                        os.kill(os.getpid(), signal.SIGTERM)
                    elif action == "pause":
                        logger.info("ws_server: pause command received")
                        config.WS_PAUSE_FLAG = True
                    elif action == "resume":
                        logger.info("ws_server: resume command received")
                        config.WS_PAUSE_FLAG = False
                    else:
                        logger.info("ws_server: laptop_command action=%r (unhandled)", action)

            _ws_server.on_message(_laptop_on_message)
            asyncio.run_coroutine_threadsafe(_ws_server.start(), _ws_loop)
            _ws_endpoint = _ws_server
            logger.info("ws: laptop mode — WsServer started on port %d", config.WS_PORT)
        else:
            _ws_client = WsClient(server_host=config.WS_SERVER_HOST, port=config.WS_PORT)
            asyncio.run_coroutine_threadsafe(_ws_client.start(), _ws_loop)
            _ws_endpoint = _ws_client
            logger.info("ws: Pi mode — WsClient connecting to %s:%d", config.WS_SERVER_HOST, config.WS_PORT)

        # Wire the WS endpoint into the handler so it can emit per-turn status.
        if hasattr(handler, "set_ws_client"):
            handler.set_ws_client(_ws_endpoint)

    log_checkpoint("handler ready", logger)

    # Kiosk-mode startup screen: opt-in welcome prompt + dynamic persona listing.
    # Runs before the per-persona greeting so the two never overlap.
    if not args.sim and config.STARTUP_SCREEN_ENABLED:
        try:
            from robot_comic.startup_screen import run_startup_screen

            asyncio.get_event_loop().run_until_complete(
                run_startup_screen(
                    chatterbox_url=config.CHATTERBOX_URL,
                    profiles_dir=config.PROFILES_DIRECTORY,
                    persona_order=config.STARTUP_SCREEN_PERSONA_ORDER,
                )
            )
        except Exception as _ss_exc:
            logger.warning("startup_screen: failed (continuing): %s", _ss_exc)

    stream_manager: Any = None

    if args.sim:
        # Sim/dev mode — load fastrtc + gradio only now (deferred from top-level
        # import to avoid ~10 s cold-start cost in headless on-robot mode).
        import gradio as gr
        from fastapi import FastAPI
        from fastrtc import Stream

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        chatbot = gr.Chatbot(
            type="messages",
            resizable=True,
            allow_tags=False,
            avatar_images=(
                os.path.join(current_file_path, "images", "user_avatar.png"),
                os.path.join(current_file_path, "images", "reachymini_avatar.png"),
            ),
        )
        log_checkpoint("chatbot ready", logger)

        # Bridge audio to the browser via FastRTC; admin UI at "/" (same as
        # headless), FastRTC chat UI at "/chat".
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        admin_only_stream = LocalStream(
            handler=None,
            robot=None,
            settings_app=app,
            instance_path=instance_path,
            app_stop_event=app_stop_event,
            restart_requested_event=restart_requested_event,
            pause_controller=pause_controller,
            movement_manager=movement_manager,
        )
        admin_only_stream.init_admin_ui()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[chatbot],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Robot Comic"},
        )
        stream_manager = stream.ui

        app = gr.mount_gradio_app(app, stream.ui, path="/chat")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
            app_stop_event=app_stop_event,
            restart_requested_event=restart_requested_event,
            pause_controller=pause_controller,
            movement_manager=movement_manager,
        )
        log_checkpoint("LocalStream constructed", logger)

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    # camera_worker is deliberately NOT started here: the gstreamer pipeline
    # adds ~5 s to the cold-start budget and the boot greeting does not need
    # vision. CameraWorker.ensure_started() is invoked lazily by the first
    # camera-touching tool call (camera / greet / roast / head_tracking),
    # which is the only time we actually need a live capture loop. See #323.

    # Emit boot event now that all handlers are ready (Pi side only).
    if config.WS_ENABLED and _ws_loop is not None and _ws_endpoint is not None:
        from robot_comic.ws_client import WsClient as _WsClient
        from robot_comic.ws_protocol import make_pi_event as _make_pi_event

        if isinstance(_ws_endpoint, _WsClient):
            _boot_future = asyncio.run_coroutine_threadsafe(_ws_endpoint.send(_make_pi_event("boot")), _ws_loop)
            try:
                _boot_future.result(timeout=1.0)
            except Exception as _boot_exc:
                logger.warning("ws: failed to emit boot event: %s", _boot_exc)

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        if args.sim:
            stream_manager.launch(server_name="0.0.0.0")
        else:
            stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()

        # Emit shutdown event (Pi side) and stop WS endpoint.
        if config.WS_ENABLED and _ws_loop is not None and _ws_endpoint is not None:
            from robot_comic.ws_client import WsClient as _WsClientShutdown
            from robot_comic.ws_protocol import make_pi_event as _make_pi_event_shutdown

            if isinstance(_ws_endpoint, _WsClientShutdown):
                _sd_future = asyncio.run_coroutine_threadsafe(
                    _ws_endpoint.send(_make_pi_event_shutdown("shutdown")), _ws_loop
                )
                try:
                    _sd_future.result(timeout=1.0)
                except Exception as _sd_exc:
                    logger.warning("ws: failed to emit shutdown event: %s", _sd_exc)

            _stop_future = asyncio.run_coroutine_threadsafe(_ws_endpoint.stop(), _ws_loop)
            try:
                _stop_future.result(timeout=2.0)
            except Exception as _stop_exc:
                logger.warning("ws: error stopping ws endpoint: %s", _stop_exc)

            _ws_loop.call_soon_threadsafe(_ws_loop.stop)

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        try:
            robot.goto_sleep()
            # Pin the motor controller's stored target to the physical sleep
            # position. Without this, enable_motors() on next boot snaps to
            # the previous neutral target (set by movement_manager.stop()),
            # causing a visible head jerk before the wake_up animation runs.
            try:
                sleep_head = robot.get_current_head_pose()
                _, sleep_antennas = robot.get_current_joint_positions()
                robot.set_target(head=sleep_head, antennas=list(sleep_antennas), body_yaw=0.0)
            except Exception as snap_e:
                logger.debug("Could not pin sleep position target: %s", snap_e)
            robot.disable_motors()
        except Exception as e:
            logger.warning(f"Error during goto_sleep on app shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")

    if restart_requested_event.is_set():
        # Signal the autostart unit to relaunch us. The unit pairs this with
        # ``RestartForceExitStatus=75`` so admin-driven restarts come back up
        # even though ``Restart=on-failure`` ignores clean exits.
        logger.info("Admin restart requested — exiting with code %d to trigger relaunch", ADMIN_RESTART_EXIT_CODE)
        sys.exit(ADMIN_RESTART_EXIT_CODE)


class RobotComic(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for Robot Comic."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run Robot Comic."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        data_dir = Path.home() / ".robot_comic"
        data_dir.mkdir(exist_ok=True)
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=str(data_dir),
        )


if __name__ == "__main__":
    app = RobotComic()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
