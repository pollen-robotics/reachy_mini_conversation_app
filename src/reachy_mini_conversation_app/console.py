"""Bidirectional local audio stream with optional settings UI.

In headless mode, there is no Gradio UI. If the OpenAI API key is not
available via environment/.env, we expose a minimal settings page via the
Reachy Mini Apps settings server to let non-technical users enter it.

The settings UI is served from this package's ``static/`` folder and offers a
single password field to set ``OPENAI_API_KEY``. Once set, we persist it to the
app instance's ``.env`` file (if available) and proceed to start streaming.
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore[assignment]
    FileResponse = object  # type: ignore[assignment]
    JSONResponse = object  # type: ignore[assignment]
    StaticFiles = object  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ):
        """Initialize the stream with an OpenAI realtime handler and pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None

    # ---- Settings UI (only when API key is missing) ----
    def _persist_api_key(self, key: str) -> None:
        """Persist API key to environment and instance ``.env`` if possible.

        Behavior:
        - Always sets ``OPENAI_API_KEY`` in process env and in-memory config.
        - Writes/updates ``<instance_path>/.env``:
          * If ``.env`` exists, replaces/append OPENAI_API_KEY line.
          * Else, copies template from ``<instance_path>/.env.example`` when present,
            otherwise falls back to the packaged template
            ``reachy_mini_conversation_app/.env.example``.
          * Ensures the resulting file contains the full template plus the key.
        - Loads the written ``.env`` into the current process environment.
        """
        k = (key or "").strip()
        if not k:
            return
        # Update live process env and config so consumers see it immediately
        try:
            os.environ["OPENAI_API_KEY"] = k
        except Exception:  # best-effort
            pass
        try:
            config.OPENAI_API_KEY = k  # type: ignore[attr-defined]
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines: list[str]
            if env_path.exists():
                try:
                    lines = env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    lines = []
            else:
                # Try instance template first
                template_text = None
                ex = inst / ".env.example"
                if ex.exists():
                    try:
                        template_text = ex.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
                # Fallback to CWD template
                if template_text is None:
                    try:
                        cwd_example = Path.cwd() / ".env.example"
                        if cwd_example.exists():
                            template_text = cwd_example.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None

                # Fallback to packaged template
                if template_text is None:
                    packaged = Path(__file__).parent / ".env.example"
                    if packaged.exists():
                        try:
                            template_text = packaged.read_text(encoding="utf-8")
                        except Exception:
                            template_text = None
                lines = template_text.splitlines() if template_text else []
            replaced = False
            for i, ln in enumerate(lines):
                if ln.strip().startswith("OPENAI_API_KEY="):
                    lines[i] = f"OPENAI_API_KEY={k}"
                    replaced = True
                    break
            if not replaced:
                lines.append(f"OPENAI_API_KEY={k}")
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted OPENAI_API_KEY to %s", env_path)

            # Load the newly written .env into this process to ensure downstream imports see it
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist OPENAI_API_KEY: %s", e)

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                # Serve /static/* assets
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")  # type: ignore[arg-type]
            except Exception:
                pass

        class ApiKeyPayload(BaseModel):  # type: ignore[misc,valid-type]
            openai_api_key: str

        # GET / -> index.html
        @self._settings_app.get("/")  # type: ignore[union-attr]
        def _root() -> FileResponse:  # type: ignore[no-redef]
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")  # type: ignore[union-attr]
        def _favicon() -> Response:  # type: ignore[no-redef]
            return Response(status_code=204)

        # GET /status -> whether key is set
        @self._settings_app.get("/status")  # type: ignore[union-attr]
        def _status() -> JSONResponse:  # type: ignore[no-redef]
            has_key = bool(config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip())
            return JSONResponse({"has_key": has_key})

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")  # type: ignore[union-attr]
        def _ready() -> JSONResponse:  # type: ignore[no-redef]
            try:
                mod = sys.modules.get("reachy_mini_conversation_app.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # POST /openai_api_key -> set/persist key
        @self._settings_app.post("/openai_api_key")  # type: ignore[union-attr]
        def _set_key(payload: ApiKeyPayload) -> JSONResponse:  # type: ignore[no-redef]
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            self._persist_api_key(key)
            return JSONResponse({"ok": True})

        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the OpenAI key is missing, expose a tiny settings UI via the
        Reachy Mini settings server to collect it before starting streams.
        """
        self._stop_event.clear()

        # Always expose settings UI if a settings app is available
        self._init_settings_ui_if_needed()

        # Try to load an existing instance .env first (covers subsequent runs)
        if not (config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip()) and self._instance_path:
            try:
                from dotenv import load_dotenv

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    # Update config with newly loaded value
                    new_key = os.getenv("OPENAI_API_KEY", "").strip()
                    if new_key:
                        try:
                            config.OPENAI_API_KEY = new_key  # type: ignore[attr-defined]
                        except Exception:
                            pass
            except Exception:
                pass

        # If key is still missing -> wait until provided via the settings UI
        if not (config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip()):
            logger.warning("OPENAI_API_KEY not found. Open the app settings page to enter it.")
            # Poll until the key becomes available (set via the settings UI)
            try:
                while not (config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip()):
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for API key.")
                return

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(self._settings_app, self.handler, lambda: self._asyncio_loop)
            except Exception:
                pass
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        - Stops audio recording and playback
        """
        logger.info("Stopping LocalStream...")
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        self._robot.media.stop_recording()
        self._robot.media.stop_playing()

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            # Directly flush gstreamer audio pipe
            self._robot.media.audio.clear_player()
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                # Reshape if needed
                if audio_data.ndim == 2:
                    # Scipy channels last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
