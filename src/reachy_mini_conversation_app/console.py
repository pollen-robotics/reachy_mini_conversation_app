"""Bidirectional local audio stream with optional settings UI.

In headless mode, there is no Gradio UI. If the selected backend is missing
its required API key, we expose a minimal settings page via the Reachy Mini
Apps settings server so users can pick a backend and provide any missing
credentials.

The settings UI is served from this package's ``static/`` folder. It persists
the selected backend and any provided API keys into the app instance's ``.env``
file when available.
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
from reachy_mini_conversation_app.config import (
    HF_BACKEND,
    GEMINI_BACKEND,
    LOCKED_PROFILE,
    OPENAI_BACKEND,
    HF_REALTIME_WS_URL_ENV,
    HF_LOCAL_CONNECTION_MODE,
    HF_DEPLOYED_CONNECTION_MODE,
    HF_REALTIME_CONNECTION_MODE_ENV,
    config,
    get_backend_choice,
    get_hf_session_url,
    get_hf_direct_ws_url,
    build_hf_direct_ws_url,
    has_hf_realtime_target,
    parse_hf_direct_target,
    get_model_name_for_backend,
    get_hf_connection_selection,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.startup_settings import read_startup_settings, write_startup_settings
from reachy_mini_conversation_app.audio.startup_config import apply_audio_startup_config
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore


logger = logging.getLogger(__name__)

LOCAL_PLAYER_BACKEND = (
    getattr(MediaBackend, "LOCAL", None)
    or getattr(MediaBackend, "GSTREAMER", None)
    or getattr(MediaBackend, "DEFAULT", None)
)

LEGACY_STARTUP_ENV_NAMES = (
    "REACHY_MINI_CUSTOM_PROFILE",
    "REACHY_MINI_VOICE_OVERRIDE",
)


def _estimate_pending_playback_seconds(robot: ReachyMini) -> float:
    """Best-effort estimate of audio still queued in the local player."""
    media = getattr(robot, "media", None)
    audio = getattr(media, "audio", None)
    if audio is None:
        return 0.0

    next_pts_ns = getattr(audio, "_playback_next_pts_ns", None)
    get_running_time_ns = getattr(audio, "_get_playback_running_time_ns", None)
    if next_pts_ns is None or not callable(get_running_time_ns):
        return 0.0

    try:
        pending_ns = int(next_pts_ns) - int(get_running_time_ns())
    except Exception:
        return 0.0

    return max(0.0, pending_ns / 1e9)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: ConversationHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ):
        """Initialize the stream with a realtime handler and pipelines.

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
        self._active_backend_name = get_backend_choice()

    # ---- Settings UI ----
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []

    def _active_backend(self) -> str:
        """Return the backend family of the currently running handler."""
        return self._active_backend_name

    @staticmethod
    def _has_key(value: Optional[str]) -> bool:
        """Return whether a runtime credential value is present."""
        return bool(value and str(value).strip())

    def _has_required_key(self, backend: str) -> bool:
        """Return whether the requested backend has its required credential."""
        if backend == GEMINI_BACKEND:
            return self._has_key(config.GEMINI_API_KEY)
        if backend == HF_BACKEND:
            return has_hf_realtime_target()
        return self._has_key(config.OPENAI_API_KEY)

    @staticmethod
    def _requirement_name(backend: str) -> str:
        """Return the env var users need for a backend, if any."""
        if backend == GEMINI_BACKEND:
            return "GEMINI_API_KEY"
        if backend == HF_BACKEND:
            return HF_REALTIME_WS_URL_ENV
        return "OPENAI_API_KEY"

    def _persist_env_value(self, env_name: str, value: str) -> None:
        """Persist a non-empty environment value in memory and in the instance `.env`."""
        self._persist_env_values({env_name: value})

    def _persist_env_values(self, updates: dict[str, str]) -> None:
        """Persist non-empty environment values in memory and in the instance `.env`."""
        normalized_updates = {name: (value or "").strip() for name, value in updates.items()}
        normalized_updates = {name: value for name, value in normalized_updates.items() if value}
        if not normalized_updates:
            return

        for env_name, value in normalized_updates.items():
            try:
                os.environ[env_name] = value
            except Exception:
                pass
        refresh_runtime_config_from_env()

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            for env_name, value in normalized_updates.items():
                replaced = False
                for i, ln in enumerate(lines):
                    if ln.strip().startswith(f"{env_name}="):
                        lines[i] = f"{env_name}={value}"
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"{env_name}={value}")
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted %s to %s", ", ".join(sorted(normalized_updates)), env_path)

            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path))
            except Exception:
                pass
            refresh_runtime_config_from_env()
        except Exception as e:
            logger.warning("Failed to persist %s: %s", ", ".join(sorted(normalized_updates)), e)

    def _remove_persisted_env_values(self, env_names: tuple[str, ...]) -> None:
        """Remove keys from the instance `.env` without mutating the current runtime."""
        normalized_names = tuple(sorted({name.strip() for name in env_names if name and name.strip()}))
        if not normalized_names or not self._instance_path:
            return

        env_path = Path(self._instance_path) / ".env"
        if not env_path.exists():
            return

        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
            filtered_lines = [
                line
                for line in lines
                if not any(line.strip().startswith(f"{env_name}=") for env_name in normalized_names)
            ]
            if filtered_lines == lines:
                return

            final_text = "\n".join(filtered_lines)
            if final_text:
                final_text += "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Removed %s from %s", ", ".join(normalized_names), env_path)
        except Exception as e:
            logger.warning("Failed to remove %s: %s", ", ".join(normalized_names), e)

    def _persist_hf_direct_connection(self, host: str, port: int) -> None:
        """Persist a direct Hugging Face websocket target."""
        self._persist_env_values(
            {
                HF_REALTIME_CONNECTION_MODE_ENV: HF_LOCAL_CONNECTION_MODE,
                HF_REALTIME_WS_URL_ENV: build_hf_direct_ws_url(host, port),
            }
        )

    def _persist_hf_allocator_connection(self) -> None:
        """Persist the deployed Hugging Face allocator mode."""
        self._persist_env_value(HF_REALTIME_CONNECTION_MODE_ENV, HF_DEPLOYED_CONNECTION_MODE)
        self._remove_persisted_env_values(("HF_REALTIME_SESSION_URL",))

    def _persist_api_key(self, key: str) -> None:
        """Persist OPENAI_API_KEY to environment and instance `.env`."""
        self._persist_env_value("OPENAI_API_KEY", key)

    def _persist_gemini_api_key(self, key: str) -> None:
        """Persist GEMINI_API_KEY to environment and instance `.env`."""
        self._persist_env_value("GEMINI_API_KEY", key)

    def _persist_backend_choice(self, backend: str) -> None:
        """Persist the selected backend without clobbering explicit model overrides."""
        current_backend = get_backend_choice()
        current_model_name = (os.getenv("MODEL_NAME") or "").strip()
        updates = {"BACKEND_PROVIDER": backend}
        if backend == HF_BACKEND:
            self._persist_env_values(updates)
            try:
                os.environ.pop("MODEL_NAME", None)
            except Exception:
                pass
            self._remove_persisted_env_values(("MODEL_NAME",))
            refresh_runtime_config_from_env()
            return

        if current_model_name and current_model_name != get_model_name_for_backend(current_backend):
            updates["MODEL_NAME"] = current_model_name
        else:
            updates["MODEL_NAME"] = get_model_name_for_backend(backend)
        self._persist_env_values(updates)

    def _persist_personality(self, profile: Optional[str], voice_override: Optional[str] = None) -> None:
        """Persist startup profile and voice in instance-local UI settings."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        normalized_voice_override = (voice_override or "").strip() or None
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            write_startup_settings(
                self._instance_path,
                profile=selection,
                voice=normalized_voice_override,
            )
            self._remove_persisted_env_values(LEGACY_STARTUP_ENV_NAMES)
            logger.info("Persisted startup personality settings to %s", Path(self._instance_path))
        except Exception as e:
            logger.warning("Failed to persist startup personality settings: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read the saved startup personality from instance-local UI settings."""
        return read_startup_settings(self._instance_path).profile

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
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        class ApiKeyPayload(BaseModel):
            openai_api_key: str

        class BackendPayload(BaseModel):
            backend: str
            api_key: Optional[str] = None
            hf_mode: Optional[str] = None
            hf_host: Optional[str] = None
            hf_port: Optional[int] = None

        def _status_payload() -> dict[str, object]:
            backend_provider = get_backend_choice()
            active_backend = self._active_backend()
            has_openai_key = self._has_required_key(OPENAI_BACKEND)
            has_gemini_key = self._has_required_key(GEMINI_BACKEND)
            hf_session_url = get_hf_session_url()
            hf_ws_url = get_hf_direct_ws_url()
            hf_direct_host, hf_direct_port = parse_hf_direct_target(hf_ws_url)
            has_hf_session_url = bool(hf_session_url)
            has_hf_ws_url = bool(hf_ws_url)
            hf_connection_selection = get_hf_connection_selection()
            hf_connection_mode = hf_connection_selection.mode
            has_hf_connection = hf_connection_selection.has_target
            can_proceed_with_openai = has_openai_key
            can_proceed_with_gemini = has_gemini_key
            can_proceed_with_hf = has_hf_connection
            can_proceed = self._has_required_key(active_backend)
            requires_restart = backend_provider != active_backend
            return {
                "active_backend": active_backend,
                "backend_provider": backend_provider,
                "has_key": can_proceed,
                "has_openai_key": has_openai_key,
                "has_gemini_key": has_gemini_key,
                "has_hf_session_url": has_hf_session_url,
                "has_hf_ws_url": has_hf_ws_url,
                "has_hf_connection": has_hf_connection,
                "hf_connection_mode": hf_connection_mode,
                "hf_direct_host": hf_direct_host,
                "hf_direct_port": hf_direct_port,
                "can_proceed": can_proceed,
                "can_proceed_with_openai": can_proceed_with_openai,
                "can_proceed_with_gemini": can_proceed_with_gemini,
                "can_proceed_with_hf": can_proceed_with_hf,
                "requires_restart": requires_restart,
            }

        # GET / -> index.html
        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # GET /status -> whether key is set
        @self._settings_app.get("/status")
        def _status() -> JSONResponse:
            return JSONResponse(_status_payload())

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("reachy_mini_conversation_app.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # POST /openai_api_key -> set/persist key
        @self._settings_app.post("/openai_api_key")
        def _set_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            self._persist_api_key(key)
            return JSONResponse({"ok": True, **_status_payload()})

        @self._settings_app.post("/backend_config")
        def _set_backend(payload: BackendPayload) -> JSONResponse:
            backend = payload.backend.strip().lower()
            if backend not in {OPENAI_BACKEND, GEMINI_BACKEND, HF_BACKEND}:
                return JSONResponse({"ok": False, "error": "invalid_backend"}, status_code=400)

            api_key = (payload.api_key or "").strip()
            if backend == GEMINI_BACKEND and not api_key and not self._has_required_key(GEMINI_BACKEND):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)

            if backend == OPENAI_BACKEND and api_key:
                self._persist_api_key(api_key)
            if backend == GEMINI_BACKEND and api_key:
                self._persist_gemini_api_key(api_key)
            if backend == HF_BACKEND:
                hf_selection = get_hf_connection_selection()
                hf_mode = (payload.hf_mode or hf_selection.mode).strip().lower()
                if hf_mode == HF_LOCAL_CONNECTION_MODE:
                    existing_host, existing_port = parse_hf_direct_target(hf_selection.direct_ws_url)
                    host = (payload.hf_host or "").strip() or existing_host or ""
                    if not host:
                        return JSONResponse({"ok": False, "error": "empty_hf_host"}, status_code=400)
                    if "://" in host or "/" in host or "?" in host or "#" in host:
                        return JSONResponse({"ok": False, "error": "invalid_hf_host"}, status_code=400)

                    port = payload.hf_port if payload.hf_port is not None else existing_port or 8765
                    if port < 1 or port > 65535:
                        return JSONResponse({"ok": False, "error": "invalid_hf_port"}, status_code=400)

                    self._persist_hf_direct_connection(host, port)
                elif hf_mode == HF_DEPLOYED_CONNECTION_MODE:
                    if not bool(get_hf_session_url()):
                        return JSONResponse({"ok": False, "error": "missing_hf_session_url"}, status_code=400)
                    self._persist_hf_allocator_connection()
                else:
                    return JSONResponse({"ok": False, "error": "invalid_hf_mode"}, status_code=400)

            self._persist_backend_choice(backend)
            payload_data = _status_payload()
            message = "Backend saved."
            if payload_data["requires_restart"]:
                message = "Backend saved. Restart Reachy Mini Conversation from the desktop app to apply it."
            return JSONResponse(
                {
                    "ok": True,
                    "message": message,
                    **payload_data,
                }
            )

        # POST /validate_api_key -> validate key without persisting it
        @self._settings_app.post("/validate_api_key")
        async def _validate_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"valid": False, "error": "empty_key"}, status_code=400)

            # Try to validate by checking if we can fetch the models
            try:
                import httpx

                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("https://api.openai.com/v1/models", headers=headers)
                    if response.status_code == 200:
                        return JSONResponse({"valid": True})
                    elif response.status_code == 401:
                        return JSONResponse({"valid": False, "error": "invalid_api_key"}, status_code=401)
                    else:
                        return JSONResponse(
                            {"valid": False, "error": "validation_failed"}, status_code=response.status_code
                        )
            except Exception as e:
                logger.warning(f"API key validation failed: {e}")
                return JSONResponse({"valid": False, "error": "validation_error"}, status_code=500)

        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the selected backend is missing its required key, expose a tiny
        settings UI via the Reachy Mini settings server to collect it before
        starting streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    refresh_runtime_config_from_env()
            except Exception:
                pass  # Instance .env loading is optional; continue with defaults

        active_backend = self._active_backend()

        # Always expose settings UI if a settings app is available
        # (do this AFTER loading the instance .env so status endpoint sees the right value)
        self._init_settings_ui_if_needed()

        # If key is still missing -> wait until provided via the settings UI
        if not self._has_required_key(active_backend):
            requirement_name = self._requirement_name(active_backend)
            if active_backend == HF_BACKEND:
                logger.error(
                    "%s not found. Set it in the app .env before starting the Hugging Face backend.", requirement_name
                )
                return
            else:
                logger.warning("%s not found. Open the app settings page to enter it.", requirement_name)
            # Poll until the key becomes available (set via the settings UI)
            try:
                while not self._has_required_key(active_backend):
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for API key.")
                return

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start
        apply_audio_startup_config(self._robot, logger=logger)

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
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
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Now signal async loops to stop
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        backend = getattr(self._robot.media, "backend", None)
        audio = getattr(self._robot.media, "audio", None)
        if audio is not None:
            if (
                LOCAL_PLAYER_BACKEND is not None
                and backend == LOCAL_PLAYER_BACKEND
                and hasattr(audio, "clear_player")
                and callable(audio.clear_player)
            ):
                audio.clear_player()
            elif (
                backend == MediaBackend.WEBRTC
                and hasattr(audio, "clear_output_buffer")
                and callable(audio.clear_output_buffer)
            ):
                audio.clear_output_buffer()
            elif hasattr(audio, "clear_output_buffer") and callable(audio.clear_output_buffer):
                audio.clear_output_buffer()
            elif hasattr(audio, "clear_player") and callable(audio.clear_player):
                audio.clear_player()
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

                # Skip empty audio frames
                if audio_data.size == 0:
                    continue

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
                    num_samples = int(len(audio_frame) * output_sample_rate / input_sample_rate)
                    if num_samples == 0:
                        continue
                    audio_frame = resample(
                        audio_frame,
                        num_samples,
                    )

                head_wobbler = self.handler.deps.head_wobbler
                if head_wobbler is not None:
                    playback_delay_s = _estimate_pending_playback_seconds(self._robot)
                    head_wobbler.feed_pcm(audio_data.reshape(1, -1), input_sample_rate, start_delay_s=playback_delay_s)

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
