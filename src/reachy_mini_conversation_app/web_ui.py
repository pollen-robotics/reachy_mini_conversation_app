"""Web UI server: FastAPI + fastrtc WebRTC for audio, SSE for control messages.

Uses fastrtc's WebRTC transport for bidirectional audio with built-in
echo cancellation. Control messages (transcripts, tool events, state)
are streamed to the frontend via Server-Sent Events (SSE).

Architecture
------------
Audio:    Browser <--WebRTC--> fastrtc Stream <--> OpenaiRealtimeHandler <--> OpenAI
Control:  Browser <---SSE---- /api/events (AdditionalOutputs from handler)
Config:   Browser <---REST--- /api/* endpoints
"""

import os
import json
import base64
import asyncio
import logging
from typing import Any, Optional
from pathlib import Path

import numpy as np
import cv2
import uvicorn
from numpy.typing import NDArray
from fastapi import FastAPI, Request
from fastrtc import Stream, AdditionalOutputs, audio_to_float32
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import resample

from reachy_mini_conversation_app.config import config, set_custom_profile
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.headless_personality import (
    DEFAULT_OPTION,
    _sanitize_name,
    _write_profile,
    list_personalities,
    available_tools_for,
    resolve_profile_dir,
    read_instructions_for,
)


logger = logging.getLogger(__name__)

WEB_APP_DIR = Path(__file__).parent / "web-app"
SETTINGS_STATIC_DIR = Path(__file__).parent / "static"


def _format_control_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an AdditionalOutputs dict to a frontend-friendly JSON event."""
    if "_state" in msg:
        return {"type": "state", "state": msg["_state"]}

    role = msg.get("role", "")
    content = msg.get("content", "")
    metadata = msg.get("metadata")

    if isinstance(content, str) and content.startswith("data:image"):
        return {"type": "image", "data": content}

    call_id = msg.get("call_id")

    if metadata:
        event: dict[str, Any] = {
            "type": "tool",
            "title": metadata.get("title", ""),
            "content": content if isinstance(content, str) else json.dumps(content),
            "status": "done",
        }
        if call_id:
            event["call_id"] = call_id
        return event

    if isinstance(content, str) and content.startswith("\U0001f6e0\ufe0f Used tool"):
        parts = content.split(" with args ", 1)
        tool_name = parts[0].replace("\U0001f6e0\ufe0f Used tool ", "")
        args_str = parts[1].split(". The tool is now running.")[0] if len(parts) > 1 else "{}"
        event = {"type": "tool", "title": tool_name, "content": args_str, "status": "running"}
        if call_id:
            event["call_id"] = call_id
        return event

    if isinstance(content, str) and content.startswith("[error]"):
        return {"type": "error", "content": content[len("[error] "):]}

    return {
        "type": "transcript",
        "role": role,
        "content": content if isinstance(content, str) else str(content),
    }


class WebUI:
    """Web server bridging a React frontend to the realtime handler via WebRTC."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        host: str = "0.0.0.0",
        port: int = 7860,
        instance_path: Optional[str] = None,
        dev_mode: bool = False,
        robot: Optional[Any] = None,
    ):
        """Initialize the web UI server with a realtime handler."""
        self.handler = handler
        self.host = host
        self.port = port
        self._instance_path = instance_path
        self.dev_mode = dev_mode
        self._robot = robot
        self.app = FastAPI(title="Reachy Mini Conversation")
        self._active_handler: Optional[OpenaiRealtimeHandler] = None

        self._patch_handler_copy()

        self.stream = Stream(
            handler=self.handler,
            mode="send-receive",
            modality="audio",
            concurrency_limit=None,
        )

        self._setup_middleware()
        self._setup_api_routes()

        # Mount fastrtc WebRTC routes before the catch-all static files
        self.stream.mount(self.app)

        self._setup_events_and_static()

    def _push_audio_to_robot_sync(self, sample_rate: int, audio_data: NDArray[np.int16]) -> None:
        """Send an audio frame to the robot's physical speaker (blocking, run in executor)."""
        try:
            output_sr = self._robot.media.get_output_audio_samplerate()
            data = audio_data
            if data.ndim == 2:
                if data.shape[1] > data.shape[0]:
                    data = data.T
                if data.shape[1] > 1:
                    data = data[:, 0]
            frame = audio_to_float32(data)
            if sample_rate != output_sr:
                frame = resample(frame, int(len(frame) * output_sr / sample_rate))
            self._robot.media.push_audio_sample(frame)
        except Exception as e:
            logger.debug("Robot speaker push failed: %s", e)

    def _patch_handler_for_robot(self, handler: OpenaiRealtimeHandler) -> None:
        """Route all audio I/O through the robot hardware instead of WebRTC.

        - emit() pushes audio to the robot speaker and returns None to WebRTC
          (AdditionalOutputs still flow through for SSE events).
        - receive() becomes a no-op so browser mic audio is ignored;
          the original receive is saved as _real_receive for the record loop.
        """
        if self._robot is None:
            return

        _original_emit = handler.emit

        async def _robot_emit():
            result = await _original_emit()
            if isinstance(result, tuple):
                sr, audio = result
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, self._push_audio_to_robot_sync, sr, audio)
                return None
            return result

        handler.emit = _robot_emit  # type: ignore[assignment]

        handler._real_receive = handler.receive  # type: ignore[attr-defined]

        async def _noop_receive(frame):
            pass

        handler.receive = _noop_receive  # type: ignore[assignment]

    async def _robot_record_loop(self) -> None:
        """Read audio from the robot's mic and feed it to the active handler."""
        input_sr = self._robot.media.get_input_audio_samplerate()
        loop = asyncio.get_event_loop()
        logger.info("Robot mic recording loop started at %d Hz", input_sr)
        while True:
            try:
                audio_frame = await loop.run_in_executor(
                    None, self._robot.media.get_audio_sample
                )
                if audio_frame is not None and self._active_handler is not None:
                    real_recv = getattr(self._active_handler, "_real_receive", self._active_handler.receive)
                    await real_recv((input_sr, audio_frame))
                elif audio_frame is None:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Robot mic read error: %s", e)
            await asyncio.sleep(0)

    def _patch_handler_copy(self) -> None:
        """Override handler.copy() to track the active session and wire interrupts."""
        _original_copy = self.handler.copy

        def _tracked_copy() -> OpenaiRealtimeHandler:
            new_handler = _original_copy()
            self._active_handler = new_handler

            def _clear_queue() -> None:
                while not new_handler.output_queue.empty():
                    try:
                        new_handler.output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                new_handler.output_queue.put_nowait(AdditionalOutputs({"_state": "interrupt"}))

            new_handler._clear_queue = _clear_queue  # type: ignore[attr-defined]
            self._patch_handler_for_robot(new_handler)
            return new_handler

        self.handler.copy = _tracked_copy  # type: ignore[assignment]

    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Startup personality persistence (same pattern as console.py)
    # ------------------------------------------------------------------
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        try:
            if env_path.exists():
                return env_path.read_text(encoding="utf-8").splitlines()
            for candidate in [
                env_path.parent / ".env.example",
                Path.cwd() / ".env.example",
                Path(__file__).parent / ".env.example",
            ]:
                if candidate.exists():
                    return candidate.read_text(encoding="utf-8").splitlines()
        except Exception:
            pass
        return []

    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        selection = (profile or "").strip() or None
        set_custom_profile(selection)
        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    def _startup_choice(self) -> str:
        """Return the persisted startup personality or default."""
        persisted = self._read_persisted_personality()
        if persisted:
            return persisted
        env_val = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        return env_val if env_val else DEFAULT_OPTION

    # ------------------------------------------------------------------
    # REST API for settings (personalities, voices, API key)
    # ------------------------------------------------------------------
    def _setup_api_routes(self) -> None:

        @self.app.get("/api/config")
        def api_config() -> dict[str, Any]:
            """Expose the OpenAI API key (from HF Secret or env) to the TS frontend."""
            key = os.environ.get("OPENAI_API_KEY", "")
            api_key = key.strip() if key.strip() else None
            if not api_key:
                api_key = str(config.OPENAI_API_KEY).strip() if config.OPENAI_API_KEY else None
            return {"openai_api_key": api_key, "dev_mode": self.dev_mode}

        @self.app.get("/api/status")
        def api_status() -> dict[str, Any]:
            has_key = bool(config.OPENAI_API_KEY and str(config.OPENAI_API_KEY).strip())
            cur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or DEFAULT_OPTION
            return {"has_key": has_key, "current_profile": cur}

        @self.app.get("/api/personalities")
        def api_personalities() -> dict[str, Any]:
            choices = [DEFAULT_OPTION, *list_personalities()]
            cur = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None) or DEFAULT_OPTION
            return {"choices": choices, "current": cur, "startup": self._startup_choice()}

        @self.app.get("/api/personalities/load")
        def api_load_personality(name: str) -> dict[str, Any]:
            instr = read_instructions_for(name)
            tools_txt = ""
            voice = "cedar"
            if name != DEFAULT_OPTION:
                pdir = resolve_profile_dir(name)
                tp = pdir / "tools.txt"
                if tp.exists():
                    tools_txt = tp.read_text(encoding="utf-8")
                vf = pdir / "voice.txt"
                if vf.exists():
                    v = vf.read_text(encoding="utf-8").strip()
                    voice = v or "cedar"
            avail = available_tools_for(name)
            enabled = [ln.strip() for ln in tools_txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            return {
                "instructions": instr,
                "tools_text": tools_txt,
                "voice": voice,
                "available_tools": avail,
                "enabled_tools": enabled,
            }

        @self.app.post("/api/personalities/save")
        async def api_save_personality(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            name = str(raw.get("name", ""))
            instructions = str(raw.get("instructions", ""))
            tools_text = str(raw.get("tools_text", ""))
            voice = str(raw.get("voice", "cedar")) if raw.get("voice") is not None else "cedar"

            name_s = _sanitize_name(name)
            if not name_s:
                return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
            try:
                _write_profile(name_s, instructions, tools_text, voice or "cedar")
                value = f"user_personalities/{name_s}"
                choices = [DEFAULT_OPTION, *list_personalities()]
                return JSONResponse({"ok": True, "value": value, "choices": choices})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        @self.app.post("/api/personalities/apply")
        async def api_apply_personality(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            sel_name = str(raw.get("name", DEFAULT_OPTION))
            persist = bool(raw.get("persist", False))
            handler = self._active_handler or self.handler
            sel = None if sel_name == DEFAULT_OPTION else sel_name
            try:
                status = await handler.apply_personality(sel)
                startup = self._startup_choice()
                if persist:
                    self._persist_personality(sel)
                    startup = self._startup_choice()
                return JSONResponse({"ok": True, "status": status, "startup": startup})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        @self.app.get("/api/voices")
        async def api_voices() -> list[str]:
            handler = self._active_handler or self.handler
            try:
                return await handler.get_available_voices()
            except Exception:
                return ["cedar"]

        @self.app.post("/api/validate_api_key")
        async def api_validate_key(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            key = str(raw.get("openai_api_key", "")).strip()
            if not key:
                return JSONResponse({"valid": False, "error": "empty_key"}, status_code=400)
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
                        return JSONResponse({"valid": False, "error": "validation_failed"}, status_code=response.status_code)
            except Exception as e:
                logger.warning("API key validation failed: %s", e)
                return JSONResponse({"valid": False, "error": "validation_error"}, status_code=500)

        @self.app.get("/api/camera/snapshot")
        def api_camera_snapshot() -> Response:
            """Return the latest camera frame as a base64-encoded JPEG."""
            try:
                cam = self.handler.deps.camera_worker
                if cam is None:
                    return JSONResponse({"error": "Camera worker not available"}, status_code=503)
                frame = cam.get_latest_frame()
                if frame is None:
                    return JSONResponse({"error": "No frame available"}, status_code=503)
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    return JSONResponse({"error": "Failed to encode frame"}, status_code=500)
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                return JSONResponse({"b64": b64})
            except Exception as e:
                logger.error("Camera snapshot error: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.post("/api/openai_api_key")
        async def api_set_key(request: Request) -> JSONResponse:
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            key = str(raw.get("openai_api_key", "")).strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            os.environ["OPENAI_API_KEY"] = key
            try:
                config.OPENAI_API_KEY = key
            except Exception:
                pass
            return JSONResponse({"ok": True})

    # ------------------------------------------------------------------
    # SSE events + static files
    # ------------------------------------------------------------------
    def _setup_events_and_static(self) -> None:

        @self.app.get("/api/events")
        async def sse_events(webrtc_id: str) -> StreamingResponse:
            """Stream control messages (transcripts, tools, state) via SSE."""
            async def event_stream() -> Any:
                yield "retry: 1000\n\n"
                try:
                    async for output in self.stream.output_stream(webrtc_id):
                        for msg in output.args:
                            if not isinstance(msg, dict):
                                continue
                            event = _format_control_message(msg)
                            if event:
                                yield f"data: {json.dumps(event)}\n\n"
                except asyncio.CancelledError:
                    logger.debug("SSE stream cancelled for %s (shutdown)", webrtc_id)
                except Exception as e:
                    logger.debug("SSE stream ended for %s: %s", webrtc_id, e)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        if SETTINGS_STATIC_DIR.exists():
            self.app.mount(
                "/settings",
                StaticFiles(directory=str(SETTINGS_STATIC_DIR), html=True),
                name="settings-static",
            )

        if WEB_APP_DIR.exists():
            self.app.mount("/", StaticFiles(directory=str(WEB_APP_DIR), html=True), name="web-app")
        else:
            logger.warning("Web app directory %s not found - frontend will not be served", WEB_APP_DIR)

    def launch(self) -> None:
        """Start the web server (blocking)."""
        if self._robot is not None:
            try:
                self._robot.media.start_playing()
                self._robot.media.start_recording()
                import time
                time.sleep(1)
                logger.info("Robot audio I/O enabled (mic + speaker)")
            except Exception as e:
                logger.warning("Could not start robot media pipelines: %s", e)

            @self.app.on_event("startup")
            async def _start_robot_mic() -> None:
                self._record_task = asyncio.create_task(self._robot_record_loop())

        logger.info("Starting web UI on http://%s:%s", self.host, self.port)
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self._server.run()

    def close(self) -> None:
        """Gracefully stop the web server and robot media."""
        server = getattr(self, "_server", None)
        if server is not None:
            server.should_exit = True

        if self._robot is not None:
            task = getattr(self, "_record_task", None)
            if task and not task.done():
                task.cancel()
            try:
                self._robot.media.stop_recording()
            except Exception:
                pass
            try:
                self._robot.media.stop_playing()
            except Exception:
                pass
