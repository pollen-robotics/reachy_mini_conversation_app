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
from typing import Any, List, Optional, Protocol, cast
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from robot_comic.config import (
    HF_BACKEND,
    GEMINI_BACKEND,
    LOCKED_PROFILE,
    OPENAI_BACKEND,
    AUDIO_OUTPUT_HF,
    LLM_BACKEND_ENV,
    LLM_BACKEND_LLAMA,
    LOCAL_STT_BACKEND,
    PIPELINE_MODE_ENV,
    CHATTERBOX_URL_ENV,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_CHOICES,
    LOCAL_STT_MODEL_ENV,
    AUDIO_OUTPUT_CHOICES,
    CHATTERBOX_VOICE_ENV,
    AUDIO_INPUT_MOONSHINE,
    PIPELINE_MODE_CHOICES,
    CHATTERBOX_DEFAULT_URL,
    HF_REALTIME_WS_URL_ENV,
    LOCAL_STT_LANGUAGE_ENV,
    LOCAL_STT_PROVIDER_ENV,
    AUDIO_INPUT_BACKEND_ENV,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    LOCAL_STT_CACHE_DIR_ENV,
    LOCAL_STT_MODEL_CHOICES,
    AUDIO_OUTPUT_BACKEND_ENV,
    CHATTERBOX_DEFAULT_VOICE,
    HF_LOCAL_CONNECTION_MODE,
    PIPELINE_MODE_COMPOSABLE,
    PIPELINE_MODE_GEMINI_LIVE,
    PIPELINE_MODE_HF_REALTIME,
    AUDIO_CAPTURE_PATH_ALSA_RW,
    HF_DEPLOYED_CONNECTION_MODE,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    LOCAL_STT_UPDATE_INTERVAL_ENV,
    PIPELINE_MODE_OPENAI_REALTIME,
    HF_REALTIME_CONNECTION_MODE_ENV,
    config,
    get_provider_id,
    get_hf_session_url,
    get_hf_direct_ws_url,
    build_hf_direct_ws_url,
    has_hf_realtime_target,
    parse_hf_direct_target,
    provider_id_from_pipeline,
    get_hf_connection_selection,
    refresh_runtime_config_from_env,
)
from robot_comic.pause_settings import (
    PausePhraseSettings,
    read_pause_settings,
    write_pause_settings,
    settings_from_payload,
)
from robot_comic.startup_settings import read_startup_settings, write_startup_settings
from robot_comic.audio.startup_config import apply_audio_startup_config
from robot_comic.conversation_handler import ConversationHandler
from robot_comic.headless_personality_ui import mount_personality_routes


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
CROWD_HISTORY_DIRNAME = ".comedy_sessions"
# Legacy directory name retained so the admin UI can still surface old sessions
# from before the rename in commit 141f761.
LEGACY_CROWD_HISTORY_DIRNAME = ".rickles_sessions"


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


_BATTERY_CACHE_TTL_S = 5.0
_battery_cache: dict[str, Any] = {}
_battery_warned_once = False


def _read_battery(robot: Optional[ReachyMini]) -> dict[str, Any]:
    """Read battery info from the robot SDK, with a 5-second cache.

    Returns a dict with at minimum a ``source`` key:
    - ``"sim"``    — no real robot present (sim / headless-without-robot mode)
    - ``"unknown"``— robot present but SDK exposes no battery API
    - ``"robot"``  — real reading (not yet reachable with current SDK v1.7.1)
    """
    global _battery_warned_once

    now = time.monotonic()
    cached = _battery_cache.get("entry")
    if cached is not None and (now - cached["ts"]) < _BATTERY_CACHE_TTL_S:
        return cast(dict[str, Any], cached["value"])

    if robot is None:
        result: dict[str, Any] = {"source": "sim", "percent": None}
    else:
        # The Reachy Mini SDK v1.7.1 does not expose a public battery API.
        # Check defensively in case a future SDK version adds it.
        battery_attr = getattr(robot, "battery", None)
        if battery_attr is not None:
            try:
                percent = getattr(battery_attr, "percent", None)
                voltage = getattr(battery_attr, "voltage", None)
                charging = getattr(battery_attr, "charging", None)
                result = {
                    "source": "robot",
                    "percent": int(percent) if percent is not None else None,
                    "voltage": float(voltage) if voltage is not None else None,
                    "charging": bool(charging) if charging is not None else None,
                }
            except Exception as exc:
                if not _battery_warned_once:
                    logger.debug("Battery read failed: %s", exc)
                    _battery_warned_once = True
                result = {"source": "unknown", "percent": None}
        else:
            if not _battery_warned_once:
                logger.debug("Reachy Mini SDK does not expose a battery attribute (SDK v1.7.1)")
                _battery_warned_once = True
            result = {"source": "unknown", "percent": None}

    _battery_cache["entry"] = {"ts": now, "value": result}
    return result


class _AudioSource(Protocol):
    """Uniform shape for record_loop's audio producer.

    Both _DaemonAudioSource (legacy path via r.media) and AlsaRwCapture
    (direct arecord) conform to this Protocol so record_loop is agnostic.
    """

    @property
    def sample_rate(self) -> int: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_audio_sample(self) -> Any: ...


class _DaemonAudioSource:
    """Adapter that exposes r.media.get_audio_sample as the _AudioSource shape."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self._sample_rate = int(robot.media.get_input_audio_samplerate())

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        # Daemon's media.start_recording() is called separately in launch().
        # No additional work needed here.
        return None

    def stop(self) -> None:
        return None

    def get_audio_sample(self) -> Any:
        return self._robot.media.get_audio_sample()


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: Optional[ConversationHandler],
        robot: Optional[ReachyMini],
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
        app_stop_event: Optional[Any] = None,
        restart_requested_event: Optional[Any] = None,
        pause_controller: Optional[Any] = None,
        movement_manager: Optional[Any] = None,
    ):
        """Initialize the stream with a realtime handler and pipelines.

        - ``handler``/``robot`` may be ``None`` when constructing a settings-only
          LocalStream (e.g. ``--sim`` mode, which uses FastRTC for audio but still
          serves the static admin UI through ``init_admin_ui``). ``launch`` and
          the audio-pipeline methods require both to be provided.
        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        - ``app_stop_event``: optional ``threading.Event`` used by the admin restart endpoint to
          trigger the graceful shutdown path (the autostart service is expected to relaunch).
        - ``restart_requested_event``: optional ``threading.Event`` set by the admin
          restart endpoint in addition to ``app_stop_event``. ``main`` uses this to
          distinguish admin-requested restarts (exit 75 → systemd relaunches) from
          plain stops (exit 0).
        - ``pause_controller``: optional ``PauseController`` whose phrase lists the admin
          endpoints can read and hot-reload.
        - ``movement_manager``: optional ``MovementManager`` used by the admin
          ``/movement_speed`` endpoints (read/write the playback speed factor).
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        self._audio_source: Optional[_AudioSource] = None
        # Allow the handler to flush the player queue when appropriate.
        if self.handler is not None:
            self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._app_stop_event = app_stop_event
        self._restart_requested_event = restart_requested_event
        self._pause_controller = pause_controller
        self._movement_manager = movement_manager
        self._settings_initialized = False
        self._asyncio_loop = None
        # Snapshot of the active pipeline at handler-construction time.  Compared
        # against the live values in ``_status_payload`` to flag
        # ``requires_restart`` whenever the operator's saved selection no longer
        # matches what's running.  Pre-Phase-4f this was a single string
        # (``get_backend_choice()``); post-4f it's a pair so we catch composable-
        # vs-bundled flips that resolve to the same ``provider_id``.
        self._active_backend_name = get_provider_id()
        self._active_pipeline_mode = getattr(config, "PIPELINE_MODE", PIPELINE_MODE_COMPOSABLE)
        self._active_audio_output_backend = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_OPENAI_REALTIME)
        # Dedup state for the role=user_partial INFO log. Many STT backends
        # emit the same partial transcript repeatedly while waiting for the
        # next token; we demote identical consecutive partials to DEBUG so
        # the journal only shows INFO when the transcript text actually
        # changes. Reset implicitly whenever any non-user_partial row is
        # logged (e.g. the final role=user line that closes the utterance),
        # so the first partial of the next utterance always re-INFOs.
        self._last_logged_user_partial_text: Optional[str] = None

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

    def _has_required_key(self, provider_id: str) -> bool:
        """Return whether the configured pipeline has its required credential.

        ``provider_id`` is one of ``OPENAI_BACKEND``, ``GEMINI_BACKEND``,
        ``HF_BACKEND``, ``LOCAL_STT_BACKEND``. For the composable LOCAL_STT
        pipeline the answer depends on the currently configured
        ``AUDIO_OUTPUT_BACKEND`` + ``LLM_BACKEND`` (the latter only matters
        when the chosen TTS adapter needs its own LLM credentials, e.g.
        ElevenLabs paired with Gemini).
        """
        if provider_id == GEMINI_BACKEND:
            return self._has_key(config.GEMINI_API_KEY)
        if provider_id == HF_BACKEND:
            return has_hf_realtime_target()
        if provider_id == LOCAL_STT_BACKEND:
            audio_output = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_OPENAI_REALTIME)
            llm_backend = getattr(config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
            if audio_output == AUDIO_OUTPUT_HF:
                return has_hf_realtime_target()
            if audio_output == AUDIO_OUTPUT_GEMINI_TTS:
                return self._has_key(config.GEMINI_API_KEY)
            if audio_output == AUDIO_OUTPUT_ELEVENLABS:
                # ElevenLabs + Gemini LLM needs both ElevenLabs + Gemini keys;
                # ElevenLabs + llama only needs the ElevenLabs key.
                if llm_backend == LLM_BACKEND_GEMINI:
                    return self._has_key(config.ELEVENLABS_API_KEY) and self._has_key(config.GEMINI_API_KEY)
                return self._has_key(config.ELEVENLABS_API_KEY)
            if audio_output == AUDIO_OUTPUT_CHATTERBOX:
                return True  # no API key needed; server URL has a usable default
            return self._has_key(config.OPENAI_API_KEY)
        return self._has_key(config.OPENAI_API_KEY)

    @staticmethod
    def _requirement_name(provider_id: str) -> str:
        """Return the env var users need for a pipeline, if any."""
        if provider_id == GEMINI_BACKEND:
            return "GEMINI_API_KEY"
        if provider_id == HF_BACKEND:
            return HF_REALTIME_WS_URL_ENV
        if provider_id == LOCAL_STT_BACKEND:
            audio_output = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_OPENAI_REALTIME)
            if audio_output == AUDIO_OUTPUT_HF:
                return HF_REALTIME_WS_URL_ENV
            if audio_output == AUDIO_OUTPUT_GEMINI_TTS:
                return "GEMINI_API_KEY"
            if audio_output == AUDIO_OUTPUT_ELEVENLABS:
                return "ELEVENLABS_API_KEY"
            return "OPENAI_API_KEY"
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

    def _persist_elevenlabs_api_key(self, key: str) -> None:
        """Persist ELEVENLABS_API_KEY to environment and instance `.env`."""
        self._persist_env_value("ELEVENLABS_API_KEY", key)

    def _persist_elevenlabs_voice(self, voice: str) -> None:
        """Persist ELEVENLABS_VOICE override to environment and instance `.env`."""
        self._persist_env_value("ELEVENLABS_VOICE", voice)

    def _persist_local_stt_settings(
        self,
        *,
        cache_dir: str,
        language: str,
        model: str,
        update_interval: float,
        chatterbox_url: Optional[str] = None,
        chatterbox_voice: Optional[str] = None,
        llm_backend: Optional[str] = None,
    ) -> None:
        """Persist local STT settings to environment and instance `.env`.

        Moonshine-specific knobs only.  The audio output backend (and the
        provider's required credentials) are persisted separately by
        :meth:`_persist_pipeline_choice`.
        """
        values: dict[str, str] = {
            LOCAL_STT_PROVIDER_ENV: "moonshine",
            LOCAL_STT_CACHE_DIR_ENV: cache_dir,
            LOCAL_STT_LANGUAGE_ENV: language,
            LOCAL_STT_MODEL_ENV: model,
            LOCAL_STT_UPDATE_INTERVAL_ENV: f"{update_interval:.2f}",
        }
        if llm_backend is not None:
            values[LLM_BACKEND_ENV] = llm_backend
            config.LLM_BACKEND = llm_backend
        if chatterbox_url is not None:
            values[CHATTERBOX_URL_ENV] = chatterbox_url
        if chatterbox_voice is not None:
            values[CHATTERBOX_VOICE_ENV] = chatterbox_voice
        self._persist_env_values(values)

    def _persist_pipeline_choice(
        self,
        pipeline_mode: str,
        audio_input_backend: Optional[str],
        audio_output_backend: Optional[str],
    ) -> None:
        """Persist a pipeline selection into the instance `.env`.

        Writes ``REACHY_MINI_PIPELINE_MODE`` plus (for the composable pipeline)
        ``REACHY_MINI_AUDIO_INPUT_BACKEND`` / ``REACHY_MINI_AUDIO_OUTPUT_BACKEND``.
        Replaces the pre-Phase-4f ``_persist_backend_choice`` helper.
        """
        updates: dict[str, str] = {PIPELINE_MODE_ENV: pipeline_mode}
        if pipeline_mode == PIPELINE_MODE_COMPOSABLE:
            if audio_input_backend:
                updates[AUDIO_INPUT_BACKEND_ENV] = audio_input_backend
            if audio_output_backend:
                updates[AUDIO_OUTPUT_BACKEND_ENV] = audio_output_backend
        else:
            # Bundled modes ignore the audio dials.  Drop any leftover overrides
            # so the bundled handler isn't accidentally re-routed on next reload.
            self._remove_persisted_env_values((AUDIO_INPUT_BACKEND_ENV, AUDIO_OUTPUT_BACKEND_ENV))
            for env_name in (AUDIO_INPUT_BACKEND_ENV, AUDIO_OUTPUT_BACKEND_ENV):
                try:
                    os.environ.pop(env_name, None)
                except Exception:
                    pass
        self._persist_env_values(updates)

    def _persist_personality(self, profile: Optional[str], voice_override: Optional[str] = None) -> None:
        """Persist startup profile and voice in instance-local UI settings."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        normalized_voice_override = (voice_override or "").strip() or None
        try:
            from robot_comic.config import set_custom_profile

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

    def _crowd_history_path(self) -> Path:
        """Return the local crowd-work session directory for the running app.

        Prefers the instance-local path so the admin UI sees the same files the
        tools write. Falls back to a CWD-relative path when no instance_path is
        configured (dev/sim mode).
        """
        if self._instance_path is not None:
            return Path(self._instance_path) / CROWD_HISTORY_DIRNAME
        return Path(CROWD_HISTORY_DIRNAME)

    def _legacy_crowd_history_path(self) -> Path:
        """Return the pre-rename ``.rickles_sessions`` directory for back-compat."""
        if self._instance_path is not None:
            return Path(self._instance_path) / LEGACY_CROWD_HISTORY_DIRNAME
        return Path(LEGACY_CROWD_HISTORY_DIRNAME)

    def _crowd_history_files(self) -> list[Path]:
        """Return persisted crowd-work session files, newest first.

        Includes any leftover legacy ``.rickles_sessions`` files so renaming the
        directory didn't strand pre-existing history.
        """
        files: list[Path] = []
        for session_dir in (self._crowd_history_path(), self._legacy_crowd_history_path()):
            if session_dir.exists():
                files.extend(session_dir.glob("session_*.json"))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def _crowd_history_status(self) -> dict[str, object]:
        """Return admin-facing metadata for persisted crowd-work sessions."""
        session_dir = self._crowd_history_path()
        files = self._crowd_history_files()
        latest = files[0] if files else None
        return {
            "crowd_history_dir": str(session_dir.resolve()),
            "crowd_history_count": len(files),
            "crowd_history_latest": str(latest.resolve()) if latest else None,
        }

    def _clear_crowd_history(self) -> dict[str, object]:
        """Remove local crowd-work session files and return the new status."""
        removed = 0
        for path in self._crowd_history_files():
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
        return {"removed": removed, **self._crowd_history_status()}

    def init_admin_ui(self) -> None:
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
            # Phase 4f: ``backend`` / ``local_stt_response_backend`` are gone.
            # The picker now sends the canonical pipeline dials directly.
            pipeline_mode: str
            audio_input_backend: Optional[str] = None
            audio_output_backend: Optional[str] = None
            api_key: Optional[str] = None
            hf_mode: Optional[str] = None
            hf_host: Optional[str] = None
            hf_port: Optional[int] = None
            local_stt_cache_dir: Optional[str] = None
            local_stt_language: Optional[str] = None
            local_stt_model: Optional[str] = None
            local_stt_update_interval: Optional[float] = None
            chatterbox_url: Optional[str] = None
            chatterbox_voice: Optional[str] = None
            elevenlabs_api_key: Optional[str] = None
            elevenlabs_voice: Optional[str] = None
            # LLM axis from the 3-column pipeline picker (#245)
            llm_backend: Optional[str] = None

        def _status_payload() -> dict[str, object]:
            provider_id = get_provider_id()
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
            has_local_stt_key = self._has_required_key(LOCAL_STT_BACKEND)
            can_proceed_with_openai = has_openai_key
            can_proceed_with_gemini = has_gemini_key
            can_proceed_with_hf = has_hf_connection
            can_proceed_with_local_stt = has_local_stt_key
            can_proceed_with_chatterbox = True  # no API key required
            can_proceed = self._has_required_key(active_backend)
            current_pipeline_mode = getattr(config, "PIPELINE_MODE", PIPELINE_MODE_COMPOSABLE)
            current_audio_output = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_OPENAI_REALTIME)
            requires_restart = (
                provider_id != active_backend
                or current_pipeline_mode != self._active_pipeline_mode
                or current_audio_output != self._active_audio_output_backend
            )
            return {
                # Pipeline-mode-first contract (Phase 4f). The legacy
                # ``backend_provider`` / ``local_stt_response_backend`` fields
                # have been removed; clients now read ``pipeline_mode`` +
                # ``audio_input_backend`` + ``audio_output_backend`` directly.
                "active_backend": active_backend,
                "pipeline_mode": getattr(config, "PIPELINE_MODE", PIPELINE_MODE_COMPOSABLE),
                "pipeline_mode_choices": list(PIPELINE_MODE_CHOICES),
                "audio_input_backend": getattr(config, "AUDIO_INPUT_BACKEND", AUDIO_INPUT_MOONSHINE),
                "audio_input_backend_choices": list(AUDIO_INPUT_CHOICES),
                "audio_output_backend": getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_OPENAI_REALTIME),
                "audio_output_backend_choices": list(AUDIO_OUTPUT_CHOICES),
                "has_key": can_proceed,
                "has_openai_key": has_openai_key,
                "has_gemini_key": has_gemini_key,
                "has_hf_session_url": has_hf_session_url,
                "has_hf_ws_url": has_hf_ws_url,
                "has_hf_connection": has_hf_connection,
                "has_local_stt_key": has_local_stt_key,
                "hf_connection_mode": hf_connection_mode,
                "hf_direct_host": hf_direct_host,
                "hf_direct_port": hf_direct_port,
                "local_stt_provider": getattr(config, "LOCAL_STT_PROVIDER", "moonshine"),
                "llm_backend": getattr(config, "LLM_BACKEND", LLM_BACKEND_LLAMA),
                "local_stt_cache_dir": getattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice"),
                "local_stt_language": getattr(config, "LOCAL_STT_LANGUAGE", "en"),
                "local_stt_model": getattr(config, "LOCAL_STT_MODEL", "tiny_streaming"),
                "local_stt_update_interval": getattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35),
                "local_stt_model_choices": list(LOCAL_STT_MODEL_CHOICES),
                "chatterbox_url": getattr(config, "CHATTERBOX_URL", CHATTERBOX_DEFAULT_URL),
                "chatterbox_voice": getattr(config, "CHATTERBOX_VOICE", CHATTERBOX_DEFAULT_VOICE),
                "has_elevenlabs_key": self._has_key(config.ELEVENLABS_API_KEY),
                "elevenlabs_voice": getattr(config, "ELEVENLABS_VOICE", "") or os.environ.get("ELEVENLABS_VOICE", ""),
                "can_proceed": can_proceed,
                "can_proceed_with_openai": can_proceed_with_openai,
                "can_proceed_with_gemini": can_proceed_with_gemini,
                "can_proceed_with_hf": can_proceed_with_hf,
                "can_proceed_with_local_stt": can_proceed_with_local_stt,
                "can_proceed_with_chatterbox": can_proceed_with_chatterbox,
                "requires_restart": requires_restart,
                **self._crowd_history_status(),
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

        @self._settings_app.post("/crowd_history/clear")
        def _clear_history() -> JSONResponse:
            try:
                return JSONResponse({"ok": True, **self._clear_crowd_history()})
            except Exception as e:
                logger.warning("Failed to clear crowd history: %s", e)
                return JSONResponse({"ok": False, "error": "clear_failed"}, status_code=500)

        # GET /elevenlabs/voices -> fetch the live ElevenLabs voice catalog for the
        # admin UI dropdown. Requires ELEVENLABS_API_KEY; falls back to a 500 with
        # the error string when the API call fails so the UI can show a placeholder.
        @self._settings_app.get("/elevenlabs/voices")
        async def _elevenlabs_voices() -> JSONResponse:
            try:
                from robot_comic.elevenlabs_voices import fetch_elevenlabs_voices

                voices = await fetch_elevenlabs_voices()
                return JSONResponse(
                    {
                        "ok": True,
                        "voices": [{"name": n, "voice_id": v} for n, v in voices.items()],
                    }
                )
            except Exception as e:
                logger.warning("Failed to fetch ElevenLabs voices: %s", e)
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

        # GET /api/voices/catalog -> richer ElevenLabs voice catalog (name +
        # voice_id + category) for the admin UI table (#304). Returns 503 when
        # ElevenLabs is not configured so the UI can render a "not configured"
        # placeholder instead of treating it as a transient error.
        @self._settings_app.get("/api/voices/catalog")
        async def _voice_catalog() -> JSONResponse:
            if not config.ELEVENLABS_API_KEY:
                return JSONResponse(
                    {"error": "elevenlabs not configured"},
                    status_code=503,
                )
            try:
                from robot_comic.elevenlabs_voices import (
                    fetch_elevenlabs_voices,
                    get_elevenlabs_voice_records,
                )

                # Populate the cache on first call (no-op if already cached).
                await fetch_elevenlabs_voices()
                records = get_elevenlabs_voice_records()
                return JSONResponse(
                    {
                        "voices": [
                            {
                                "voice_id": rec["voice_id"],
                                "name": rec["name"],
                                "category": rec.get("category", ""),
                            }
                            for rec in records
                        ],
                    }
                )
            except Exception as e:
                logger.warning("Failed to fetch ElevenLabs voice catalog: %s", e)
                return JSONResponse(
                    {"error": f"catalog fetch failed: {e}"},
                    status_code=500,
                )

        # GET /api/xtts/voices -> fetch the xtts-v2 LAN speaker list for the
        # admin UI dropdown (#438). Constructs a throw-away XttsTTSAdapter from
        # the current config, calls get_available_voices(), then shuts it down.
        # Falls back gracefully when the xtts service is unreachable.
        @self._settings_app.get("/api/xtts/voices")
        async def _xtts_voices() -> JSONResponse:
            from robot_comic.adapters.xtts_tts_adapter import XttsTTSAdapter

            adapter = XttsTTSAdapter(
                base_url=config.XTTS_URL,
                default_speaker=config.XTTS_DEFAULT_SPEAKER_KEY,
                language=config.XTTS_LANGUAGE,
                timeout_s=config.XTTS_TIMEOUT_S,
            )
            try:
                voices = await adapter.get_available_voices()
                return JSONResponse({"ok": True, "voices": voices})
            except Exception as e:
                logger.warning("Failed to fetch xtts voices: %s", e)
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
            finally:
                await adapter.shutdown()

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("robot_comic.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # GET /api/battery -> Reachy battery status (cached 5 s)
        @self._settings_app.get("/api/battery")
        def _battery() -> JSONResponse:
            return JSONResponse(_read_battery(self._robot))

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
            # Phase 4f: the picker sends ``pipeline_mode`` +
            # ``audio_input_backend`` + ``audio_output_backend`` directly.
            pipeline_mode = (payload.pipeline_mode or "").strip().lower()
            if pipeline_mode not in PIPELINE_MODE_CHOICES:
                return JSONResponse({"ok": False, "error": "invalid_pipeline_mode"}, status_code=400)

            # For bundled modes the audio dials are ignored.  For composable
            # we need a valid (input, output) pair.
            audio_input_backend: Optional[str] = None
            audio_output_backend: Optional[str] = None
            if pipeline_mode == PIPELINE_MODE_COMPOSABLE:
                audio_input_backend = (payload.audio_input_backend or "").strip().lower() or AUDIO_INPUT_MOONSHINE
                audio_output_backend = (payload.audio_output_backend or "").strip().lower()
                if not audio_output_backend:
                    audio_output_backend = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_OPENAI_REALTIME)
                if audio_input_backend not in AUDIO_INPUT_CHOICES:
                    return JSONResponse({"ok": False, "error": "invalid_audio_input_backend"}, status_code=400)
                if audio_output_backend not in AUDIO_OUTPUT_CHOICES:
                    return JSONResponse({"ok": False, "error": "invalid_audio_output_backend"}, status_code=400)

            # Effective provider_id for credential routing.
            provider_id = provider_id_from_pipeline(pipeline_mode, audio_output_backend or "")

            api_key = (payload.api_key or "").strip()
            if provider_id == GEMINI_BACKEND and not api_key and not self._has_key(config.GEMINI_API_KEY):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)

            # LLM backend axis from the 3-column pipeline picker (#245).
            # Defaults to "llama" for back-compat when field is absent.
            _llm_backend_raw = (payload.llm_backend or "").strip().lower() or LLM_BACKEND_LLAMA
            if pipeline_mode == PIPELINE_MODE_COMPOSABLE and _llm_backend_raw not in {
                LLM_BACKEND_LLAMA,
                LLM_BACKEND_GEMINI,
            }:
                return JSONResponse({"ok": False, "error": "invalid_llm_backend"}, status_code=400)

            # Validate (output, llm) combination on the composable path.
            # Bundled output adapters (openai_realtime_output / hf_output) and
            # gemini_tts ship their own LLM and cannot be paired with the
            # gemini text-LLM axis.
            _GEMINI_LLM_UNSUPPORTED_OUTPUTS = {
                AUDIO_OUTPUT_GEMINI_TTS,
                AUDIO_OUTPUT_OPENAI_REALTIME,
                AUDIO_OUTPUT_HF,
            }
            if (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE
                and _llm_backend_raw == LLM_BACKEND_GEMINI
                and audio_output_backend in _GEMINI_LLM_UNSUPPORTED_OUTPUTS
            ):
                return JSONResponse(
                    {"ok": False, "error": "unsupported_pipeline_combination"},
                    status_code=400,
                )

            if (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE
                and audio_output_backend == AUDIO_OUTPUT_OPENAI_REALTIME
                and not api_key
                and not self._has_key(config.OPENAI_API_KEY)
            ):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            if (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE
                and audio_output_backend == AUDIO_OUTPUT_GEMINI_TTS
                and not api_key
                and not self._has_key(config.GEMINI_API_KEY)
            ):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            # Chatterbox needs no API key — always allowed

            elevenlabs_key_val = (payload.elevenlabs_api_key or "").strip()
            elevenlabs_voice_val = (payload.elevenlabs_voice or "").strip()
            if (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE
                and audio_output_backend == AUDIO_OUTPUT_ELEVENLABS
                and not elevenlabs_key_val
                and not self._has_key(config.ELEVENLABS_API_KEY)
            ):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)

            if pipeline_mode == PIPELINE_MODE_OPENAI_REALTIME and api_key:
                self._persist_api_key(api_key)
            if (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE
                and audio_output_backend == AUDIO_OUTPUT_OPENAI_REALTIME
                and api_key
            ):
                self._persist_api_key(api_key)
            if pipeline_mode == PIPELINE_MODE_GEMINI_LIVE and api_key:
                self._persist_gemini_api_key(api_key)
            if (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE
                and audio_output_backend == AUDIO_OUTPUT_GEMINI_TTS
                and api_key
            ):
                self._persist_gemini_api_key(api_key)
            if pipeline_mode == PIPELINE_MODE_COMPOSABLE and audio_output_backend == AUDIO_OUTPUT_ELEVENLABS:
                if elevenlabs_key_val:
                    self._persist_elevenlabs_api_key(elevenlabs_key_val)
                if elevenlabs_voice_val:
                    self._persist_elevenlabs_voice(elevenlabs_voice_val)
            if pipeline_mode == PIPELINE_MODE_COMPOSABLE and audio_input_backend == AUDIO_INPUT_MOONSHINE:
                cache_dir = (
                    payload.local_stt_cache_dir
                    or getattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")
                    or "./cache/moonshine_voice"
                ).strip()
                if not cache_dir:
                    return JSONResponse({"ok": False, "error": "invalid_local_stt_cache_dir"}, status_code=400)

                language = (payload.local_stt_language or getattr(config, "LOCAL_STT_LANGUAGE", "en") or "en").strip()
                if not language or "/" in language or "\\" in language:
                    return JSONResponse({"ok": False, "error": "invalid_local_stt_language"}, status_code=400)

                model = (
                    (
                        payload.local_stt_model
                        or getattr(config, "LOCAL_STT_MODEL", "tiny_streaming")
                        or "tiny_streaming"
                    )
                    .strip()
                    .lower()
                    .replace("-", "_")
                )
                if model not in LOCAL_STT_MODEL_CHOICES:
                    return JSONResponse({"ok": False, "error": "invalid_local_stt_model"}, status_code=400)

                update_interval = (
                    payload.local_stt_update_interval
                    if payload.local_stt_update_interval is not None
                    else float(getattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35))
                )
                if update_interval < 0.1 or update_interval > 2.0:
                    return JSONResponse({"ok": False, "error": "invalid_local_stt_update_interval"}, status_code=400)

                chatterbox_url_val = (payload.chatterbox_url or "").strip() or None
                chatterbox_voice_val = (payload.chatterbox_voice or "").strip() or None
                self._persist_local_stt_settings(
                    cache_dir=cache_dir,
                    language=language.lower(),
                    model=model,
                    update_interval=update_interval,
                    chatterbox_url=chatterbox_url_val if audio_output_backend == AUDIO_OUTPUT_CHATTERBOX else None,
                    chatterbox_voice=chatterbox_voice_val if audio_output_backend == AUDIO_OUTPUT_CHATTERBOX else None,
                    llm_backend=_llm_backend_raw,
                )
            if pipeline_mode == PIPELINE_MODE_HF_REALTIME or (
                pipeline_mode == PIPELINE_MODE_COMPOSABLE and audio_output_backend == AUDIO_OUTPUT_HF
            ):
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

            self._persist_pipeline_choice(pipeline_mode, audio_input_backend, audio_output_backend)
            payload_data = _status_payload()
            message = "Pipeline saved."
            if payload_data["requires_restart"]:
                message = "Pipeline saved. Restart Robot Comic from the desktop app to apply it."
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

        def _pause_phrases_payload(settings: PausePhraseSettings) -> dict[str, object]:
            return {
                "saved": {
                    "stop": list(settings.stop) if settings.stop is not None else None,
                    "resume": list(settings.resume) if settings.resume is not None else None,
                    "shutdown": list(settings.shutdown) if settings.shutdown is not None else None,
                    "switch": list(settings.switch) if settings.switch is not None else None,
                },
                "effective": {
                    "stop": list(settings.resolved_stop()),
                    "resume": list(settings.resolved_resume()),
                    "shutdown": list(settings.resolved_shutdown()),
                    "switch": list(settings.resolved_switch()),
                },
            }

        @self._settings_app.get("/pause_phrases")
        def _get_pause_phrases() -> JSONResponse:
            settings = read_pause_settings(self._instance_path)
            return JSONResponse({"ok": True, **_pause_phrases_payload(settings)})

        @self._settings_app.post("/pause_phrases")
        def _set_pause_phrases(payload: dict[str, object]) -> JSONResponse:
            new_settings = settings_from_payload(payload)
            persisted = write_pause_settings(self._instance_path, new_settings)
            applied_live = False
            if self._pause_controller is not None:
                try:
                    self._pause_controller.update_phrases(
                        stop=persisted.resolved_stop(),
                        resume=persisted.resolved_resume(),
                        shutdown=persisted.resolved_shutdown(),
                        switch=persisted.resolved_switch(),
                    )
                    applied_live = True
                except Exception as e:
                    logger.warning("Failed to apply pause phrases to running controller: %s", e)
            return JSONResponse(
                {
                    "ok": True,
                    "applied_live": applied_live,
                    **_pause_phrases_payload(persisted),
                }
            )

        @self._settings_app.post("/admin/restart")
        def _restart_app() -> JSONResponse:
            if self._app_stop_event is None:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "no_stop_event",
                        "message": "Restart hook is unavailable. Stop and start Robot Comic manually to apply changes.",
                    },
                    status_code=503,
                )
            logger.info("Admin requested restart — setting app_stop_event")
            try:
                # Signal restart intent BEFORE the stop event so main can see it
                # in time to exit with the restart sentinel code.
                if self._restart_requested_event is not None:
                    self._restart_requested_event.set()
                self._app_stop_event.set()
            except Exception as e:
                logger.error("Failed to set app_stop_event: %s", e)
                return JSONResponse({"ok": False, "error": "stop_event_failed"}, status_code=500)
            return JSONResponse(
                {
                    "ok": True,
                    "message": (
                        "Robot Comic is shutting down gracefully. The autostart service will "
                        "relaunch it; if you are running manually, restart the process."
                    ),
                }
            )

        @self._settings_app.get("/movement_speed")
        def _get_movement_speed() -> JSONResponse:
            if self._movement_manager is None:
                return JSONResponse({"ok": False, "error": "no_movement_manager"}, status_code=503)
            value = float(getattr(self._movement_manager, "speed_factor", 1.0))
            return JSONResponse({"ok": True, "value": value, "min": 0.1, "max": 2.0, "step": 0.05})

        @self._settings_app.post("/movement_speed")
        def _set_movement_speed(payload: dict[str, object]) -> JSONResponse:
            if self._movement_manager is None:
                return JSONResponse({"ok": False, "error": "no_movement_manager"}, status_code=503)
            raw = payload.get("value")
            try:
                value = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return JSONResponse({"ok": False, "error": "invalid_value"}, status_code=400)
            try:
                self._movement_manager.set_speed_factor(value)
            except Exception as e:
                logger.warning("set_speed_factor(%r) failed: %s", value, e)
                return JSONResponse({"ok": False, "error": "set_failed"}, status_code=500)
            persisted_value = float(self._movement_manager.speed_factor)
            if self._instance_path:
                try:
                    write_startup_settings(self._instance_path, movement_speed=persisted_value)
                except Exception as e:
                    logger.warning("Failed to persist movement_speed=%r: %s", persisted_value, e)
            return JSONResponse({"ok": True, "value": persisted_value})

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
        self.init_admin_ui()

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

        # launch() requires both handler and robot to be non-None
        assert self._robot is not None, "launch() requires a robot instance"
        assert self.handler is not None, "launch() requires a handler instance"
        _robot = self._robot
        _handler = self.handler

        # Start media after key is set/available
        _robot.media.start_recording()
        _robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start
        apply_audio_startup_config(_robot, logger=logger)

        # Build + start the audio source (daemon shim or direct ALSA RW).
        # If AUDIO_CAPTURE_PATH=alsa_rw on Linux, this spawns an arecord
        # subprocess alongside the daemon's still-running MMAP capture.
        self._audio_source = self._build_audio_source()
        if self._audio_source is not None:
            try:
                self._audio_source.start()
                logger.info(
                    "Audio source ready: %s (path=%s)",
                    type(self._audio_source).__name__,
                    config.AUDIO_CAPTURE_PATH,
                )
            except Exception as e:
                logger.error(
                    "Failed to start audio source %s: %s — falling back to daemon path",
                    type(self._audio_source).__name__,
                    e,
                )
                self._audio_source = _DaemonAudioSource(_robot)
                self._audio_source.start()

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]

            # Fetch ElevenLabs voice catalog at startup (one-time, cached for process lifetime)
            try:
                from robot_comic import config

                await config.refresh_elevenlabs_voices()
            except Exception as e:
                logger.debug("Failed to refresh ElevenLabs voice catalog: %s", e)

            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        _handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                pass
            from robot_comic.startup_timer import log_checkpoint

            async def _start_up_with_checkpoints() -> None:
                # The ``handler.start_up.complete`` supporting event is now
                # emitted by the handler itself the moment it becomes ready
                # to accept audio (#337). Emitting it here was wrong because
                # ``await _handler.start_up()`` blocks for the handler's full
                # lifetime, so the row only landed at shutdown.
                log_checkpoint("handler.start_up begin", logger)
                try:
                    await _handler.start_up()
                except Exception:
                    log_checkpoint("handler.start_up end", logger)
                    raise
                else:
                    log_checkpoint("handler.start_up end", logger)

            _startup_task = asyncio.create_task(_start_up_with_checkpoints(), name="openai-handler")
            # Emit a dispatched checkpoint immediately so the startup timeline
            # records when the handler session was handed off to its task, not
            # when the (long-running) session eventually tears down.
            log_checkpoint("handler.start_up dispatched", logger)

            self._tasks = [
                _startup_task,
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await _handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop our audio source (terminates arecord subprocess if active).
        try:
            if self._audio_source is not None:
                self._audio_source.stop()
                self._audio_source = None
        except Exception as e:
            logger.debug(f"Error stopping audio source (may already be stopped): {e}")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            if self._robot is not None:
                self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            if self._robot is not None:
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
        if self._robot is None or self.handler is None:
            return
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

    def _build_audio_source(self) -> Optional[_AudioSource]:
        """Select the audio source per config.AUDIO_CAPTURE_PATH.

        Returns None when self._robot is None (sim mode) — record_loop
        never runs in that case, but __init__ may still call this to set
        the field.
        """
        if self._robot is None:
            return None
        path = getattr(config, "AUDIO_CAPTURE_PATH", "daemon")
        if path == AUDIO_CAPTURE_PATH_ALSA_RW:
            from robot_comic.audio_input import AlsaRwCapture

            return AlsaRwCapture()
        return _DaemonAudioSource(self._robot)

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        assert self._robot is not None and self.handler is not None
        assert self._audio_source is not None, "record_loop requires _audio_source — launch() must have populated it"
        input_sample_rate = self._audio_source.sample_rate
        logger.debug(
            "Audio recording started at %d Hz via %s",
            input_sample_rate,
            type(self._audio_source).__name__,
        )

        while not self._stop_event.is_set():
            # Snapshot the source per-iteration: close() nulls _audio_source
            # before setting _stop_event, so a naked attribute read can race
            # to AttributeError on shutdown.
            source = self._audio_source
            if source is None:
                break
            audio_frame = source.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    def _log_handler_message(self, role: Any, content: str) -> None:
        """Log one handler text message, deduping repeated user_partial lines.

        STT backends emit ``role=user_partial`` once per intermediate
        transcript update. While the recognizer is settling on a token, the
        same partial text can repeat many times in a row — flooding the
        journal with identical INFO lines. Demote consecutive duplicates to
        DEBUG; emit INFO only when the partial text actually changes.

        The tracker is reset whenever a non-``user_partial`` row is logged
        (e.g. the final ``role=user`` line that closes the utterance), so
        the first partial of the next utterance is always INFO regardless
        of the prior utterance's last text.
        """
        rendered = content if len(content) < 500 else content[:500] + "…"
        if role == "user_partial":
            if rendered == self._last_logged_user_partial_text:
                logger.debug("role=%s content=%s", role, rendered)
                return
            self._last_logged_user_partial_text = rendered
        else:
            self._last_logged_user_partial_text = None
        logger.info("role=%s content=%s", role, rendered)

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        from fastrtc import AdditionalOutputs, audio_to_float32  # deferred: fastrtc pulls gradio at boot
        from scipy.signal import resample

        assert self._robot is not None and self.handler is not None
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        self._log_handler_message(msg.get("role"), content)

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
