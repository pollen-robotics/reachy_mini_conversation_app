import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlsplit, parse_qsl, urlunsplit
from importlib.resources import files

from dotenv import find_dotenv, load_dotenv


# Locked profile: set to a profile name (e.g., "astronomer") to lock the app
# to that profile and disable all profile switching. Leave as None for normal behavior.
LOCKED_PROFILE: str | None = None
PROJECT_ROOT = Path(__file__).parents[2].resolve()


def _is_source_checkout_root(root: Path) -> bool:
    """Return whether the given root looks like this project's source checkout."""
    return (root / "pyproject.toml").is_file() and (root / "src" / "robot_comic").is_dir()


def _packaged_profiles_directory() -> Path | None:
    """Return the installed wheel's packaged profiles directory when available."""
    try:
        return Path(str(files("reachy_talk_data").joinpath("profiles")))
    except Exception:
        return None


def _resolve_default_profiles_directory() -> Path:
    """Resolve built-in profiles from source checkout or installed package data."""
    source_profiles = PROJECT_ROOT / "profiles"
    if _is_source_checkout_root(PROJECT_ROOT) and source_profiles.is_dir():
        return source_profiles

    packaged_profiles = _packaged_profiles_directory()
    if packaged_profiles is not None and packaged_profiles.is_dir():
        return packaged_profiles

    return source_profiles


DEFAULT_PROFILES_DIRECTORY = _resolve_default_profiles_directory()

# Full list of voices supported by the OpenAI Realtime / TTS API.
# Source: https://developers.openai.com/api/docs/guides/text-to-speech/#voice-options
# "marin" and "cedar" are recommended for gpt-realtime.
AVAILABLE_VOICES: list[str] = [
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]
OPENAI_DEFAULT_VOICE = "cedar"

# Qwen3-TTS CustomVoice speaker catalog from the deployed Hugging Face backend.
HF_AVAILABLE_VOICES: list[str] = [
    "Aiden",
    "Ryan",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Serena",
    "Sohee",
    "Uncle_Fu",
    "Vivian",
]

# Voices supported by the Gemini Live API
GEMINI_AVAILABLE_VOICES: list[str] = [
    "Aoede",
    "Charon",
    "Fenrir",
    "Kore",
    "Leda",
    "Orus",
    "Puck",
    "Zephyr",
]

# Voices supported by the Gemini TTS API (gemini-3.1-flash-tts-preview)
GEMINI_TTS_AVAILABLE_VOICES: list[str] = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]
GEMINI_TTS_DEFAULT_VOICE = "Algenib"

# Voices supported by the ElevenLabs TTS API (populated at startup from /v1/voices API)
# Fallback list below is used until fetch_elevenlabs_voices_async() is called
ELEVENLABS_AVAILABLE_VOICES: list[str] = [
    "Adam",
    "Bella",
    "Antoni",
    "Domi",
    "Elli",
    "Gigi",
    "Freya",
    "Harry",
    "Liam",
    "Rachel",
    "River",
    "Sam",
]
ELEVENLABS_DEFAULT_VOICE = "Adam"
ELEVENLABS_API_KEY_ENV = "ELEVENLABS_API_KEY"
ELEVENLABS_VOICE_ENV = "ELEVENLABS_VOICE"
ELEVENLABS_OUTPUT = "elevenlabs"

OPENAI_BACKEND = "openai"
GEMINI_BACKEND = "gemini"
HF_BACKEND = "huggingface"
LOCAL_STT_BACKEND = "local_stt"
GEMINI_TTS_OUTPUT = "gemini_tts"
HF_REALTIME_CONNECTION_MODE_ENV = "HF_REALTIME_CONNECTION_MODE"
HF_REALTIME_WS_URL_ENV = "HF_REALTIME_WS_URL"
HF_LOCAL_CONNECTION_MODE = "local"
HF_DEPLOYED_CONNECTION_MODE = "deployed"
HF_REALTIME_SESSION_PROXY_URL = "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session"


@dataclass(frozen=True)
class HFBackendDefaults:
    """Defaults for the Hugging Face realtime backend."""

    connection_mode: str = HF_DEPLOYED_CONNECTION_MODE
    # App-managed Hugging Face Space proxy. The Space forwards to the current
    # session allocator, so allocator changes do not require app releases.
    # Users who need a custom target should use HF_REALTIME_CONNECTION_MODE=local
    # with HF_REALTIME_WS_URL.
    session_url: str = HF_REALTIME_SESSION_PROXY_URL
    voice: str = "Aiden"
    model_name: str = ""
    direct_port: int = 8765


HF_DEFAULTS = HFBackendDefaults()
DEFAULT_MODEL_NAME_BY_BACKEND = {
    OPENAI_BACKEND: "gpt-realtime",
    GEMINI_BACKEND: "gemini-3.1-flash-live-preview",
    HF_BACKEND: HF_DEFAULTS.model_name,
    LOCAL_STT_BACKEND: "moonshine",
}
BACKEND_LABEL_BY_PROVIDER = {
    OPENAI_BACKEND: "OpenAI Realtime",
    GEMINI_BACKEND: "Gemini Live",
    HF_BACKEND: "Hugging Face",
    LOCAL_STT_BACKEND: "Local STT",
}
CHATTERBOX_OUTPUT = "chatterbox"
CHATTERBOX_URL_ENV = "CHATTERBOX_URL"
CHATTERBOX_VOICE_ENV = "CHATTERBOX_VOICE"
CHATTERBOX_DEFAULT_URL = "http://astralplane.lan:8004"
CHATTERBOX_DEFAULT_VOICE = "don_rickles"
CHATTERBOX_DEFAULT_EXAGGERATION = 0.75
CHATTERBOX_DEFAULT_CFG_WEIGHT = 0.30
CHATTERBOX_DEFAULT_TEMPERATURE = 0.6
CHATTERBOX_DEFAULT_GAIN = 2.0
CHATTERBOX_AUTO_GAIN_ENABLED_ENV = "REACHY_MINI_CHATTERBOX_AUTO_GAIN_ENABLED"
CHATTERBOX_TARGET_DBFS_ENV = "REACHY_MINI_CHATTERBOX_TARGET_DBFS"
CHATTERBOX_DEFAULT_TARGET_DBFS = -16.0

LLAMA_CPP_URL_ENV = "LLAMA_CPP_URL"
LLAMA_CPP_DEFAULT_URL = "http://astralplane.lan:8080"
LLAMA_HEALTH_CHECK_ENV = "REACHY_MINI_LLAMA_HEALTH_CHECK"
LANGUAGE_DISSECT_LLM_FALLBACK_ENV = "REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK"

# Wake-on-LAN config env-var names and defaults.
WOL_MAC_ENV = "REACHY_MINI_WOL_MAC"
WOL_BROADCAST_ENV = "REACHY_MINI_WOL_BROADCAST"
WOL_RETRY_AFTER_ENV = "REACHY_MINI_WOL_RETRY_AFTER_S"
WOL_DEFAULT_BROADCAST = "255.255.255.255"
WOL_DEFAULT_RETRY_AFTER_S = 3.0

LLAMA_GEMINI_TTS_OUTPUT = "llama_gemini_tts"
LLAMA_ELEVENLABS_TTS_OUTPUT = "llama_elevenlabs_tts"

# ---------------------------------------------------------------------------
# LLM backend selection (orthogonal to the audio input/output axis).
# ---------------------------------------------------------------------------

# Env-var that selects which text-LLM powers the local-STT pipelines.
LLM_BACKEND_ENV = "REACHY_MINI_LLM_BACKEND"
# Allowed values for LLM_BACKEND
LLM_BACKEND_LLAMA = "llama"
LLM_BACKEND_GEMINI = "gemini"

# Env-var for the Gemini model to use when LLM_BACKEND=gemini.
GEMINI_LLM_MODEL_ENV = "REACHY_MINI_GEMINI_LLM_MODEL"
_GEMINI_LLM_MODEL_DEFAULT = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Modular audio pipeline: separate input (STT) and output (TTS) backend IDs.
# These are the canonical string values for AUDIO_INPUT_BACKEND and
# AUDIO_OUTPUT_BACKEND, set explicitly via environment variables.
# ---------------------------------------------------------------------------

# Input (STT) backend identifiers
AUDIO_INPUT_MOONSHINE = "moonshine"
AUDIO_INPUT_FASTER_WHISPER = "faster_whisper"
AUDIO_INPUT_OPENAI_REALTIME = "openai_realtime_input"
AUDIO_INPUT_GEMINI_LIVE = "gemini_live_input"
AUDIO_INPUT_HF = "hf_input"

AUDIO_INPUT_CHOICES: tuple[str, ...] = (
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_FASTER_WHISPER,
    AUDIO_INPUT_OPENAI_REALTIME,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_INPUT_HF,
)

# Output (TTS) backend identifiers
AUDIO_OUTPUT_CHATTERBOX = "chatterbox"
AUDIO_OUTPUT_GEMINI_TTS = "gemini_tts"
AUDIO_OUTPUT_ELEVENLABS = "elevenlabs"
AUDIO_OUTPUT_OPENAI_REALTIME = "openai_realtime_output"
AUDIO_OUTPUT_GEMINI_LIVE = "gemini_live_output"
AUDIO_OUTPUT_HF = "hf_output"

AUDIO_OUTPUT_CHOICES: tuple[str, ...] = (
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_HF,
)

# Env-var names for explicit overrides
AUDIO_INPUT_BACKEND_ENV = "REACHY_MINI_AUDIO_INPUT_BACKEND"
AUDIO_OUTPUT_BACKEND_ENV = "REACHY_MINI_AUDIO_OUTPUT_BACKEND"

# ---------------------------------------------------------------------------
# Audio capture path: bypass ALSA MMAP attenuation on reachymini_audio_src
# by spawning our own arecord subprocess in RW mode. See
# docs/superpowers/specs/2026-05-15-stream-a-alsa-rw-capture-design.md.
# ---------------------------------------------------------------------------

AUDIO_CAPTURE_PATH_ENV = "REACHY_MINI_AUDIO_CAPTURE_PATH"
AUDIO_CAPTURE_PATH_DAEMON = "daemon"
AUDIO_CAPTURE_PATH_ALSA_RW = "alsa_rw"
AUDIO_CAPTURE_PATH_CHOICES: tuple[str, ...] = (
    AUDIO_CAPTURE_PATH_DAEMON,
    AUDIO_CAPTURE_PATH_ALSA_RW,
)


def _resolve_audio_capture_path() -> str:
    """Return the AUDIO_CAPTURE_PATH for this process.

    Default is alsa_rw on Linux (the only platform with arecord +
    reachymini_audio_src) and daemon elsewhere. The env var overrides either
    direction; invalid values log a warning and fall back to the platform
    default.
    """
    platform_default = AUDIO_CAPTURE_PATH_ALSA_RW if sys.platform == "linux" else AUDIO_CAPTURE_PATH_DAEMON
    raw = (os.getenv(AUDIO_CAPTURE_PATH_ENV) or "").strip().lower()
    if not raw:
        return platform_default
    if raw in AUDIO_CAPTURE_PATH_CHOICES:
        return raw
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Using platform default %r.",
        AUDIO_CAPTURE_PATH_ENV,
        raw,
        ", ".join(AUDIO_CAPTURE_PATH_CHOICES),
        platform_default,
    )
    return platform_default


# ---------------------------------------------------------------------------
# 4th config dial: pipeline mode (composable vs bundled-realtime).
# The STT/LLM/TTS dials only describe a meaningful pipeline when we're in
# composable mode; bundled "speech-to-speech" backends (OpenAI Realtime,
# Gemini Live, HF Realtime) fuse all three phases into one websocket session
# and ignore the other dials. PIPELINE_MODE makes that choice explicit
# instead of inferring it from the (input, output) pair.
# ---------------------------------------------------------------------------

PIPELINE_MODE_ENV = "REACHY_MINI_PIPELINE_MODE"
PIPELINE_MODE_COMPOSABLE = "composable"
PIPELINE_MODE_OPENAI_REALTIME = "openai_realtime"
PIPELINE_MODE_GEMINI_LIVE = "gemini_live"
PIPELINE_MODE_HF_REALTIME = "hf_realtime"

PIPELINE_MODE_CHOICES: tuple[str, ...] = (
    PIPELINE_MODE_COMPOSABLE,
    PIPELINE_MODE_OPENAI_REALTIME,
    PIPELINE_MODE_GEMINI_LIVE,
    PIPELINE_MODE_HF_REALTIME,
)


def derive_pipeline_mode(input_backend: str, output_backend: str) -> str:
    """Return the implied pipeline mode for a resolved (input, output) pair.

    Bundled realtime backends fuse STT/LLM/TTS in one session — they're
    identified by their input/output strings matching the canonical bundled
    pair. Everything else is a composable 3-phase pipeline.

    This is the backwards-compatibility bridge: when ``REACHY_MINI_PIPELINE_MODE``
    is unset, ``refresh_runtime_config_from_env`` calls this function with the
    resolved input/output so existing ``.env`` deployments keep working without
    operator action.
    """
    if input_backend == AUDIO_INPUT_OPENAI_REALTIME and output_backend == AUDIO_OUTPUT_OPENAI_REALTIME:
        return PIPELINE_MODE_OPENAI_REALTIME
    if input_backend == AUDIO_INPUT_GEMINI_LIVE and output_backend == AUDIO_OUTPUT_GEMINI_LIVE:
        return PIPELINE_MODE_GEMINI_LIVE
    if input_backend == AUDIO_INPUT_HF and output_backend == AUDIO_OUTPUT_HF:
        return PIPELINE_MODE_HF_REALTIME
    return PIPELINE_MODE_COMPOSABLE


def _normalize_pipeline_mode(value: str | None) -> str | None:
    """Validate an explicit ``REACHY_MINI_PIPELINE_MODE`` value.

    Returns the normalised string, or None if the value was absent/empty/invalid
    (in which case ``refresh_runtime_config_from_env`` falls back to
    :func:`derive_pipeline_mode`). Logs a warning on invalid values.
    """
    candidate = (value or "").strip().lower()
    if not candidate:
        return None
    if candidate in PIPELINE_MODE_CHOICES:
        return candidate
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Falling back to (input,output)-derived value.",
        PIPELINE_MODE_ENV,
        value,
        ", ".join(PIPELINE_MODE_CHOICES),
    )
    return None


# Supported (bundled) combinations — the only ones that map cleanly to an
# existing handler class.  Any combo not in this set is aspirational and will
# trigger a warning + fallback.
_SUPPORTED_AUDIO_COMBINATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # Bundled hf_realtime
        (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF),
        # Bundled openai_realtime
        (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME),
        # Bundled gemini_live
        (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE),
        # Composable moonshine STT + various TTS outputs
        (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX),
        (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_GEMINI_TTS),
        (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_ELEVENLABS),
        (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_OPENAI_REALTIME),
        (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_HF),
        # Composable faster-whisper STT + composable TTS outputs (Phase 5f).
        # No realtime-output hybrids — those still use LocalSTTInputMixin
        # (Phase 4c-tris Option B "legacy forever").
        (AUDIO_INPUT_FASTER_WHISPER, AUDIO_OUTPUT_CHATTERBOX),
        (AUDIO_INPUT_FASTER_WHISPER, AUDIO_OUTPUT_GEMINI_TTS),
        (AUDIO_INPUT_FASTER_WHISPER, AUDIO_OUTPUT_ELEVENLABS),
    }
)

_LOCAL_STT_OUTPUT_BASE_LABELS: dict[str, str] = {
    "openai_realtime_output": "Local STT + OpenAI voice",
    "hf_output": "Local STT + Hugging Face voice",
    "gemini_tts": "Local STT + Gemini TTS",
    "chatterbox": "Local STT + Chatterbox TTS",
    "elevenlabs": "Local STT + ElevenLabs TTS",
}
DEFAULT_VOICE_BY_BACKEND = {
    OPENAI_BACKEND: OPENAI_DEFAULT_VOICE,
    GEMINI_BACKEND: "Kore",
    HF_BACKEND: HF_DEFAULTS.voice,
    LOCAL_STT_BACKEND: OPENAI_DEFAULT_VOICE,
}

LOCAL_STT_PROVIDER_ENV = "LOCAL_STT_PROVIDER"
LOCAL_STT_CACHE_DIR_ENV = "LOCAL_STT_CACHE_DIR"
LOCAL_STT_LANGUAGE_ENV = "LOCAL_STT_LANGUAGE"
LOCAL_STT_MODEL_ENV = "LOCAL_STT_MODEL"
LOCAL_STT_UPDATE_INTERVAL_ENV = "LOCAL_STT_UPDATE_INTERVAL"
LOCAL_STT_DEFAULT_PROVIDER = "moonshine"
LOCAL_STT_DEFAULT_CACHE_DIR = "./cache/moonshine_voice"

LOCAL_STT_DEFAULT_LANGUAGE = "en"
LOCAL_STT_DEFAULT_MODEL = "tiny_streaming"
LOCAL_STT_MODEL_CHOICES = ("tiny_streaming", "small_streaming")
LOCAL_STT_DEFAULT_UPDATE_INTERVAL = 0.35

# Cap how many user turns are kept in handler-managed conversation history.
# 0 disables trimming; live realtime backends (OpenAI/HF/Gemini Live) manage
# history server-side and ignore this. See history_trim.py.
MAX_HISTORY_TURNS_ENV = "REACHY_MINI_MAX_HISTORY_TURNS"
DEFAULT_MAX_HISTORY_TURNS = 20

# Echo-guard cooldown added on top of the byte-count-derived playback deadline.
# Lower than the old 500ms queue-size estimate because the byte-count is more
# accurate. Phase 5f.2 (2026-05-16 hardware finding): raised from 300 → 800
# to swallow room-acoustic reverb tail in the chassis enclosure under the
# faster-whisper STT backend; the old 300ms covered device-buffer + scheduling
# jitter but not physical decay, so reverb leaked past the guard and webrtcvad
# + whisper hallucinated echo-driven turns ("You", "Thank you") that
# interrupted the assistant. Operators who need tighter barge-in can set
# ``REACHY_MINI_ECHO_COOLDOWN_MS=300`` to restore the old behaviour.
ECHO_COOLDOWN_MS_ENV = "REACHY_MINI_ECHO_COOLDOWN_MS"
DEFAULT_ECHO_COOLDOWN_MS = 800

logger = logging.getLogger(__name__)


# NOTE (Phase 4f, #337): The retired ``BACKEND_PROVIDER`` and
# ``LOCAL_STT_RESPONSE_BACKEND`` dials have been deleted along with their
# normalisation / model-name-derivation helpers
# (``_normalize_backend_provider``, ``_resolve_model_name``,
# ``_normalize_local_stt_response_backend``). The active pipeline is now
# expressed via ``PIPELINE_MODE`` + ``AUDIO_INPUT_BACKEND`` +
# ``AUDIO_OUTPUT_BACKEND`` + ``LLM_BACKEND``. See
# docs/superpowers/specs/2026-05-16-phase-4f-backend-provider-retirement.md.


def _is_gemini_model_name(model_name: str | None) -> bool:
    """Return True when the provided model name targets Gemini."""
    candidate = (model_name or "").strip().lower()
    return candidate.startswith("gemini")


_VALID_PROVIDER_IDS: frozenset[str] = frozenset({OPENAI_BACKEND, GEMINI_BACKEND, HF_BACKEND, LOCAL_STT_BACKEND})


def _normalize_provider_id(provider_id: str | None) -> str:
    """Validate that ``provider_id`` is a known PROVIDER_ID, defaulting to HF.

    Unknown strings raise; ``None``/empty falls back to :data:`HF_BACKEND`
    to keep the legacy default-on-unset semantic.
    """
    candidate = (provider_id or "").strip().lower()
    if not candidate:
        return HF_BACKEND
    if candidate not in _VALID_PROVIDER_IDS:
        raise ValueError(f"Invalid provider_id={provider_id!r}. Expected one of: {sorted(_VALID_PROVIDER_IDS)}.")
    return candidate


def _resolve_model_name_from_env(env_value: str | None, pipeline_mode: str | None = None) -> str:
    """Resolve ``MODEL_NAME`` from the environment, applying pipeline defaults.

    Provider defaults come from :data:`DEFAULT_MODEL_NAME_BY_BACKEND` keyed by
    the ``PROVIDER_ID`` implied by ``pipeline_mode``.  The Hugging Face
    realtime pipeline intentionally returns ``""`` so the OpenAI-compat
    client omits the ``model`` connect kwarg (the HF server picks its own).
    """
    candidate = (env_value or "").strip()
    if candidate:
        return candidate
    if pipeline_mode == PIPELINE_MODE_HF_REALTIME:
        return DEFAULT_MODEL_NAME_BY_BACKEND[HF_BACKEND]
    if pipeline_mode == PIPELINE_MODE_GEMINI_LIVE:
        return DEFAULT_MODEL_NAME_BY_BACKEND[GEMINI_BACKEND]
    return DEFAULT_MODEL_NAME_BY_BACKEND[OPENAI_BACKEND]


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    Accepted truthy values: 1, true, yes, on
    Accepted falsy values: 0, false, no, off
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _env_float_clamped(name: str, default: float, lo: float, hi: float) -> float:
    """Parse a float env var, clamping to [lo, hi]."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default=%.2f", name, raw, default)
        return default
    clamped = max(lo, min(hi, value))
    if clamped != value:
        logger.warning("%s=%.2f clamped to %.2f", name, value, clamped)
    return clamped


def _normalize_hf_connection_mode(value: str | None) -> str | None:
    """Normalize the Hugging Face connection mode, if explicitly configured."""
    candidate = (value or "").strip().lower()
    if not candidate:
        return None

    if candidate not in {HF_LOCAL_CONNECTION_MODE, HF_DEPLOYED_CONNECTION_MODE}:
        logger.warning(
            "Invalid %s=%r. Expected local or deployed.",
            HF_REALTIME_CONNECTION_MODE_ENV,
            value,
        )
        return None
    return candidate


def _normalize_local_stt_model(value: str | None) -> str:
    """Normalize the local STT model selector."""
    candidate = (value or "").strip().lower().replace("-", "_")
    if not candidate:
        return LOCAL_STT_DEFAULT_MODEL
    if candidate not in LOCAL_STT_MODEL_CHOICES:
        logger.warning(
            "Invalid %s=%r. Expected one of %s; using %s.",
            LOCAL_STT_MODEL_ENV,
            value,
            ", ".join(LOCAL_STT_MODEL_CHOICES),
            LOCAL_STT_DEFAULT_MODEL,
        )
        return LOCAL_STT_DEFAULT_MODEL
    return candidate


def _normalize_local_stt_language(value: str | None) -> str:
    """Normalize the local STT language code."""
    candidate = (value or "").strip().lower()
    return candidate or LOCAL_STT_DEFAULT_LANGUAGE


def _normalize_local_stt_update_interval(value: str | None) -> float:
    """Normalize the local STT partial update interval in seconds."""
    if value is None or not value.strip():
        return LOCAL_STT_DEFAULT_UPDATE_INTERVAL
    try:
        interval = float(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r. Using %.2f.",
            LOCAL_STT_UPDATE_INTERVAL_ENV,
            value,
            LOCAL_STT_DEFAULT_UPDATE_INTERVAL,
        )
        return LOCAL_STT_DEFAULT_UPDATE_INTERVAL
    if interval < 0.1 or interval > 2.0:
        logger.warning(
            "Invalid %s=%r. Expected 0.1-2.0 seconds; using %.2f.",
            LOCAL_STT_UPDATE_INTERVAL_ENV,
            value,
            LOCAL_STT_DEFAULT_UPDATE_INTERVAL,
        )
        return LOCAL_STT_DEFAULT_UPDATE_INTERVAL
    return interval


def derive_audio_backends(
    provider_id: str,
    response_backend: str | None = None,
) -> tuple[str, str]:
    """Derive ``(AUDIO_INPUT_BACKEND, AUDIO_OUTPUT_BACKEND)`` from a ``PROVIDER_ID``.

    Pure helper retained as a library / test convenience.  ``provider_id``
    is one of :data:`OPENAI_BACKEND`, :data:`GEMINI_BACKEND`,
    :data:`HF_BACKEND`, :data:`LOCAL_STT_BACKEND` — the same coarse provider
    strings the realtime handlers carry on their ``PROVIDER_ID`` ClassVar.

    For ``provider_id == "local_stt"`` the ``response_backend`` argument
    selects which TTS output is paired with moonshine.  Recognised values
    are the historical response-backend strings (``chatterbox``,
    ``gemini_tts``, ``elevenlabs``, ``llama_*_tts`` variants, ``openai``,
    ``huggingface``).  When unset/unknown, falls back to
    :data:`AUDIO_OUTPUT_CHATTERBOX`.
    """
    if provider_id not in _VALID_PROVIDER_IDS:
        raise ValueError(f"Invalid provider_id={provider_id!r}. Expected one of: {sorted(_VALID_PROVIDER_IDS)}.")
    if provider_id == HF_BACKEND:
        return (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)
    if provider_id == OPENAI_BACKEND:
        return (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME)
    if provider_id == GEMINI_BACKEND:
        return (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE)
    # LOCAL_STT_BACKEND — input is always moonshine; output mirrors the
    # legacy response-backend table for back-compat with operators that
    # still feed the function those strings (the live config path no
    # longer reads them — only the test suite + library callers do).
    _legacy_response_to_audio_output: dict[str, str] = {
        CHATTERBOX_OUTPUT: AUDIO_OUTPUT_CHATTERBOX,
        ELEVENLABS_OUTPUT: AUDIO_OUTPUT_ELEVENLABS,
        LLAMA_ELEVENLABS_TTS_OUTPUT: AUDIO_OUTPUT_ELEVENLABS,
        GEMINI_TTS_OUTPUT: AUDIO_OUTPUT_GEMINI_TTS,
        LLAMA_GEMINI_TTS_OUTPUT: AUDIO_OUTPUT_GEMINI_TTS,
        OPENAI_BACKEND: AUDIO_OUTPUT_OPENAI_REALTIME,
        HF_BACKEND: AUDIO_OUTPUT_HF,
    }
    output = _legacy_response_to_audio_output.get(
        (response_backend or "").strip().lower(),
        AUDIO_OUTPUT_CHATTERBOX,
    )
    return (AUDIO_INPUT_MOONSHINE, output)


def _normalize_audio_input_backend(value: str | None) -> str | None:
    """Validate and normalize an AUDIO_INPUT_BACKEND value.

    Returns the normalised string, or None if the value was absent/empty.
    Logs a warning and returns None for unrecognised values.
    """
    candidate = (value or "").strip().lower()
    if not candidate:
        return None
    if candidate in AUDIO_INPUT_CHOICES:
        return candidate
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Ignoring.",
        AUDIO_INPUT_BACKEND_ENV,
        value,
        ", ".join(AUDIO_INPUT_CHOICES),
    )
    return None


def _normalize_audio_output_backend(value: str | None) -> str | None:
    """Validate and normalize an AUDIO_OUTPUT_BACKEND value.

    Returns the normalised string, or None if the value was absent/empty.
    Logs a warning and returns None for unrecognised values.
    """
    candidate = (value or "").strip().lower()
    if not candidate:
        return None
    if candidate in AUDIO_OUTPUT_CHOICES:
        return candidate
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Ignoring.",
        AUDIO_OUTPUT_BACKEND_ENV,
        value,
        ", ".join(AUDIO_OUTPUT_CHOICES),
    )
    return None


def resolve_audio_backends(
    provider_id: str,
    explicit_input: str | None,
    explicit_output: str | None,
    response_backend: str | None = None,
) -> tuple[str, str]:
    """Return the resolved ``(input, output)`` audio backend pair.

    Resolution order:

    1. If *both* explicit overrides are provided and the combination is
       supported, use them directly.
    2. If *both* are provided but the combination is unsupported, log a
       WARNING and fall back to the ``provider_id``-derived defaults.
    3. If only one override is set, treat it the same as unsupported
       (partial overrides are not yet implemented) — log a WARNING and
       fall back.
    4. If neither is set, derive from ``provider_id``.

    Args:
        provider_id:       A coarse provider id (``OPENAI_BACKEND``,
            ``GEMINI_BACKEND``, ``HF_BACKEND``, ``LOCAL_STT_BACKEND``).
        explicit_input:    Normalised ``AUDIO_INPUT_BACKEND`` override, or ``None``.
        explicit_output:   Normalised ``AUDIO_OUTPUT_BACKEND`` override, or ``None``.
        response_backend:  Optional legacy response-backend string, only
            consulted on the ``local_stt`` derivation fallback path; ignored
            when explicit overrides win.

    Returns:
        A ``(input_backend, output_backend)`` tuple of canonical backend IDs.

    """
    derived = derive_audio_backends(provider_id, response_backend)

    both_set = explicit_input is not None and explicit_output is not None
    neither_set = explicit_input is None and explicit_output is None

    if neither_set:
        return derived

    if not both_set:
        set_var = AUDIO_INPUT_BACKEND_ENV if explicit_input is not None else AUDIO_OUTPUT_BACKEND_ENV
        logger.warning(
            "Partial audio backend override detected: %s is set but the other is not. "
            "Both %s and %s must be set together to take effect. "
            "Falling back to provider_id=%r defaults: input=%r, output=%r.",
            set_var,
            AUDIO_INPUT_BACKEND_ENV,
            AUDIO_OUTPUT_BACKEND_ENV,
            provider_id,
            derived[0],
            derived[1],
        )
        return derived

    assert explicit_input is not None and explicit_output is not None
    combo = (explicit_input, explicit_output)
    if combo in _SUPPORTED_AUDIO_COMBINATIONS:
        return combo

    logger.warning(
        "Unsupported audio backend combination: %s=%r + %s=%r. "
        "This pairing has no handler implementation yet. "
        "Falling back to provider_id=%r defaults: input=%r, output=%r. "
        "See docs/audio-backends.md for the supported matrix.",
        AUDIO_INPUT_BACKEND_ENV,
        explicit_input,
        AUDIO_OUTPUT_BACKEND_ENV,
        explicit_output,
        provider_id,
        derived[0],
        derived[1],
    )
    return derived


@dataclass(frozen=True)
class HFConnectionSelection:
    """Resolved Hugging Face connection mode and target availability."""

    mode: str
    has_target: bool
    session_url: str | None = None
    direct_ws_url: str | None = None


@dataclass(frozen=True)
class HFRealtimeURLParts:
    """Parsed Hugging Face realtime URL components used by UI and client setup."""

    base_url: str
    websocket_base_url: str
    connect_query: dict[str, str]
    host: str | None
    port: int | None
    has_realtime_path: bool


def parse_hf_realtime_url(realtime_url: str) -> HFRealtimeURLParts:
    """Parse a Hugging Face realtime URL into OpenAI-compatible client endpoints."""
    parsed = urlsplit(realtime_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss", "http", "https"}:
        raise ValueError(
            "Expected Hugging Face realtime URL to start with ws://, wss://, http://, or https://, "
            f"got: {realtime_url}"
        )

    path = parsed.path.rstrip("/")
    has_realtime_path = path.endswith("/realtime")
    if has_realtime_path:
        base_path = path[: -len("/realtime")]
    else:
        base_path = path

    connect_query = {key: value for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "model"}
    http_scheme = "https" if scheme in {"wss", "https"} else "http"
    websocket_scheme = "wss" if scheme in {"wss", "https"} else "ws"
    base_url = urlunsplit((http_scheme, parsed.netloc, base_path, "", ""))
    websocket_base_url = urlunsplit((websocket_scheme, parsed.netloc, base_path, "", ""))
    return HFRealtimeURLParts(
        base_url=base_url,
        websocket_base_url=websocket_base_url,
        connect_query=connect_query,
        host=parsed.hostname,
        port=parsed.port or HF_DEFAULTS.direct_port,
        has_realtime_path=has_realtime_path,
    )


def parse_hf_direct_target(ws_url: str | None) -> tuple[str | None, int | None]:
    """Extract host and port from a direct Hugging Face realtime URL."""
    if not ws_url:
        return None, None
    try:
        parsed = parse_hf_realtime_url(ws_url)
        return parsed.host, parsed.port
    except Exception:
        return None, None


def build_hf_direct_ws_url(host: str, port: int) -> str:
    """Build the direct Hugging Face realtime websocket URL used by the app."""
    return f"ws://{host}:{port}/v1/realtime"


def _collect_profile_names(profiles_root: Path) -> set[str]:
    """Return profile folder names from a profiles root directory."""
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


def _collect_tool_module_names(tools_root: Path) -> set[str]:
    """Return tool module names from a tools directory."""
    if not tools_root.exists() or not tools_root.is_dir():
        return set()
    ignored = {"__init__", "core_tools"}
    return {p.stem for p in tools_root.glob("*.py") if p.is_file() and p.stem not in ignored}


def _raise_on_name_collisions(
    *,
    label: str,
    external_root: Path,
    internal_root: Path,
    external_names: set[str],
    internal_names: set[str],
) -> None:
    """Raise with a clear message when external/internal names collide."""
    collisions = sorted(external_names & internal_names)
    if not collisions:
        return

    raise RuntimeError(
        f"Config.__init__(): Ambiguous {label} names found in both external and built-in libraries: {collisions}. "
        f"External {label} root: {external_root}. Built-in {label} root: {internal_root}. "
        f"Please rename the conflicting external {label}(s) to continue."
    )


# Validate LOCKED_PROFILE at startup
if LOCKED_PROFILE is not None:
    _profiles_dir = DEFAULT_PROFILES_DIRECTORY
    _profile_path = _profiles_dir / LOCKED_PROFILE
    _instructions_file = _profile_path / "instructions.txt"
    if not _profile_path.is_dir():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' does not exist in {_profiles_dir}", file=sys.stderr)
        sys.exit(1)
    if not _instructions_file.is_file():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' has no instructions.txt", file=sys.stderr)
        sys.exit(1)

_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)

if _skip_dotenv:
    logger.info("Skipping .env loading because REACHY_MINI_SKIP_DOTENV is set")
else:
    # Locate .env file (search upward from current working directory)
    dotenv_path = find_dotenv(usecwd=True)

    if dotenv_path:
        # Load .env and override environment variables
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Configuration loaded from {dotenv_path}")
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for Robot Comic."""

    # Required (one of these depending on the active pipeline)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ELEVENLABS_API_KEY = os.getenv(ELEVENLABS_API_KEY_ENV)
    ELEVENLABS_VOICE = os.getenv(ELEVENLABS_VOICE_ENV) or ""

    # MODEL_NAME — env override falls back to the pipeline-mode default (see
    # ``_resolve_model_name_from_env``).  Resolved a second time at the bottom
    # of the class body once ``PIPELINE_MODE`` is known so the HF-realtime
    # path correctly gets the empty-string default.
    MODEL_NAME = _resolve_model_name_from_env(os.getenv("MODEL_NAME"))
    HF_REALTIME_CONNECTION_MODE = (
        _normalize_hf_connection_mode(os.getenv(HF_REALTIME_CONNECTION_MODE_ENV)) or HF_DEFAULTS.connection_mode
    )
    # Deliberately ignore HF_REALTIME_SESSION_URL from the environment; the app-managed proxy is HF_DEFAULTS.session_url.
    HF_REALTIME_SESSION_URL = HF_DEFAULTS.session_url
    HF_REALTIME_WS_URL = os.getenv(HF_REALTIME_WS_URL_ENV)
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set
    LOCAL_STT_PROVIDER = (os.getenv(LOCAL_STT_PROVIDER_ENV) or LOCAL_STT_DEFAULT_PROVIDER).strip().lower()
    LOCAL_STT_CACHE_DIR = os.getenv(LOCAL_STT_CACHE_DIR_ENV, LOCAL_STT_DEFAULT_CACHE_DIR)
    LOCAL_STT_LANGUAGE = _normalize_local_stt_language(os.getenv(LOCAL_STT_LANGUAGE_ENV))
    LOCAL_STT_MODEL = _normalize_local_stt_model(os.getenv(LOCAL_STT_MODEL_ENV))
    LOCAL_STT_UPDATE_INTERVAL = _normalize_local_stt_update_interval(os.getenv(LOCAL_STT_UPDATE_INTERVAL_ENV))

    GEMINI_LIVE_VIDEO_STREAMING = _env_flag("GEMINI_LIVE_VIDEO_STREAMING", default=False)

    # Gemini Live presence-backoff: re-prompt the user on exponential backoff
    # when they go silent after the robot asks a question.
    # GEMINI_LIVE_PRESENCE_ENABLED: 0/false disables the feature entirely.
    # GEMINI_LIVE_PRESENCE_FIRST_S: seconds before the first re-prompt (default 10).
    # GEMINI_LIVE_PRESENCE_MAX_ATTEMPTS: max re-prompts before entering silent-wait (default 3).
    # GEMINI_LIVE_PRESENCE_BACKOFF_FACTOR: multiplier per successive re-prompt (default 2.0).
    GEMINI_LIVE_PRESENCE_ENABLED = _env_flag("GEMINI_LIVE_PRESENCE_ENABLED", default=False)
    GEMINI_LIVE_PRESENCE_FIRST_S = float(os.getenv("GEMINI_LIVE_PRESENCE_FIRST_S", "10"))
    GEMINI_LIVE_PRESENCE_MAX_ATTEMPTS = int(os.getenv("GEMINI_LIVE_PRESENCE_MAX_ATTEMPTS", "3"))
    GEMINI_LIVE_PRESENCE_BACKOFF_FACTOR = float(os.getenv("GEMINI_LIVE_PRESENCE_BACKOFF_FACTOR", "2.0"))

    # Gemini Live server-side VAD tuning. Defaults raise the end-of-speech bar
    # well above the SDK's eager defaults so brief pauses / breaths during the
    # user's reply don't fire a fresh turn while the model is still responding.
    #
    # 2026-05-14 field-tune (issue #140): original 800 ms silence threshold made
    # turns feel sluggish during natural conversation on the robot.  Dropping to
    # 600 ms (the candidate value from #140 field observation) restores
    # responsiveness while still preventing premature barge-in.  prefix_ms and
    # sensitivity levels left at their PR #136 values — no field complaint there.
    GEMINI_LIVE_VAD_SILENCE_MS = int(os.getenv("GEMINI_LIVE_VAD_SILENCE_MS", "600"))
    GEMINI_LIVE_VAD_PREFIX_MS = int(os.getenv("GEMINI_LIVE_VAD_PREFIX_MS", "200"))
    # One of: HIGH | LOW | UNSPECIFIED. LOW = harder to trigger (fewer false positives).
    GEMINI_LIVE_VAD_START_SENSITIVITY = os.getenv("GEMINI_LIVE_VAD_START_SENSITIVITY", "LOW").upper()
    GEMINI_LIVE_VAD_END_SENSITIVITY = os.getenv("GEMINI_LIVE_VAD_END_SENSITIVITY", "LOW").upper()
    MOVEMENT_SPEED_FACTOR = _env_float_clamped("MOVEMENT_SPEED_FACTOR", default=0.3, lo=0.1, hi=2.0)
    IDLE_ANIMATION_ENABLED = _env_flag("IDLE_ANIMATION_ENABLED", default=False)
    MOONSHINE_HEARTBEAT = _env_flag("MOONSHINE_HEARTBEAT", default=False)
    AUDIO_CAPTURE_PATH = _resolve_audio_capture_path()
    # OTel instrumentation mode: unset=disabled, "trace"=console only, "remote"=console+OTLP
    ROBOT_INSTRUMENTATION = os.getenv("ROBOT_INSTRUMENTATION", "")

    CHATTERBOX_URL = os.getenv(CHATTERBOX_URL_ENV, CHATTERBOX_DEFAULT_URL)
    CHATTERBOX_VOICE = os.getenv(CHATTERBOX_VOICE_ENV, CHATTERBOX_DEFAULT_VOICE)
    CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", str(CHATTERBOX_DEFAULT_EXAGGERATION)))
    CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", str(CHATTERBOX_DEFAULT_CFG_WEIGHT)))
    CHATTERBOX_TEMPERATURE = float(os.getenv("CHATTERBOX_TEMPERATURE", str(CHATTERBOX_DEFAULT_TEMPERATURE)))
    CHATTERBOX_GAIN = float(os.getenv("CHATTERBOX_GAIN", str(CHATTERBOX_DEFAULT_GAIN)))
    CHATTERBOX_AUTO_GAIN_ENABLED = _env_flag(CHATTERBOX_AUTO_GAIN_ENABLED_ENV, default=True)
    CHATTERBOX_TARGET_DBFS = float(os.getenv(CHATTERBOX_TARGET_DBFS_ENV, str(CHATTERBOX_DEFAULT_TARGET_DBFS)))
    REACHY_MINI_TTS_SLOW_WARN_S = float(os.getenv("REACHY_MINI_TTS_SLOW_WARN_S", "10.0"))
    LLAMA_CPP_URL = os.getenv(LLAMA_CPP_URL_ENV, LLAMA_CPP_DEFAULT_URL)
    LLAMA_HEALTH_CHECK_ENABLED = _env_flag(LLAMA_HEALTH_CHECK_ENV, default=True)
    # When True, language_dissect will call llama-server as a fallback for phrases
    # that score below the 60% decomposition threshold.  Default off — adds latency.
    LANGUAGE_DISSECT_LLM_FALLBACK_ENABLED = _env_flag(LANGUAGE_DISSECT_LLM_FALLBACK_ENV, default=False)
    LLM_WARMUP_ENABLED = _env_flag("REACHY_MINI_LLM_WARMUP_ENABLED", default=True)
    LLM_BACKEND = (os.getenv(LLM_BACKEND_ENV) or LLM_BACKEND_LLAMA).strip().lower()
    GEMINI_LLM_MODEL = os.getenv(GEMINI_LLM_MODEL_ENV, _GEMINI_LLM_MODEL_DEFAULT)

    # Wake-on-LAN: send a magic packet when llama-server is unreachable.
    # WOL_MAC is unset by default; set it to enable the fallback.
    WOL_MAC = os.getenv(WOL_MAC_ENV) or None
    WOL_BROADCAST = os.getenv(WOL_BROADCAST_ENV, WOL_DEFAULT_BROADCAST)
    WOL_RETRY_AFTER_S = float(os.getenv(WOL_RETRY_AFTER_ENV, str(WOL_DEFAULT_RETRY_AFTER_S)))

    WARMUP_WAV_ENABLED = _env_flag("ROBOT_COMIC_WARMUP_WAV_ENABLED", default=True)
    WARMUP_WAV_PATH = os.getenv("ROBOT_COMIC_WARMUP_WAV") or None
    # When True, a tiny synthesised tone is played immediately at startup (before
    # any disk I/O) so the robot sounds alive within ~1 s of process launch even
    # if heavy imports take longer. Opt-in; default False.
    WARMUP_BLIP_ENABLED = _env_flag("REACHY_MINI_WARMUP_BLIP_ENABLED", default=False)

    # Disengagement guardrail for high-intensity personas (e.g. Bill Hicks).
    # When unset, the guardrail activates automatically for profiles in
    # guardrail.GUARDRAIL_PROFILES.  Set to 1/true to force-enable for any
    # profile, or 0/false to force-disable globally.
    GUARDRAIL_ENABLED = _env_flag("REACHY_MINI_GUARDRAIL_ENABLED", default=False)

    # When True, EngagementMonitor.analyze() uses a one-shot LLM call
    # (llama-server /completion) to score user discomfort instead of the
    # keyword heuristic.  More accurate but incurs a local LLM round-trip.
    # Falls back to heuristic on network/parse errors.  Default False.
    GUARDRAIL_LLM_SCORING_ENABLED = _env_flag("REACHY_MINI_GUARDRAIL_LLM_SCORING", default=False)

    # Persist recent punchlines across sessions and inject a "don't repeat"
    # section into the system prompt.  Set to 0/false to disable.
    JOKE_HISTORY_ENABLED = _env_flag("REACHY_MINI_JOKE_HISTORY_ENABLED", default=True)

    # When True, use a one-shot LLM call to extract the punchline and topic
    # from each completed turn instead of the last-sentence heuristic.
    # Falls back to the heuristic on timeout / network error.
    # Set to 0/false to disable and always use the heuristic.
    JOKE_HISTORY_LLM_EXTRACT_ENABLED = _env_flag("REACHY_MINI_JOKE_HISTORY_LLM_EXTRACT_ENABLED", default=True)

    # When set to 1/true, forces the GEMINI TTS DELIVERY TAGS section to be
    # included in the assembled system prompt regardless of the active backend.
    # Useful for debugging or testing delivery-tag behaviour on any backend.
    FORCE_DELIVERY_TAGS = _env_flag("REACHY_MINI_FORCE_DELIVERY_TAGS", default=False)

    # Welcome-gate: when enabled, the handler is blocked until the user speaks
    # the persona wake-name.  Only meaningful when the composable pipeline
    # uses the local STT (moonshine) input.  Default is False (gate disabled).
    WELCOME_GATE_ENABLED = _env_flag("REACHY_MINI_WELCOME_GATE_ENABLED", default=False)

    # Echo-guard cooldown after assistant audio ends.
    ECHO_COOLDOWN_MS = int(os.getenv(ECHO_COOLDOWN_MS_ENV, str(DEFAULT_ECHO_COOLDOWN_MS)))

    # Kiosk-mode startup screen: plays a short generic-voice intro + persona
    # listing before the per-persona greeting begins.
    STARTUP_SCREEN_ENABLED = _env_flag("REACHY_MINI_STARTUP_SCREEN", default=False)
    STARTUP_SCREEN_PERSONA_ORDER = os.getenv("REACHY_MINI_STARTUP_SCREEN_PERSONA_ORDER", "")

    # Boot-time opener strategy (#290). "canned" reads a per-persona line from
    # profiles/<persona>/openers.txt and enqueues it directly to TTS, bypassing
    # the LLM entirely. "llm" preserves the legacy synthetic
    # "[conversation started]" round-trip — fragile under Gemini load but kept
    # behind the flag for A/B comparison. Default: canned.
    STARTUP_TRIGGER_MODE: str = (os.getenv("REACHY_MINI_STARTUP_TRIGGER_MODE") or "canned").strip().lower()

    # Face recognition: when enabled, the camera pipeline will attempt to match
    # incoming visitors against the stored face-embedding database and surface
    # a name for repeat-visitor callbacks.  Requires a real FaceEmbedder
    # implementation (follow-up PR); keep False until then.
    FACE_RECOGNITION_ENABLED = _env_flag("REACHY_MINI_FACE_RECOGNITION_ENABLED", default=False)

    # WebSocket Pi ↔ laptop channel (opt-in, default disabled).
    # REACHY_MINI_WS_ENABLED: set to 1/true to enable the bidirectional channel.
    # REACHY_MINI_WS_PORT: TCP port (server listens, client connects); default 8765.
    # REACHY_MINI_WS_SERVER_HOST: hostname/IP of the laptop (Pi-side client only).
    # WS_PAUSE_FLAG: runtime flag set by a laptop_command(action="pause") to signal
    # the handler to pause between turns. Reset to False on action="resume".
    WS_ENABLED = _env_flag("REACHY_MINI_WS_ENABLED", default=False)
    WS_PORT = int(os.getenv("REACHY_MINI_WS_PORT", "8765"))
    WS_SERVER_HOST = os.getenv("REACHY_MINI_WS_SERVER_HOST", "localhost")
    WS_PAUSE_FLAG: bool = False

    # ---------------------------------------------------------------------------
    # Modular audio pipeline.  Operators set REACHY_MINI_AUDIO_INPUT_BACKEND +
    # REACHY_MINI_AUDIO_OUTPUT_BACKEND (and optionally REACHY_MINI_PIPELINE_MODE)
    # directly in the .env.  Defaults are the bundled HF realtime pair so a
    # blank deploy still boots.  Setting only one of the two is treated as if
    # neither were set (with a warning).  See docs/audio-backends.md.
    # ---------------------------------------------------------------------------
    _raw_audio_input = _normalize_audio_input_backend(os.getenv(AUDIO_INPUT_BACKEND_ENV))
    _raw_audio_output = _normalize_audio_output_backend(os.getenv(AUDIO_OUTPUT_BACKEND_ENV))
    # ``resolve_audio_backends`` handles every case (both set / one set /
    # neither set) by falling back to the bundled HF pair when the explicit
    # override is partial or unsupported.  Operators flip to a different
    # pipeline via the admin UI, which writes both env vars + PIPELINE_MODE.
    _resolved_audio = resolve_audio_backends(HF_BACKEND, _raw_audio_input, _raw_audio_output)
    AUDIO_INPUT_BACKEND: str = _resolved_audio[0]
    AUDIO_OUTPUT_BACKEND: str = _resolved_audio[1]

    # 4th dial: PIPELINE_MODE. Composable means the STT/LLM/TTS dials decide
    # the pipeline; bundled values (openai_realtime / gemini_live /
    # hf_realtime) fuse all three phases into one session.  When the env var
    # is unset we derive from the resolved (input, output) pair — see
    # ``derive_pipeline_mode``.
    _raw_pipeline_mode = _normalize_pipeline_mode(os.getenv(PIPELINE_MODE_ENV))
    PIPELINE_MODE: str = _raw_pipeline_mode or derive_pipeline_mode(AUDIO_INPUT_BACKEND, AUDIO_OUTPUT_BACKEND)

    # Resolve MODEL_NAME a second time now that PIPELINE_MODE is known so the
    # HF-realtime pipeline gets the correct empty-string default.
    MODEL_NAME = _resolve_model_name_from_env(os.getenv("MODEL_NAME"), PIPELINE_MODE)

    logger.debug(
        "Pipeline mode: %s, Audio: %s → %s, Model: %s, "
        "HF mode: %s, HF session URL set: %s, HF direct URL set: %s, HF_HOME: %s, "
        "Vision Model: %s, Local STT: %s/%s/%s cache=%s",
        PIPELINE_MODE,
        AUDIO_INPUT_BACKEND,
        AUDIO_OUTPUT_BACKEND,
        MODEL_NAME,
        HF_REALTIME_CONNECTION_MODE,
        bool(HF_REALTIME_SESSION_URL and HF_REALTIME_SESSION_URL.strip()),
        bool(HF_REALTIME_WS_URL and HF_REALTIME_WS_URL.strip()),
        HF_HOME,
        LOCAL_VISION_MODEL,
        LOCAL_STT_PROVIDER,
        LOCAL_STT_LANGUAGE,
        LOCAL_STT_MODEL,
        LOCAL_STT_CACHE_DIR,
    )

    # Filesystem root containing profile directories, not a Python import path.
    _profiles_directory_env = os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY")
    PROFILES_DIRECTORY = Path(_profiles_directory_env) if _profiles_directory_env else DEFAULT_PROFILES_DIRECTORY
    _tools_directory_env = os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    TOOLS_DIRECTORY = Path(_tools_directory_env) if _tools_directory_env else None
    AUTOLOAD_EXTERNAL_TOOLS = _env_flag("AUTOLOAD_EXTERNAL_TOOLS", default=False)
    REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    def __init__(self) -> None:
        """Initialize the configuration."""
        if self.REACHY_MINI_CUSTOM_PROFILE and self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            selected_profile_path = self.PROFILES_DIRECTORY / self.REACHY_MINI_CUSTOM_PROFILE
            if not selected_profile_path.is_dir():
                available_profiles = sorted(_collect_profile_names(self.PROFILES_DIRECTORY))
                raise RuntimeError(
                    "Config.__init__(): Selected profile "
                    f"'{self.REACHY_MINI_CUSTOM_PROFILE}' was not found in external profiles root "
                    f"{self.PROFILES_DIRECTORY}. "
                    f"Available external profiles: {available_profiles}. "
                    "Either set 'REACHY_MINI_CUSTOM_PROFILE' to one of the available external profiles "
                    "or unset 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' to use built-in profiles."
                )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            external_profiles = _collect_profile_names(self.PROFILES_DIRECTORY)
            internal_profiles = _collect_profile_names(DEFAULT_PROFILES_DIRECTORY)
            _raise_on_name_collisions(
                label="profile",
                external_root=self.PROFILES_DIRECTORY,
                internal_root=DEFAULT_PROFILES_DIRECTORY,
                external_names=external_profiles,
                internal_names=internal_profiles,
            )

        if self.TOOLS_DIRECTORY is not None:
            builtin_tools_root = Path(__file__).parent / "tools"
            external_tools = _collect_tool_module_names(self.TOOLS_DIRECTORY)
            internal_tools = _collect_tool_module_names(builtin_tools_root)
            _raise_on_name_collisions(
                label="tool",
                external_root=self.TOOLS_DIRECTORY,
                internal_root=builtin_tools_root,
                external_names=external_tools,
                internal_names=internal_tools,
            )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is set. "
                "Profiles (instructions.txt, ...) will be loaded from %s.",
                self.PROFILES_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is not set. Using built-in profiles from %s.",
                DEFAULT_PROFILES_DIRECTORY,
            )

        if self.TOOLS_DIRECTORY is not None:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is set. "
                "External tools will be loaded from %s.",
                self.TOOLS_DIRECTORY,
            )
        else:
            logger.info("'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is not set. Using built-in shared tools only.")


config = Config()


def refresh_runtime_config_from_env() -> None:
    """Refresh mutable runtime config fields from the current environment."""
    config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    config.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    config.ELEVENLABS_API_KEY = os.getenv(ELEVENLABS_API_KEY_ENV)
    config.ELEVENLABS_VOICE = os.getenv(ELEVENLABS_VOICE_ENV) or ""
    # First pass with no pipeline_mode so we can still pick up an env override;
    # re-resolved below once ``config.PIPELINE_MODE`` is refreshed.
    config.MODEL_NAME = _resolve_model_name_from_env(os.getenv("MODEL_NAME"))
    config.HF_REALTIME_CONNECTION_MODE = (
        _normalize_hf_connection_mode(os.getenv(HF_REALTIME_CONNECTION_MODE_ENV)) or HF_DEFAULTS.connection_mode
    )
    # Deliberately ignore HF_REALTIME_SESSION_URL from the environment; the app-managed proxy is HF_DEFAULTS.session_url.
    config.HF_REALTIME_SESSION_URL = HF_DEFAULTS.session_url
    config.HF_REALTIME_WS_URL = os.getenv(HF_REALTIME_WS_URL_ENV)
    config.HF_HOME = os.getenv("HF_HOME", "./cache")
    config.LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    config.HF_TOKEN = os.getenv("HF_TOKEN")
    config.LOCAL_STT_PROVIDER = (os.getenv(LOCAL_STT_PROVIDER_ENV) or LOCAL_STT_DEFAULT_PROVIDER).strip().lower()
    config.LOCAL_STT_CACHE_DIR = os.getenv(LOCAL_STT_CACHE_DIR_ENV, LOCAL_STT_DEFAULT_CACHE_DIR)
    config.LOCAL_STT_LANGUAGE = _normalize_local_stt_language(os.getenv(LOCAL_STT_LANGUAGE_ENV))
    config.LOCAL_STT_MODEL = _normalize_local_stt_model(os.getenv(LOCAL_STT_MODEL_ENV))
    config.LOCAL_STT_UPDATE_INTERVAL = _normalize_local_stt_update_interval(os.getenv(LOCAL_STT_UPDATE_INTERVAL_ENV))
    config.REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    config.GEMINI_LIVE_VIDEO_STREAMING = _env_flag("GEMINI_LIVE_VIDEO_STREAMING", default=False)
    config.GEMINI_LIVE_PRESENCE_ENABLED = _env_flag("GEMINI_LIVE_PRESENCE_ENABLED", default=False)
    config.GEMINI_LIVE_PRESENCE_FIRST_S = float(os.getenv("GEMINI_LIVE_PRESENCE_FIRST_S", "10"))
    config.GEMINI_LIVE_PRESENCE_MAX_ATTEMPTS = int(os.getenv("GEMINI_LIVE_PRESENCE_MAX_ATTEMPTS", "3"))
    config.GEMINI_LIVE_PRESENCE_BACKOFF_FACTOR = float(os.getenv("GEMINI_LIVE_PRESENCE_BACKOFF_FACTOR", "2.0"))
    config.GEMINI_LIVE_VAD_SILENCE_MS = int(os.getenv("GEMINI_LIVE_VAD_SILENCE_MS", "600"))
    config.GEMINI_LIVE_VAD_PREFIX_MS = int(os.getenv("GEMINI_LIVE_VAD_PREFIX_MS", "200"))
    config.GEMINI_LIVE_VAD_START_SENSITIVITY = os.getenv("GEMINI_LIVE_VAD_START_SENSITIVITY", "LOW").upper()
    config.GEMINI_LIVE_VAD_END_SENSITIVITY = os.getenv("GEMINI_LIVE_VAD_END_SENSITIVITY", "LOW").upper()
    config.MOVEMENT_SPEED_FACTOR = _env_float_clamped("MOVEMENT_SPEED_FACTOR", default=0.3, lo=0.1, hi=2.0)
    config.IDLE_ANIMATION_ENABLED = _env_flag("IDLE_ANIMATION_ENABLED", default=False)
    config.MOONSHINE_HEARTBEAT = _env_flag("MOONSHINE_HEARTBEAT", default=False)
    config.AUDIO_CAPTURE_PATH = _resolve_audio_capture_path()
    config.CHATTERBOX_URL = os.getenv(CHATTERBOX_URL_ENV, CHATTERBOX_DEFAULT_URL)
    config.CHATTERBOX_VOICE = os.getenv(CHATTERBOX_VOICE_ENV, CHATTERBOX_DEFAULT_VOICE)
    config.CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", str(CHATTERBOX_DEFAULT_EXAGGERATION)))
    config.CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", str(CHATTERBOX_DEFAULT_CFG_WEIGHT)))
    config.CHATTERBOX_TEMPERATURE = float(os.getenv("CHATTERBOX_TEMPERATURE", str(CHATTERBOX_DEFAULT_TEMPERATURE)))
    config.CHATTERBOX_GAIN = float(os.getenv("CHATTERBOX_GAIN", str(CHATTERBOX_DEFAULT_GAIN)))
    config.CHATTERBOX_AUTO_GAIN_ENABLED = _env_flag(CHATTERBOX_AUTO_GAIN_ENABLED_ENV, default=True)
    config.CHATTERBOX_TARGET_DBFS = float(os.getenv(CHATTERBOX_TARGET_DBFS_ENV, str(CHATTERBOX_DEFAULT_TARGET_DBFS)))
    config.REACHY_MINI_TTS_SLOW_WARN_S = float(os.getenv("REACHY_MINI_TTS_SLOW_WARN_S", "10.0"))
    config.LLAMA_CPP_URL = os.getenv(LLAMA_CPP_URL_ENV, LLAMA_CPP_DEFAULT_URL)
    config.LLAMA_HEALTH_CHECK_ENABLED = _env_flag(LLAMA_HEALTH_CHECK_ENV, default=True)
    config.LANGUAGE_DISSECT_LLM_FALLBACK_ENABLED = _env_flag(LANGUAGE_DISSECT_LLM_FALLBACK_ENV, default=False)
    config.LLM_BACKEND = (os.getenv(LLM_BACKEND_ENV) or LLM_BACKEND_LLAMA).strip().lower()
    config.GEMINI_LLM_MODEL = os.getenv(GEMINI_LLM_MODEL_ENV, _GEMINI_LLM_MODEL_DEFAULT)
    config.WOL_MAC = os.getenv(WOL_MAC_ENV) or None
    config.WOL_BROADCAST = os.getenv(WOL_BROADCAST_ENV, WOL_DEFAULT_BROADCAST)
    config.WOL_RETRY_AFTER_S = float(os.getenv(WOL_RETRY_AFTER_ENV, str(WOL_DEFAULT_RETRY_AFTER_S)))
    config.ROBOT_INSTRUMENTATION = os.getenv("ROBOT_INSTRUMENTATION", "")
    config.WARMUP_WAV_ENABLED = _env_flag("ROBOT_COMIC_WARMUP_WAV_ENABLED", default=True)
    config.WARMUP_WAV_PATH = os.getenv("ROBOT_COMIC_WARMUP_WAV") or None
    config.WARMUP_BLIP_ENABLED = _env_flag("REACHY_MINI_WARMUP_BLIP_ENABLED", default=False)
    config.GUARDRAIL_LLM_SCORING_ENABLED = _env_flag("REACHY_MINI_GUARDRAIL_LLM_SCORING", default=False)
    config.JOKE_HISTORY_ENABLED = _env_flag("REACHY_MINI_JOKE_HISTORY_ENABLED", default=True)
    config.JOKE_HISTORY_LLM_EXTRACT_ENABLED = _env_flag("REACHY_MINI_JOKE_HISTORY_LLM_EXTRACT_ENABLED", default=True)
    config.FORCE_DELIVERY_TAGS = _env_flag("REACHY_MINI_FORCE_DELIVERY_TAGS", default=False)
    config.WELCOME_GATE_ENABLED = _env_flag("REACHY_MINI_WELCOME_GATE_ENABLED", default=False)
    config.ECHO_COOLDOWN_MS = int(os.getenv(ECHO_COOLDOWN_MS_ENV, str(DEFAULT_ECHO_COOLDOWN_MS)))
    config.STARTUP_SCREEN_ENABLED = _env_flag("REACHY_MINI_STARTUP_SCREEN", default=False)
    config.STARTUP_SCREEN_PERSONA_ORDER = os.getenv("REACHY_MINI_STARTUP_SCREEN_PERSONA_ORDER", "")
    config.STARTUP_TRIGGER_MODE = (os.getenv("REACHY_MINI_STARTUP_TRIGGER_MODE") or "canned").strip().lower()
    config.WS_ENABLED = _env_flag("REACHY_MINI_WS_ENABLED", default=False)
    config.WS_PORT = int(os.getenv("REACHY_MINI_WS_PORT", "8765"))
    config.WS_SERVER_HOST = os.getenv("REACHY_MINI_WS_SERVER_HOST", "localhost")
    _refresh_raw_input = _normalize_audio_input_backend(os.getenv(AUDIO_INPUT_BACKEND_ENV))
    _refresh_raw_output = _normalize_audio_output_backend(os.getenv(AUDIO_OUTPUT_BACKEND_ENV))
    _refresh_audio = resolve_audio_backends(
        HF_BACKEND,
        _refresh_raw_input,
        _refresh_raw_output,
    )
    config.AUDIO_INPUT_BACKEND = _refresh_audio[0]
    config.AUDIO_OUTPUT_BACKEND = _refresh_audio[1]
    _refresh_pipeline_mode = _normalize_pipeline_mode(os.getenv(PIPELINE_MODE_ENV))
    config.PIPELINE_MODE = _refresh_pipeline_mode or derive_pipeline_mode(
        config.AUDIO_INPUT_BACKEND, config.AUDIO_OUTPUT_BACKEND
    )
    # Second pass with the resolved pipeline_mode so HF-realtime gets the
    # empty-string default that triggers the OpenAI-compat ``model`` kwarg
    # omission in ``base_realtime._run_realtime_session``.
    config.MODEL_NAME = _resolve_model_name_from_env(os.getenv("MODEL_NAME"), config.PIPELINE_MODE)


def provider_id_from_pipeline(pipeline_mode: str, audio_output_backend: str) -> str:
    """Return the realtime-handler ``PROVIDER_ID`` implied by a pipeline.

    Bundled pipeline modes map directly to their realtime provider; the
    composable pipeline derives the provider from the audio output backend
    so the voice catalog / required-credential helpers keep working.

    Returns one of ``OPENAI_BACKEND``, ``GEMINI_BACKEND``, ``HF_BACKEND``,
    ``LOCAL_STT_BACKEND``.
    """
    if pipeline_mode == PIPELINE_MODE_OPENAI_REALTIME:
        return OPENAI_BACKEND
    if pipeline_mode == PIPELINE_MODE_GEMINI_LIVE:
        return GEMINI_BACKEND
    if pipeline_mode == PIPELINE_MODE_HF_REALTIME:
        return HF_BACKEND
    # PIPELINE_MODE_COMPOSABLE — derive provider from the TTS output.  This
    # mirrors the LocalSTT*RealtimeHandler ClassVar overrides: the hybrid
    # classes report the *response* backend (openai / huggingface) rather
    # than ``local_stt`` so the voice catalog matches the spoken voice.
    if audio_output_backend == AUDIO_OUTPUT_OPENAI_REALTIME:
        return OPENAI_BACKEND
    if audio_output_backend == AUDIO_OUTPUT_HF:
        return HF_BACKEND
    return LOCAL_STT_BACKEND


def get_provider_id() -> str:
    """Return the active realtime-handler ``PROVIDER_ID``.

    Derived from ``config.PIPELINE_MODE`` and ``config.AUDIO_OUTPUT_BACKEND``.
    Replaces ``get_backend_choice()`` from before Phase 4f.
    """
    return provider_id_from_pipeline(config.PIPELINE_MODE, config.AUDIO_OUTPUT_BACKEND)


def get_backend_choice() -> str:
    """Return the active realtime-handler ``PROVIDER_ID``.

    Kept as a transitional alias during the Phase 4f cut-over for any
    external callers that hadn't migrated yet — use :func:`get_provider_id`
    in new code.
    """
    return get_provider_id()


def get_model_name_for_backend(backend: str) -> str:
    """Return the default model name for a ``PROVIDER_ID`` value."""
    return DEFAULT_MODEL_NAME_BY_BACKEND[_normalize_provider_id(backend)]


def get_backend_label(backend: str | None = None) -> str:
    """Return a human-readable label for the active (or specified) pipeline.

    When ``backend`` is supplied it is treated as a coarse ``PROVIDER_ID``
    so older call sites keep working.  When omitted, the label is derived
    from the active ``PIPELINE_MODE`` / ``AUDIO_OUTPUT_BACKEND`` /
    ``LLM_BACKEND`` so the text reflects the composable pipeline that's
    actually running.
    """
    if backend is not None:
        normalized = _normalize_provider_id(backend)
        if normalized != LOCAL_STT_BACKEND:
            return BACKEND_LABEL_BY_PROVIDER[normalized]
        # Fall through to the composable label-from-config path below.

    provider = get_provider_id()
    if provider != LOCAL_STT_BACKEND:
        return BACKEND_LABEL_BY_PROVIDER[provider]

    audio_output = getattr(config, "AUDIO_OUTPUT_BACKEND", AUDIO_OUTPUT_CHATTERBOX)
    llm_backend = getattr(config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    base_label = _LOCAL_STT_OUTPUT_BASE_LABELS.get(
        audio_output, _LOCAL_STT_OUTPUT_BASE_LABELS["openai_realtime_output"]
    )
    # The legacy ``llama_*_tts`` labels surfaced the LLM choice in the
    # human-readable string.  Preserve that for the gemini-TTS + elevenlabs
    # outputs where the LLM choice is operator-visible.
    if llm_backend == LLM_BACKEND_LLAMA and audio_output == AUDIO_OUTPUT_GEMINI_TTS:
        return "Local STT + llama.cpp + Gemini TTS"
    if llm_backend == LLM_BACKEND_LLAMA and audio_output == AUDIO_OUTPUT_ELEVENLABS:
        return "Local STT + llama.cpp + ElevenLabs TTS"
    return base_label


def get_available_voices_for_provider(provider_id: str | None = None) -> list[str]:
    """Return the curated voice list for a realtime-handler ``PROVIDER_ID``.

    The ``provider_id`` argument is one of ``OPENAI_BACKEND``,
    ``GEMINI_BACKEND``, ``HF_BACKEND``, or ``LOCAL_STT_BACKEND`` — the same
    coarse provider strings the realtime handlers carry on their
    ``PROVIDER_ID`` ClassVar.
    """
    normalized = get_provider_id() if provider_id is None else _normalize_provider_id(provider_id)
    if normalized == GEMINI_BACKEND:
        return list(GEMINI_AVAILABLE_VOICES)
    if normalized == HF_BACKEND:
        return list(HF_AVAILABLE_VOICES)
    return list(AVAILABLE_VOICES)


def get_default_voice_for_provider(provider_id: str | None = None) -> str:
    """Return the default voice for a realtime-handler ``PROVIDER_ID``.

    See :func:`get_available_voices_for_provider` for argument semantics.
    """
    normalized = get_provider_id() if provider_id is None else _normalize_provider_id(provider_id)
    return DEFAULT_VOICE_BY_BACKEND[normalized]


def get_hf_session_url() -> str | None:
    """Return the built-in Hugging Face session proxy URL, if any."""
    value = (getattr(config, "HF_REALTIME_SESSION_URL", None) or "").strip()
    return value or None


def get_hf_direct_ws_url() -> str | None:
    """Return the configured direct Hugging Face realtime URL, if any."""
    value = (getattr(config, "HF_REALTIME_WS_URL", None) or "").strip()
    return value or None


def get_hf_connection_selection() -> HFConnectionSelection:
    """Resolve the selected Hugging Face connection mode and whether it is usable."""
    session_url = get_hf_session_url()
    direct_ws_url = get_hf_direct_ws_url()
    mode = _normalize_hf_connection_mode(getattr(config, "HF_REALTIME_CONNECTION_MODE", None))
    if mode is None:
        raise RuntimeError(f"{HF_REALTIME_CONNECTION_MODE_ENV} must be set to local or deployed.")

    target = direct_ws_url if mode == HF_LOCAL_CONNECTION_MODE else session_url

    return HFConnectionSelection(
        mode=mode,
        has_target=bool(target),
        session_url=session_url,
        direct_ws_url=direct_ws_url,
    )


def has_hf_realtime_target() -> bool:
    """Return whether Hugging Face has a target for the selected mode."""
    return get_hf_connection_selection().has_target


def is_gemini_model() -> bool:
    """Return True if the configured MODEL_NAME is a Gemini Live model."""
    return get_backend_choice() == GEMINI_BACKEND


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass


async def refresh_elevenlabs_voices() -> None:
    """Fetch ElevenLabs voices from the API and update ELEVENLABS_AVAILABLE_VOICES.

    Called once at app startup to populate the voice catalog. Falls back to
    hardcoded voices if the API is unreachable.
    """
    from robot_comic.elevenlabs_voices import fetch_elevenlabs_voices

    global ELEVENLABS_AVAILABLE_VOICES
    voices = await fetch_elevenlabs_voices()
    ELEVENLABS_AVAILABLE_VOICES = list(voices.keys())
