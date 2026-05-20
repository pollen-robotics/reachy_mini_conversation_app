"""Handler-factory + lifecycle + config-resolution smoke matrix — issue #261.

Three test surfaces that close the coverage gap exposed by the production crashes
hotfixed in #260:

1. **Factory instantiation matrix** (catches MRO/init bugs like #260 bug #2):
   walk every entry in ``_SUPPORTED_AUDIO_COMBINATIONS`` and assert
   ``HandlerFactory.build`` returns without raising — including an
   ``LLM_BACKEND=gemini`` variant for the two Moonshine outputs that branch on
   it inside ``handler_factory.py``.

2. **Lifecycle smoke** (catches Moonshine + voice bugs like #260 bug #1 and #3):
   for each factory product, ``await handler._prepare_startup_credentials()`` with
   the same mocks.  Asserts:
   * ``Transcriber`` mock is called with ``model_path`` being a ``str`` (not
     ``Path``) AND the path is a directory (or its parent exists).
   * For TTS handlers, ``get_current_voice()`` is callable and returns a string
     (not raising ``NotImplementedError``).

3. **Config-resolution matrix** (catches env-var routing bugs):
   parametrise over the documented matrix in ``docs/audio-backends.md`` plus the
   legacy ``response_backend`` values, asserting that
   ``resolve_audio_backends`` returns the expected tuple.  Uses the post-#262
   ``derive_audio_backends`` ``response_backend`` kwarg behaviour.

No network, no real model loading, no subprocess spawning.  Heavy SDK boundaries
are patched.
"""

from __future__ import annotations
import sys
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import make_tool_deps
from robot_comic.config import (
    HF_BACKEND,
    AUDIO_INPUT_HF,
    GEMINI_BACKEND,
    OPENAI_BACKEND,
    AUDIO_OUTPUT_HF,
    XTTS_DEFAULT_URL,
    AUDIO_OUTPUT_XTTS,
    CHATTERBOX_OUTPUT,
    ELEVENLABS_OUTPUT,
    GEMINI_TTS_OUTPUT,
    LLM_BACKEND_LLAMA,
    LOCAL_STT_BACKEND,
    LLM_BACKEND_GEMINI,
    XTTS_DEFAULT_SPEAKER,
    AUDIO_INPUT_MOONSHINE,
    XTTS_DEFAULT_LANGUAGE,
    XTTS_DEFAULT_TIMEOUT_S,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_INPUT_FASTER_WHISPER,
    AUDIO_INPUT_OPENAI_REALTIME,
    LLAMA_ELEVENLABS_TTS_OUTPUT,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    _SUPPORTED_AUDIO_COMBINATIONS,
    derive_audio_backends,
    resolve_audio_backends,
)
from robot_comic.config import (
    HF_BACKEND as HF_RESPONSE_BACKEND,
)
from robot_comic.handler_factory import HandlerFactory


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_MOONSHINE_OUTPUTS_BRANCHING_ON_LLM = (AUDIO_OUTPUT_CHATTERBOX, AUDIO_OUTPUT_ELEVENLABS)


def _expected_handler_name(input_b: str, output_b: str, llm_backend: str) -> str:
    """Map a (input, output, llm) tuple to the handler class name selected by HandlerFactory.

    Phase 4e (#337) replaced every composable triple's concrete handler with
    :class:`ComposableConversationHandler`; the bundled-realtime and
    Moonshine+realtime-output hybrids still return their legacy concrete
    classes. Tests on the composable triples should additionally inspect
    ``result._tts_handler`` to confirm the right host was composed.
    """
    if input_b == AUDIO_INPUT_HF and output_b == AUDIO_OUTPUT_HF:
        return "HuggingFaceRealtimeHandler"
    if input_b == AUDIO_INPUT_OPENAI_REALTIME and output_b == AUDIO_OUTPUT_OPENAI_REALTIME:
        return "OpenaiRealtimeHandler"
    if input_b == AUDIO_INPUT_GEMINI_LIVE and output_b == AUDIO_OUTPUT_GEMINI_LIVE:
        return "GeminiLiveHandler"
    if input_b == AUDIO_INPUT_MOONSHINE:
        # Hybrids (LocalSTT + realtime output) still return concrete classes.
        if output_b == AUDIO_OUTPUT_OPENAI_REALTIME:
            return "LocalSTTOpenAIRealtimeHandler"
        if output_b == AUDIO_OUTPUT_HF:
            return "LocalSTTHuggingFaceRealtimeHandler"
        # Composable triples — always wrapped.
        if output_b in (AUDIO_OUTPUT_CHATTERBOX, AUDIO_OUTPUT_ELEVENLABS, AUDIO_OUTPUT_GEMINI_TTS, AUDIO_OUTPUT_XTTS):
            return "ComposableConversationHandler"
    if input_b == AUDIO_INPUT_FASTER_WHISPER:
        # Phase 5f: faster-whisper only pairs with composable triples; the
        # realtime-output hybrids stay on the LocalSTTInputMixin path
        # (faster-whisper does not slot into those).
        if output_b in (AUDIO_OUTPUT_CHATTERBOX, AUDIO_OUTPUT_ELEVENLABS, AUDIO_OUTPUT_GEMINI_TTS, AUDIO_OUTPUT_XTTS):
            return "ComposableConversationHandler"
    raise AssertionError(f"unexpected combo: {input_b!r} → {output_b!r} (llm={llm_backend!r})")


# ---------------------------------------------------------------------------
# Parametrize the factory matrix:
#   - every entry in _SUPPORTED_AUDIO_COMBINATIONS with the default (llama) LLM
#   - plus an LLM_BACKEND=gemini variant for the two moonshine outputs that
#     branch on LLM_BACKEND inside handler_factory.py.
# ---------------------------------------------------------------------------


def _factory_matrix() -> list[tuple[str, str, str]]:
    """Build the parametrize list as (input, output, llm_backend)."""
    base = [(i, o, LLM_BACKEND_LLAMA) for (i, o) in sorted(_SUPPORTED_AUDIO_COMBINATIONS)]
    gemini_variants = [(AUDIO_INPUT_MOONSHINE, out, LLM_BACKEND_GEMINI) for out in _MOONSHINE_OUTPUTS_BRANCHING_ON_LLM]
    return base + gemini_variants


# ---------------------------------------------------------------------------
# 1. Factory instantiation matrix
# ---------------------------------------------------------------------------


class TestFactoryInstantiationMatrix:
    """HandlerFactory.build returns a real handler instance for every supported combo."""

    @pytest.mark.parametrize("input_b, output_b, llm_backend", _factory_matrix())
    def test_build_returns_instance_without_raising(
        self,
        input_b: str,
        output_b: str,
        llm_backend: str,
    ) -> None:
        deps = make_tool_deps()
        with patch("robot_comic.handler_factory.config") as mock_cfg:
            mock_cfg.LLM_BACKEND = llm_backend
            mock_cfg.XTTS_URL = XTTS_DEFAULT_URL
            mock_cfg.XTTS_DEFAULT_SPEAKER_KEY = XTTS_DEFAULT_SPEAKER
            mock_cfg.XTTS_LANGUAGE = XTTS_DEFAULT_LANGUAGE
            mock_cfg.XTTS_TIMEOUT_S = XTTS_DEFAULT_TIMEOUT_S
            handler = HandlerFactory.build(input_b, output_b, deps)

        expected_name = _expected_handler_name(input_b, output_b, llm_backend)
        assert type(handler).__name__ == expected_name, (
            f"Expected {expected_name} for ({input_b!r}, {output_b!r}, llm={llm_backend!r}); "
            f"got {type(handler).__name__}"
        )


# ---------------------------------------------------------------------------
# 2. Lifecycle smoke
# ---------------------------------------------------------------------------


def _install_fake_moonshine_voice() -> tuple[MagicMock, Any]:
    """Install a fake ``moonshine_voice`` module into ``sys.modules``.

    Returns ``(Transcriber_mock, fake_module)``.  Caller is responsible for
    restoring ``sys.modules`` (use ``monkeypatch`` for that).
    """
    fake_module = SimpleNamespace()
    transcriber_mock = MagicMock(name="Transcriber")
    transcriber_mock.return_value = SimpleNamespace(
        create_stream=MagicMock(
            return_value=SimpleNamespace(
                add_listener=MagicMock(),
                start=MagicMock(),
            )
        )
    )

    fake_module.Transcriber = transcriber_mock
    fake_module.TranscriptEventListener = type("TranscriptEventListener", (), {})

    class _ModelArch:
        TINY_STREAMING = "tiny_streaming"
        SMALL_STREAMING = "small_streaming"

    fake_module.ModelArch = _ModelArch

    def _get_model_for_language(language: str, *, wanted_model_arch: Any = None, cache_root: Path) -> tuple[Path, Any]:
        # Return a directory that actually exists so the smoke test asserts
        # ``isinstance(model_path, str)`` AND the path is a directory.
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        return Path(cache_root), wanted_model_arch or _ModelArch.TINY_STREAMING

    fake_module.get_model_for_language = _get_model_for_language

    sys.modules["moonshine_voice"] = fake_module  # type: ignore[assignment]
    return transcriber_mock, fake_module


def _build_handler_with_mocks(
    input_b: str,
    output_b: str,
    llm_backend: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, MagicMock | None]:
    """Build a handler with all external SDKs / heavy paths mocked.

    Returns ``(handler, transcriber_mock_or_None)``.
    """
    # 1. genai.Client — used by gemini_tts, elevenlabs_tts, gemini_text_*.
    monkeypatch.setattr("google.genai.Client", MagicMock(name="genai.Client"))

    # 2. httpx.AsyncClient — used by llama_base, chatterbox_tts, elevenlabs_tts.
    fake_http = MagicMock(name="AsyncClient")
    fake_http.return_value.aclose = AsyncMock()
    fake_http.return_value.get = AsyncMock(return_value=SimpleNamespace(status_code=200, json=lambda: {}))
    fake_http.return_value.post = AsyncMock(
        return_value=SimpleNamespace(status_code=200, content=b"", json=lambda: {})
    )
    monkeypatch.setattr("httpx.AsyncClient", fake_http)

    # 3. Cap llama-server health probes — already disabled via env in conftest
    #    but be explicit so ChatterboxTTS startup doesn't sit for 3 seconds.
    monkeypatch.setenv("REACHY_MINI_LLAMA_HEALTH_CHECK", "0")
    monkeypatch.setenv("CHATTERBOX_WARMUP_ENABLED", "0")

    # 4. Force LOCAL_STT_CACHE_DIR to a real directory so ``Transcriber`` is
    #    called with a string that points to an extant directory.
    monkeypatch.setenv("LOCAL_STT_CACHE_DIR", str(tmp_path))
    # Refresh config so the new env var is picked up.
    from robot_comic.config import refresh_runtime_config_from_env

    refresh_runtime_config_from_env()

    transcriber_mock: MagicMock | None = None
    if input_b == AUDIO_INPUT_MOONSHINE:
        transcriber_mock, _ = _install_fake_moonshine_voice()
        # Ensure no real moonshine_voice gets imported.
        monkeypatch.setitem(sys.modules, "moonshine_voice", sys.modules["moonshine_voice"])

    deps = make_tool_deps()
    with patch("robot_comic.handler_factory.config") as mock_cfg:
        mock_cfg.LLM_BACKEND = llm_backend
        mock_cfg.XTTS_URL = XTTS_DEFAULT_URL
        mock_cfg.XTTS_DEFAULT_SPEAKER_KEY = XTTS_DEFAULT_SPEAKER
        mock_cfg.XTTS_LANGUAGE = XTTS_DEFAULT_LANGUAGE
        mock_cfg.XTTS_TIMEOUT_S = XTTS_DEFAULT_TIMEOUT_S
        handler = HandlerFactory.build(input_b, output_b, deps)

    return handler, transcriber_mock


class TestLifecycleSmoke:
    """For each factory product, _prepare_startup_credentials() must complete cleanly."""

    @pytest.mark.parametrize("input_b, output_b, llm_backend", _factory_matrix())
    @pytest.mark.asyncio
    async def test_prepare_startup_credentials_does_not_raise(
        self,
        input_b: str,
        output_b: str,
        llm_backend: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        handler, transcriber_mock = _build_handler_with_mocks(input_b, output_b, llm_backend, monkeypatch, tmp_path)

        if not hasattr(handler, "_prepare_startup_credentials"):
            # GeminiLiveHandler does not inherit BaseRealtimeHandler and so has
            # no _prepare_startup_credentials hook; the equivalent credential
            # plumbing happens inside start_up().  Nothing to smoke-check here
            # beyond construction (covered by the factory matrix test).
            pytest.skip(f"{type(handler).__name__} has no _prepare_startup_credentials hook")

        await handler._prepare_startup_credentials()

        # Moonshine-based handlers MUST have called Transcriber(model_path=str(<dir>)).
        if input_b == AUDIO_INPUT_MOONSHINE and transcriber_mock is not None:
            assert transcriber_mock.call_count >= 1, (
                "Moonshine handlers must call Transcriber during _prepare_startup_credentials"
            )
            call_kwargs = transcriber_mock.call_args.kwargs
            model_path_arg = call_kwargs.get("model_path")
            assert isinstance(model_path_arg, str), (
                f"Transcriber model_path must be str (not Path) — Moonshine .encode()s it. "
                f"Got {type(model_path_arg).__name__}: {model_path_arg!r}"
            )
            resolved = Path(model_path_arg)
            assert resolved.is_dir() or resolved.parent.exists(), (
                f"Transcriber model_path {model_path_arg!r} must be a directory or have an extant parent; "
                "this catches the regression where model_path is a non-existent file."
            )

    @pytest.mark.parametrize("input_b, output_b, llm_backend", _factory_matrix())
    @pytest.mark.asyncio
    async def test_get_current_voice_returns_string(
        self,
        input_b: str,
        output_b: str,
        llm_backend: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``get_current_voice`` must return a string (not raise NotImplementedError).

        Catches #260 bug #3 where BaseLlamaResponseHandler.get_current_voice (which raises
        NotImplementedError) was found before ElevenLabsTTSResponseHandler.get_current_voice
        in the MRO.
        """
        handler, _ = _build_handler_with_mocks(input_b, output_b, llm_backend, monkeypatch, tmp_path)
        # _prepare_startup_credentials needs to run for some handlers (e.g.
        # ElevenLabs sets _client before voice resolution).  But for the voice
        # MRO check we only need the constructor to have set _voice_override
        # and any class-level defaults, which it does.
        assert callable(handler.get_current_voice), f"{type(handler).__name__}.get_current_voice must be callable"
        try:
            voice = handler.get_current_voice()
        except NotImplementedError as exc:
            pytest.fail(
                f"{type(handler).__name__}.get_current_voice raised NotImplementedError: "
                f"{exc}. This is the #260 bug #3 regression — MRO must resolve to a concrete "
                "subclass before reaching BaseLlamaResponseHandler.get_current_voice."
            )
        assert isinstance(voice, str), (
            f"{type(handler).__name__}.get_current_voice must return str, got {type(voice).__name__}"
        )


# ---------------------------------------------------------------------------
# 3. Config-resolution matrix
#
# Parametrise over the documented matrix in docs/audio-backends.md plus the
# legacy response_backend values.  Uses the post-#262 behaviour:
# derive_audio_backends takes an optional ``response_backend`` kwarg that maps
# the legacy local_stt response selector to the canonical AUDIO_OUTPUT_*.
# ---------------------------------------------------------------------------


class TestConfigResolutionMatrix:
    """resolve_audio_backends + derive_audio_backends route the env-var matrix correctly."""

    # ----- pure derivation from provider_id (no overrides, no response selector)

    @pytest.mark.parametrize(
        "backend_provider, expected",
        [
            (HF_BACKEND, (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)),
            (OPENAI_BACKEND, (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME)),
            (GEMINI_BACKEND, (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE)),
            (LOCAL_STT_BACKEND, (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)),
        ],
    )
    def test_derive_from_backend_provider_default(self, backend_provider: str, expected: tuple[str, str]) -> None:
        result = derive_audio_backends(backend_provider)
        assert result == expected, f"derive_audio_backends({backend_provider!r}) → {result}, expected {expected}"

    # ----- post-#262: derive_audio_backends honours response_backend

    @pytest.mark.parametrize(
        "response_backend, expected_output",
        [
            (CHATTERBOX_OUTPUT, AUDIO_OUTPUT_CHATTERBOX),
            (ELEVENLABS_OUTPUT, AUDIO_OUTPUT_ELEVENLABS),
            (LLAMA_ELEVENLABS_TTS_OUTPUT, AUDIO_OUTPUT_ELEVENLABS),
            (GEMINI_TTS_OUTPUT, AUDIO_OUTPUT_GEMINI_TTS),
            (OPENAI_BACKEND, AUDIO_OUTPUT_OPENAI_REALTIME),
            (HF_RESPONSE_BACKEND, AUDIO_OUTPUT_HF),
        ],
    )
    def test_local_stt_response_backend_maps_to_output(self, response_backend: str, expected_output: str) -> None:
        result = derive_audio_backends(LOCAL_STT_BACKEND, response_backend=response_backend)
        assert result == (AUDIO_INPUT_MOONSHINE, expected_output), (
            f"derive_audio_backends(local_stt, response_backend={response_backend!r}) → "
            f"{result}, expected (moonshine, {expected_output})"
        )

    def test_local_stt_response_backend_unset_falls_back_to_chatterbox(self) -> None:
        result = derive_audio_backends(LOCAL_STT_BACKEND, response_backend=None)
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)

    def test_local_stt_response_backend_unknown_falls_back_to_chatterbox(self) -> None:
        result = derive_audio_backends(LOCAL_STT_BACKEND, response_backend="totally_bogus")
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)

    def test_non_local_stt_provider_ignores_response_backend(self) -> None:
        """response_backend only applies to LOCAL_STT_BACKEND; others derive normally."""
        result = derive_audio_backends(HF_BACKEND, response_backend=ELEVENLABS_OUTPUT)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)

    # ----- resolve_audio_backends — explicit overrides win for supported pairs

    @pytest.mark.parametrize(
        "input_b, output_b",
        sorted(_SUPPORTED_AUDIO_COMBINATIONS),
    )
    def test_supported_pair_explicit_overrides_win(self, input_b: str, output_b: str) -> None:
        """Every supported pair from _SUPPORTED_AUDIO_COMBINATIONS must be accepted as-is."""
        # Use a deliberately mismatched provider so we can prove the explicit pair won.
        result = resolve_audio_backends(LOCAL_STT_BACKEND, input_b, output_b)
        assert result == (input_b, output_b), (
            f"Explicit ({input_b!r}, {output_b!r}) must win over provider_id fallback"
        )

    def test_unsupported_pair_falls_back_to_derived(self, caplog: pytest.LogCaptureFixture) -> None:
        """An unsupported explicit pair logs WARNING and falls back to derived defaults."""
        import logging

        with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
            result = resolve_audio_backends(GEMINI_BACKEND, AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_CHATTERBOX)
        assert result == (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE)
        assert any("Unsupported" in r.message for r in caplog.records)

    def test_partial_override_falls_back_to_derived(self, caplog: pytest.LogCaptureFixture) -> None:
        """When only one of the two AUDIO_*_BACKEND env vars is set, fall back."""
        import logging

        with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
            result = resolve_audio_backends(HF_BACKEND, AUDIO_INPUT_HF, None)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)
        assert any("Partial" in r.message for r in caplog.records)

    def test_no_overrides_returns_derived(self) -> None:
        """Neither override set → pure provider_id derivation."""
        result = resolve_audio_backends(OPENAI_BACKEND, None, None)
        assert result == (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME)

    # ----- the cross-product that resolve_audio_backends ends up using in production:
    # provider_id=local_stt with a response_backend env, no explicit pair.

    @pytest.mark.parametrize(
        "response_backend, expected_output",
        [
            (CHATTERBOX_OUTPUT, AUDIO_OUTPUT_CHATTERBOX),
            (ELEVENLABS_OUTPUT, AUDIO_OUTPUT_ELEVENLABS),
            (GEMINI_TTS_OUTPUT, AUDIO_OUTPUT_GEMINI_TTS),
            (OPENAI_BACKEND, AUDIO_OUTPUT_OPENAI_REALTIME),
            (HF_RESPONSE_BACKEND, AUDIO_OUTPUT_HF),
        ],
    )
    def test_resolve_local_stt_with_response_backend(self, response_backend: str, expected_output: str) -> None:
        """End-to-end: resolve_audio_backends forwards response_backend through to derive_audio_backends."""
        result = resolve_audio_backends(
            LOCAL_STT_BACKEND,
            None,
            None,
            response_backend=response_backend,
        )
        assert result == (AUDIO_INPUT_MOONSHINE, expected_output), (
            f"resolve_audio_backends(local_stt, None, None, response_backend={response_backend!r}) "
            f"→ {result}, expected (moonshine, {expected_output})"
        )


# Tag the whole module as integration so it is opt-in (matches existing harness).
pytestmark = pytest.mark.integration
