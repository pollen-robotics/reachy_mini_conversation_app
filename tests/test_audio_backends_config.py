"""Tests for the modular audio-pipeline config scaffold (issue #54).

Covers:
- derive_audio_backends() derivation from provider_id values.
- resolve_audio_backends() with explicit env-var overrides (happy path).
- Fallback behaviour for unsupported combinations.
- Fallback behaviour for partial overrides (only one of the two set).
- Config class attributes AUDIO_INPUT_BACKEND / AUDIO_OUTPUT_BACKEND reflect
  the env vars at class-load time.
"""

import os
import logging
from unittest.mock import patch

import pytest

import robot_comic.config as cfg_module
from robot_comic.config import (
    HF_BACKEND,
    AUDIO_INPUT_HF,
    GEMINI_BACKEND,
    OPENAI_BACKEND,
    AUDIO_OUTPUT_HF,
    CHATTERBOX_OUTPUT,
    ELEVENLABS_OUTPUT,
    GEMINI_TTS_OUTPUT,
    LOCAL_STT_BACKEND,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_INPUT_BACKEND_ENV,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_BACKEND_ENV,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_INPUT_OPENAI_REALTIME,
    LLAMA_ELEVENLABS_TTS_OUTPUT,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    derive_audio_backends,
    resolve_audio_backends,
)


# ---------------------------------------------------------------------------
# derive_audio_backends — deterministic derivation from provider_id
# ---------------------------------------------------------------------------


class TestDeriveAudioBackends:
    def test_chatterbox_derives_moonshine_chatterbox(self) -> None:
        # local_stt defaults to moonshine + chatterbox
        result = derive_audio_backends(LOCAL_STT_BACKEND)
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)

    def test_gemini_live_derives_gemini_live_pair(self) -> None:
        result = derive_audio_backends(GEMINI_BACKEND)
        assert result == (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE)

    def test_huggingface_derives_hf_pair(self) -> None:
        result = derive_audio_backends(HF_BACKEND)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)

    def test_openai_derives_openai_pair(self) -> None:
        result = derive_audio_backends(OPENAI_BACKEND)
        assert result == (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME)

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            derive_audio_backends("does_not_exist")


# ---------------------------------------------------------------------------
# derive_audio_backends — response_backend lookup (issue #262)
# ---------------------------------------------------------------------------


class TestDeriveAudioBackendsLocalSttResponseLookup:
    """derive_audio_backends() must map response_backend to the
    correct AUDIO_OUTPUT_* constant per the table in issue #262, rather than
    unconditionally returning chatterbox.
    """

    @pytest.mark.parametrize(
        ("response_backend", "expected_output"),
        [
            (CHATTERBOX_OUTPUT, AUDIO_OUTPUT_CHATTERBOX),
            (ELEVENLABS_OUTPUT, AUDIO_OUTPUT_ELEVENLABS),
            (LLAMA_ELEVENLABS_TTS_OUTPUT, AUDIO_OUTPUT_ELEVENLABS),
            (GEMINI_TTS_OUTPUT, AUDIO_OUTPUT_GEMINI_TTS),
            (OPENAI_BACKEND, AUDIO_OUTPUT_OPENAI_REALTIME),
            (HF_BACKEND, AUDIO_OUTPUT_HF),
        ],
    )
    def test_response_backend_selects_output(
        self,
        response_backend: str,
        expected_output: str,
    ) -> None:
        result = derive_audio_backends(LOCAL_STT_BACKEND, response_backend)
        assert result == (AUDIO_INPUT_MOONSHINE, expected_output)

    def test_response_backend_none_defaults_to_chatterbox(self) -> None:
        # When response_backend is unset (None), keep the historical
        # chatterbox default so existing deployments aren't disturbed.
        result = derive_audio_backends(LOCAL_STT_BACKEND, None)
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)

    def test_response_backend_empty_string_defaults_to_chatterbox(self) -> None:
        result = derive_audio_backends(LOCAL_STT_BACKEND, "")
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)

    def test_response_backend_unknown_value_defaults_to_chatterbox(self) -> None:
        # Unrecognised values shouldn't crash; fall back to the safe default.
        result = derive_audio_backends(LOCAL_STT_BACKEND, "not_a_real_backend")
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_CHATTERBOX)

    def test_response_backend_ignored_for_non_local_stt_providers(self) -> None:
        # response_backend is only consulted when backend_provider==local_stt.
        # For other providers it must be ignored entirely.
        result = derive_audio_backends(HF_BACKEND, ELEVENLABS_OUTPUT)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)


# ---------------------------------------------------------------------------
# resolve_audio_backends — override + validation logic
# ---------------------------------------------------------------------------


class TestResolveAudioBackends:
    """Test resolve_audio_backends() in isolation (no env-var side effects)."""

    # -- Neither set → pure derivation from provider

    def test_neither_set_returns_derived(self) -> None:
        result = resolve_audio_backends(GEMINI_BACKEND, None, None)
        assert result == (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE)

    # -- Both set to a supported combination → override wins

    def test_supported_combo_moonshine_gemini_tts(self) -> None:
        result = resolve_audio_backends(LOCAL_STT_BACKEND, AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_GEMINI_TTS)
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_GEMINI_TTS)

    def test_supported_combo_moonshine_elevenlabs(self) -> None:
        result = resolve_audio_backends(LOCAL_STT_BACKEND, AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_ELEVENLABS)
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_ELEVENLABS)

    def test_supported_combo_hf_pair(self) -> None:
        result = resolve_audio_backends(HF_BACKEND, AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)

    # -- Both set to an unsupported combination → WARNING + fallback to derived

    def test_unsupported_combo_logs_warning_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        # gemini_live_input + chatterbox is not a supported pair
        with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
            result = resolve_audio_backends(
                GEMINI_BACKEND,
                AUDIO_INPUT_GEMINI_LIVE,
                AUDIO_OUTPUT_CHATTERBOX,
            )
        assert result == (AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE), (
            "Should fall back to provider_id-derived defaults"
        )
        assert any("Unsupported" in r.message for r in caplog.records), (
            "Expected a WARNING about unsupported combination"
        )

    def test_realtime_input_chatterbox_output_is_unsupported(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
            result = resolve_audio_backends(
                OPENAI_BACKEND,
                AUDIO_INPUT_OPENAI_REALTIME,
                AUDIO_OUTPUT_CHATTERBOX,
            )
        # Fallback to derived openai pair
        assert result == (AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME)
        assert any("Unsupported" in r.message for r in caplog.records)

    # -- Only one of the two set → WARNING + fallback

    def test_only_input_set_logs_warning_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
            result = resolve_audio_backends(HF_BACKEND, AUDIO_INPUT_HF, None)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)
        assert any("Partial" in r.message for r in caplog.records)

    def test_only_output_set_logs_warning_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
            result = resolve_audio_backends(HF_BACKEND, None, AUDIO_OUTPUT_CHATTERBOX)
        assert result == (AUDIO_INPUT_HF, AUDIO_OUTPUT_HF)
        assert any("Partial" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Config class attributes — env-var overrides applied at class-definition time.
# We monkey-patch the module-level resolved values rather than re-importing
# (which would re-execute module-level code with side effects).
# ---------------------------------------------------------------------------


class TestConfigAttributes:
    """Verify that Config.AUDIO_INPUT_BACKEND / AUDIO_OUTPUT_BACKEND reflect
    the result of resolve_audio_backends at the time the class body ran.
    These tests inspect the *already-resolved* values rather than reloading the
    module (which would re-trigger dotenv loading and other side effects)."""

    def test_config_has_audio_input_backend_attr(self) -> None:
        assert hasattr(cfg_module.config, "AUDIO_INPUT_BACKEND")
        assert cfg_module.config.AUDIO_INPUT_BACKEND in cfg_module.AUDIO_INPUT_CHOICES

    def test_config_has_audio_output_backend_attr(self) -> None:
        assert hasattr(cfg_module.config, "AUDIO_OUTPUT_BACKEND")
        assert cfg_module.config.AUDIO_OUTPUT_BACKEND in cfg_module.AUDIO_OUTPUT_CHOICES

    def test_resolve_with_both_env_vars_overrides(self) -> None:
        """resolve_audio_backends honours both env vars when a supported combo."""
        result = resolve_audio_backends(
            LOCAL_STT_BACKEND,
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_GEMINI_TTS,
        )
        assert result == (AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_GEMINI_TTS)

    def test_refresh_picks_up_env_vars(self) -> None:
        """refresh_runtime_config_from_env updates AUDIO_INPUT/OUTPUT_BACKEND."""
        original_input = cfg_module.config.AUDIO_INPUT_BACKEND
        original_output = cfg_module.config.AUDIO_OUTPUT_BACKEND
        try:
            # Patch env to force a valid supported pair
            with patch.dict(
                os.environ,
                {
                    AUDIO_INPUT_BACKEND_ENV: AUDIO_INPUT_MOONSHINE,
                    AUDIO_OUTPUT_BACKEND_ENV: AUDIO_OUTPUT_CHATTERBOX,
                    "REACHY_MINI_SKIP_DOTENV": "1",
                },
            ):
                cfg_module.refresh_runtime_config_from_env()
                assert cfg_module.config.AUDIO_INPUT_BACKEND == AUDIO_INPUT_MOONSHINE
                assert cfg_module.config.AUDIO_OUTPUT_BACKEND == AUDIO_OUTPUT_CHATTERBOX
        finally:
            # Restore originals so later tests are not polluted
            cfg_module.config.AUDIO_INPUT_BACKEND = original_input
            cfg_module.config.AUDIO_OUTPUT_BACKEND = original_output
