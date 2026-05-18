"""Factory-matrix tests for the (local-STT, *, xtts) composable triples (#438).

Verifies that ``HandlerFactory.build()`` correctly routes every supported
(input_backend, xtts) combination to a :class:`ComposableConversationHandler`
whose pipeline TTS backend is an :class:`XttsTTSAdapter` instance.

Coverage:
- (faster_whisper, xtts) + LLM_BACKEND=llama → ComposableConversationHandler,
  pipeline.tts is XttsTTSAdapter, _tts_handler is XttsTTSAdapter.
- (moonshine, xtts) + LLM_BACKEND=llama → same shape.
- (faster_whisper, xtts) + LLM_BACKEND=gemini → XttsTTSAdapter.
- (moonshine, xtts) + LLM_BACKEND=gemini → XttsTTSAdapter.
- Unsupported combo (openai_realtime_input, xtts) → NotImplementedError with
  the supported-combos help text.

All handler __init__ methods that touch network or hardware are left to their
own lazy-import paths; the MagicMock deps object satisfies the
``ToolDependencies`` surface. Config attributes are patched via monkeypatch so
tests don't depend on the current machine's env vars.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from robot_comic.config import (
    AUDIO_OUTPUT_XTTS,
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_MOONSHINE,
    PIPELINE_MODE_COMPOSABLE,
    AUDIO_INPUT_FASTER_WHISPER,
    AUDIO_INPUT_OPENAI_REALTIME,
)
from robot_comic.handler_factory import HandlerFactory
from robot_comic.adapters.xtts_tts_adapter import XttsTTSAdapter
from robot_comic.composable_conversation_handler import ComposableConversationHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_deps() -> MagicMock:
    """Return a mock ToolDependencies object — mirrors other factory test files."""
    return MagicMock(name="ToolDependencies")


# ---------------------------------------------------------------------------
# Llama LLM + xtts TTS
# ---------------------------------------------------------------------------


class TestLlamaXttsCombinations:
    """HandlerFactory produces a composable pipeline with XttsTTSAdapter for llama+xtts."""

    def _build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_deps: MagicMock,
        input_backend: str,
    ) -> ComposableConversationHandler:
        from robot_comic import config as cfg_mod

        monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)

        result = HandlerFactory.build(
            input_backend,
            AUDIO_OUTPUT_XTTS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
        return result  # type: ignore[return-value]

    def test_faster_whisper_xtts_llama_returns_composable(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(faster_whisper, xtts) + llama → ComposableConversationHandler."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_FASTER_WHISPER)
        assert isinstance(result, ComposableConversationHandler)

    def test_faster_whisper_xtts_llama_pipeline_tts_is_xtts_adapter(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(faster_whisper, xtts) + llama → pipeline.tts is XttsTTSAdapter."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_FASTER_WHISPER)
        assert isinstance(result.pipeline.tts, XttsTTSAdapter)

    def test_faster_whisper_xtts_llama_tts_handler_is_xtts_adapter(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(faster_whisper, xtts) + llama → _tts_handler is the XttsTTSAdapter instance."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_FASTER_WHISPER)
        assert isinstance(result._tts_handler, XttsTTSAdapter)

    def test_moonshine_xtts_llama_returns_composable(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(moonshine, xtts) + llama → ComposableConversationHandler."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        assert isinstance(result, ComposableConversationHandler)

    def test_moonshine_xtts_llama_pipeline_tts_is_xtts_adapter(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(moonshine, xtts) + llama → pipeline.tts is XttsTTSAdapter."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        assert isinstance(result.pipeline.tts, XttsTTSAdapter)

    def test_moonshine_xtts_llama_tool_dispatcher_is_wired(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(moonshine, xtts) + llama → tool_dispatcher is set (phase-5b regression guard)."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        assert result.pipeline.tool_dispatcher is not None

    def test_xtts_adapter_uses_config_values(self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock) -> None:
        """The XttsTTSAdapter is constructed from the live config values."""
        from robot_comic import config as cfg_mod

        monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
        monkeypatch.setattr(cfg_mod.config, "XTTS_URL", "http://test-xtts.lan:9999")
        monkeypatch.setattr(cfg_mod.config, "XTTS_DEFAULT_SPEAKER_KEY", "test_speaker")
        monkeypatch.setattr(cfg_mod.config, "XTTS_LANGUAGE", "fr")
        monkeypatch.setattr(cfg_mod.config, "XTTS_TIMEOUT_S", 15.0)

        result = HandlerFactory.build(
            AUDIO_INPUT_FASTER_WHISPER,
            AUDIO_OUTPUT_XTTS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

        tts = result.pipeline.tts
        assert isinstance(tts, XttsTTSAdapter)
        assert tts._base_url == "http://test-xtts.lan:9999"
        assert tts.get_current_voice() == "test_speaker"
        assert tts._language == "fr"
        assert tts._timeout_s == 15.0


# ---------------------------------------------------------------------------
# Gemini LLM + xtts TTS
# ---------------------------------------------------------------------------


class TestGeminiXttsCombinations:
    """HandlerFactory produces a composable pipeline with XttsTTSAdapter for gemini+xtts."""

    def _build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_deps: MagicMock,
        input_backend: str,
    ) -> ComposableConversationHandler:
        from robot_comic import config as cfg_mod

        monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)

        result = HandlerFactory.build(
            input_backend,
            AUDIO_OUTPUT_XTTS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
        return result  # type: ignore[return-value]

    def test_faster_whisper_xtts_gemini_returns_composable(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(faster_whisper, xtts) + gemini → ComposableConversationHandler."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_FASTER_WHISPER)
        assert isinstance(result, ComposableConversationHandler)

    def test_faster_whisper_xtts_gemini_pipeline_tts_is_xtts_adapter(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(faster_whisper, xtts) + gemini → pipeline.tts is XttsTTSAdapter."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_FASTER_WHISPER)
        assert isinstance(result.pipeline.tts, XttsTTSAdapter)

    def test_faster_whisper_xtts_gemini_tts_handler_is_xtts_adapter(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(faster_whisper, xtts) + gemini → _tts_handler is the XttsTTSAdapter instance."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_FASTER_WHISPER)
        assert isinstance(result._tts_handler, XttsTTSAdapter)

    def test_moonshine_xtts_gemini_returns_composable(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(moonshine, xtts) + gemini → ComposableConversationHandler."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        assert isinstance(result, ComposableConversationHandler)

    def test_moonshine_xtts_gemini_pipeline_tts_is_xtts_adapter(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(moonshine, xtts) + gemini → pipeline.tts is XttsTTSAdapter."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        assert isinstance(result.pipeline.tts, XttsTTSAdapter)

    def test_moonshine_xtts_gemini_tool_dispatcher_is_wired(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """(moonshine, xtts) + gemini → tool_dispatcher is set (phase-5b regression guard)."""
        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        assert result.pipeline.tool_dispatcher is not None

    def test_gemini_llm_host_is_gemini_text_handler(
        self, monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
    ) -> None:
        """gemini+xtts triple wraps a GeminiTextChatterboxResponseHandler as LLM host."""
        from robot_comic.gemini_text_handlers import GeminiTextChatterboxResponseHandler

        result = self._build(monkeypatch, mock_deps, AUDIO_INPUT_MOONSHINE)
        # The LLM adapter wraps the gemini host; verify via the adapter's handler.
        llm_adapter = result.pipeline.llm
        assert isinstance(llm_adapter._handler, GeminiTextChatterboxResponseHandler)


# ---------------------------------------------------------------------------
# Unsupported combinations — NotImplementedError
# ---------------------------------------------------------------------------


class TestXttsUnsupportedCombinations:
    """Unsupported (input, xtts) pairs raise NotImplementedError with the help text."""

    def test_openai_realtime_input_xtts_output_raises(self, mock_deps: MagicMock) -> None:
        """(openai_realtime_input, xtts) is not a supported combo → NotImplementedError."""
        with pytest.raises(NotImplementedError):
            HandlerFactory.build(
                AUDIO_INPUT_OPENAI_REALTIME,
                AUDIO_OUTPUT_XTTS,
                mock_deps,
            )

    def test_error_message_contains_xtts_combos(self, mock_deps: MagicMock) -> None:
        """The NotImplementedError message lists the (moonshine, xtts) and
        (faster_whisper, xtts) supported combos so operators can self-serve."""
        with pytest.raises(NotImplementedError) as exc_info:
            HandlerFactory.build(
                AUDIO_INPUT_OPENAI_REALTIME,
                AUDIO_OUTPUT_XTTS,
                mock_deps,
            )
        msg = str(exc_info.value)
        assert AUDIO_OUTPUT_XTTS in msg
        assert AUDIO_INPUT_MOONSHINE in msg
        assert AUDIO_INPUT_FASTER_WHISPER in msg

    def test_error_message_references_docs(self, mock_deps: MagicMock) -> None:
        """The NotImplementedError message references docs/audio-backends.md."""
        with pytest.raises(NotImplementedError) as exc_info:
            HandlerFactory.build(
                AUDIO_INPUT_OPENAI_REALTIME,
                AUDIO_OUTPUT_XTTS,
                mock_deps,
            )
        assert "docs/audio-backends.md" in str(exc_info.value)
