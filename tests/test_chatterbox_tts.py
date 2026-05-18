"""Tests for :class:`ChatterboxTTSResponseHandler` lifecycle.

Most chatterbox unit-test coverage lives in adjacent files
(:mod:`test_chatterbox_auto_gain`, :mod:`test_chatterbox_voice_clone`,
:mod:`test_llama_health_check`, :mod:`test_llm_warmup`). This module
captures the Phase 5e.3 idempotency guard on
:meth:`ChatterboxTTSResponseHandler._prepare_startup_credentials`,
the Phase 5e.4 sibling guard on the leaf
:class:`GeminiTextChatterboxResponseHandler` (which inherits from the
chatterbox handler but adds its own ``self._gemini_llm`` reassignment),
and the non-chatterbox-backend skip guard (issue #438/PR fix) that
prevents spurious ConnectTimeout warnings when the active TTS backend is
something other than chatterbox (e.g. ``AUDIO_OUTPUT_BACKEND=xtts``).
"""

from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import robot_comic.config as cfg_module
from robot_comic.config import AUDIO_OUTPUT_CHATTERBOX
from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


# ---------------------------------------------------------------------------
# Phase 5e.3 — _prepare_startup_credentials idempotency guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_startup_credentials_is_idempotent() -> None:
    """Second call must NOT rebuild ``_http`` or re-run the warmup chain.

    Pre-5e.3, the idempotency guard lived on
    :meth:`LocalSTTInputMixin._prepare_startup_credentials` (the host
    shell wraps the handler's prepare). Post-5e.3 the migrated triple's
    factory composes a plain handler — no shell — and the LLM and TTS
    adapters each call ``handler._prepare_startup_credentials`` once.
    Without a per-handler guard the second call leaks an extra
    ``httpx.AsyncClient`` and re-fires the llama-server health probe +
    KV-cache warmup (a streaming POST) + Chatterbox TTS warmup (a
    real TTS round-trip).

    This test forces ``AUDIO_OUTPUT_BACKEND=chatterbox`` so the probes
    fire — the non-chatterbox skip guard is tested separately below.
    """
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    original_output = cfg_module.config.AUDIO_OUTPUT_BACKEND
    cfg_module.config.AUDIO_OUTPUT_BACKEND = AUDIO_OUTPUT_CHATTERBOX
    try:
        handler = ChatterboxTTSResponseHandler(_make_deps())
        handler.tool_manager = MagicMock()
        handler._probe_llama_health = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_llm_kv_cache = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_tts = AsyncMock()  # type: ignore[method-assign]

        await handler._prepare_startup_credentials()
        first_http = handler._http
        assert first_http is not None
        assert handler._probe_llama_health.await_count == 1
        assert handler._warmup_llm_kv_cache.await_count == 1
        assert handler._warmup_tts.await_count == 1

        await handler._prepare_startup_credentials()
        # Same client; no leak. Warmups not re-run.
        assert handler._http is first_http
        assert handler._probe_llama_health.await_count == 1
        assert handler._warmup_llm_kv_cache.await_count == 1
        assert handler._warmup_tts.await_count == 1
    finally:
        cfg_module.config.AUDIO_OUTPUT_BACKEND = original_output


# ---------------------------------------------------------------------------
# Phase 5e.4 — leaf guard on GeminiTextChatterboxResponseHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_chatterbox_prepare_startup_credentials_is_idempotent() -> None:
    """Second call must NOT re-instantiate ``GeminiLLMClient``.

    Pre-5e.4, the mixin shell's ``_prepare_startup_credentials`` gated
    the whole chain. Post-5e.4 the migrated triple's factory composes
    a plain leaf handler (no shell), and the LLM and TTS adapters each
    call ``handler._prepare_startup_credentials`` once. The inherited
    chatterbox guard (5e.3) short-circuits the chained
    ``ChatterboxTTSResponseHandler._prepare_startup_credentials`` call
    on the second invocation, but without a leaf-level guard the
    leaf-body's ``self._gemini_llm = GeminiLLMClient(...)`` reassignment
    still runs every time — leaking a fresh ``genai.Client`` per
    duplicate call.

    This test forces ``AUDIO_OUTPUT_BACKEND=chatterbox`` so the probes
    fire — the non-chatterbox skip guard is tested separately below.
    """
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxResponseHandler

    original_output = cfg_module.config.AUDIO_OUTPUT_BACKEND
    cfg_module.config.AUDIO_OUTPUT_BACKEND = AUDIO_OUTPUT_CHATTERBOX
    try:
        handler = GeminiTextChatterboxResponseHandler(_make_deps())
        handler.tool_manager = MagicMock()
        handler._probe_llama_health = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_llm_kv_cache = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_tts = AsyncMock()  # type: ignore[method-assign]

        with patch("robot_comic.gemini_llm.GeminiLLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock(name="GeminiLLMClient_instance")
            await handler._prepare_startup_credentials()
            first_client = handler._gemini_llm
            assert first_client is not None
            assert mock_client_cls.call_count == 1
            assert handler._probe_llama_health.await_count == 1
            assert handler._warmup_llm_kv_cache.await_count == 1
            assert handler._warmup_tts.await_count == 1

            await handler._prepare_startup_credentials()
            # Same Gemini client instance; no SDK-client leak.
            assert handler._gemini_llm is first_client
            assert mock_client_cls.call_count == 1
            # Inherited chatterbox guard still short-circuits its warmups.
            assert handler._probe_llama_health.await_count == 1
            assert handler._warmup_llm_kv_cache.await_count == 1
            assert handler._warmup_tts.await_count == 1
    finally:
        cfg_module.config.AUDIO_OUTPUT_BACKEND = original_output


# ---------------------------------------------------------------------------
# Non-chatterbox backend skip guard (PR fix for issue #438)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_startup_credentials_skips_probes_when_not_chatterbox_backend() -> None:
    """Probes must NOT fire when ``AUDIO_OUTPUT_BACKEND`` is not ``chatterbox``.

    Root cause: when ``AUDIO_OUTPUT_BACKEND=xtts`` (or any non-chatterbox
    backend), :class:`GeminiTextChatterboxResponseHandler` is used as the
    LLM host in the composable pipeline.  Its
    ``_prepare_startup_credentials`` chain always calls
    ``ChatterboxTTSResponseHandler._prepare_startup_credentials``, which
    fired the llama-server health probe + LLM KV-cache warmup + TTS warmup
    even though Chatterbox is not the active TTS.  On every boot this
    produced ConnectTimeout warnings like::

        WARNING robot_comic.chatterbox_tts | TTS attempt 1/3 failed: ConnectTimeout
        WARNING robot_comic.chatterbox_tts | TTS attempt 2/3 failed: ConnectTimeout
        WARNING robot_comic.chatterbox_tts | TTS attempt 3/3 failed: ConnectTimeout

    Fix: gate the probes behind ``config.AUDIO_OUTPUT_BACKEND == "chatterbox"``.
    The httpx client setup (``super()._prepare_startup_credentials()``) still
    runs — the LLM host needs it regardless of TTS backend.
    """
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler

    original_output = cfg_module.config.AUDIO_OUTPUT_BACKEND
    cfg_module.config.AUDIO_OUTPUT_BACKEND = "xtts"
    try:
        handler = ChatterboxTTSResponseHandler(_make_deps())
        handler.tool_manager = MagicMock()
        handler._probe_llama_health = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_llm_kv_cache = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_tts = AsyncMock()  # type: ignore[method-assign]

        await handler._prepare_startup_credentials()

        # The base httpx client must be set up regardless of output backend
        # (the LLM host needs it for llama-server calls).
        assert handler._http is not None
        # None of the Chatterbox-specific probes must fire.
        assert handler._probe_llama_health.await_count == 0, (
            "_probe_llama_health fired even though AUDIO_OUTPUT_BACKEND=xtts"
        )
        assert handler._warmup_llm_kv_cache.await_count == 0, (
            "_warmup_llm_kv_cache fired even though AUDIO_OUTPUT_BACKEND=xtts"
        )
        assert handler._warmup_tts.await_count == 0, "_warmup_tts fired even though AUDIO_OUTPUT_BACKEND=xtts"
    finally:
        cfg_module.config.AUDIO_OUTPUT_BACKEND = original_output


@pytest.mark.asyncio
async def test_gemini_chatterbox_skips_probes_when_not_chatterbox_backend() -> None:
    """Probes must NOT fire for ``GeminiTextChatterboxResponseHandler`` when backend is not chatterbox.

    The diamond-MRO leaf used as the LLM host in ``_build_composable_gemini_xtts``
    chains through ``ChatterboxTTSResponseHandler._prepare_startup_credentials``.
    When ``AUDIO_OUTPUT_BACKEND=xtts`` the skip guard must prevent the
    chatterbox probes from firing, while still completing the Gemini LLM
    client initialisation.
    """
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxResponseHandler

    original_output = cfg_module.config.AUDIO_OUTPUT_BACKEND
    cfg_module.config.AUDIO_OUTPUT_BACKEND = "xtts"
    try:
        handler = GeminiTextChatterboxResponseHandler(_make_deps())
        handler.tool_manager = MagicMock()
        handler._probe_llama_health = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_llm_kv_cache = AsyncMock()  # type: ignore[method-assign]
        handler._warmup_tts = AsyncMock()  # type: ignore[method-assign]

        with patch("robot_comic.gemini_llm.GeminiLLMClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock(name="GeminiLLMClient_instance")
            await handler._prepare_startup_credentials()

        # httpx client initialised (base class runs regardless of backend).
        assert handler._http is not None
        # Chatterbox-specific probes must NOT fire.
        assert handler._probe_llama_health.await_count == 0, (
            "_probe_llama_health fired even though AUDIO_OUTPUT_BACKEND=xtts"
        )
        assert handler._warmup_llm_kv_cache.await_count == 0, (
            "_warmup_llm_kv_cache fired even though AUDIO_OUTPUT_BACKEND=xtts"
        )
        assert handler._warmup_tts.await_count == 0, "_warmup_tts fired even though AUDIO_OUTPUT_BACKEND=xtts"
        # The Gemini LLM client must still be initialised (LLM host still works).
        assert handler._gemini_llm is not None
    finally:
        cfg_module.config.AUDIO_OUTPUT_BACKEND = original_output
