"""Factory selection tests for STT process isolation (#540).

When ``REACHY_MINI_STT_PROCESS_ISOLATION`` is enabled, the Moonshine
input backend must resolve to the out-of-process
:class:`SubprocessSTTAdapter`; otherwise it keeps the in-process
:class:`MoonshineSTTAdapter`. Faster-whisper is unaffected.
"""

from __future__ import annotations
import os
from unittest.mock import patch

from robot_comic import config as cfg_module
from robot_comic.config import AUDIO_INPUT_MOONSHINE, config
from robot_comic.handler_factory import _build_stt_adapter
from robot_comic.adapters.stt_process import SubprocessSTTAdapter
from robot_comic.adapters.moonshine_stt_adapter import MoonshineSTTAdapter


def _drop() -> bool:
    return False


def test_moonshine_in_process_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Default (isolation off) keeps the in-process Moonshine adapter."""
    monkeypatch.setattr(config, "LOCAL_STT_PROCESS_ISOLATION", False, raising=False)
    adapter = _build_stt_adapter(AUDIO_INPUT_MOONSHINE, _drop)
    assert isinstance(adapter, MoonshineSTTAdapter)
    assert not isinstance(adapter, SubprocessSTTAdapter)


def test_moonshine_uses_subprocess_when_isolation_enabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Isolation flag swaps in the subprocess adapter for Moonshine."""
    monkeypatch.setattr(config, "LOCAL_STT_PROCESS_ISOLATION", True, raising=False)
    monkeypatch.setattr(config, "LOCAL_STT_STALL_TIMEOUT", 12.5, raising=False)
    adapter = _build_stt_adapter(AUDIO_INPUT_MOONSHINE, _drop)
    assert isinstance(adapter, SubprocessSTTAdapter)
    # The configured stall timeout should flow through to the proxy.
    assert adapter._stall_timeout == 12.5
    # The echo-guard predicate should be wired through.
    assert adapter._should_drop_frame is _drop


def test_refresh_runtime_config_reads_new_knobs() -> None:
    """The two new knobs must flow through refresh_runtime_config_from_env.

    Top-level module reads fire before main.py's load_dotenv(), so any new
    env knob must also be re-read in refresh (the two-track convention).
    """
    original_iso = config.LOCAL_STT_PROCESS_ISOLATION
    original_stall = config.LOCAL_STT_STALL_TIMEOUT
    try:
        with patch.dict(
            os.environ,
            {
                "REACHY_MINI_STT_PROCESS_ISOLATION": "1",
                "REACHY_MINI_STT_STALL_TIMEOUT": "42",
                "REACHY_MINI_SKIP_DOTENV": "1",
            },
        ):
            cfg_module.refresh_runtime_config_from_env()
            assert config.LOCAL_STT_PROCESS_ISOLATION is True
            assert config.LOCAL_STT_STALL_TIMEOUT == 42.0
    finally:
        config.LOCAL_STT_PROCESS_ISOLATION = original_iso
        config.LOCAL_STT_STALL_TIMEOUT = original_stall
