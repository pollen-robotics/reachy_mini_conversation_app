import sys
import importlib

import pytest


@pytest.fixture(autouse=True)
def _restore_config_object():
    """Save the original config singleton and restore it after each test.

    importlib.reload() replaces robot_comic.config.__dict__['config'] with a
    new Config instance. Imported functions (get_backend_choice etc.) look up
    'config' via __globals__ which IS that same __dict__, so they start reading
    the new instance. Restoring the original object keeps all callers
    consistent after the test.
    """
    cfg_mod = importlib.import_module("robot_comic.config")
    original = cfg_mod.config
    yield
    cfg_mod.config = original


def _reload_config(monkeypatch, env: dict):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import robot_comic.config as cfg_mod

    importlib.reload(cfg_mod)
    return cfg_mod.config


def test_gemini_live_video_streaming_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.GEMINI_LIVE_VIDEO_STREAMING is False


def test_gemini_live_video_streaming_env_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"GEMINI_LIVE_VIDEO_STREAMING": "true"})
    assert cfg.GEMINI_LIVE_VIDEO_STREAMING is True


def test_movement_speed_factor_defaults_0_3(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(0.3)


def test_movement_speed_factor_clamped_high(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOVEMENT_SPEED_FACTOR": "5.0"})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(2.0)


def test_movement_speed_factor_clamped_low(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOVEMENT_SPEED_FACTOR": "0.0"})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(0.1)


def test_moonshine_heartbeat_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.MOONSHINE_HEARTBEAT is False


def test_moonshine_heartbeat_env_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOONSHINE_HEARTBEAT": "true"})
    assert cfg.MOONSHINE_HEARTBEAT is True


def test_audio_capture_path_default_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = _reload_config(monkeypatch, {})
    assert cfg.AUDIO_CAPTURE_PATH == "alsa_rw"


def test_audio_capture_path_default_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = _reload_config(monkeypatch, {})
    assert cfg.AUDIO_CAPTURE_PATH == "daemon"


def test_audio_capture_path_explicit_daemon_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = _reload_config(monkeypatch, {"REACHY_MINI_AUDIO_CAPTURE_PATH": "daemon"})
    assert cfg.AUDIO_CAPTURE_PATH == "daemon"


def test_audio_capture_path_explicit_alsa_rw_on_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    cfg = _reload_config(monkeypatch, {"REACHY_MINI_AUDIO_CAPTURE_PATH": "alsa_rw"})
    assert cfg.AUDIO_CAPTURE_PATH == "alsa_rw"


def test_audio_capture_path_invalid_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = _reload_config(monkeypatch, {"REACHY_MINI_AUDIO_CAPTURE_PATH": "bogus"})
    assert cfg.AUDIO_CAPTURE_PATH == "alsa_rw"


def test_llama_cpp_default_url_is_port_8080(monkeypatch):
    import robot_comic.config as cfg_mod

    assert cfg_mod.LLAMA_CPP_DEFAULT_URL == "http://astralplane.lan:8080"


def test_llama_cpp_url_env_override(monkeypatch):
    cfg = _reload_config(monkeypatch, {"LLAMA_CPP_URL": "http://myhost.lan:9999"})
    assert cfg.LLAMA_CPP_URL == "http://myhost.lan:9999"
