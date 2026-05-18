"""Argument-parsing and utility tests for scripts/elevenlabs_clone.py.

These tests do NOT make network calls. They exercise:
- create-ivc argument parsing (required/optional flags)
- --max-bytes pre-flight size check
- write_voice_id_to_local_config create/update logic
- cmd_list state dict flattening (Windows crash fix)
"""

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Load the script as a module without executing main()
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "elevenlabs_clone.py"


def _load_script():
    """Load elevenlabs_clone.py as a module for white-box testing."""
    spec = importlib.util.spec_from_file_location("elevenlabs_clone", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


script = _load_script()


# ---------------------------------------------------------------------------
# Argument-parsing helpers
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse argv using a minimal replica of the create-ivc subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ivc = sub.add_parser("create-ivc")
    p_ivc.add_argument("--profile", required=True)
    p_ivc.add_argument("name")
    p_ivc.add_argument("files", nargs="*")
    p_ivc.add_argument("--all-clips", action="store_true")
    p_ivc.add_argument("--description", default=None)
    p_ivc.add_argument("--max-bytes", type=int, default=script.DEFAULT_MAX_BYTES)
    p_ivc.add_argument(
        "--write-profile-override",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_ivc.set_defaults(func=lambda a: None)

    return parser.parse_args(argv)


def _fake_ivc_args(max_bytes: int = script.DEFAULT_MAX_BYTES, write_override: bool = True) -> MagicMock:
    """Return a minimal Namespace-alike for create-ivc tests."""
    args = MagicMock()
    args.profile = "house_comedian"
    args.name = "House Comedian"
    args.description = None
    args.all_clips = True
    args.files = []
    args.max_bytes = max_bytes
    args.write_profile_override = write_override
    return args


def _run_write_local(profile: str, voice_id: str, repo_root: Path) -> None:
    """Call write_voice_id_to_local_config with a patched REPO_ROOT."""
    with patch.object(script, "REPO_ROOT", repo_root):
        script.write_voice_id_to_local_config(profile, voice_id)


def _run_cmd_list(state_value, capsys) -> str:
    """Run cmd_list with a mocked /voices response and return captured stdout."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "voices": [
            {
                "voice_id": "abc123",
                "name": "House Comedian",
                "category": "professional",
                "fine_tuning": {"state": state_value},
            }
        ]
    }
    with (
        patch.object(script, "get_api_key", return_value="sk_test"),
        patch("requests.get", return_value=mock_resp),
    ):
        script.cmd_list(MagicMock())
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests: argument parsing
# ---------------------------------------------------------------------------


def test_parse_all_clips_flag():
    """--all-clips sets all_clips=True and captures profile/name."""
    args = _parse(["create-ivc", "--profile", "house_comedian", "House Comedian", "--all-clips"])
    assert args.all_clips is True
    assert args.profile == "house_comedian"
    assert args.name == "House Comedian"


def test_parse_explicit_files():
    """Positional file args are collected into args.files."""
    args = _parse(["create-ivc", "--profile", "house_comedian", "House Comedian", "clip_a.wav", "clip_b.wav"])
    assert args.all_clips is False
    assert args.files == ["clip_a.wav", "clip_b.wav"]


def test_parse_default_write_profile_override_is_true():
    """--write-profile-override defaults to True."""
    args = _parse(["create-ivc", "--profile", "x", "Name", "--all-clips"])
    assert args.write_profile_override is True


def test_parse_no_write_profile_override():
    """--no-write-profile-override sets the flag to False."""
    args = _parse(["create-ivc", "--profile", "x", "Name", "--all-clips", "--no-write-profile-override"])
    assert args.write_profile_override is False


def test_parse_default_max_bytes():
    """--max-bytes defaults to DEFAULT_MAX_BYTES."""
    args = _parse(["create-ivc", "--profile", "x", "Name", "--all-clips"])
    assert args.max_bytes == script.DEFAULT_MAX_BYTES


def test_parse_custom_max_bytes():
    """--max-bytes accepts a custom integer value."""
    args = _parse(["create-ivc", "--profile", "x", "Name", "--all-clips", "--max-bytes", "5000000"])
    assert args.max_bytes == 5_000_000


# ---------------------------------------------------------------------------
# Tests: max-bytes pre-flight check
# ---------------------------------------------------------------------------


def test_max_bytes_aborts_when_over_limit(tmp_path):
    """cmd_create_ivc exits 1 when combined clip size exceeds --max-bytes."""
    clip1 = tmp_path / "clip_a.wav"
    clip2 = tmp_path / "clip_b.wav"
    clip1.write_bytes(b"x" * 6_000_000)
    clip2.write_bytes(b"x" * 6_000_000)

    args = _fake_ivc_args(max_bytes=10_000_000)

    with (
        patch.object(script, "resolve_clip_files", return_value=[clip1, clip2]),
        pytest.raises(SystemExit) as exc_info,
    ):
        script.cmd_create_ivc(args)

    assert exc_info.value.code == 1


def test_max_bytes_passes_when_under_limit(tmp_path):
    """cmd_create_ivc completes without error when clips are within the limit."""
    clip1 = tmp_path / "clip_a.wav"
    clip1.write_bytes(b"x" * 100)

    args = _fake_ivc_args(write_override=False)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"voice_id": "test_voice_id_abc"}

    with (
        patch.object(script, "resolve_clip_files", return_value=[clip1]),
        patch.object(script, "get_api_key", return_value="sk_test"),
        patch("requests.post", return_value=mock_response),
    ):
        script.cmd_create_ivc(args)  # should not raise


# ---------------------------------------------------------------------------
# Tests: write_voice_id_to_local_config
# ---------------------------------------------------------------------------


def test_write_local_config_creates_new_file(tmp_path):
    """Creates elevenlabs.local.txt when the file does not exist."""
    (tmp_path / "profiles" / "testpersona").mkdir(parents=True)
    _run_write_local("testpersona", "voice_abc123", tmp_path)
    out = (tmp_path / "profiles" / "testpersona" / "elevenlabs.local.txt").read_text(encoding="utf-8")
    assert "voice_id=voice_abc123" in out


def test_write_local_config_updates_existing_voice_id(tmp_path):
    """Replaces an existing voice_id= line while preserving other keys."""
    profile_dir = tmp_path / "profiles" / "testpersona"
    profile_dir.mkdir(parents=True)
    config = profile_dir / "elevenlabs.local.txt"
    config.write_text("voice=House Comedian\nvoice_id=old_id\n", encoding="utf-8")

    _run_write_local("testpersona", "new_id_xyz", tmp_path)

    content = config.read_text(encoding="utf-8")
    assert "voice_id=new_id_xyz" in content
    assert "old_id" not in content
    assert "voice=House Comedian" in content


def test_write_local_config_appends_when_missing(tmp_path):
    """Appends voice_id= when the file exists but has no voice_id key."""
    profile_dir = tmp_path / "profiles" / "testpersona"
    profile_dir.mkdir(parents=True)
    config = profile_dir / "elevenlabs.local.txt"
    config.write_text("voice=House Comedian\n", encoding="utf-8")

    _run_write_local("testpersona", "brand_new_id", tmp_path)

    content = config.read_text(encoding="utf-8")
    assert "voice_id=brand_new_id" in content
    assert "voice=House Comedian" in content


def test_write_local_config_replaces_commented_voice_id(tmp_path):
    """Replaces a commented-out voice_id= line."""
    profile_dir = tmp_path / "profiles" / "testpersona"
    profile_dir.mkdir(parents=True)
    config = profile_dir / "elevenlabs.local.txt"
    config.write_text("# voice_id=old_commented\n", encoding="utf-8")

    _run_write_local("testpersona", "active_id", tmp_path)

    content = config.read_text(encoding="utf-8")
    assert "voice_id=active_id" in content
    assert "old_commented" not in content


# ---------------------------------------------------------------------------
# Tests: cmd_list state dict flattening (Windows crash fix)
# ---------------------------------------------------------------------------


def test_cmd_list_state_plain_string(capsys):
    """cmd_list prints the state string without crashing."""
    out = _run_cmd_list("fine_tuned", capsys)
    assert "fine_tuned" in out


def test_cmd_list_state_dict_does_not_crash(capsys):
    """cmd_list handles fine_tuning.state as a per-model dict (Windows crash regression)."""
    out = _run_cmd_list({"eleven_multilingual_v2": "fine_tuned"}, capsys)
    assert "fine_tuned" in out


def test_cmd_list_state_empty_dict(capsys):
    """cmd_list falls back to 'n/a' when fine_tuning.state is an empty dict."""
    out = _run_cmd_list({}, capsys)
    assert "n/a" in out
