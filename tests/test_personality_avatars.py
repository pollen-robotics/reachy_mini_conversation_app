"""Coverage for avatar resolution and the bulk personality listing.

These back the phone's "load everything from the daemon instead of bundling
the defaults" path: ``personalities.all`` returns every persona in one call
(no inline SVG), and ``personalities.avatar`` serves one SVG on demand with a
default fallback.
"""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.personality as personality_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.avatars import (
    DEFAULT_AVATAR_FILE,
    avatar_id_for,
    read_avatar_svg,
)
from reachy_mini_conversation_app.personality import DEFAULT_OPTION
from reachy_mini_conversation_app.personality_routes import (
    RouteError,
    PersonalityOps,
    build_personality_ops,
)


def _ops() -> PersonalityOps:
    """Build ops with stub callbacks; get_all/avatar touch neither loop nor handler."""
    return build_personality_ops(object(), lambda: None)  # type: ignore[arg-type]


def test_avatar_id_maps_builtin_to_slug() -> None:
    """A built-in profile resolves to its shared avatar slug (no .svg suffix)."""
    assert avatar_id_for("mad_scientist_assistant") == "mad-scientist"
    assert avatar_id_for(DEFAULT_OPTION) == "default"


def test_avatar_id_falls_back_to_default_for_unknown() -> None:
    """An unmapped selection (e.g. a user persona) falls back to the default id."""
    assert avatar_id_for("user_personalities/whatever") == "default"


def test_read_avatar_svg_returns_builtin_markup() -> None:
    """A built-in persona yields real SVG markup from static/avatars/."""
    svg = read_avatar_svg("mad_scientist_assistant")
    assert svg is not None
    assert "<svg" in svg


def test_read_avatar_svg_defaults_for_unknown() -> None:
    """An unknown selection falls back to the default avatar, not None."""
    default_svg = (Path(personality_mod.__file__).parent / "static" / "avatars" / DEFAULT_AVATAR_FILE).read_text(
        encoding="utf-8"
    )
    assert read_avatar_svg("user_personalities/nope") == default_svg


def test_profile_local_avatar_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A persona that ships its own avatar.svg wins over the mapping/default."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    personality_mod._write_profile("selfie", "Be brief.", "")
    own = tmp_path / "user_personalities" / "selfie" / "avatar.svg"
    own.write_text("<svg id='own'></svg>", encoding="utf-8")

    assert read_avatar_svg("user_personalities/selfie") == "<svg id='own'></svg>"
    assert avatar_id_for("user_personalities/selfie") == "user_personalities/selfie"


def test_get_all_lists_every_persona_without_inline_svg() -> None:
    """get_all returns the built-in default first plus each persona, avatar_id only."""
    result = _ops().get_all()
    personalities = result["personalities"]

    assert personalities[0]["name"] == DEFAULT_OPTION
    names = {p["name"] for p in personalities}
    assert "mad_scientist_assistant" in names
    for entry in personalities:
        assert "avatar_id" in entry
        assert "instructions" in entry
        assert "svg" not in entry  # avatars are fetched lazily, never inlined here


def test_avatar_method_returns_svg_and_id() -> None:
    """personalities.avatar returns the markup plus a cache id."""
    payload = _ops().avatar("mad_scientist_assistant")
    assert payload["avatar_id"] == "mad-scientist"
    assert "<svg" in payload["svg"]


def test_avatar_method_falls_back_for_unknown() -> None:
    """An unknown selection still yields the default SVG rather than erroring."""
    payload = _ops().avatar("user_personalities/nope")
    assert "<svg" in payload["svg"]
    assert payload["avatar_id"] == "default"


def test_avatar_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If even the default avatar cannot be read, avatar() surfaces a RouteError."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.personality_routes.read_avatar_svg",
        lambda name: None,
    )
    with pytest.raises(RouteError) as ei:
        _ops().avatar("mad_scientist_assistant")
    assert ei.value.reason == "avatar_unavailable"
