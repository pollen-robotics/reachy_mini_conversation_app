"""Round-trip and collision tests for the personality tag codec."""

import pytest

from reachy_mini_conversation_app.personality_tag import to_tag_token, from_tag_token


@pytest.mark.parametrize(
    "selection, token",
    [
        ("noir_detective", "hf_noir_detective"),
        ("default", "hf_default"),
        ("user_personalities/my_bot", "usr_my_bot"),
    ],
)
def test_round_trip(selection, token):
    """A selection encodes to its token and decodes back unchanged."""
    assert to_tag_token(selection) == token
    assert from_tag_token(token) == selection


def test_a_user_personality_never_collides_with_a_built_in_one():
    """The reason both sides are prefixed rather than only the built-ins.

    The name sanitizer allows a user to call their personality
    ``hf_noir_detective``. With only the built-ins prefixed, that name would
    encode to the same tag text as the built-in ``noir_detective`` and one
    would silently shadow the other.
    """
    builtin = to_tag_token("noir_detective")
    impostor = to_tag_token("user_personalities/hf_noir_detective")
    assert builtin != impostor
    assert from_tag_token(builtin) == "noir_detective"
    assert from_tag_token(impostor) == "user_personalities/hf_noir_detective"


def test_a_user_may_reuse_a_built_in_name():
    """Same bare name on both sides stays distinguishable."""
    assert to_tag_token("mars_rover") != to_tag_token("user_personalities/mars_rover")


def test_an_empty_selection_has_no_token():
    """Nothing to write for an empty name."""
    assert to_tag_token("") is None
    assert to_tag_token("   ") is None


@pytest.mark.parametrize(
    "token",
    [
        "976F33E2",  # an opaque code from the old mapping scheme
        "",
        "hello winnie",
        "hf_",
        "usr_",
    ],
)
def test_foreign_tag_content_decodes_to_nothing(token):
    """A tag written by something else must not resolve to a personality."""
    assert from_tag_token(token) is None
