"""Tests for the LanguageDissect tool and the euphemisms dictionary."""

from __future__ import annotations
import sys
import json
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_TOOL_PATH = Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "language_dissect.py"
_EUPHEMISMS_PATH = Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "euphemisms.json"


def _load_language_dissect_module():
    """Load language_dissect from its source path, bypassing the package registry."""
    spec = importlib.util.spec_from_file_location("language_dissect_test_module", _TOOL_PATH)
    assert spec and spec.loader, f"Cannot load module from {_TOOL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["language_dissect_test_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def ld_module():
    return _load_language_dissect_module()


@pytest.fixture(scope="module")
def euphemisms_dict():
    """Load the raw euphemisms JSON for independent validation."""
    with _EUPHEMISMS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_deps() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Dictionary integrity
# ---------------------------------------------------------------------------


class TestEuphemismsDictionary:
    def test_dict_loads_as_mapping(self, euphemisms_dict):
        """euphemisms.json must parse as a non-empty dict."""
        assert isinstance(euphemisms_dict, dict)
        assert len(euphemisms_dict) >= 30, f"Expected at least 30 entries, got {len(euphemisms_dict)}"

    def test_every_entry_has_required_keys(self, euphemisms_dict):
        """Every entry must have literal_words, euphemism_target, dissection_suggestion."""
        required = {"literal_words", "euphemism_target", "dissection_suggestion"}
        for phrase, entry in euphemisms_dict.items():
            missing = required - entry.keys()
            assert not missing, f"Entry '{phrase}' is missing keys: {missing}"

    def test_literal_words_is_dict(self, euphemisms_dict):
        """literal_words must be a dict mapping word strings to meaning strings."""
        for phrase, entry in euphemisms_dict.items():
            lw = entry["literal_words"]
            assert isinstance(lw, dict), f"Entry '{phrase}': literal_words must be a dict, got {type(lw).__name__}"
            for word, meaning in lw.items():
                assert isinstance(word, str) and word, f"Entry '{phrase}': literal_words key must be non-empty string"
                assert isinstance(meaning, str) and meaning, (
                    f"Entry '{phrase}': literal_words value must be non-empty string"
                )

    def test_euphemism_target_is_non_empty_string(self, euphemisms_dict):
        for phrase, entry in euphemisms_dict.items():
            target = entry["euphemism_target"]
            assert isinstance(target, str) and target.strip(), (
                f"Entry '{phrase}': euphemism_target must be a non-empty string"
            )

    def test_dissection_suggestion_is_non_empty_string(self, euphemisms_dict):
        for phrase, entry in euphemisms_dict.items():
            suggestion = entry["dissection_suggestion"]
            assert isinstance(suggestion, str) and suggestion.strip(), (
                f"Entry '{phrase}': dissection_suggestion must be a non-empty string"
            )

    def test_known_phrases_present(self, euphemisms_dict):
        """A selection of canonical Carlin targets must be in the dictionary."""
        expected = [
            "thoughts and prayers",
            "passed away",
            "human resources",
            "collateral damage",
            "enhanced interrogation",
            "downsizing",
            "correctional facility",
        ]
        for phrase in expected:
            assert phrase in euphemisms_dict, f"Expected canonical phrase '{phrase}' not found in euphemisms.json"


# ---------------------------------------------------------------------------
# dissect_phrase — known entry
# ---------------------------------------------------------------------------


class TestDissectPhraseKnown:
    def test_returns_expected_structure(self, ld_module):
        """dissect_phrase returns all four required keys for a known phrase."""
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert set(result.keys()) >= {"phrase", "literal_words", "euphemism_target", "dissection_suggestion"}

    def test_phrase_field_preserved(self, ld_module):
        """The phrase field echoes the original input, not the normalised version."""
        original = "Thoughts and Prayers"
        result = ld_module.dissect_phrase(original)
        assert result["phrase"] == original

    def test_literal_words_is_dict(self, ld_module):
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert isinstance(result["literal_words"], dict)
        assert len(result["literal_words"]) >= 1

    def test_euphemism_target_populated(self, ld_module):
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert isinstance(result["euphemism_target"], str)
        assert result["euphemism_target"] != "unknown"

    def test_dissection_suggestion_populated(self, ld_module):
        result = ld_module.dissect_phrase("thoughts and prayers")
        assert isinstance(result["dissection_suggestion"], str)
        assert result["dissection_suggestion"].strip()

    def test_case_insensitive_lookup(self, ld_module):
        """Lookup must be case-insensitive."""
        lower = ld_module.dissect_phrase("passed away")
        upper = ld_module.dissect_phrase("PASSED AWAY")
        mixed = ld_module.dissect_phrase("Passed Away")
        assert lower["euphemism_target"] == upper["euphemism_target"] == mixed["euphemism_target"]

    def test_human_resources_target(self, ld_module):
        """human resources — the euphemism_target should describe interchangeable workers."""
        result = ld_module.dissect_phrase("human resources")
        assert result["euphemism_target"] != "unknown"
        assert "literal_words" in result
        assert "human" in result["literal_words"] or "resources" in result["literal_words"]

    def test_collateral_damage_literal_words(self, ld_module):
        """collateral damage should have at least 'collateral' and 'damage' in literal_words."""
        result = ld_module.dissect_phrase("collateral damage")
        assert "collateral" in result["literal_words"]
        assert "damage" in result["literal_words"]


# ---------------------------------------------------------------------------
# dissect_phrase — unknown phrase fallback
# ---------------------------------------------------------------------------


class TestDissectPhraseUnknown:
    def test_returns_all_required_keys(self, ld_module):
        """Unknown phrase still returns all four structural keys."""
        result = ld_module.dissect_phrase("synergistic paradigm shift")
        assert set(result.keys()) >= {"phrase", "literal_words", "euphemism_target", "dissection_suggestion"}

    def test_phrase_field_preserved(self, ld_module):
        phrase = "synergistic paradigm shift"
        result = ld_module.dissect_phrase(phrase)
        assert result["phrase"] == phrase

    def test_literal_words_contains_each_word(self, ld_module):
        """For an unknown phrase, literal_words should have one entry per word."""
        result = ld_module.dissect_phrase("synergistic paradigm shift")
        lw = result["literal_words"]
        assert isinstance(lw, dict)
        assert "synergistic" in lw
        assert "paradigm" in lw
        assert "shift" in lw

    def test_euphemism_target_indicates_unknown(self, ld_module):
        """Unknown phrases should signal that the phrase is not in the dictionary."""
        result = ld_module.dissect_phrase("totally made up phrase")
        assert "unknown" in result["euphemism_target"].lower()

    def test_dissection_suggestion_non_empty(self, ld_module):
        result = ld_module.dissect_phrase("totally made up phrase")
        assert isinstance(result["dissection_suggestion"], str)
        assert result["dissection_suggestion"].strip()

    def test_single_word_phrase(self, ld_module):
        """A single unknown word should produce a single-key literal_words dict."""
        result = ld_module.dissect_phrase("quibble")
        lw = result["literal_words"]
        assert "quibble" in lw

    def test_empty_phrase_returns_error(self, ld_module):
        """Empty string should not crash — the tool layer handles it, but dissect_phrase
        should still return a dict (the tool wraps it in an error check)."""
        # dissect_phrase with empty string produces an empty literal_words dict
        result = ld_module.dissect_phrase("   ")
        # Whitespace-only normalises to empty — just verify it doesn't raise
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# LanguageDissect tool class
# ---------------------------------------------------------------------------


class TestLanguageDissectTool:
    @pytest.fixture
    def tool(self, ld_module):
        return ld_module.LanguageDissect()

    def test_name(self, tool):
        assert tool.name == "language_dissect"

    def test_description_non_empty(self, tool):
        assert isinstance(tool.description, str) and tool.description.strip()

    def test_parameters_schema_has_phrase(self, tool):
        props = tool.parameters_schema.get("properties", {})
        assert "phrase" in props
        assert tool.parameters_schema.get("required") == ["phrase"]

    @pytest.mark.asyncio
    async def test_call_known_phrase(self, tool):
        """Tool returns expected structure for a known euphemism via async __call__."""
        result = await tool(make_deps(), phrase="thoughts and prayers")
        assert result["phrase"] == "thoughts and prayers"
        assert isinstance(result["literal_words"], dict)
        assert result["euphemism_target"] != "unknown"
        assert result["dissection_suggestion"].strip()

    @pytest.mark.asyncio
    async def test_call_unknown_phrase(self, tool):
        """Tool returns reasonable defaults for an unknown phrase."""
        result = await tool(make_deps(), phrase="leveraged synergy")
        assert result["phrase"] == "leveraged synergy"
        assert isinstance(result["literal_words"], dict)
        assert "leveraged" in result["literal_words"]
        assert "synergy" in result["literal_words"]
        assert "unknown" in result["euphemism_target"].lower()

    @pytest.mark.asyncio
    async def test_call_empty_phrase_returns_error(self, tool):
        """Empty phrase argument returns an error dict without raising."""
        result = await tool(make_deps(), phrase="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_call_missing_phrase_returns_error(self, tool):
        """Missing phrase kwarg returns an error dict without raising."""
        result = await tool(make_deps())
        assert "error" in result

    def test_spec_structure(self, tool):
        """spec() returns a well-formed function-call spec."""
        spec = tool.spec()
        assert spec["type"] == "function"
        assert spec["name"] == "language_dissect"
        assert "description" in spec
        assert "parameters" in spec


# ---------------------------------------------------------------------------
# Per-word fallback decomposition (Issue #220)
# ---------------------------------------------------------------------------


class TestDecomposeWord:
    """Unit tests for _decompose_word — the per-word lexicon lookup."""

    def test_exact_match(self, ld_module):
        """A word present verbatim in the lexicon returns its gloss."""
        result = ld_module._decompose_word("institution")
        assert result is not None
        root, gloss = result
        assert root == "institution"
        assert "organization" in gloss.lower() or "authority" in gloss.lower()

    def test_suffix_stripping_plural(self, ld_module):
        """'institutions' stems to 'institution' via -s suffix."""
        result = ld_module._decompose_word("institutions")
        assert result is not None
        root, gloss = result
        assert root == "institution"

    def test_suffix_stripping_tion(self, ld_module):
        """A word ending in -tion decomposes via its stem."""
        # 'operation' stems to 'operation' (direct hit), confirm the mechanism works
        result = ld_module._decompose_word("operations")
        assert result is not None
        root, _ = result
        assert root == "operation"

    def test_suffix_stripping_ing(self, ld_module):
        """A gerund form strips -ing to find the root."""
        result = ld_module._decompose_word("leveraging")
        # 'leverag' is not a valid stem — no match expected for this specific word,
        # but 'managing' → 'manag' is also invalid. Test a clean case: 'targeting'
        # → 'target' (5 chars, in lexicon)
        result = ld_module._decompose_word("targeting")
        assert result is not None
        root, _ = result
        assert root == "target"

    def test_unknown_word_returns_none(self, ld_module):
        """A word with no lexicon match (even after stemming) returns None."""
        result = ld_module._decompose_word("xylophonically")
        assert result is None

    def test_short_stem_not_matched(self, ld_module):
        """Stemming that produces a stem shorter than 3 chars should not match."""
        # 'es' → stem '' (empty) — should return None without crashing
        result = ld_module._decompose_word("es")
        assert result is None

    def test_punctuation_stripped(self, ld_module):
        """Non-alpha characters are removed before lookup."""
        result = ld_module._decompose_word("institution.")
        assert result is not None
        root, _ = result
        assert root == "institution"


class TestDecomposePhrase:
    """Unit tests for _decompose_phrase."""

    def test_all_words_decompose(self, ld_module):
        """A phrase made of known roots returns all words as decomposed."""
        breakdown = ld_module._decompose_phrase("human resources")
        assert len(breakdown) == 2
        assert all(item["decomposed"] == "true" for item in breakdown)

    def test_partial_decomposition(self, ld_module):
        """A phrase with one unknown word returns a mixed breakdown."""
        # 'quux' is not in the lexicon; 'human' is
        breakdown = ld_module._decompose_phrase("human quux")
        decomposed = [item for item in breakdown if item["decomposed"] == "true"]
        undecomposed = [item for item in breakdown if item["decomposed"] == "false"]
        assert len(decomposed) == 1
        assert len(undecomposed) == 1
        assert decomposed[0]["word"] == "human"

    def test_returns_list_of_dicts(self, ld_module):
        """Return value is always a list of dicts with expected keys."""
        breakdown = ld_module._decompose_phrase("strategic realignment")
        assert isinstance(breakdown, list)
        for item in breakdown:
            assert "word" in item
            assert "root" in item
            assert "gloss" in item
            assert "decomposed" in item


class TestDissectPhraseDecomposition:
    """Integration tests for the per-word fallback path in dissect_phrase."""

    def test_high_quality_decomposition(self, ld_module):
        """Phrase composed of lexicon roots returns decomposition_quality 'high'."""
        result = ld_module.dissect_phrase("human resources")
        # 'human resources' IS in the curated dictionary, so test with a close variant
        # that isn't: use words known to be in the lexicon
        result = ld_module.dissect_phrase("institutional behavior")
        assert result["decomposition_quality"] == "high"
        assert isinstance(result["literal_words"], dict)
        assert "institutional" in result["literal_words"] or "behavior" in result["literal_words"]

    def test_high_quality_with_partial_match(self, ld_module):
        """2 out of 3 words decompose successfully → quality still 'high' (>60%)."""
        # 'human' and 'resource' are in lexicon; 'zorp' is not
        result = ld_module.dissect_phrase("human zorp resource")
        assert result["decomposition_quality"] == "high"

    def test_low_quality_all_unknown(self, ld_module):
        """All-unknown words produce decomposition_quality 'low' and v1-style literal_words."""
        result = ld_module.dissect_phrase("zorp quux blorp")
        assert result["decomposition_quality"] == "low"
        lw = result["literal_words"]
        assert "zorp" in lw
        assert "quux" in lw
        assert "blorp" in lw
        assert lw["zorp"] == "plain meaning of 'zorp'"

    def test_stemming_works_for_institutions(self, ld_module):
        """'institutions' decomposes via the 'institution' root."""
        # Use a phrase where all words stem cleanly so quality is 'high'.
        # 'targeting' -> 'target'; 'institutions' -> 'institution' (both in lexicon)
        result = ld_module.dissect_phrase("targeting institutions")
        assert result["decomposition_quality"] == "high"
        # Confirm the stemmed root appeared in the literal_words entry
        lw = result["literal_words"]
        assert any("institution" in v for v in lw.values())

    def test_required_keys_present_high_quality(self, ld_module):
        """High-quality fallback result has all required keys."""
        result = ld_module.dissect_phrase("institutional behavior")
        required = {"phrase", "literal_words", "euphemism_target", "dissection_suggestion", "decomposition_quality"}
        assert required.issubset(result.keys())

    def test_required_keys_present_low_quality(self, ld_module):
        """Low-quality fallback result has all required keys."""
        result = ld_module.dissect_phrase("zorp quux blorp")
        required = {"phrase", "literal_words", "euphemism_target", "dissection_suggestion", "decomposition_quality"}
        assert required.issubset(result.keys())

    def test_phrase_field_preserved_in_fallback(self, ld_module):
        """The phrase field echoes the original input in both fallback paths."""
        original = "Institutional Behavior"
        result = ld_module.dissect_phrase(original)
        assert result["phrase"] == original

    def test_dissection_suggestion_mentions_words_high(self, ld_module):
        """High-quality suggestion references word meanings."""
        result = ld_module.dissect_phrase("institutional behavior")
        assert isinstance(result["dissection_suggestion"], str)
        assert result["dissection_suggestion"].strip()

    def test_curated_entry_not_affected(self, ld_module):
        """Curated entries are unaffected by the new decomposition logic — no quality key."""
        result = ld_module.dissect_phrase("thoughts and prayers")
        # Curated entries do not include decomposition_quality
        assert "decomposition_quality" not in result

    def test_lexicon_entry_count(self, ld_module):
        """The bundled lexicon must have at least 200 entries."""
        lexicon = ld_module._get_lexicon()
        assert len(lexicon) >= 200, f"Expected >= 200 lexicon entries, got {len(lexicon)}"


# ---------------------------------------------------------------------------
# LLM-assisted fallback (Issue #236)
# ---------------------------------------------------------------------------

_NOVEL_PHRASE = "zorp quux blorp"  # Gibberish — guaranteed low-quality decomposition.

# Canonical well-formed LLM response payload.
_GOOD_LLM_RESPONSE = {
    "literal_words": [
        "zorp (nonsense filler word)",
        "quux (placeholder jargon)",
        "blorp (empty buzzword)",
    ],
    "euphemism_target": "meaningless corporate noise",
    "dissection_suggestion": "Strip the syllables away and you're left with nothing.",
}


def _make_http_mock(response_json: dict | None = None, *, raise_exc: Exception | None = None) -> AsyncMock:
    """Build an AsyncMock httpx.AsyncClient whose .post() returns *response_json*."""
    mock_client = MagicMock()
    if raise_exc is not None:
        mock_client.post = AsyncMock(side_effect=raise_exc)
    else:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(response_json or _GOOD_LLM_RESPONSE)}}]
        }
        mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


def _make_deps(http_client=None, llama_url: str | None = "http://llama.lan:11434") -> MagicMock:
    """Build a MagicMock ToolDependencies with optional http_client and llama_url."""
    deps = MagicMock()
    deps.http_client = http_client
    deps.llama_url = llama_url
    return deps


class TestLLMFallbackDisabled:
    """When LANGUAGE_DISSECT_LLM_FALLBACK_ENABLED is False, LLM must never be called."""

    @pytest.mark.asyncio
    async def test_env_var_false_no_llm_call(self, ld_module):
        """LLM is never called when the feature flag is off."""
        http_mock = _make_http_mock()
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "0"}):
            # Re-read config via the module's internal import.
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()
            result = await ld_module.dissect_phrase_async(
                _NOVEL_PHRASE,
                http_client=http_mock,
                llama_url="http://llama.lan:11434",
            )
        http_mock.post.assert_not_called()
        assert result["decomposition_quality"] == "low"

    @pytest.mark.asyncio
    async def test_default_no_llm_call_no_http_client(self, ld_module):
        """With no http_client, the LLM path is skipped and quality stays 'low'."""
        result = await ld_module.dissect_phrase_async(_NOVEL_PHRASE, http_client=None, llama_url=None)
        assert result["decomposition_quality"] == "low"

    @pytest.mark.asyncio
    async def test_curated_path_unaffected(self, ld_module):
        """Curated entries are never sent to the LLM regardless of flag state."""
        http_mock = _make_http_mock()
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "1"}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()
            result = await ld_module.dissect_phrase_async(
                "thoughts and prayers",
                http_client=http_mock,
                llama_url="http://llama.lan:11434",
            )
        http_mock.post.assert_not_called()
        assert "decomposition_quality" not in result

    @pytest.mark.asyncio
    async def test_high_quality_decomposition_unaffected(self, ld_module):
        """High-quality decompositions don't hit the LLM even when flag is on."""
        http_mock = _make_http_mock()
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "1"}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()
            result = await ld_module.dissect_phrase_async(
                "institutional behavior",
                http_client=http_mock,
                llama_url="http://llama.lan:11434",
            )
        http_mock.post.assert_not_called()
        assert result["decomposition_quality"] == "high"


class TestLLMFallbackEnabled:
    """Behaviour when REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK=1."""

    @pytest.fixture(autouse=True)
    def enable_flag(self):
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "1"}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()
            yield
        # Restore default (disabled) after each test.
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "0"}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()

    @pytest.mark.asyncio
    async def test_llm_success_returns_quality_llm(self, ld_module):
        """Successful LLM call returns decomposition_quality 'llm'."""
        http_mock = _make_http_mock(_GOOD_LLM_RESPONSE)
        with patch.object(ld_module, "_cache_llm_result"):
            result = await ld_module.dissect_phrase_async(
                _NOVEL_PHRASE,
                http_client=http_mock,
                llama_url="http://llama.lan:11434",
            )
        assert result["decomposition_quality"] == "llm"
        assert result["phrase"] == _NOVEL_PHRASE
        assert isinstance(result["literal_words"], dict)
        assert result["euphemism_target"] == _GOOD_LLM_RESPONSE["euphemism_target"]

    @pytest.mark.asyncio
    async def test_llm_success_contains_all_required_keys(self, ld_module):
        """LLM result includes all standard dissection keys."""
        http_mock = _make_http_mock(_GOOD_LLM_RESPONSE)
        with patch.object(ld_module, "_cache_llm_result"):
            result = await ld_module.dissect_phrase_async(
                _NOVEL_PHRASE,
                http_client=http_mock,
                llama_url="http://llama.lan:11434",
            )
        required = {"phrase", "literal_words", "euphemism_target", "dissection_suggestion", "decomposition_quality"}
        assert required.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_llm_parse_error_falls_back_to_low(self, ld_module):
        """Non-JSON LLM response falls back to 'low' quality without raising."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "not valid json {"}}]}
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await ld_module.dissect_phrase_async(
            _NOVEL_PHRASE,
            http_client=mock_client,
            llama_url="http://llama.lan:11434",
        )
        assert result["decomposition_quality"] == "low"

    @pytest.mark.asyncio
    async def test_llm_network_error_falls_back_to_low(self, ld_module):
        """Network error during LLM call falls back to 'low' without raising."""
        import httpx

        http_mock = _make_http_mock(raise_exc=httpx.ConnectError("refused"))
        result = await ld_module.dissect_phrase_async(
            _NOVEL_PHRASE,
            http_client=http_mock,
            llama_url="http://llama.lan:11434",
        )
        assert result["decomposition_quality"] == "low"

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_to_low(self, ld_module):
        """Timeout during LLM call falls back to 'low' without raising."""
        import httpx

        http_mock = _make_http_mock(raise_exc=httpx.TimeoutException("timed out"))
        result = await ld_module.dissect_phrase_async(
            _NOVEL_PHRASE,
            http_client=http_mock,
            llama_url="http://llama.lan:11434",
        )
        assert result["decomposition_quality"] == "low"

    @pytest.mark.asyncio
    async def test_llm_empty_literal_words_falls_back_to_low(self, ld_module):
        """Degenerate LLM response (empty literal_words) falls back to 'low'."""
        bad_response = {
            "literal_words": [],  # empty — sanity check should reject this
            "euphemism_target": "nothing",
            "dissection_suggestion": "nothing",
        }
        http_mock = _make_http_mock(bad_response)
        result = await ld_module.dissect_phrase_async(
            _NOVEL_PHRASE,
            http_client=http_mock,
            llama_url="http://llama.lan:11434",
        )
        assert result["decomposition_quality"] == "low"

    @pytest.mark.asyncio
    async def test_llm_result_cached_to_lexicon(self, ld_module, tmp_path):
        """Successful LLM result is written atomically to the lexicon JSON.

        Two sub-checks:
        1. ``dissect_phrase_async`` calls ``_cache_llm_result`` with the right data.
        2. ``_cache_llm_result`` itself writes the expected JSON structure.
        """
        lexicon_file = tmp_path / "language_lexicon.json"
        http_mock = _make_http_mock(_GOOD_LLM_RESPONSE)
        captured_cache_calls: list[tuple] = []

        # Intercept cache writes to avoid touching the real lexicon file.
        def _spy_cache(phrase: str, result: dict, **kwargs) -> None:
            captured_cache_calls.append((phrase, result))

        with patch.object(ld_module, "_cache_llm_result", side_effect=_spy_cache):
            result = await ld_module.dissect_phrase_async(
                _NOVEL_PHRASE,
                http_client=http_mock,
                llama_url="http://llama.lan:11434",
            )

        assert result["decomposition_quality"] == "llm"
        # Verify the cache was called once with the phrase and LLM payload.
        assert len(captured_cache_calls) == 1
        assert captured_cache_calls[0][0] == _NOVEL_PHRASE
        captured_result = captured_cache_calls[0][1]
        assert captured_result.get("euphemism_target") == _GOOD_LLM_RESPONSE["euphemism_target"]
        assert isinstance(captured_result.get("literal_words"), list)

        # Verify the write mechanism directly using the real function.
        ld_module._cache_llm_result(_NOVEL_PHRASE, _GOOD_LLM_RESPONSE, lexicon_path=lexicon_file)
        assert lexicon_file.exists(), "Lexicon file should have been created"
        with lexicon_file.open(encoding="utf-8") as fh:
            cached = json.load(fh)
        assert _NOVEL_PHRASE.strip().lower() in cached
        entry = cached[_NOVEL_PHRASE.strip().lower()]
        assert entry.get("source") == "llm-cached"
        assert isinstance(entry.get("literal_words"), dict)
        assert entry.get("euphemism_target") == _GOOD_LLM_RESPONSE["euphemism_target"]

    @pytest.mark.asyncio
    async def test_cache_degenerate_result_not_written(self, ld_module, tmp_path):
        """Degenerate LLM result (empty literal_words) is NOT cached."""
        lexicon_file = tmp_path / "language_lexicon.json"
        bad_result = {"literal_words": [], "euphemism_target": "x", "dissection_suggestion": "x"}
        ld_module._cache_llm_result(_NOVEL_PHRASE, bad_result, lexicon_path=lexicon_file)
        assert not lexicon_file.exists(), "Degenerate result should not be cached"

    @pytest.mark.asyncio
    async def test_cache_source_tag_is_llm_cached(self, ld_module, tmp_path):
        """Cached entries carry source='llm-cached' to distinguish from curated."""
        lexicon_file = tmp_path / "language_lexicon.json"
        ld_module._cache_llm_result(_NOVEL_PHRASE, _GOOD_LLM_RESPONSE, lexicon_path=lexicon_file)
        with lexicon_file.open(encoding="utf-8") as fh:
            cached = json.load(fh)
        assert cached[_NOVEL_PHRASE.strip().lower()]["source"] == "llm-cached"

    @pytest.mark.asyncio
    async def test_cache_write_is_atomic(self, ld_module, tmp_path):
        """Cache write uses tmp+rename so existing file is not left corrupt."""
        lexicon_file = tmp_path / "language_lexicon.json"
        # Pre-populate so we can verify the file is replaced, not truncated.
        existing = {"existing_entry": {"literal_words": {}, "euphemism_target": "old", "dissection_suggestion": ""}}
        with lexicon_file.open("w", encoding="utf-8") as fh:
            json.dump(existing, fh)

        ld_module._cache_llm_result(_NOVEL_PHRASE, _GOOD_LLM_RESPONSE, lexicon_path=lexicon_file)

        with lexicon_file.open(encoding="utf-8") as fh:
            cached = json.load(fh)
        # Both old and new entries should be present.
        assert "existing_entry" in cached
        assert _NOVEL_PHRASE.strip().lower() in cached


class TestLanguageDissectToolLLMWiring:
    """Verify the LanguageDissect Tool class wires deps.http_client + deps.llama_url."""

    @pytest.fixture
    def tool(self, ld_module):
        return ld_module.LanguageDissect()

    @pytest.mark.asyncio
    async def test_tool_passes_http_client_from_deps(self, tool, ld_module):
        """Tool.__call__ forwards deps.http_client to the async dissect path."""
        http_mock = _make_http_mock(_GOOD_LLM_RESPONSE)
        deps = _make_deps(http_client=http_mock, llama_url="http://llama.lan:11434")
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "1"}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()
            with patch.object(ld_module, "_cache_llm_result"):
                result = await tool(deps, phrase=_NOVEL_PHRASE)
        # LLM should have been called via the injected http_client.
        assert result["decomposition_quality"] == "llm"
        http_mock.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_no_http_client_skips_llm(self, tool):
        """Tool.__call__ skips LLM when deps.http_client is None."""
        deps = _make_deps(http_client=None, llama_url=None)
        with patch.dict("os.environ", {"REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK": "1"}):
            from robot_comic.config import refresh_runtime_config_from_env

            refresh_runtime_config_from_env()
            result = await tool(deps, phrase=_NOVEL_PHRASE)
        assert result["decomposition_quality"] == "low"
