"""Language dissection tool — breaks down euphemisms and hollow language patterns.

The tool provides a heuristic analysis of a phrase without making any additional LLM calls.
It looks the phrase up in the curated euphemisms dictionary and, for unknown phrases, falls
back to a per-word decomposition via a bundled plain-English lexicon with suffix stripping.
When both approaches yield poor coverage (< 60%), an optional LLM-assisted fallback can be
enabled via REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK=1 — it calls llama-server with a
strict-schema dissection prompt and caches successful results back into the
lexicon for future hits.
The result is a structured hook the comedian persona can riff against.
"""

from __future__ import annotations
import os
import re
import json
import logging
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List
from pathlib import Path


if TYPE_CHECKING:
    import httpx

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Resolve the bundled dictionary relative to this file so it works whether the
# package is installed editable or as a wheel.
_EUPHEMISMS_PATH = Path(__file__).parent / "euphemisms.json"
_LEXICON_PATH = Path(__file__).parent / "language_lexicon.json"

# Common suffixes to strip when looking up a stem in the lexicon, ordered from
# longest to shortest so multi-character endings are tried first.
_SUFFIXES = (
    "ization",
    "isation",
    "ations",
    "ation",
    "ments",
    "ness",
    "ment",
    "ity",
    "ies",
    "ize",
    "ise",
    "ers",
    "ing",
    "ous",
    "ive",
    "ble",
    "ful",
    "al",
    "ly",
    "ed",
    "er",
    "es",
    "s",
)


def _load_euphemisms(path: Path = _EUPHEMISMS_PATH) -> Dict[str, Any]:
    """Load the curated euphemisms dictionary from *path*.

    Returns an empty dict if the file cannot be read so the tool degrades
    gracefully rather than raising at import time.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("euphemisms.json is not a dict — returning empty dict")
            return {}
        return data
    except FileNotFoundError:
        logger.warning("euphemisms.json not found at %s — tool will use fallback only", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load euphemisms.json: %s", exc)
        return {}


def _load_lexicon(path: Path = _LEXICON_PATH) -> Dict[str, str]:
    """Load the plain-English lexicon from *path*.

    Returns an empty dict on failure so the tool degrades to v1 fallback.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("language_lexicon.json is not a dict — returning empty dict")
            return {}
        return data
    except FileNotFoundError:
        logger.warning("language_lexicon.json not found at %s — decomposition disabled", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load language_lexicon.json: %s", exc)
        return {}


# Module-level caches — loaded once on first use.
_EUPHEMISMS: Dict[str, Any] | None = None
_LEXICON: Dict[str, str] | None = None


def _get_euphemisms() -> Dict[str, Any]:
    global _EUPHEMISMS
    if _EUPHEMISMS is None:
        _EUPHEMISMS = _load_euphemisms()
    return _EUPHEMISMS


def _get_lexicon() -> Dict[str, str]:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = _load_lexicon()
    return _LEXICON


# ---------------------------------------------------------------------------
# LLM-assisted fallback
# ---------------------------------------------------------------------------

# Prompt for the LLM fallback — strict schema for easy JSON parsing.
_LLM_FALLBACK_PROMPT = (
    "Dissect this phrase as comedian George Carlin would. "
    'Return JSON: {{"literal_words": ["word (plain meaning)", ...], '
    '"euphemism_target": "what it hides or null", '
    '"dissection_suggestion": "one-sentence Carlin angle"}}. '
    "Be terse. Phrase: {phrase}"
)
# Keep the LLM call short — we only need a structured decomposition.
_LLM_MAX_TOKENS: int = 200
_LLM_TEMPERATURE: float = 0.3
# Tight timeout so a slow llama-server doesn't block the tool indefinitely.
_LLM_TIMEOUT_S: float = 0.8


async def _dissect_via_llm(
    phrase: str,
    http_client: "httpx.AsyncClient",
    llama_url: str,
) -> Dict[str, Any] | None:
    """Ask llama-server to dissect *phrase* in the Carlin style.

    Returns a dict with keys ``literal_words`` (list[str]),
    ``euphemism_target`` (str | None), and ``dissection_suggestion`` (str)
    on success, or ``None`` on parse error / network failure / timeout.

    The caller is responsible for checking whether the feature is enabled
    before invoking this function.

    Args:
        phrase: The phrase to dissect (original casing).
        http_client: Shared ``httpx.AsyncClient`` from the calling handler.
        llama_url: Base URL of the llama-server instance.

    """
    import httpx as _httpx

    prompt = _LLM_FALLBACK_PROMPT.format(phrase=phrase)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _LLM_MAX_TOKENS,
        "temperature": _LLM_TEMPERATURE,
        "stream": False,
    }
    _timeout = _httpx.Timeout(_LLM_TIMEOUT_S)

    try:
        resp = await http_client.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            timeout=_timeout,
        )
        resp.raise_for_status()
        raw_content: str = resp.json()["choices"][0]["message"]["content"]
        # Strip markdown code fences if present.
        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```[a-z]*\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content.strip())
        parsed = json.loads(raw_content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned non-dict JSON")
        # Validate required keys are present and non-empty.
        literal_words = parsed.get("literal_words")
        if not isinstance(literal_words, list):
            raise ValueError("literal_words must be a list")
        euphemism_target = parsed.get("euphemism_target")
        dissection_suggestion = str(parsed.get("dissection_suggestion") or "").strip()
        logger.debug(
            "_dissect_via_llm: phrase=%r literal_words=%r target=%r",
            phrase,
            literal_words,
            euphemism_target,
        )
        return {
            "literal_words": literal_words,
            "euphemism_target": euphemism_target,
            "dissection_suggestion": dissection_suggestion,
        }
    except (_httpx.TimeoutException, _httpx.NetworkError, _httpx.ConnectError) as exc:
        logger.debug("_dissect_via_llm network/timeout error (%s) — skipping LLM fallback", exc)
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.debug("_dissect_via_llm parse error (%s) — skipping LLM fallback", exc)
        return None
    except Exception as exc:
        logger.debug("_dissect_via_llm unexpected error (%s) — skipping LLM fallback", exc)
        return None


def _is_valid_llm_result(result: Dict[str, Any]) -> bool:
    """Return True when *result* looks like a usable LLM dissection.

    Rejects degenerate results (empty literal_words list) so we don't
    pollute the lexicon cache with useless entries.
    """
    literal_words = result.get("literal_words")
    return isinstance(literal_words, list) and len(literal_words) > 0


def _cache_llm_result(phrase: str, result: Dict[str, Any], lexicon_path: Path = _LEXICON_PATH) -> None:
    """Write a successful LLM dissection back into the lexicon JSON atomically.

    The phrase is stored under its normalised (lower-stripped) form with
    ``source: "llm-cached"`` so curated entries remain distinguishable.
    Skips the write when *result* fails the sanity check.

    File writes use a tmp-file + os.replace pattern (same as joke_history)
    so an interrupted write never corrupts the stored lexicon.
    """
    if not _is_valid_llm_result(result):
        logger.debug("_cache_llm_result: degenerate result for %r — skipping cache write", phrase)
        return

    normalised = phrase.strip().lower()

    # Build the lexicon entry in the same schema as curated entries.
    # literal_words from LLM is a list[str]; convert to dict for consistency
    # (word → plain meaning extracted from "word (meaning)" or kept as-is).
    lw_list: List[str] = result.get("literal_words") or []
    lw_dict: Dict[str, str] = {}
    for item in lw_list:
        item = str(item).strip()
        # Try to parse "word (meaning)" format.
        m = re.match(r"^(\S+)\s+\((.+)\)$", item)
        if m:
            lw_dict[m.group(1).lower()] = m.group(2)
        elif item:
            # Fallback: use the whole item as both key and value.
            first_word = item.split()[0].lower()
            lw_dict[first_word] = item

    entry: Dict[str, Any] = {
        "literal_words": lw_dict,
        "euphemism_target": str(result.get("euphemism_target") or "unknown").strip() or "unknown",
        "dissection_suggestion": str(result.get("dissection_suggestion") or "").strip(),
        "source": "llm-cached",
    }

    # Load the current lexicon (or start fresh if the file is missing).
    try:
        with lexicon_path.open(encoding="utf-8") as fh:
            current: Dict[str, Any] = json.load(fh)
        if not isinstance(current, dict):
            current = {}
    except FileNotFoundError:
        current = {}
    except Exception as exc:
        logger.warning("_cache_llm_result: failed to read lexicon for caching: %s", exc)
        return

    # Skip write if the entry is already present (avoid unnecessary I/O).
    if normalised in current:
        logger.debug("_cache_llm_result: %r already in lexicon — skipping", normalised)
        return

    current[normalised] = entry

    try:
        lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=lexicon_path.parent,
            prefix=".language_lexicon_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(current, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, lexicon_path)
            logger.info("_cache_llm_result: cached LLM dissection for %r", normalised)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.warning("_cache_llm_result: failed to write lexicon cache: %s", exc)


def _llm_result_to_dissection(phrase: str, llm: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an ``_dissect_via_llm`` result into the standard dissection dict."""
    lw_list: List[str] = llm.get("literal_words") or []
    lw_dict: Dict[str, str] = {}
    for item in lw_list:
        item = str(item).strip()
        m = re.match(r"^(\S+)\s+\((.+)\)$", item)
        if m:
            lw_dict[m.group(1).lower()] = m.group(2)
        elif item:
            first_word = item.split()[0].lower()
            lw_dict[first_word] = item

    euphemism_target = str(llm.get("euphemism_target") or "unknown — LLM dissection").strip()
    dissection_suggestion = str(llm.get("dissection_suggestion") or "").strip()

    return {
        "phrase": phrase,
        "literal_words": lw_dict,
        "euphemism_target": euphemism_target,
        "dissection_suggestion": dissection_suggestion,
        "decomposition_quality": "llm",
    }


def _make_literal_words(phrase: str) -> Dict[str, str]:
    """Return a plain dictionary gloss for each word in an unknown phrase."""
    return {word: f"plain meaning of '{word}'" for word in phrase.lower().split()}


def _decompose_word(word: str, lexicon: Dict[str, str] | None = None) -> tuple[str, str] | None:
    """Try to find *word* (or a stemmed form) in the lexicon.

    Returns ``(root, gloss)`` on success or ``None`` if no match is found.

    The function first tries the bare word, then progressively strips known
    suffixes until the stem matches a lexicon entry.  A minimum stem length of
    three characters is enforced to avoid nonsense matches (e.g. stripping
    ``-ation`` from ``nation`` → ``n``).
    """
    if lexicon is None:
        lexicon = _get_lexicon()

    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return None

    # Exact match first.
    if clean in lexicon:
        return clean, lexicon[clean]

    # Try each suffix in order.
    for suffix in _SUFFIXES:
        if clean.endswith(suffix):
            stem = clean[: -len(suffix)]
            if len(stem) >= 3 and stem in lexicon:
                return stem, lexicon[stem]

    return None


def _decompose_phrase(phrase: str, lexicon: Dict[str, str] | None = None) -> List[Dict[str, str]]:
    """Decompose each word in *phrase* using the lexicon.

    Returns a list of per-word dicts, each containing:
    - ``word``: the original word token
    - ``root``: the matched lexicon entry key (may equal ``word``)
    - ``gloss``: the plain-English meaning
    - ``decomposed``: whether a lexicon match was found

    Words not found in the lexicon receive ``decomposed: false`` and an empty
    gloss so the caller can decide how to handle them.
    """
    if lexicon is None:
        lexicon = _get_lexicon()

    results: List[Dict[str, str]] = []
    for token in phrase.lower().split():
        match = _decompose_word(token, lexicon)
        if match is not None:
            root, gloss = match
            results.append({"word": token, "root": root, "gloss": gloss, "decomposed": "true"})
        else:
            results.append({"word": token, "root": token, "gloss": "", "decomposed": "false"})
    return results


def _dissect_phrase_sync(phrase: str, euphemisms: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """Run the curated-dict + per-word decomposition path synchronously.

    Returns the dissection dict when either path succeeds, or ``None`` when
    decomposition quality is low (caller should try the LLM fallback).
    An *explicit* ``None`` return signals the low-quality sentinel so the
    async wrapper can decide whether to invoke the LLM.
    """
    if euphemisms is None:
        euphemisms = _get_euphemisms()

    normalised = phrase.strip().lower()

    entry = euphemisms.get(normalised)
    if entry is not None:
        return {
            "phrase": phrase,
            "literal_words": entry.get("literal_words", _make_literal_words(normalised)),
            "euphemism_target": entry.get("euphemism_target", "unknown"),
            "dissection_suggestion": entry.get("dissection_suggestion", ""),
        }

    # Fallback: phrase not in dictionary — attempt per-word decomposition.
    words = normalised.split()
    if not words:
        # Whitespace-only input: return a minimal scaffold — no LLM needed.
        return {
            "phrase": phrase,
            "literal_words": {},
            "euphemism_target": "unknown — phrase not in curated dictionary",
            "dissection_suggestion": "No words to dissect.",
            "decomposition_quality": "low",
        }

    breakdown = _decompose_phrase(normalised)
    decomposed_count = sum(1 for item in breakdown if item["decomposed"] == "true")
    ratio = decomposed_count / len(breakdown) if breakdown else 0.0

    if ratio >= 0.6:
        # Rich per-word breakdown.
        literal_words = {
            item["word"]: (f"{item['root']} ({item['gloss']})" if item["root"] != item["word"] else item["gloss"])
            for item in breakdown
        }
        glosses = [f"{item['word']} — {item['gloss']}" for item in breakdown if item["decomposed"] == "true"]
        dissection_suggestion = (
            f"'{phrase}' isn't in the playbook, but the words give it away: "
            + "; ".join(glosses)
            + ". Strip the fancy packaging and say what you mean."
        )
        return {
            "phrase": phrase,
            "literal_words": literal_words,
            "euphemism_target": "unknown — phrase not in curated dictionary",
            "dissection_suggestion": dissection_suggestion,
            "decomposition_quality": "high",
        }

    # Low-quality result — signal to the async wrapper to try the LLM fallback.
    return None


def _low_quality_fallback(phrase: str) -> Dict[str, Any]:
    """Return the v1 flat fallback with ``decomposition_quality: 'low'``."""
    normalised = phrase.strip().lower()
    return {
        "phrase": phrase,
        "literal_words": _make_literal_words(normalised),
        "euphemism_target": "unknown — phrase not in curated dictionary",
        "dissection_suggestion": (
            f"The phrase '{phrase}' isn't in the standard playbook. "
            "Pull each word apart on its own: what does it literally mean, "
            "and what is the speaker using it to avoid saying?"
        ),
        "decomposition_quality": "low",
    }


def dissect_phrase(phrase: str, euphemisms: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a structured dissection of *phrase* (synchronous, no LLM).

    Looks up the normalised phrase in the curated dictionary first.  If not
    found, attempts per-word decomposition via the plain-English lexicon.
    When at least 60% of words decompose successfully the result carries
    ``decomposition_quality: "high"``; otherwise the v1 flat fallback is
    returned with ``decomposition_quality: "low"``.

    For LLM-assisted fallback on low-quality results, use
    :func:`dissect_phrase_async` instead.

    Args:
        phrase: The target phrase to dissect, e.g. ``"thoughts and prayers"``.
        euphemisms: Override the dictionary (mainly for tests).

    Returns:
        A dict with keys ``phrase``, ``literal_words``, ``euphemism_target``,
        ``dissection_suggestion``, and (for fallback paths) ``decomposition_quality``.

    """
    result = _dissect_phrase_sync(phrase, euphemisms)
    if result is None:
        return _low_quality_fallback(phrase)
    return result


async def dissect_phrase_async(
    phrase: str,
    euphemisms: Dict[str, Any] | None = None,
    *,
    http_client: "httpx.AsyncClient | None" = None,
    llama_url: str | None = None,
) -> Dict[str, Any]:
    """Return a structured dissection of *phrase*, with optional LLM fallback.

    Identical to :func:`dissect_phrase` for curated-dict hits and high-quality
    per-word decompositions.  When decomposition quality is low AND
    ``REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK=1`` AND ``http_client`` is
    provided, makes a single llama-server call for a Carlin-style dissection
    and caches the result into the lexicon for future calls.

    Args:
        phrase: The target phrase to dissect.
        euphemisms: Override the dictionary (mainly for tests).
        http_client: Shared ``httpx.AsyncClient``; pass ``None`` to skip LLM.
        llama_url: Base URL of the llama-server instance.

    Returns:
        Same structure as :func:`dissect_phrase`, with an additional
        ``decomposition_quality: "llm"`` value for LLM-sourced results.

    """
    from robot_comic.config import config as _config  # avoid circular at module load

    result = _dissect_phrase_sync(phrase, euphemisms)
    if result is not None:
        # Curated hit or high-quality decomposition — no LLM needed.
        return result

    # Low-quality path: try LLM fallback if enabled and HTTP client is present.
    fallback_enabled = getattr(_config, "LANGUAGE_DISSECT_LLM_FALLBACK_ENABLED", False)
    if fallback_enabled and http_client is not None and llama_url:
        llm_result = await _dissect_via_llm(phrase, http_client, llama_url)
        if llm_result is not None and _is_valid_llm_result(llm_result):
            # Cache for next time, then return the LLM dissection.
            _cache_llm_result(phrase, llm_result)
            # Also invalidate the in-memory lexicon cache so the next call for
            # the same phrase hits the curated dict path.
            global _LEXICON
            _LEXICON = None
            return _llm_result_to_dissection(phrase, llm_result)

    # Final fallback — v1 flat structure with quality flag.
    return _low_quality_fallback(phrase)


class LanguageDissect(Tool):
    """Dissect a phrase or euphemism into its structural components for riff material.

    Returns a structured analysis — for novel phrases the tool may optionally
    invoke llama-server (when REACHY_MINI_LANGUAGE_DISSECT_LLM_FALLBACK=1) to
    produce a Carlin-style dissection and caches the result for future calls.
    The result gives the persona the raw material (literal meanings, what is
    being hidden, a suggested angle) to run the quote-name-strip deconstruction
    pattern.
    """

    name = "language_dissect"
    description = (
        "Analyse a phrase or euphemism to expose what it is hiding. "
        "Returns: the original phrase, a word-by-word literal gloss, what the phrase is "
        "euphemistically concealing, and a short dissection angle for the Carlin pattern "
        "(quote the word → name what it's doing → strip it to the plain version). "
        "Call this whenever the conversation surfaces an institutional euphemism, "
        "a corporate buzzword, a political phrase, or any soft language worth dissecting."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "phrase": {
                "type": "string",
                "description": (
                    "The word or phrase to dissect. Use the exact wording the person said, "
                    "or a well-known euphemism (e.g. 'thoughts and prayers', "
                    "'passed away', 'human resources', 'collateral damage')."
                ),
            },
        },
        "required": ["phrase"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dissect the supplied phrase and return structured riff material."""
        phrase: str = kwargs.get("phrase", "").strip()
        logger.info("Tool call: language_dissect phrase=%r", phrase)

        if not phrase:
            return {"error": "No phrase provided — pass a non-empty 'phrase' argument."}

        http_client = getattr(deps, "http_client", None)
        llama_url: str | None = getattr(deps, "llama_url", None)
        return await dissect_phrase_async(phrase, http_client=http_client, llama_url=llama_url)
