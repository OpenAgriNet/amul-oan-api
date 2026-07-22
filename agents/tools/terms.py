import json
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz
import re
from helpers.utils import get_logger

logger = get_logger(__name__)

def _load_glossary_file(filename: str, *, warn_on_fail: bool = False) -> list[dict]:
    candidates = [
        Path.cwd() / "assets" / filename,
        Path(__file__).resolve().parents[2] / "assets" / filename,
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    return loaded if isinstance(loaded, list) else []
            except Exception as e:
                if warn_on_fail:
                    logger.warning("Failed loading glossary file %s at %s: %s", filename, path, e)
                return []
    if warn_on_fail:
        logger.warning(
            "Glossary file %s not found in expected locations: %s",
            filename,
            ", ".join(str(p) for p in candidates),
        )
    return []


# Load term pairs from JSON files with UTF-8 encoding.
term_pairs = _load_glossary_file("glossary_terms.json")
hindi_term_pairs = _load_glossary_file("glossary_terms_hindi_openrouter.json", warn_on_fail=True)


def _load_gu_term_policy() -> dict:
    candidates = [
        Path.cwd() / "assets/gu_term_policy.json",
        Path(__file__).resolve().parents[2] / "assets/gu_term_policy.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


GU_TERM_POLICY = _load_gu_term_policy()
PREFERRED_GU_BY_EN = {
    str(k).strip().lower(): str(v).strip()
    for k, v in (GU_TERM_POLICY.get("preferred", {}) if isinstance(GU_TERM_POLICY, dict) else {}).items()
    if str(k).strip() and str(v).strip()
}
ALLOWED_ALIASES_BY_EN = {
    str(k).strip().lower(): [str(v).strip() for v in vals if str(v).strip()]
    for k, vals in (GU_TERM_POLICY.get("allowed_aliases", {}) if isinstance(GU_TERM_POLICY, dict) else {}).items()
    if str(k).strip() and isinstance(vals, list)
}
INPUT_ALIASES_BY_EN = {
    str(k).strip().lower(): [str(v).strip() for v in vals if str(v).strip()]
    for k, vals in (GU_TERM_POLICY.get("input_aliases", {}) if isinstance(GU_TERM_POLICY, dict) else {}).items()
    if str(k).strip() and isinstance(vals, list)
}

class Language(str, Enum):
    ENGLISH = "en"
    GUJARATI = "gu"
    HINDI = "hi"
    TRANSLITERATION = "transliteration"

class TermPair(BaseModel):
    en: str = Field(description="English term")
    gu: str = Field(description="Gujarati term")
    hi: str = Field(default="", description="Hindi term")
    transliteration: str = Field(description="Transliteration of Gujarati term to English")
    mr: str = Field(default="", description="Marathi term (for backward compatibility)")

    def __str__(self):
        return f"{self.en} -> {self.gu} ({self.transliteration})"

# Convert raw dictionaries to TermPair objects
# Handle backward compatibility: if JSON has 'mr' but not 'gu', map 'mr' to 'gu'
TERM_PAIRS = []
for pair in term_pairs:
    # If 'gu' is not present but 'mr' is, use 'mr' as 'gu'
    if 'gu' not in pair and 'mr' in pair:
        pair['gu'] = pair['mr']
    en_key = str(pair.get("en", "")).strip().lower()
    if en_key in PREFERRED_GU_BY_EN:
        pair["gu"] = PREFERRED_GU_BY_EN[en_key]
    TERM_PAIRS.append(TermPair(**pair))

HI_TERM_PAIRS = []
for pair in hindi_term_pairs:
    try:
        parsed = TermPair(
            en=str(pair.get("en", "")),
            gu="",
            hi=str(pair.get("hi", "")),
            transliteration=str(pair.get("transliteration", "")),
        )
        if parsed.en.strip() and parsed.hi.strip():
            HI_TERM_PAIRS.append(parsed)
    except Exception:
        continue


async def search_terms(
    term: str, 
    max_results: int = 5,
    threshold: float = 0.7,
    language: Language = None
) -> str:
    """Search for terms using fuzzy partial string matching across all fields.
    
    Args:
        term: The term to search for
        max_results: Maximum number of results to return
        threshold: Minimum similarity score (0-1) to consider a match (default is 0.7)
        language: Optional language to restrict search to (en/gu/transliteration)
        
    Returns:
        str: Formatted string with matching results and their scores
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
        
    matches = []
    term = term.lower()
    
    for term_pair in TERM_PAIRS:
        max_score = 0
        
        # Check English term if no language specified or language is English
        if language in [None, Language.ENGLISH]:
            en_score = fuzz.ratio(term, term_pair.en.lower()) / 100.0
            max_score = max(max_score, en_score)
            
        # Check Gujarati term if no language specified or language is Gujarati    
        if language in [None, Language.GUJARATI]:
            gu_score = fuzz.ratio(term, term_pair.gu.lower()) / 100.0
            max_score = max(max_score, gu_score)

        # Check Hindi term if no language specified or language is Hindi
        if language in [None, Language.HINDI]:
            hi_score = fuzz.ratio(term, term_pair.hi.lower()) / 100.0
            max_score = max(max_score, hi_score)
            
        # Check transliteration if no language specified or language is transliteration
        if language in [None, Language.TRANSLITERATION]:
            tr_score = fuzz.ratio(term, term_pair.transliteration.lower()) / 100.0
            max_score = max(max_score, tr_score)
            
        if max_score >= threshold:
            matches.append((term_pair, max_score))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)    
    
    if len(matches) > 0:
        matches = matches[:max_results]
        return f"Matching Terms for `{term}`\n\n" + "\n".join([f"{match[0]} [{match[1]:.0%}]" for match in matches])
    else:
        return f"No matching terms found for `{term}`"


### Utility functions for Correcting Document Search Results
from rapidfuzz import process

# Build English index from glossary
EN_INDEX = {tp.en.lower(): tp for tp in TERM_PAIRS}
EN_TERMS = list(EN_INDEX.keys())
HI_EN_INDEX = {tp.en.lower(): tp for tp in HI_TERM_PAIRS}
HI_EN_TERMS = list(HI_EN_INDEX.keys())


def _normalize_lookup_key(text: str) -> str:
    text = str(text or "").lower().strip()
    if not text:
        return ""
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s/\-().'&]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_parenthetical_suffix(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", _normalize_lookup_key(text)).strip()


def _canonical_display_term(canonical_en: str) -> str:
    for tp in TERM_PAIRS:
        if _normalize_lookup_key(tp.en) == canonical_en:
            return tp.en
    return canonical_en.title()


def _build_canonical_alias_map() -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    canonical_terms: dict[str, tuple[str, str]] = {}
    alias_to_canonical: dict[str, str] = {}

    for canonical_en, preferred_gu in PREFERRED_GU_BY_EN.items():
        canonical_terms[canonical_en] = (_canonical_display_term(canonical_en), preferred_gu)
        aliases = {canonical_en}
        aliases.update(_normalize_lookup_key(alias) for alias in INPUT_ALIASES_BY_EN.get(canonical_en, []))
        aliases.update(_normalize_lookup_key(alias) for alias in ALLOWED_ALIASES_BY_EN.get(canonical_en, []))

        for tp in TERM_PAIRS:
            norm_en = _normalize_lookup_key(tp.en)
            stripped_en = _strip_parenthetical_suffix(tp.en)
            if not norm_en:
                continue
            if norm_en == canonical_en or stripped_en == canonical_en or tp.gu == preferred_gu:
                aliases.add(norm_en)

        for alias in aliases:
            if alias:
                alias_to_canonical[alias] = canonical_en

    return canonical_terms, alias_to_canonical


CANONICAL_TERMS, ALIAS_TO_CANONICAL_EN = _build_canonical_alias_map()
CANONICAL_ALIAS_TERMS = list(ALIAS_TO_CANONICAL_EN.keys())


def _tokenize_lookup_key(text: str) -> list[str]:
    return [token for token in re.split(r"[\s/\-().&]+", _normalize_lookup_key(text)) if token]


def _has_meaningful_token_overlap(left: str, right: str) -> bool:
    left_tokens = {token for token in _tokenize_lookup_key(left) if len(token) >= 3}
    right_tokens = {token for token in _tokenize_lookup_key(right) if len(token) >= 3}
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens)


def _normalize_target_lang(target_lang: str) -> str:
    normalized = str(target_lang or "").strip().lower()
    if normalized in {"gu", "gujarati"}:
        return "gu"
    if normalized in {"hi", "hindi"}:
        return "hi"
    return "gu"

def build_glossary_pattern(terms):
    """
    Build a regex alternation pattern for glossary terms.
    Longer terms first to ensure multi-word phrases
    (e.g., 'Yellow Mosaic Virus') are matched before 'Virus'.
    """
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    return r"\b(" + "|".join(escaped) + r")\b"

# Precompile regex pattern once
GLOSSARY_PATTERN = re.compile(build_glossary_pattern(EN_TERMS), flags=re.IGNORECASE)

def normalize_text_with_glossary(text: str, threshold=97):
    """
    Append Gujarati term in brackets next to English glossary terms. Preserves formatting & avoids spacing issues.
    NOTE: Adds about 100ms of latency to the search results. Can it be optimized?
    """

    def replacer(match):
        word = match.group(0)
        lw = _normalize_lookup_key(word)

        canonical_en = ALIAS_TO_CANONICAL_EN.get(lw)
        if canonical_en:
            gujarati = CANONICAL_TERMS[canonical_en][1]
        elif lw in EN_INDEX:
            gujarati = EN_INDEX[lw].gu
        else:
            alias_match = process.extractOne(lw, CANONICAL_ALIAS_TERMS, score_cutoff=threshold) or (None, 0, None)
            if alias_match and alias_match[0]:
                gujarati = CANONICAL_TERMS[ALIAS_TO_CANONICAL_EN[alias_match[0]]][1]
            else:
                match_term, score, _ = process.extractOne(
                    lw, EN_TERMS, score_cutoff=threshold, scorer=fuzz.ratio
                ) or (None, 0, None)
                if not match_term:
                    return word
                gujarati = EN_INDEX[match_term].gu

        # Decide spacing: if next char is alphanumeric, add space after replacement
        after = match.end()
        if after < len(text) and text[after].isalnum():
            return f"{word} [{gujarati}] "
        else:
            return f"{word} [{gujarati}]"

    return GLOSSARY_PATTERN.sub(replacer, text)


def get_mini_glossary_for_text(
    text: str,
    threshold: float = 0.95,
    max_terms: int = 25,
    target_lang: str = "gu",
) -> str:
    """
    Find glossary terms that appear in `text` (exact or fuzzy match with high threshold)
    and return a mini-glossary string for injection into a translation prompt.

    Uses word and multi-word phrase spans (1–4 words) from the text, fuzzy-matched
    against the target glossary index, so Gemma can use consistent terminology.

    Args:
        text: The sentence or batch to be translated (English).
        threshold: Minimum similarity 0–1 (default 0.95). Converted to 0–100 for rapidfuzz.
        max_terms: Maximum number of (en -> target) pairs to include (default 25).
        target_lang: Target language for glossary values ("gu"/"gujarati" or "hi"/"hindi").

    Returns:
        Formatted string like "Mastitis -> <target term>\\nMilk Production -> ..."
        or empty string if no matches.
    """
    if not text or not text.strip():
        return ""
    normalized_target = _normalize_target_lang(target_lang)
    if normalized_target == "hi":
        target_en_index = HI_EN_INDEX
        target_en_terms = HI_EN_TERMS
        target_value_field = "hi"
    else:
        target_en_index = EN_INDEX
        target_en_terms = EN_TERMS
        target_value_field = "gu"
    if not target_en_terms:
        return ""
    score_cutoff = int(threshold * 100) if 0 < threshold <= 1 else 95
    # Normalize phrase candidates so punctuation doesn't reduce fuzzy matches.
    words = re.findall(r"[A-Za-z0-9\-/]+", text)
    if not words:
        return ""
    # Dedupe by canonical English term (lowercase key)
    term_to_target: dict[str, tuple[str, str]] = {}  # en_lower -> (en_display, target_term)
    seen_phrases: set[str] = set()

    # Longer phrases first so we match "Milk Production" before "Milk"
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n]).strip()
            if not phrase or phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            normalized_phrase = _normalize_lookup_key(phrase)
            canonical_en = ALIAS_TO_CANONICAL_EN.get(normalized_phrase)

            if not canonical_en or canonical_en not in target_en_index:
                alias_match = None
                if n >= 2 and len(normalized_phrase) >= 6:
                    alias_match = process.extractOne(
                        normalized_phrase,
                        CANONICAL_ALIAS_TERMS,
                        score_cutoff=score_cutoff,
                        scorer=fuzz.ratio,
                    )
                if alias_match and _has_meaningful_token_overlap(normalized_phrase, alias_match[0]):
                    canonical_en = ALIAS_TO_CANONICAL_EN.get(alias_match[0])
                if canonical_en and canonical_en not in target_en_index:
                    canonical_en = None

            if canonical_en:
                if canonical_en in term_to_target:
                    continue
                target_pair = target_en_index[canonical_en]
                target_value = getattr(target_pair, target_value_field, "").strip()
                if not target_value:
                    continue
                term_to_target[canonical_en] = (target_pair.en, target_value)
            else:
                if n == 1:
                    exact_term = target_en_index.get(normalized_phrase)
                    if not exact_term:
                        continue
                    match = (exact_term.en.lower(), 100, None)
                else:
                    match = process.extractOne(
                        normalized_phrase,
                        target_en_terms,
                        score_cutoff=score_cutoff,
                        scorer=fuzz.ratio,
                    )
                    if match and not _has_meaningful_token_overlap(normalized_phrase, match[0]):
                        match = None
                if not match:
                    continue
                en_term, score, _ = match
                if en_term in term_to_target:
                    continue
                tp = target_en_index[en_term]
                target_value = getattr(tp, target_value_field, "").strip()
                if not target_value:
                    continue
                # Use original casing from glossary for display
                term_to_target[en_term] = (tp.en, target_value)
            if len(term_to_target) >= max_terms:
                break
        if len(term_to_target) >= max_terms:
            break

    if not term_to_target:
        return ""
    lines = [f"{en} -> {target_value}" for en, target_value in term_to_target.values()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ambiguity hints — dynamically injected into system prompt based on query
# ---------------------------------------------------------------------------

def _load_ambiguity_terms() -> list:
    candidates = [
        Path.cwd() / "assets/ambiguity_terms.json",
        Path(__file__).resolve().parents[2] / "assets/ambiguity_terms.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
    return []


_AMBIGUITY_TERMS = _load_ambiguity_terms()


def get_ambiguity_hints_for_query(query: str, threshold: float | None = None, include_ask: bool = True) -> str:
    """
    Fuzzy-match incoming query (any language) against ambiguity_terms.json.
    Returns a formatted string of matching rules to inject into the system prompt,
    or empty string if no matches.

    Args:
        query: The raw user query string.
        threshold: Minimum similarity 0-1. When None (default), falls back to
                   settings.ambiguity_match_threshold, then to 0.80. Lower than
                   glossary matching since Gujarati term matching is fuzzier.
        include_ask: If False, skip entries with type == "ask" (those rules tell
                     the answering agent to ask a clarifying question and must
                     NOT leak into the pretranslation prompt — otherwise the
                     translator dutifully appends the clarifying question to
                     its English output).

    Returns:
        Formatted rules string, e.g.:
          "- 'ઉથલા' always means repeat breeder, NOT vomiting."
        or "" if no terms matched.
    """
    if not query or not _AMBIGUITY_TERMS:
        return ""

    # Allow callers to override; fall back to env setting, then hard-coded default
    if threshold is None:
        try:
            from app.config import settings as _settings
            threshold = _settings.ambiguity_match_threshold
        except Exception:
            threshold = 0.80

    score_cutoff = int(threshold * 100)
    matched_rules = []
    seen = set()

    query_lower = query.lower().strip()

    for entry in _AMBIGUITY_TERMS:
        gu_terms = entry.get("gu_terms", [])
        rule = entry.get("rule", "").strip()
        if not rule or not gu_terms:
            continue
        if not include_ask and entry.get("type") == "ask":
            continue

        # Check each trigger term against the query using substring + fuzzy
        for term in gu_terms:
            term_lower = term.lower().strip()
            if not term_lower:
                continue
            # Substring match first (fast path)
            if term_lower in query_lower:
                if rule not in seen:
                    matched_rules.append(f"- {rule}")
                    seen.add(rule)
                break
            # Fuzzy match fallback
            score = fuzz.partial_ratio(term_lower, query_lower)
            if score >= score_cutoff:
                if rule not in seen:
                    matched_rules.append(f"- {rule}")
                    seen.add(rule)
                break

    return "\n".join(matched_rules)
