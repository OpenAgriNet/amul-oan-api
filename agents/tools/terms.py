import json
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz

# Load term pairs from JSON file with UTF-8 encoding
term_pairs = json.load(open('assets/glossary_terms.json', 'r', encoding='utf-8'))

class Language(str, Enum):
    ENGLISH = "en"
    GUJARATI = "gu"
    TRANSLITERATION = "transliteration"

class TermPair(BaseModel):
    en: str = Field(description="English term")
    gu: str = Field(description="Gujarati term")
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
    TERM_PAIRS.append(TermPair(**pair))

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

import re
from rapidfuzz import process

# Build English index from glossary
EN_INDEX = {tp.en.lower(): tp for tp in TERM_PAIRS}
EN_TERMS = list(EN_INDEX.keys())

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
        lw = word.lower().strip()

        # Exact match
        if lw in EN_INDEX:
            gujarati = EN_INDEX[lw].gu
        else:
            # Fuzzy fallback (very high threshold to avoid false positives)
            match_term, score, _ = process.extractOne(
                lw, EN_TERMS, score_cutoff=threshold
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
) -> str:
    """
    Find glossary terms that appear in `text` (exact or fuzzy match with high threshold)
    and return a mini-glossary string for injection into a translation prompt.

    Uses word and multi-word phrase spans (1–4 words) from the text, fuzzy-matched
    against EN_TERMS, so Gemma can use consistent Gujarati terminology.

    Args:
        text: The sentence or batch to be translated (English).
        threshold: Minimum similarity 0–1 (default 0.95). Converted to 0–100 for rapidfuzz.
        max_terms: Maximum number of (en -> gu) pairs to include (default 25).

    Returns:
        Formatted string like "Mastitis -> આઉનો/બાવલાનો સોજો\\nMilk Production -> ..."
        or empty string if no matches.
    """
    if not text or not text.strip():
        return ""
    score_cutoff = int(threshold * 100) if 0 < threshold <= 1 else 95
    words = text.split()
    if not words:
        return ""
    # Dedupe by canonical English term (lowercase key)
    term_to_gu: dict[str, tuple[str, str]] = {}  # en_lower -> (en_display, gu)
    seen_phrases: set[str] = set()

    # Longer phrases first so we match "Milk Production" before "Milk"
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n]).strip()
            if not phrase or phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            match = process.extractOne(phrase, EN_TERMS, score_cutoff=score_cutoff)
            if not match:
                continue
            en_term, score, _ = match
            if en_term in term_to_gu:
                continue
            tp = EN_INDEX[en_term]
            # Use original casing from glossary for display
            term_to_gu[en_term] = (tp.en, tp.gu)
            if len(term_to_gu) >= max_terms:
                break
        if len(term_to_gu) >= max_terms:
            break

    if not term_to_gu:
        return ""
    lines = [f"{en} -> {gu}" for en, gu in term_to_gu.values()]
    return "\n".join(lines)