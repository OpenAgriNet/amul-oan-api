"""Voice's pretranslation: the caller's words into the English the agent reads.

From voice-oan-api at amul-dev ``3b19835``: the pretranslation step of
``stream_voice_message`` and the conversation it gives the translator
(``app/services/voice.py``), with the prompt and parsing it uses
(``app/services/translation.py``). The translator sees the recent understood
conversation, the glossary and the ambiguity rules, and an exact glossary term
the model only transliterated is put back as its English label. The ambiguity
rules are voice's own (``assets/voice_ambiguity_terms.json``): voice's include
rules chat's do not, such as reading an ASR'd નિદાન as insemination rather than
diagnosis. The glossary is chat's with voice's own rows on top
(``agents/voice/tools/terms.py``), as amul-dev reads abortion and silage.

Each attempt runs on the pretranslation tier ``llm_core`` picks for the turn,
with that tier's client and model, where voice built them from the environment.
Voice's TranslateGemma fallback for when the chain is switched off is not
brought over; the chain does that job.

When nothing usable comes back, the caller is asked to repeat, unless moderation
rejected the query, which is declined instead.
"""
from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional, Sequence, Union

from openai import AsyncOpenAI

from agents.tools import terms as _terms
from agents.voice.tools.terms import TERM_PAIRS
from app.config import settings
from app.llm_core import Step
from app.llm_core.config_model import StepClientKind
from app.llm_core.execution import classify
from app.services.translation import LANG_CODES, LANG_NAMES, _get_langfuse
from app.turn.types import ClassifierResult, Pretranslated, Pretranslation, Turn
from app.voice.classifiers import _FRAGMENT_RESPONSES, RenderForCaller, _canned_for_caller
from app.voice.history import HISTORY_MARKERS as _HISTORY_MARKERS
from app.voice.history import history_pair as _history_pair
from app.voice.moderation import pretranslation_timeout_s
from app.voice.stt_signals import detect_stt_signal
from app.voice.trace import current_trace, step_tiers
from helpers.utils import get_logger

logger = get_logger(__name__)


def _load_voice_ambiguity_terms() -> list:
    path = Path(__file__).resolve().parents[2] / "assets" / "voice_ambiguity_terms.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# The prompt below asks for hints the way voice does; they come from voice's rules.
get_ambiguity_hints_for_query = partial(
    _terms.get_ambiguity_hints_for_query, terms=_load_voice_ambiguity_terms()
)

# Pretranslation sees the recent conversation, minus the turns that carry no
# meaning. Generous on purpose: prod assistant turns are p50 140 / max 589 chars,
# so 8 exchanges fit comfortably and the prefill cost is a few hundred tokens.
PRETRANSLATION_CONTEXT_MAX_EXCHANGES = 8
PRETRANSLATION_CONTEXT_MAX_CHARS = 3000

# User-side history markers for a turn nobody understood. The exchange they open
# — marker plus the agent's "please repeat" — is dropped whole.
_GARBLED_USER_MARKERS = frozenset(
    {
        _HISTORY_MARKERS["fragment"],
        _HISTORY_MARKERS["low_confidence"],
        _HISTORY_MARKERS["pretranslation_failed"],
        _HISTORY_MARKERS["stt_no_audio"],
        _HISTORY_MARKERS["stt_unclear"],
        _HISTORY_MARKERS["moderation_reject"],
    }
)
# Bookkeeping markers: the user side says nothing, the agent's turn is real.
_SILENT_USER_MARKERS = frozenset(
    {_HISTORY_MARKERS["greeting"], _HISTORY_MARKERS["outbound_intro"]}
)
# The agent asking the farmer to say it again. Those exchanges are noise, and
# keeping them hides the question the farmer is still answering: session
# 601f1db4 asked "cow or buffalo?", then "please repeat", then "I could not
# understand" — and the farmer's પસ (buffalo) became "pus".
_REPEAT_REQUEST_RE = re.compile(
    r"\b(please (repeat|say (that|it) again|ask your question again)|"
    r"(did not|didn't|could not|couldn't|can't|cannot|wasn't able to) "
    r"(quite )?(understand|hear|catch)|"
    r"(was not|wasn't) clear|"
    r"having (some )?trouble (processing|answering))",
    re.IGNORECASE,
)
# Searched, not matched from the start: a turn that told the caller a booking
# they had not heard saved that note ahead of their words (agents/tools/ai_call.py).
_USER_QUERY_WRAPPER_RE = re.compile(r'\*\*User:\*\*\s*"(.*)"$', re.DOTALL)


def _history_exchanges(history: list) -> list[tuple[Optional[str], str]]:
    """Group history into (user text, assistant text) exchanges, oldest first.

    An exchange starts at a user prompt and collects every assistant text part
    until the next one (a hold line and the answer after a tool call both
    belong to it). Tool calls and returns are skipped.
    """
    exchanges: list[list] = []
    for message in history or []:
        for part in getattr(message, "parts", []) or []:
            kind = getattr(part, "part_kind", "")
            content = getattr(part, "content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if kind == "user-prompt":
                exchanges.append([content.strip(), []])
            elif kind == "text":
                if not exchanges:
                    exchanges.append([None, []])
                exchanges[-1][1].append(content.strip())
    return [(user, " ".join(texts)) for user, texts in exchanges]


def _pretranslation_context(history: list) -> list[tuple[str, str]]:
    """The recent conversation for pretranslation, as (speaker, English text).

    Keeps what was understood and drops what was not: an exchange goes when the
    farmer's side was an STT/garble marker or the agent's side was a request to
    repeat (or an error line). A farmer turn translated with an ``[unclear ...]``
    token loses only its own side — the agent's reply to it can be the very
    question the farmer is answering now.
    What remains is bounded by exchange count and characters, newest kept.

    Read straight off the in-memory ``history`` already passed in — no session
    or Redis read, because pretranslation is on the latency path.
    """
    kept: list[list[tuple[str, str]]] = []
    for user, assistant in _history_exchanges(history):
        if user is not None:
            if user.startswith("Runtime context for this turn"):
                user = None
            elif user in _GARBLED_USER_MARKERS or detect_stt_signal(user) is not None:
                continue
            elif user in _SILENT_USER_MARKERS:
                user = None
        if user is not None:
            match = _USER_QUERY_WRAPPER_RE.search(user)
            if match:
                user = match.group(1).strip()
            if "[unclear" in user.lower():
                # Only the farmer's side is unreliable. The agent's reply may be
                # the real follow-up question they are answering now.
                user = None
        if assistant and _REPEAT_REQUEST_RE.search(assistant):
            continue
        turns = []
        if user:
            turns.append(("Farmer", user))
        if assistant:
            turns.append(("Assistant", assistant))
        if turns:
            kept.append(turns)

    window: list[list[tuple[str, str]]] = []
    used = 0
    for turns in reversed(kept[-PRETRANSLATION_CONTEXT_MAX_EXCHANGES:]):
        size = sum(len(text) for _, text in turns)
        if window and used + size > PRETRANSLATION_CONTEXT_MAX_CHARS:
            break
        window.append(turns)
        used += size
    return [turn for turns in reversed(window) for turn in turns]


def _canonical_history_user_text(kind: str, fallback: str = "") -> str:
    return _HISTORY_MARKERS.get(kind, fallback or kind)


# A species' young ('ઊંટનું બચ્ચું') fuzzy-matches any '<animal>નું બચ્ચું', so a cow
# or buffalo calf would get a Lamb or Camel calf hint. Hint these rows only when
# the species itself is in the farmer's words.
_SPECIES_YOUNG_HINT_STEMS = {"camel calf": "ઊંટ", "goat kid": "બકર", "lamb": "ઘેટ"}


# Glossary terms are matched word by word (#610). A fuzzy score over the whole
# message matched words that only look alike, and the prompt then made the
# translator use that term's English: "દૂધની અંદર લોહી આવે છે" (blood comes in the
# milk) scored 82 against "લોહી જરે છે" and was translated "BLOOD IN URINE";
# સાયડામાં (in the shade) became diarrhoea through ઝાડા. A word may take a case
# ending, and a term's last vowel may change only as its noun declines (આફરો ->
# આફરા), so પાડો (a male buffalo calf) does not match પાડી or પાડવા.
_GU_WORD_RE = re.compile(r"[\u0a80-\u0aff]+|[A-Za-z]+")
# Long and short ી/િ and ૂ/ુ are one vowel here, as speech-to-text writes either
# (બિજદાન for બીજદાન): words, terms and the endings below are in that form.
_GU_VOWEL_LENGTH = str.maketrans({"ી": "િ", "ૂ": "ુ", "ઈ": "ઇ", "ઊ": "ઉ"})
_GU_CASE_ENDING = "(?:ને|નો|નિ|નું|ના|નાં|થિ|માં|મા|માંથિ|પર)?"
_GU_FINAL_VOWELS = ("ું", "ો", "િ", "ા", "ે", "ુ", "ૈ", "ૌ")
_GU_DECLINED = {"ો": ("ો", "ા"), "ું": ("ું", "ુ", "ા", "ાં", "ો")}
# A vowel sign speech-to-text doubles (વિિયાણ) is one.
_GU_DOUBLED_SIGN_RE = re.compile(r"([\u0a81-\u0a83\u0abe-\u0acc])\1+")
# Speech-to-text's spellings of વિયાણ (calving) in prod voice, 22 Oct-18 Nov, put
# back in the caller's words before they are matched. Input only: the text the
# translator gets is unchanged. વિયાવાની (about to calve) stays as it is.
_GU_CALLER_SPELLINGS = (("વયાણ", "વિયાણ"), ("વ્યાણ", "વિયાણ"), ("બિયાણ", "વિયાણ"), ("વિહાણ", "વિયાણ"))


def _gu_words(text: str) -> list[str]:
    text = text.replace("\u200c", "").replace("\u200d", "").lower().translate(_GU_VOWEL_LENGTH)
    return _GU_WORD_RE.findall(_GU_DOUBLED_SIGN_RE.sub(r"\1", text))


def _caller_spelling(word: str) -> str:
    for heard, said in _GU_CALLER_SPELLINGS:
        if word.startswith(heard):
            return said + word[len(heard):]
    return word


@lru_cache(maxsize=None)
def _gu_word_rule(term: str) -> tuple[str, Optional[re.Pattern]]:
    """The stem a word matching ``term`` starts with, and the pattern it must
    match: ``term`` with a case ending, or with its last vowel declined. A Latin
    word matches only itself."""
    if not ("\u0a80" <= term[:1] <= "\u0aff"):
        return term, None
    ending = next((v for v in _GU_FINAL_VOWELS if term.endswith(v) and len(term) > len(v) + 1), None)
    if ending is None:
        # A short term takes a case ending only: જર (placenta) is not જરે (oozes).
        stem, endings = term, (r"[ાિુેૈોૌ]?" if len(term) > 2 else "")
    else:
        stem = term[: -len(ending)]
        endings = "(?:" + "|".join(map(re.escape, _GU_DECLINED.get(ending, (ending,)))) + ")"
    return stem, re.compile(re.escape(stem) + endings + _GU_CASE_ENDING)


def _gu_word_matches(word: str, term: str) -> bool:
    if word == term:
        return True
    pattern = _gu_word_rule(term)[1]
    return pattern is not None and pattern.fullmatch(word) is not None


def _glossary_variants(field: str) -> list[tuple[str, ...]]:
    """The words of each spelling a glossary field gives: "ગેસ આફરો,ફૂલી જવું" and
    "કાળીયો તાવ (એન્થ્રેક્સ)" hold two each."""
    parts = re.split(r"[,/;|()]", field or "")
    return [tuple(words) for words in (_gu_words(p) for p in parts) if words]


@lru_cache(maxsize=1)
def _glossary_index() -> dict:
    """Each glossary spelling by the stem of its first word: {stem: [(row number,
    row, words)]}, so a message is checked only against spellings it can match."""
    index: dict = {}
    for order, tp in enumerate(TERM_PAIRS):
        for variant in _glossary_variants(tp.gu) + _glossary_variants(tp.transliteration):
            index.setdefault(_gu_word_rule(variant[0])[0], []).append((order, tp, variant))
    return index


def _get_glossary_hints_for_gu_query(text: str, max_results: int = 7) -> str:
    """Glossary terms whose words are in the Gujarati input (or their
    transliteration in its Latin words).

    Returns a compact hint string like:
      આફરો = Bloat (rumen tympany)
      આંચળ = Udder / Teat
    """
    if not text or not text.strip():
        return ""

    words = [_caller_spelling(word) for word in _gu_words(text)]
    index = _glossary_index()
    # Row number -> (row, word count of its longest spelling found).
    found: dict[int, tuple] = {}
    for i, word in enumerate(words):
        for end in range(1, len(word) + 1):
            for order, tp, variant in index.get(word[:end], ()):
                k = len(variant)
                if k > found.get(order, (None, 0))[1] and i + k <= len(words) \
                        and all(_gu_word_matches(words[i + j], variant[j]) for j in range(k)):
                    found[order] = (tp, k)

    scored: list[tuple[str, str, float]] = []
    for order in sorted(found):
        tp, k = found[order]
        stem = _SPECIES_YOUNG_HINT_STEMS.get((tp.en or "").strip().lower())
        if stem and stem not in text:
            continue
        # A longer term found is the more specific hint.
        scored.append((tp.gu, tp.en, float(k)))

    if not scored:
        return ""

    # Deduplicate by English term, keep highest score
    seen_en: dict[str, tuple[str, str, float]] = {}
    for gu, en, score in scored:
        en_key = en.lower()
        if en_key not in seen_en or score > seen_en[en_key][2]:
            seen_en[en_key] = (gu, en, score)

    top = sorted(seen_en.values(), key=lambda x: x[2], reverse=True)[:max_results]
    return "\n".join(f"  {gu} = {en}" for gu, en, _ in top)


def _whole_ascii_token_pattern(term: str) -> str:
    escaped = re.escape(term.strip())
    escaped = re.sub(r"\\\s+", r"\\s+", escaped)
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def _apply_exact_glossary_transliteration_replacements(source_text: str, translation: str) -> str:
    """Replace model transliterations with glossary labels for exact Gujarati term hits."""
    if not source_text or not translation:
        return translation

    source_lower = source_text.lower()
    cleaned = translation

    for tp in TERM_PAIRS:
        gu_term = (tp.gu or "").strip()
        transliteration = (tp.transliteration or "").strip()
        english_label = (tp.en or "").strip()
        if not gu_term or not transliteration or not english_label:
            continue
        if len(transliteration) < 3 or transliteration.lower() == english_label.lower():
            continue
        if gu_term.lower() not in source_lower:
            continue
        if re.search(_whole_ascii_token_pattern(english_label), cleaned, flags=re.IGNORECASE):
            continue

        cleaned = re.sub(
            _whole_ascii_token_pattern(transliteration),
            english_label,
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


# ગેસ is a real veterinary term (bloat), so the species corruption table below
# is the one thing that stays conditional — applied on a turn that did not ask
# the species, it would turn "my cow has gas" into "cow".
_BOOKING_QUESTION_RE = re.compile(r"insemination|confirm", re.IGNORECASE)
_SPECIES_QUESTION_RE = re.compile(r"cow\s+or\s+(a\s+)?buffalo|buffalo\s+or\s+(a\s+)?cow", re.IGNORECASE)


def _conversation_context(conversation: Optional[Sequence[tuple[str, str]]]) -> str:
    """Give the translator the recent, understood conversation.

    Built by ``voice._pretranslation_context``: garbled exchanges and "please
    repeat" turns are already gone, so the last assistant turn here is the
    question the farmer is still answering even when a repeat request came in
    between. That gap is what sank session 601f1db4: "cow or buffalo?", a
    garbled answer, "please repeat", "I could not understand" — and the one-turn
    context quoted the last of those, so પસ (buffalo) became "pus" and the agent
    searched for pus treatment.

    The earlier turns resolve what one turn cannot: an animal named three turns
    back, the service the farmer already asked for, a technician list the agent
    read out before a confirmation. The answer-slot rule still points at the
    last assistant turn only — that is where the named options live.
    """
    turns = [
        (speaker, (text or "").strip())
        for speaker, text in (conversation or [])
        if (text or "").strip()
    ]
    if not turns:
        return ""
    last_assistant = next(
        (text for speaker, text in reversed(turns) if speaker == "Assistant"), None
    )
    transcript = "\n".join(f"{speaker}: {text}" for speaker, text in turns)
    block = (
        "\nConversation so far (English; turns that were not understood are omitted):\n"
        f"{transcript}\n"
        "Use it to resolve what the farmer is referring to — an animal, a service, a name "
        "already mentioned — but translate ONLY the new message; never copy earlier turns "
        "into it, and never add a meaning the new message does not carry.\n"
    )
    if last_assistant:
        block += (
            f'The assistant\'s last question was: "{last_assistant}"\n'
            "The message you are translating is most likely the farmer's ANSWER to it.\n"
            "Context rule: if that turn offered a specific set of NAMED options — two species, a "
            "list of technician names — and this message is a garbled token in the answer "
            "slot, translate it as the option it clearly corresponds to, "
            "spelled EXACTLY as that option appears above. Gujarati proper nouns survive ASR "
            "badly: syllables split, double, or drop (BAHECHARBHAI -> 'be be char bhai'), and "
            "callers say the parts out of order or give only one. If the token fits TWO of the "
            "options, or none, do NOT choose — transliterate it and leave it ambiguous, because "
            "acting on the wrong option is worse than re-asking. If the turn offered no options, "
            "this rule adds nothing: translate conservatively as the rules above require.\n"
            "Agreement is never inferred. If that question asked for a yes/no confirmation, "
            "render an affirmation or a refusal ONLY when the utterance actually carries one "
            "(હા -> yes, ના -> no). Never supply a 'yes' the farmer did not say, never drop a "
            "'ના'/'no' that they did, and if they answered with something else entirely — a "
            "question, a symptom, a name — translate THAT and leave the confirmation unanswered.\n"
        )
    if last_assistant and _SPECIES_QUESTION_RE.search(last_assistant):
        # Issue #306: 525 real answer turns, 46.1% -> 79.6% usable species, 0 regressions.
        block += (
            "Species hint: a token that plausibly sounds like ગાય (cow) or ભેંસ (buffalo) is "
            "that species. Common ASR corruptions: ગેસ/ગસ/ગ્યાસ/કેસ/ગાડી -> cow; "
            "બસ/બેસ/મેસ/બેંસ/પસ/દેસ -> buffalo. This narrows the general 'do not infer animal "
            "species' rule for THIS turn only; if it matches neither, still say 'unclear animal'.\n"
        )
    return block


def _heat_alert_context(script: str, conversation: Optional[Sequence[tuple[str, str]]]) -> str:
    """The reply a heat-alert call's booking turns on: the first one, to the script
    Sarlaben speaks outside our history, and the yes/no on the booking after it.
    The species and technician answers keep the rules they already have: this text
    in front of them turned the ASR's કાય માટે (for cow) into "For what?"."""
    last = next((text for speaker, text in reversed(conversation or []) if speaker == "Assistant"), None)
    if last is not None and not _BOOKING_QUESTION_RE.search(last):
        return ""
    block = ""
    if last is None:
        block += (
            "\nThis is Sarlaben's outbound heat-alert call: Sarlaben rang the farmer and opened with "
            f"«{script}»\nThe message is the farmer's answer to that offer to register an artificial "
            "insemination ticket.\n"
        )
    block += (
        "\nOn this call a short reply to the booking question is an answer, not noise: હા, હ, હં, હા જી, "
        "જી, ઓકે, બરાબર, કરો, કરી દો -> 'Yes.'; ના, નહી, નથી, ના ના -> 'No.'; હમ, હમ્મ -> 'Hmm.' (only these two, "
        "spelled with મ: હં is 'Yes.'). A greeting stays a greeting.\n"
    )
    if last is not None and "confirm" in last.lower():
        block += (
            "The last assistant turn read a booking back: translate an agreement as 'Yes, confirm the booking.', "
            "and only a clear agreement.\n"
        )
    return block


def _build_openai_pretranslation_messages(
    source_name: str,
    source_code: str,
    text: str,
    conversation: Optional[Sequence[tuple[str, str]]] = None,
    call_context: str = "",
) -> list[dict[str, str]]:
    # -- Domain context ------------------------------------------------
    domain_preamble = (
        "You are translating messages from Indian dairy farmers calling the Amul AI helpline (voiced as 'Sarlaben' / સરલાબેન). "
        "The farmers speak Gujarati and ask about animal health, milk production, fodder, breeding, and dairy cooperative services.\n\n"
        "IMPORTANT translation rules:\n"
        "- Your job is faithful pretranslation for safe routing, not correction, completion, or advice.\n"
        "- Preserve uncertainty from the original speech. Do not repair missing words, fill missing slots, or choose a clean interpretation when the audio transcript is ambiguous.\n"
        "- Words that look like human names (e.g. સલાદ, સરલા, ગંગા) are almost always ANIMAL NAMES (cow/buffalo names). Transliterate them as-is, do NOT translate literally.\n"
        "- If a garbled token does not clearly map to a real medicine, feed, symptom, or service term, do NOT invent a meaning. Keep the translation conservative.\n"
        "- Kinship words like બેન, બહેન, ભાઈ are often address markers for Sarlaben or filler in phone speech. Do not turn them into the caller's gender. Use 'Sarlaben' only if the caller is clearly addressing the assistant; otherwise omit the address marker.\n"
        "- 'ભાઈ' in livestock context may refer to a male animal (bull/ox); keep it generic if the word could also be an address marker.\n"
        "- Prefer veterinary/agricultural meanings only when the term is clear in the original transcript. If choosing the agricultural meaning requires guessing, preserve the uncertain token.\n"
        "- Do not infer animal species. If cow/buffalo/sheep/goat is unclear, write 'unclear animal' or keep the uncertain token.\n"
    )

    domain_preamble += _conversation_context(conversation) + call_context

    # -- Ambiguity hints from ambiguity_terms.json ---------------------
    # include_ask=False so "ask" type entries (clarifying-question rules
    # meant for the answering agent) don't leak into the translator prompt
    # and get echoed back as appended follow-up questions.
    ambiguity_hints = get_ambiguity_hints_for_query(text, include_ask=False)
    if ambiguity_hints:
        domain_preamble += f"\nDomain-specific disambiguation rules for terms in this message:\n{ambiguity_hints}\n"

    # -- Glossary hints (top matching gu→en terms) ---------------------
    glossary_hints = _get_glossary_hints_for_gu_query(text, max_results=7)
    if glossary_hints:
        domain_preamble += (
            f"\nGlossary (Gujarati → English) for terms likely in this message:\n{glossary_hints}\n"
            "Glossary usage rule: If the user's term clearly matches a glossary line above, use the right-hand English label "
            "from that line instead of transliterating the Gujarati token. Do not output the romanized/transliterated form "
            "when a matching glossary English label is available. Domain-specific disambiguation rules above override "
            "glossary lines if they conflict.\n"
        )

    system_content = (
        f"{domain_preamble}\n"
        "Translate the user's message to faithful spoken English for an internal agent. "
        "Respond with JSON: {\"translation\": \"...\"}.\n\n"
        "Do not preserve markdown, bullets, bracketed duplicates, or other formatting clutter, but do preserve the meaning uncertainty.\n"
        "When the input is garbled noise, random syllables, fragmentary, contradictory, or when any key noun, animal species, medicine, feed, product, disease, symptom, or requested action is uncertain, still provide the most faithful translation possible, using markers such as 'unclear animal', 'unclear feed name', 'unclear symptom', or '[unclear token]' instead of inventing missing meaning.\n"
        "Never convert a doubtful token into a specific medicine, feed, disease, animal species, or service term just because it would make a plausible livestock question."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": text.strip()},
    ]


async def _create_pretranslation_response(
    client: AsyncOpenAI,
    model: str,
    *,
    source_name: str,
    source_code: str,
    text: str,
    max_tokens: int,
    conversation: Optional[Sequence[tuple[str, str]]] = None,
    call_context: str = "",
):
    """Single OpenAI-compatible pretranslation call, parametrized by (client, model).

    Replaces the former ``_create_openai_pretranslation_response`` /
    ``_create_oss_pretranslation_response`` twins (identical bodies bar the model)."""
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=_build_openai_pretranslation_messages(
                source_name, source_code, text, conversation, call_context
            ),
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        ),
        timeout=pretranslation_timeout_s(),
    )


def _extract_translation_from_response(response) -> str:
    """Extract the translation string from an OpenAI JSON response."""
    raw = (response.choices[0].message.content or "").strip()
    return _extract_translation_from_raw(raw)


def _extract_translation_from_raw(raw: str) -> str:
    """Extract the translation string from raw model output."""
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        return (data.get("translation") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        # Fallback: use raw content if JSON parsing fails
        return raw


async def _translate_to_english_pretranslation(
    text: str,
    source_lang: str,
    *,
    client: AsyncOpenAI,
    model: str,
    label: str,
    translation_provider_label: str,
    extra_metadata: Optional[dict] = None,
    max_tokens: int = 1024,
    conversation: Optional[Sequence[tuple[str, str]]] = None,
    call_context: str = "",
) -> str:
    """Single parametrized pretranslation body — the collapse of the former
    ``translate_to_english_with_gpt5_mini`` / ``translate_to_english_with_oss_vllm``
    twins (identical bodies bar the client/model/langfuse-label). The two public
    wrappers below supply the managed-OpenAI vs OSS-vLLM (client, model, label);
    everything else — early-returns, glossary replacement, empty/timeout handling,
    the Langfuse ``query_pretranslation`` observation — is shared verbatim.

    Returns the translated text, or the original text on empty output.
    """
    if not text or not text.strip():
        return text

    if source_lang.lower() in {"english", "en"}:
        return text

    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())

    langfuse = _get_langfuse()
    try:
        if not langfuse:
            response = await _create_pretranslation_response(
                client, model,
                source_name=source_name,
                source_code=source_code,
                text=text,
                max_tokens=max_tokens,
                conversation=conversation,
                call_context=call_context,
            )
            translated_text = _extract_translation_from_response(response)
            if not translated_text:
                logger.warning(
                    "%s pretranslation returned empty - source_lang=%s query_chars=%s",
                    label, source_lang, len(text or ""),
                )
                return text
            translated_text = _apply_exact_glossary_transliteration_replacements(text, translated_text)
            return translated_text

        with langfuse.start_as_current_observation(
            name="query_pretranslation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "target_lang": "english",
                "text": text,
            },
            model=model,
            metadata={
                "translation_provider": translation_provider_label,
                "pipeline_stage": "query_pretranslation",
                **(extra_metadata or {}),
            },
        ) as observation:
            response = await _create_pretranslation_response(
                client, model,
                source_name=source_name,
                source_code=source_code,
                text=text,
                max_tokens=max_tokens,
                conversation=conversation,
                call_context=call_context,
            )
            translated_text = _extract_translation_from_response(response)
            if not translated_text:
                logger.warning(
                    "%s pretranslation returned empty - source_lang=%s query_chars=%s",
                    label, source_lang, len(text or ""),
                )
                observation.update(output="__EMPTY__")
                return text
            translated_text = _apply_exact_glossary_transliteration_replacements(text, translated_text)
            observation.update(output=translated_text)
            return translated_text
    except asyncio.TimeoutError as e:
        logger.error(
            "%s pretranslation timed out - source_lang=%s model=%s timeout_seconds=%.2f query_chars=%s",
            label,
            source_lang,
            model,
            pretranslation_timeout_s(),
            len(text or ""),
        )
        raise TimeoutError(f"{label} pretranslation timed out") from e


async def _pretranslate_on(
    target,
    *,
    text: str,
    source_lang: str,
    conversation: Sequence[tuple[str, str]],
    pipeline_profile: str,
    call_context: str = "",
) -> str:
    """One attempt on the tier ``llm_core`` chose: voice's OSS or managed call,
    by the tier's kind."""
    if target.kind == "oss":
        return await _translate_to_english_pretranslation(
            text,
            source_lang,
            client=target.handle,
            model=target.model_name,
            label="OSS vLLM",
            translation_provider_label="vllm",
            extra_metadata={"pipeline_profile": pipeline_profile or "oss"},
            conversation=conversation,
            call_context=call_context,
        )
    return await _translate_to_english_pretranslation(
        text,
        source_lang,
        client=target.handle,
        model=target.model_name,
        label="OpenAI",
        translation_provider_label=target.provider,
        conversation=conversation,
        call_context=call_context,
    )


async def _voice_pretranslation(
    turn: Turn,
    *,
    execution,
    background,
    render: RenderForCaller,
) -> Union[Pretranslated, ClassifierResult]:
    query = turn.query
    requested_source_lang = (turn.source_lang or "gu").strip().lower()
    requested_target_lang = (turn.target_lang or "gu").strip().lower()
    process_id = turn.call.process_id if turn.call is not None else None
    processing_query = query
    history_user_text = query
    trace = current_trace()

    if requested_source_lang not in {"en", "english"}:
        _pre_mt = execution.info(Step.PRE_TRANSLATION)
        _pretrans_provider_label = "vllm" if _pre_mt.kind == "oss" else "openai"
        _pretrans_model = _pre_mt.model_name
        _pretrans_requested_tier = _pre_mt.kind
        _pretranslation_attempts: list[dict[str, object]] = []
        _pretranslation_actual_tier = _pretrans_requested_tier
        _pretranslation_actual_provider = _pretrans_provider_label
        _pretranslation_actual_model = _pretrans_model
        logger.info(
            "Translation pipeline enabled; pretranslating %s -> en with %s (variant=%s)",
            requested_source_lang,
            _pretrans_model,
            execution.profile_name,
        )
        # Recent understood conversation — see _pretranslation_context.
        _pretranslation_conversation = _pretranslation_context(list(turn.history))
        _heat_alert = turn.call.heat_alert if turn.call is not None else None
        _call_context = (
            _heat_alert_context(_heat_alert.script, _pretranslation_conversation)
            if _heat_alert and _heat_alert.script
            else ""
        )

        async def _run_pretranslation_attempt(a):
            nonlocal _pretranslation_actual_tier, _pretranslation_actual_provider, _pretranslation_actual_model
            attempt_info: dict[str, object] = {
                "tier": a.kind,
                "provider": a.provider,
                "model": a.model_name,
                "endpoint": a.endpoint,
            }
            _pretranslation_attempts.append(attempt_info)
            try:
                translated = await _pretranslate_on(
                    a,
                    text=query,
                    source_lang=requested_source_lang,
                    conversation=_pretranslation_conversation,
                    pipeline_profile=execution.profile_name,
                    call_context=_call_context,
                )
            except Exception as _attempt_exc:
                attempt_info["status"] = "error"
                attempt_info["error_class"] = type(_attempt_exc).__name__
                attempt_info["error_reason"] = classify(_attempt_exc).value
                raise
            attempt_info["status"] = "ok"
            _pretranslation_actual_tier = a.kind
            _pretranslation_actual_provider = a.provider
            _pretranslation_actual_model = a.model_name
            return translated

        try:
            # turn.v1's pre_translation step: the question in, the English out, and
            # the tier that served once the walker ran (written when the step closes).
            with trace.stage(
                "pre_translation",
                as_type="generation",
                input=trace.text(query),
                metadata={
                    "provider": _pretrans_provider_label,
                    "source_lang": requested_source_lang,
                    "pipeline_profile": execution.profile_name,
                },
                model=_pretrans_model,
            ) as step:
                try:
                    processing_query = await execution.run_adapter(
                        Step.PRE_TRANSLATION,
                        _run_pretranslation_attempt,
                        client_kind=StepClientKind.RAW_OPENAI,
                    )
                finally:
                    step.record(metadata=step_tiers(execution, Step.PRE_TRANSLATION))
                step.record(output=trace.text(processing_query))
            _pretranslation_fallback_used = (
                len(_pretranslation_attempts) > 1
                and _pretranslation_attempts[0].get("status") == "error"
            )
            trace.set_pretranslation(
                text=processing_query,
                provider=_pretrans_provider_label,
                fallback_used=_pretranslation_fallback_used,
                requested_tier=_pretrans_requested_tier,
                requested_provider=_pretrans_provider_label,
                requested_model=_pretrans_model,
                actual_tier=_pretranslation_actual_tier,
                actual_provider=_pretranslation_actual_provider,
                actual_model=_pretranslation_actual_model,
                attempts=_pretranslation_attempts,
            )
            history_user_text = processing_query or _canonical_history_user_text("low_confidence")
        except Exception as e:
            logger.error(
                "pretranslation failed (all tiers) for session_id=%s source_lang=%s error=%s",
                turn.session_id,
                requested_source_lang,
                e,
            )
            processing_query = ""
            _pretranslation_fallback_used = (
                len(_pretranslation_attempts) > 1
                and _pretranslation_attempts[0].get("status") == "error"
            )
            trace.set_pretranslation(
                text=processing_query,
                provider="failed",
                fallback_used=_pretranslation_fallback_used,
                requested_tier=_pretrans_requested_tier,
                requested_provider=_pretrans_provider_label,
                requested_model=_pretrans_model,
                actual_tier="failed",
                actual_provider="failed",
                actual_model="failed",
                attempts=_pretranslation_attempts,
            )
            history_user_text = _canonical_history_user_text("pretranslation_failed")
    else:
        trace.set_pretranslation(
            text=query,
            provider="none",
            fallback_used=False,
            requested_tier="none",
            requested_provider="none",
            requested_model="none",
            actual_tier="none",
            actual_provider="none",
            actual_model="none",
            attempts=[],
        )

    if background is not None:
        background.set_history_text(history_user_text)

    # ── Empty-pretranslation guard ───────────────────────────────
    # Only short-circuit when pretranslation produced no usable text
    # at all (i.e. both primary and fallback failed). True noise still
    # routes to the agent, which is better at asking for
    # clarification in context than a canned global retry.
    if (
        requested_source_lang not in {"en", "english"}
        and not (processing_query or "").strip()
    ):
        # This short-circuits the agent, so resolve moderation here: a
        # rejected query must be declined rather than asked to repeat.
        declined = await background.decline() if background is not None else None
        if declined is not None:
            return declined
        logger.info(
            "Pretranslation produced no usable text; asking to repeat - session_id=%s process_id=%s query_chars=%s",
            turn.session_id, process_id, len(query or ""),
        )
        low_conf_resp_for_history = _FRAGMENT_RESPONSES["en"]
        low_conf_resp_for_caller = await _canned_for_caller(
            render, low_conf_resp_for_history, requested_target_lang, _FRAGMENT_RESPONSES,
            execution=execution,
        )
        return ClassifierResult(
            canned_text=low_conf_resp_for_caller,
            label="pretranslation_empty",
            history_pair=_history_pair(
                history_user_text or _canonical_history_user_text("low_confidence"),
                low_conf_resp_for_history,
            ),
            outcome="pretranslation_empty",
        )

    return Pretranslated(query=processing_query, lang="en")


def voice_pretranslation(render: RenderForCaller) -> Pretranslation:
    """Voice's pretranslation for ``SurfaceProfile.pretranslation``.

    ``render`` puts the English "please repeat" line into the caller's language
    when there is no canned one, as it does for the classifiers.
    """
    return partial(_voice_pretranslation, render=render)
