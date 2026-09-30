"""Voice's pretranslation: the caller's words into the English the agent reads.

From voice-oan-api at amul-dev ``3b19835``: the pretranslation step of
``stream_voice_message`` and the conversation it gives the translator
(``app/services/voice.py``), with the prompt and parsing it uses
(``app/services/translation.py``). The translator sees the recent understood
conversation, the glossary and the ambiguity rules, and an exact glossary term
the model only transliterated is put back as its English label. The ambiguity
rules are voice's own (``assets/voice_ambiguity_terms.json``): voice's include
rules chat's do not, such as reading an ASR'd નિદાન as insemination rather than
diagnosis. The glossary is shared with chat.

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
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Union

from openai import AsyncOpenAI

from agents.tools import terms as _terms
from agents.tools.terms import TERM_PAIRS
from app.config import settings
from app.llm_core import Step
from app.llm_core.config_model import StepClientKind
from app.services.translation import LANG_CODES, LANG_NAMES, _get_langfuse
from app.turn.types import ClassifierResult, Pretranslated, Pretranslation, Turn
from app.voice.classifiers import _FRAGMENT_RESPONSES, RenderForCaller, _canned_for_caller
from app.voice.history import HISTORY_MARKERS as _HISTORY_MARKERS
from app.voice.history import history_pair as _history_pair
from app.voice.stt_signals import detect_stt_signal
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
_USER_QUERY_WRAPPER_RE = re.compile(r'^\*\*User:\*\*\s*"(.*)"$', re.DOTALL)


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
            match = _USER_QUERY_WRAPPER_RE.match(user)
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


def _get_glossary_hints_for_gu_query(text: str, max_results: int = 7) -> str:
    """Fuzzy-match Gujarati input against glossary gu/transliteration fields.

    Returns a compact hint string like:
      આફરો = Bloat (rumen tympany)
      આંચળ = Udder / Teat
    """
    from rapidfuzz import fuzz as _fuzz

    if not text or not text.strip():
        return ""

    text_lower = text.lower().strip()
    scored: list[tuple[str, str, float]] = []

    for tp in TERM_PAIRS:
        scores: list[float] = []
        gu_lower = (tp.gu or "").lower().strip()
        translit_lower = (tp.transliteration or "").lower().strip()

        # Check substring containment first (fast path), ignoring empty fields.
        if gu_lower:
            scores.append(100.0 if gu_lower in text_lower else _fuzz.partial_ratio(gu_lower, text_lower))
        if translit_lower:
            scores.append(
                100.0 if translit_lower in text_lower else _fuzz.partial_ratio(translit_lower, text_lower)
            )
        if not scores:
            continue
        best = max(scores)

        if best >= 75:
            scored.append((tp.gu, tp.en, best))

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


def _build_openai_pretranslation_messages(
    source_name: str,
    source_code: str,
    text: str,
    conversation: Optional[Sequence[tuple[str, str]]] = None,
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

    domain_preamble += _conversation_context(conversation)

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
):
    """Single OpenAI-compatible pretranslation call, parametrized by (client, model).

    Replaces the former ``_create_openai_pretranslation_response`` /
    ``_create_oss_pretranslation_response`` twins (identical bodies bar the model)."""
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=_build_openai_pretranslation_messages(
                source_name, source_code, text, conversation
            ),
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        ),
        timeout=settings.openai_pretranslation_timeout_seconds,
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
            )
            translated_text = _extract_translation_from_response(response)
            if not translated_text:
                logger.warning(
                    "%s pretranslation returned empty - source_lang=%s query=%r",
                    label, source_lang, (text or "")[:100],
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
            )
            translated_text = _extract_translation_from_response(response)
            if not translated_text:
                logger.warning(
                    "%s pretranslation returned empty - source_lang=%s query=%r",
                    label, source_lang, (text or "")[:100],
                )
                observation.update(output="__EMPTY__")
                return text
            translated_text = _apply_exact_glossary_transliteration_replacements(text, translated_text)
            observation.update(output=translated_text)
            return translated_text
    except asyncio.TimeoutError as e:
        logger.error(
            "%s pretranslation timed out - source_lang=%s model=%s timeout_seconds=%.2f query_chars=%s query_preview=%r",
            label,
            source_lang,
            model,
            settings.openai_pretranslation_timeout_seconds,
            len(text or ""),
            (text or "")[:160],
        )
        raise TimeoutError(f"{label} pretranslation timed out") from e


async def _pretranslate_on(
    target,
    *,
    text: str,
    source_lang: str,
    conversation: Sequence[tuple[str, str]],
    pipeline_profile: str,
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
        )
    return await _translate_to_english_pretranslation(
        text,
        source_lang,
        client=target.handle,
        model=target.model_name,
        label="OpenAI",
        translation_provider_label=target.provider,
        conversation=conversation,
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

    if requested_source_lang not in {"en", "english"}:
        logger.info(
            "Translation pipeline enabled; pretranslating %s -> en with %s (variant=%s)",
            requested_source_lang,
            execution.info(Step.PRE_TRANSLATION).model_name,
            execution.profile_name,
        )
        # Recent understood conversation — see _pretranslation_context.
        _pretranslation_conversation = _pretranslation_context(list(turn.history))
        try:
            processing_query = await execution.run_adapter(
                Step.PRE_TRANSLATION,
                partial(
                    _pretranslate_on,
                    text=query,
                    source_lang=requested_source_lang,
                    conversation=_pretranslation_conversation,
                    pipeline_profile=execution.profile_name,
                ),
                client_kind=StepClientKind.RAW_OPENAI,
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
            history_user_text = _canonical_history_user_text("pretranslation_failed")

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
            "Pretranslation produced no usable text; asking to repeat - session_id=%s process_id=%s query=%r",
            turn.session_id, process_id, query,
        )
        low_conf_resp_for_history = _FRAGMENT_RESPONSES["en"]
        low_conf_resp_for_caller = await _canned_for_caller(
            render, low_conf_resp_for_history, requested_target_lang, _FRAGMENT_RESPONSES
        )
        return ClassifierResult(
            canned_text=low_conf_resp_for_caller,
            label="pretranslation_empty",
            history_pair=_history_pair(
                history_user_text or _canonical_history_user_text("low_confidence"),
                low_conf_resp_for_history,
            ),
        )

    return Pretranslated(query=processing_query, lang="en")


def voice_pretranslation(render: RenderForCaller) -> Pretranslation:
    """Voice's pretranslation for ``SurfaceProfile.pretranslation``.

    ``render`` puts the English "please repeat" line into the caller's language
    when there is no canned one, as it does for the classifiers.
    """
    return partial(_voice_pretranslation, render=render)
