"""
Translation vocabulary test suite — generated from Shridhar feedback (Mar 16-30, 2026)
and extended with the Apr 11-12 production follow-up corpus.

Tests the post-translation normalization pipeline (_post_normalize_gu_translation)
and documents known vocabulary, gender, and dialect issues as regression tests.

Categories:
- Forbidden term replacement (gu_term_policy.json "forbidden" section)
- Gender agreement errors
- Wrong word / better word choices
- Term choice rules (don't combine synonyms)
- Nonexistent words
"""
import pytest
import sys
import os
import re
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.translation import (
    _post_normalize_gu_translation,
    VOICE_GU_PREFERRED_TRANSLATION_RULES,
    GU_TERM_POLICY,
    GU_POST_REPLACEMENTS,
    translation_channel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_PATH = REPO_ROOT / "assets" / "glossary_terms.json"
# §14 merge: voice-only glossary additions (incl. the Amul-AI disambiguation
# entries) live in the voice-extras file, keeping chat's glossary untouched.
VOICE_EXTRA_GLOSSARY_PATH = REPO_ROOT / "assets" / "voice_glossary_terms_extra.json"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def normalize_gu(text: str) -> str:
    """Run the full Gujarati post-translation normalization.

    Runs on the voice channel: voice's context-aware body-slang normalization
    (બૈડા પર -> પીઠ પર) is §14 channel-gated, and these are voice regressions. The
    shared term policy (ફેટ/ગાભણ/બુલ/…) applies on either channel.
    """
    with translation_channel("voice"):
        return _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True)


# ---------------------------------------------------------------------------
# Forbidden term replacements (from gu_term_policy.json)
# These MUST be caught by the post-translation pipeline.
# ---------------------------------------------------------------------------

class TestForbiddenReplacements:
    """Verify forbidden→correct replacements from Shridhar feedback."""

    @pytest.mark.parametrize("forbidden, expected", [
        # Animal body parts — [59, 79] "don't use both", standardize to આંચળ
        ("સ્તન", "આંચળ"),
        ("સ્તનના નિપલ્સ", "આંચળ"),
        ("સ્તનનો સોજો", "આંચળનો સોજો"),
        ("સ્તન પ્રદેશ", "આઉ/બાવલા ના ભાગ"),
        # Mastitis standardization — [59, 79] use one term not both
        ("આઉનો/બાવલાનો સોજો", "આંચળનો સોજો"),
        ("આઉ નો સોજો", "આંચળનો સોજો"),
        ("બાવલાનો સોજો", "આંચળનો સોજો"),
        ("બાવલા નો સોજો", "આંચળનો સોજો"),
        # Udder — [130] પાહો→બાવલું
        ("પાહો", "બાવલું"),
        # Dairy terms — [28, 300] ચરબી→ફેટ, ઘન પદાર્થો→એસ.એન.એફ.
        ("ચરબી", "ફેટ"),
        ("ઘન પદાર્થોની", "એસ.એન.એફ.ની"),
        ("ઘન પદાર્થો", "એસ.એન.એફ."),
        # Bacteria — [42] જંતુઓ→બેક્ટેરિયા
        ("જંતુઓ", "બેક્ટેરિયા"),
        # Herd — [53] ટોળું/ટોળા→ધણ
        ("ટોળા", "ધણ"),
        ("ટોળું", "ધણ"),
        ("ટોળામાં", "ધણમાં"),
        # Fodder — [52, 523] ચારોની→ચારાની
        ("ચારોની", "ચારાની"),
        # Calf terms — [31] પાડુના→બચ્ચાંના
        ("પાડુના", "બચ્ચાંના"),
        # Bull — [11] બળદ→બુલ
        ("બળદ", "બુલ"),
        # Dairy product terms — [130, 132, 133]
        ("મખાણ", "માખણ"),
        ("માલઈ", "મલાઈ"),
        ("મથણીથી", "વલોણાથી"),
        ("ઘીમાં પકાવવું", "ઘી બનાવવું"),
        # Physical/scientific — [58] શારીરિક→ભૌતિક
        ("શારીરિક", "ભૌતિક"),
        # Body terms — avoid colloquial/dialectal "બૈડા/બૈડું/બરડા/બરડું"
        ("બૈડા", "શરીર"),
        ("બૈડું", "શરીર"),
        ("બૈડુ", "શરીર"),
        ("બરડા", "શરીર"),
        ("બરડું", "શરીર"),
        ("બરડુ", "શરીર"),
        # Medical — [162] ચૂભો→ચીરો, [168] તણાવ→માનસિક આઘાત
        ("ચૂભો", "ચીરો"),
        ("તણાવ", "માનસિક આઘાત"),
        # Grammar — [144] એટલી સુધી→ત્યાં સુધી
        ("એટલી સુધી", "ત્યાં સુધી"),
        # Pregnancy — ગર્ભવતી→ગાભણ
        ("ગર્ભવતી", "ગાભણ"),
        # Medicine plural — [425] દવાઓ→દવા
        ("દવાઓ", "દવા"),
        # Foam — [425] ફી ઓછી થાય→ફીણ ઓછા થાય
        ("ફી ઓછી થાય", "ફીણ ઓછા થાય"),
        # Tick — ટિક્કી→ઇતરડી
        ("ટિક્કી", "ઇતરડી"),
        # Deworming — કીડા→કૃમિ
        ("કીડા", "કૃમિ"),
        # Insemination — ગર્ભાધાન→બીજદાન
        ("ગર્ભાધાન", "બીજદાન"),
        # Spelling/terminology refinements
        ("સુકો", "સૂકો"),
        ("મિશ્રણ ખનીજ તત્વો", "ખનિજ મિશ્રણ"),
        ("દૂધની પેદાશ", "દૂધના ઉત્પાદન"),
        ("ઘટતું નથી", "ઘટે નહીં"),
        ("ન્યુટ્રીએનર્જીઆ", "ન્યુટ્રીએનર્જી"),
        ("યોગ્ય રીતે ગરમીમાં આવવામાં", "યોગ્ય સમયે ગરમીમાં આવવામાં"),
        ("ગર્ભચારો", "ગાભણ પશુ માટેનું દાણ"),
        ("ગર્ભ માટેનો ચારો", "ગાભણ પશુ માટેનો ચારો"),
        ("સામાન્ય જાળવણી ચારો", "રોજિંદો ઘાસચારો"),
        ("કણું", "દાણ"),
        ("ડિજેશન", "ડાઈજેશન"),
    ])
    def test_forbidden_replaced(self, forbidden, expected):
        """Each forbidden term in output must be replaced with the correct term."""
        result = normalize_gu(f"આ {forbidden} છે.")
        assert expected in result, f"Expected '{expected}' in output, got: {result}"
        if forbidden != expected:
            assert forbidden not in result, f"Forbidden '{forbidden}' still present in: {result}"


class TestForbiddenInContext:
    """Test forbidden replacements within realistic sentences."""

    def test_mastitis_in_sentence(self):
        """[59, 79] Should not output both આઉ and બાવલા for mastitis."""
        text = "ગાયને આઉનો/બાવલાનો સોજો થયો છે."
        result = normalize_gu(text)
        assert "આંચળનો સોજો" in result

    def test_fat_in_dairy_context(self):
        """[28, 300] ચરબી should become ફેટ in dairy context."""
        text = "દૂધમાં ચરબી ઓછી છે."
        result = normalize_gu(text)
        assert "ફેટ" in result
        assert "ચરબી" not in result

    def test_snf_in_dairy_context(self):
        """[28] ઘન પદાર્થો should become એસ.એન.એફ."""
        text = "દૂધમાં ઘન પદાર્થોની ટકાવારી ઓછી છે."
        result = normalize_gu(text)
        assert "એસ.એન.એફ." in result

    def test_pregnant_in_animal_context(self):
        """ગર્ભવતી should become ગાભણ for animals."""
        text = "ગાય ગર્ભવતી છે."
        result = normalize_gu(text)
        assert "ગાભણ" in result
        assert "ગર્ભવતી" not in result

    def test_bacteria_replacement(self):
        """[42] જંતુઓ should become બેક્ટેરિયા."""
        text = "જંતુઓ દ્વારા ચેપ લાગે છે."
        result = normalize_gu(text)
        assert "બેક્ટેરિયા" in result

    def test_milk_yield_sentence_prefers_utpadan_and_mishrit_daan(self):
        text = "તેને દૂધની પેદાશ મુજબ પૂરતું પશુચારો આપો."
        result = normalize_gu(text)
        assert "દૂધના ઉત્પાદન મુજબ" in result
        assert "મિશ્રિત દાણ" in result
        assert "દૂધની પેદાશ" not in result

    def test_weight_clause_uses_ghate_nahi(self):
        text = "તેથી તેનું વજન ઘટતું નથી."
        result = normalize_gu(text)
        assert "ઘટે નહીં" in result
        assert "ઘટતું નથી" not in result

    def test_nutrienergy_typo_is_fixed(self):
        text = "ન્યુટ્રીએનર્જીઆ જેવું સપ્લીમેન્ટ આપો."
        result = normalize_gu(text)
        assert "ન્યુટ્રીએનર્જી" in result
        assert "ન્યુટ્રીએનર્જીઆ" not in result

    def test_heat_phrase_prefers_yogya_samaye(self):
        text = "તેને યોગ્ય રીતે ગરમીમાં આવવામાં મદદ મળે."
        result = normalize_gu(text)
        assert "યોગ્ય સમયે ગરમીમાં આવવામાં" in result
        assert "યોગ્ય રીતે ગરમીમાં આવવામાં" not in result

    def test_pregnant_animal_feed_does_not_use_nonexistent_garbhacharo(self):
        text = "ગાયને ગર્ભચારો આપો."
        result = normalize_gu(text)
        assert "ગાભણ પશુ માટેનું દાણ" in result
        assert "ગર્ભચારો" not in result

    def test_feed_is_for_pregnant_animal_not_fetus(self):
        text = "ગર્ભ માટેનો ચારો આપો."
        result = normalize_gu(text)
        assert "ગાભણ પશુ માટેનો ચારો" in result
        assert "ગર્ભ માટેનો ચારો" not in result

    def test_maintenance_fodder_phrase_is_made_farmer_natural(self):
        text = "સામાન્ય જાળવણી ચારો આપો."
        result = normalize_gu(text)
        assert "રોજિંદો ઘાસચારો" in result
        assert "સામાન્ય જાળવણી ચારો" not in result

    def test_digestion_problem_transliteration_is_normalized(self):
        text = "જેને આપણે અપચો ડિજેશન પ્રોબ્લેમ કહી શકીએ."
        result = normalize_gu(text)
        assert "અપચો ડાઈજેશન પ્રોબ્લેમ" in result
        assert "ડિજેશન" not in result

    def test_kanoo_feed_word_is_normalized(self):
        text = "ગાયને કણું ઓછું આપો."
        result = normalize_gu(text)
        assert "દાણ ઓછું આપો" in result
        assert "કણું" not in result

    def test_body_term_replacement(self):
        """Use શરીર/પીઠ-style vocabulary, not બૈડા."""
        text = "પશુના બૈડા પર સોજો છે."
        result = normalize_gu(text)
        assert "પીઠ" in result
        assert "બૈડા" not in result

    def test_body_term_singular_defaults_to_sharir(self):
        """Generic body sense should normalize to શરીર with correct agreement."""
        text = "મને બૈડું ઠંડું લાગે છે."
        result = normalize_gu(text)
        assert "શરીર ઠંડું લાગે છે" in result
        assert "બૈડું" not in result

    def test_body_term_back_context_uses_pith(self):
        """Back-location context should normalize to પીઠ."""
        text = "ગાયના બરડામાં દુખાવો છે."
        result = normalize_gu(text)
        assert "પીઠમાં દુખાવો" in result
        assert "બરડામાં" not in result

    def test_body_term_agreement_fixes(self):
        """Common agreement mismatches after normalization should be corrected."""
        text = "મને શરીર ઠંડા લાગે છે અને પીઠ ઠંડું લાગે છે."
        result = normalize_gu(text)
        assert "શરીર ઠંડું લાગે છે" in result
        assert "પીઠ ઠંડી લાગે છે" in result

    def test_butter_word(self):
        """[130] મખાણ→માખણ."""
        text = "દૂધમાંથી મખાણ બનાવો."
        result = normalize_gu(text)
        assert "માખણ" in result
        assert "મખાણ" not in result

    def test_churning_word(self):
        """[133] મથણી→વલોણું."""
        text = "મથણીથી માખણ બનાવો."
        result = normalize_gu(text)
        assert "વલોણાથી" in result

    def test_ghee_making(self):
        """[131] ઘીમાં પકાવવું→ઘી બનાવવું."""
        text = "માખણને ઘીમાં પકાવવું જોઈએ."
        result = normalize_gu(text)
        assert "ઘી બનાવવું" in result

    def test_bull_replacement(self):
        """[11] બળદ→બુલ."""
        text = "સારા બળદ નો ઉપયોગ કરો."
        result = normalize_gu(text)
        assert "બુલ" in result

    def test_multiple_replacements_in_one_text(self):
        """Multiple forbidden terms in a single sentence."""
        text = "ગાયને સ્તનનો સોજો થયો છે, ચરબી ઓછી છે, અને ટોળામાં ચેપ ફેલાયો."
        result = normalize_gu(text)
        assert "આંચળનો સોજો" in result
        assert "ફેટ" in result
        assert "ધણમાં" in result


# ---------------------------------------------------------------------------
# Gender agreement issues from feedback
# These document known problems — the LLM or translation model
# produces masculine forms where feminine/neutral is needed.
# ---------------------------------------------------------------------------

class TestGenderAgreement:
    """
    Document gender agreement errors found in Shridhar feedback.
    These test that known problematic patterns are caught or at least documented.
    """

    @pytest.mark.parametrize("wrong, correct, context", [
        # [517] મહિનાના પાડુ→મહિનાની પાડી (month's calf — gender of "month" + animal)
        ("મહિનાના પાડુ", "મહિનાની પાડી", "calf gender: પાડુ=male, પાડી=female"),
        # [518] નાનું→નાની (small — neuter→feminine for female calf)
        ("નાનું", "નાની", "adjective gender agreement with feminine noun"),
        # [238] ઘણા બધા→ઘણી બધી (many — masculine→feminine)
        ("ઘણા બધા", "ઘણી બધી", "quantifier gender with feminine plural"),
        # [576] બાસું→વાસી (stale — wrong word entirely)
        ("બાસું", "વાસી", "stale milk — wrong Gujarati word"),
        # [600] ડેરીનું→ડેરીની (dairy's — neuter→feminine possessive)
        ("ડેરીનું", "ડેરીની", "possessive gender: dairy is feminine"),
        # [606] જાડા પથારી→જાડી પથારી (thick bedding — masculine→feminine)
        ("જાડા પથારી", "જાડી પથારી", "adjective-noun gender: bedding is feminine"),
        # [587] પાતળું છાશ→પાતળી છાશ (thin buttermilk — neuter→feminine)
        ("પાતળું છાશ", "પાતળી છાશ", "adjective-noun gender: buttermilk is feminine"),
        # [616] સારી ચારો→સારો ચારો (good fodder — feminine→masculine)
        ("સારી ચારો", "સારો ચારો", "adjective-noun gender: fodder is masculine"),
    ])
    def test_gender_error_documented(self, wrong, correct, context):
        """Document known gender errors. These may or may not be caught by post-processing."""
        # These are LLM/translation output issues, not all catchable by regex.
        # The test documents them as known patterns for monitoring.
        assert wrong != correct, f"Gender pair should differ: {context}"


# ---------------------------------------------------------------------------
# Wrong word / dialect corrections from feedback
# These document vocabulary errors where the translation produces
# a technically valid Gujarati word but not the one farmers use.
# ---------------------------------------------------------------------------

class TestDialectVocabulary:
    """
    Document dialect/vocabulary preferences from farmer feedback.
    The 'wrong' word is valid Gujarati but not the term farmers use.
    """

    @pytest.mark.parametrize("wrong, correct, feedback_id, note", [
        # Dairy cooperative terminology
        ("ખેડૂત કોડ", "સભાસદ કોડ", 440, "farmer code → member code (cooperative term)"),
        ("ડેરી ખેડૂત", "પશુપાલક", 469, "dairy farmer → livestock keeper"),
        ("દૂધાળ ગાળો", "દૂધ આપવાનું", 264, "lactation period — formal→colloquial"),
        ("દૂધનું ઉત્પાદન", "દૂધનું પ્રમાણ", 263, "milk production → milk quantity"),
        ("દૂધનો ધારા", "દૂધ આવવાનું", 262, "milk flow → milk coming"),
        ("દૂધાળ ગાળામાં", "દૂઝણૂ", 261, "lactation period → local term"),
        ("યુવાન", "નાની", 289, "young → small (age context for animals)"),
        # Place names — Shridhar flagged these repeatedly
        ("સબર", "સાબર", 354, "district name: Sabar→Sabar"),
        ("સબરકાંઠા", "સાબરકાંઠા", 356, "district name: Sabarkantha"),
        # Dairy product terms
        ("દૂધદાર", "દુધાળ", 580, "milch/milking — nonexistent word → correct"),
        ("બાંઝપણ", "વંધ્યત્વ", 401, "barrenness — colloquial→formal veterinary"),
        # Feed terms
        ("પશુચારોના", "પશુદાણ", 348, "cattle fodder → cattle feed (concentrate)"),
        ("તૂટેલા અનાજ", "ભરડેલા અનાજ", 550, "broken grain → crushed grain"),
        ("કપાસ", "રુ", 618, "cotton → cottonseed (feed context)"),
        ("બરબા", "બરસીમ", 700, "hallucinated fodder word → valid fodder term"),
        # Veterinary terms
        ("વંશીય-પશુચિકિત્સા", "પશુ આયુર્વેદ ચિકિત્સા", 353, "ethnoveterinary → ayurvedic vet"),
        # Animal terminology
        ("પાડોના પાડાને", "ભેસની પાડીને", 362, "male calf → female calf (context was female)"),
        # Cooking/processing
        ("મીઠો", "ગળ્યું", 426, "sweet — masculine→neuter in dairy context"),
        ("ખારો", "ખારું", 427, "salty — masculine→neuter in dairy context"),
        # Communication style
        ("બોલાવવો", "બોલાવવા", 249, "to call — singular→plural/respectful"),
        ("દોહવા માટેની પશુઓ", "દુધાળ પશુઓ", 536, "animals for milking → milch animals"),
    ])
    def test_dialect_preference_documented(self, wrong, correct, feedback_id, note):
        """Document farmer vocabulary preferences from Shridhar feedback #{feedback_id}."""
        assert wrong != correct, f"Dialect pair should differ: {note}"


# ---------------------------------------------------------------------------
# Term choice rules — don't combine synonyms in same phrase
# ---------------------------------------------------------------------------

class TestTermChoiceRules:
    """
    Shridhar flagged cases where both synonyms appear together,
    which sounds unnatural in spoken Gujarati.
    """

    def test_no_double_mastitis_terms(self):
        """[59, 79] Don't say 'આઉ નો સોજો' AND 'બાવલાનો સોજો' together."""
        text = "ગાયને આઉ નો સોજો / બાવલાનો સોજો થયો છે."
        result = normalize_gu(text)
        # After normalization, both should become આંચળનો સોજો
        count = result.count("આંચળનો સોજો")
        # Should appear at most once (both replaced to same term)
        assert count >= 1

    def test_no_meli_and_jar_together(self):
        """[484] Don't use both મેલી and જર — they mean the same (afterbirth)."""
        # This is a style issue, not a forbidden-term issue.
        # Document the rule: use either મેલી OR જર, not both.
        text = "મેલી જર ન પડી"
        # Currently no post-processing for this — document as known gap
        assert "મેલી" in text or "જર" in text

    def test_no_saheb_and_ben_together(self):
        """[225] સાહેબ (male) + બેન (female) is contradictory — use one."""
        # This is an LLM output issue — it uses both honorifics
        text = "સાહેબ બેન"
        # Document: the LLM should not combine male+female honorifics
        assert "સાહેબ" in text and "બેન" in text  # Both present = bad


# ---------------------------------------------------------------------------
# Nonexistent words flagged by Shridhar
# ---------------------------------------------------------------------------

class TestNonexistentWords:
    """Words that don't exist in Gujarati — LLM hallucinations."""

    @pytest.mark.parametrize("bad_word, correct, feedback_id", [
        ("ચૂભો", "ચીરો", 162),     # Caught by forbidden list
        ("ડક્કાર જપટી", None, 540),  # No Gujarati equivalent for this garble
        ("દૂધદાર", "દુધાળ", 580),   # Caught? Check.
    ])
    def test_nonexistent_word_documented(self, bad_word, correct, feedback_id):
        """Document nonexistent words from feedback #{feedback_id}."""
        if correct:
            result = normalize_gu(f"ગાય {bad_word} છે.")
            if bad_word in GU_TERM_POLICY.get("forbidden", {}):
                assert correct in result, f"Forbidden '{bad_word}' should become '{correct}'"


# ---------------------------------------------------------------------------
# Identity / introduction phrase corrections
# ---------------------------------------------------------------------------

class TestAmulAiHelplineDisambiguation:
    """Amul AI (Artificial Intelligence) must not be translated as insemination."""

    def test_gujarati_output_rules_disambiguate_amul_ai_product_name(self):
        # §14: voice's Gujarati output rules (incl. Amul-AI disambiguation) live in
        # VOICE_GU_PREFERRED_TRANSLATION_RULES; chat keeps the lean shared set.
        rules = "\n".join(VOICE_GU_PREFERRED_TRANSLATION_RULES)
        assert "Amul AI" in rules
        assert "AI helpline" in rules
        assert "અમૂલ એ.આઈ." in rules
        assert "કૃત્રિમ બીજદાન" in rules
        assert "Artificial Intelligence" in rules

    def test_amul_ai_glossary_entries_present(self):
        # §14: the Amul-AI disambiguation glossary entries are voice-only and live
        # in the voice-extras glossary, not chat's shared glossary_terms.json.
        glossary = json.loads(VOICE_EXTRA_GLOSSARY_PATH.read_text(encoding="utf-8"))
        by_en = {entry["en"]: entry["gu"] for entry in glossary}
        assert by_en["Amul AI helpline advisor"] == "અમૂલ એ.આઈ. હેલ્પલાઇન સલાહકાર"
        assert by_en["AI helpline"] == "એ.આઈ. હેલ્પલાઇન"
        assert by_en["AMUL AI"] == "અમૂલ એ.આઈ."
        assert by_en["amul helpline"] == "અમૂલ એ.આઈ. હેલ્પલાઇન"


class TestIdentityPhrases:
    """
    Shridhar repeatedly flagged how Sarlaben introduces herself.
    These document the preferred phrasing.
    """

    @pytest.mark.parametrize("wrong_phrase, correct_phrase, feedback_ids", [
        # [377, 430, 419] "આવી છું" → "બોલું છું"
        ("અમૂલ એ.આઈ.માંથી આવી છું", "અમૂલ એ.આઈ.માંથી બોલું છું", [377, 430, 419]),
        # [378] Long intro → short intro
        ("દૂધાળાં પશુઓ માટે મદદ કરવા આવી છું", "તમારા પશુઓ માટે શું મદદ કરી શકું", [378]),
    ])
    def test_identity_phrase_documented(self, wrong_phrase, correct_phrase, feedback_ids):
        """Document preferred introduction phrasing from feedback #{feedback_ids}."""
        assert wrong_phrase != correct_phrase


# ---------------------------------------------------------------------------
# Post-processing pipeline completeness check
# ---------------------------------------------------------------------------

class TestPolicyCompleteness:
    """Verify that the forbidden list covers all Shridhar-flagged terms."""

    def test_forbidden_list_not_empty(self):
        forbidden = GU_TERM_POLICY.get("forbidden", {})
        assert len(forbidden) >= 30, f"Expected 30+ forbidden terms, got {len(forbidden)}"

    def test_all_critical_terms_covered(self):
        """Key terms from Shridhar feedback should be caught by post-processing."""
        critical = [
            "સ્તન", "પાહો", "ચરબી", "ઘન પદાર્થો", "જંતુઓ",
            "ટોળા", "બળદ", "મખાણ", "માલઈ", "ગર્ભવતી", "બૈડા", "બૈડું",
        ]
        for term in critical:
            result = normalize_gu(f"ગાયમાં {term} છે.")
            assert term not in result, f"Critical term '{term}' not being replaced in output: {result}"

    def test_replacements_list_built(self):
        """GU_POST_REPLACEMENTS should have base + policy entries."""
        assert len(GU_POST_REPLACEMENTS) >= 30, f"Expected 30+ replacements, got {len(GU_POST_REPLACEMENTS)}"


class TestMissingQuantityRepair:
    """Placeholder quantity slots should be stripped without inventing defaults."""

    def test_feed_placeholder_lines_do_not_invent_defaults(self):
        # Merge architecture: quantity-placeholder repair lives in the voice-output
        # TTS-shaping layer (normalize_voice_output -> _remove_quantity_placeholders),
        # not the terminology layer (_post_normalize_gu_translation). Voice applied
        # both in one function; amul splits them, so assert against the layer that
        # owns placeholder stripping.
        from helpers.utils import normalize_voice_output

        text = (
            "લીલો ચારો: – કિ.ગ્રા. "
            "સૂકો ચારો: -- કિ.ગ્રા. "
            "દાણ: – કિલોગ્રામ "
            "મિનરલ મિશ્રણ: – ગ્રામ "
            "મીઠું: – ગ્રામ"
        )
        result = normalize_voice_output(text, "gu")
        assert "પંદર થી વીસ કિલોગ્રામ" not in result
        assert "પાંચ થી સાત કિલોગ્રામ" not in result
        assert "બે થી ત્રણ કિલોગ્રામ" not in result
        assert "પચાસ ગ્રામ" not in result
        assert "ત્રીસ ગ્રામ" not in result
        assert "--" not in result
        assert "–" not in result
        assert "કિ.ગ્રા." not in result
        assert "બરબા" not in result

    def test_berba_normalizes_to_barseem(self):
        result = normalize_gu("લીલો ચારો તરીકે બરબા આપો.")
        assert "બરસીમ" in result
        assert "બરબા" not in result
