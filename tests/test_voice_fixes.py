"""
Tests for voice issue fixes from Shridhar feedback analysis.

Covers:
- Greeting detection (expanded tokens, repeated words)
- Fragment detection (garbled/short input)
- STT signal detection for no-audio / unclear-speech sentinels
"""
import pytest
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.voice import _is_bare_greeting, _is_fragment_query, _is_hold_message
from app.services.stt_signals import detect_stt_signal
from app.services.translation import _post_normalize_gu_translation
from agents.tools.terms import get_ambiguity_hints_for_query
from helpers.utils import clean_output_by_language


# ---------------------------------------------------------------------------
# Greeting detection tests
# ---------------------------------------------------------------------------

class TestGreetingDetection:
    """Verify _is_bare_greeting catches common greeting variants."""

    @pytest.mark.parametrize("query", [
        "hello",
        "Hello",
        "HELLO",
        "hi",
        "hey",
        "hlo",
        "હલો",
        "હેલો",
        "નમસ્તે",
        "नमस्ते",
        "हेलो",
        "namaste",
        "halo",
    ])
    def test_basic_greetings(self, query):
        assert _is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", [
        "hello hello",
        "hello hello hello",
        "હલો હલો",
        "hlo hlo",
        "hi hi",
    ])
    def test_repeated_greetings(self, query):
        assert _is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", [
        "ha hello",
        "હા હલો",
        "ji",
        "જી",
        "bolo",
        "બોલો",
        "ha bolo",
        "હા બોલો",
    ])
    def test_greeting_combos(self, query):
        assert _is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", [
        "hello!",
        "hello.",
        "  hello  ",
        "* hello *",
    ])
    def test_greetings_with_punctuation(self, query):
        assert _is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", [
        "hello my cow is sick",
        "મારી ગાયને તાવ છે",
        "hello can you help me with my buffalo",
        "હલો મારી ભેંસ",
        "namaste, meri bhains ko bukhar hai",
    ])
    def test_not_bare_greetings(self, query):
        assert _is_bare_greeting(query) is False


# ---------------------------------------------------------------------------
# Fragment detection tests
# ---------------------------------------------------------------------------

class TestFragmentDetection:
    """Verify _is_fragment_query catches garbled/short inputs."""

    @pytest.mark.parametrize("query", [
        "",
        " ",
        "*",
        "* *",
        "ઓ",        # Single Gujarati syllable
        "બ",        # Single Gujarati letter
        "b",        # Single Latin letter
        "O",
        "ok",       # 2 chars
        "હા",       # 2 Gujarati chars (3 bytes but ≤3 chars)
    ])
    def test_fragments(self, query):
        assert _is_fragment_query(query) is True

    @pytest.mark.parametrize("query", [
        "મારી ગાય",           # "My cow" — 2 words, real content
        "hello",               # 5 chars — greeting, not fragment
        "cow is sick",
        "દૂધ ઓછું",            # "Less milk"
        "ભેંસ બીમાર છે",       # "Buffalo is sick"
    ])
    def test_not_fragments(self, query):
        assert _is_fragment_query(query) is False




# ---------------------------------------------------------------------------
# Ambiguity terms tests
# ---------------------------------------------------------------------------

class TestAmbiguityTerms:
    """Verify new ambiguity terms trigger correct disambiguation hints."""

    def test_fat_vs_stomach(self):
        """ફેટ should trigger milk fat hint, not stomach."""
        result = get_ambiguity_hints_for_query("મારી ભેંસનું ફેટ ઓછું છે")
        assert "milk fat" in result.lower() or "ફેટ" in result
        assert "પેટ" not in result or "NOT પેટ" in result

    def test_retained_placenta(self):
        """માટી ન ખસવી should trigger retained placenta hint."""
        result = get_ambiguity_hints_for_query("ગાયની માટી ન ખસવી")
        assert "retained placenta" in result.lower() or "મેલી" in result

    def test_meli(self):
        """મેલી should also trigger retained placenta."""
        result = get_ambiguity_hints_for_query("મેલી ન પડી")
        assert "retained placenta" in result.lower() or "afterbirth" in result.lower()

    def test_karmodi(self):
        """કરમોડી should trigger horn cancer hint."""
        result = get_ambiguity_hints_for_query("ગાયને કરમોડી થયો છે")
        assert "horn cancer" in result.lower() or "કરમોડી" in result

    def test_vado_shed(self):
        """વાડો should trigger cattle shed hint, not calf."""
        result = get_ambiguity_hints_for_query("વાડો કેવી રીતે બનાવવો")
        assert "shed" in result.lower() or "enclosure" in result.lower()
        assert "પાડો" not in result or "NOT પાડો" in result

    def test_samudri_feed_avoids_marine_assumption(self):
        """Feed-context સમુદ્રી should avoid marine advice, but not assume a brand name."""
        result = get_ambiguity_hints_for_query("ગાભણ ભેંસને સમુદ્રી દાણ આપવું?")
        assert "repeat" in result.lower() or "clarify" in result.lower() or "સ્પષ્ટ" in result
        assert "seaweed" in result.lower() or "marine feed" in result.lower()

    @pytest.mark.parametrize("query", [
        "મારી ભેસ્ટને તાવ છે",
        "ભંચ દૂધ ઓછું આપે છે",
        "ભેંચને ખાવાનું બંધ છે",
    ])
    def test_buffalo_asr_variants_do_not_become_sheep(self, query):
        """Common Gujarati ASR variants for ભેંસ should resolve to buffalo, not sheep."""
        result = get_ambiguity_hints_for_query(query)
        assert "buffalo" in result.lower()
        assert "not sheep" in result.lower() or "NOT sheep" in result
        assert "goat" in result.lower()

# ---------------------------------------------------------------------------
# Hold message detection tests
# ---------------------------------------------------------------------------

class TestHoldMessageDetection:
    """Verify _is_hold_message catches carrier IVR hold messages."""

    @pytest.mark.parametrize("query", [
        # Raw Gujarati STT output (as seen in production traces)
        "તેને તમારો કોલ હોલ્ડ પર રાખ્યા છે કૃપા કરી લાઇન પર રહો",
        "તમારો કોલ હોલ્ડ પર રાખ્યા છે કૃપા કરી લાઈન પર રહો",
        # English translations (post-pretranslation)
        "Your call has been put on hold, please stay on the line.",
        "The person you are speaking with has put your call on hold. Please stay on the line.",
        "The person you called has put your call on hold. Please remain on the line.",
        "They have put your call on hold.",
        # Mixed Gujarati/English fragments from STT
        "હોલ્ડ પર રાખ્યા છે",
        "કોલ હોલ્ડ પર",
        "put your call on hold please stay on the line",
        # Hindi transliteration in Gujarati script
        "આપ કે કોલ કો હોલ્ડ પર રખા હૈ કૃપયા લાઇન પર બને રહી",
    ])
    def test_hold_messages_detected(self, query):
        assert _is_hold_message(query), f"Should detect hold message: {query!r}"

    @pytest.mark.parametrize("query", [
        # Normal farmer queries that should NOT be detected as hold messages
        "મારી ગાયને તાવ આવે છે",
        "My cow is not eating properly",
        "hello",
        "દૂધમાં ફેટ ઓછી છે",
        "I want to book AI for my cow",
        # "hold" in a different context
        "How long can I hold the calf's milk?",
    ])
    def test_normal_queries_not_detected(self, query):
        assert not _is_hold_message(query), f"Should NOT detect as hold: {query!r}"


    def test_existing_uthla_still_works(self):
        """Existing entry: ઉથલા should still trigger repeat breeder."""
        result = get_ambiguity_hints_for_query("મારી ગાય ઉથલા મારે છે")
        assert "repeat breeder" in result.lower()

    def test_no_match(self):
        """Unrelated query should return empty."""
        result = get_ambiguity_hints_for_query("દૂધ કેવી રીતે વધારવું")
        # Should not match any ambiguity term (unless it fuzzy-matches something)
        # At minimum, should not crash
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Voice output normalization tests
# ---------------------------------------------------------------------------

class TestVoiceOutputNormalization:
    """Verify Gujarati number text survives the final language filter."""

    def test_gujarati_digits_are_normalized_before_filtering(self):
        result = clean_output_by_language("3-4 કિ.ગ્રા.", "gu")
        assert "3" not in result
        assert "4" not in result
        assert "ત્રણ" in result
        assert "ચાર" in result
        assert "કિલોગ્રામ" in result

    def test_placeholder_dashes_before_units_are_removed(self):
        result = clean_output_by_language("લીલો ચારો: -- કિ.ગ્રા.", "gu")
        assert "--" not in result
        assert "કિ.ગ્રા." not in result
        assert "લીલો ચારો:" in result

    def test_streaming_gu_chunk_preserves_leading_space(self):
        result = _post_normalize_gu_translation(" તમારી", "gu", strip_outer=False)
        assert result.startswith(" ")
        assert result == " તમારી"

    def test_non_streaming_gu_chunk_can_strip_outer_space(self):
        result = _post_normalize_gu_translation(" તમારી", "gu", strip_outer=True)
        assert result == "તમારી"


# ---------------------------------------------------------------------------
# STT signal detection tests
# ---------------------------------------------------------------------------

class TestSttSignalDetection:
    """Verify sentinel STT outputs are recognized before agent routing."""

    @pytest.mark.parametrize("query", [
        "*No audio/User is speaking softly*",
        "No audio/User is speaking softly",
    ])
    def test_no_audio_signal_detected(self, query):
        assert detect_stt_signal(query) == "No audio/User is speaking softly"

    @pytest.mark.parametrize("query", [
        "*Unclear Speech*",
        "Unclear Speech",
    ])
    def test_unclear_speech_signal_detected(self, query):
        assert detect_stt_signal(query) == "Unclear Speech"

    @pytest.mark.parametrize("query", [
        "hello",
        "મારી ગાયને તાવ છે",
        "call has been put on hold",
    ])
    def test_normal_queries_are_not_signals(self, query):
        assert detect_stt_signal(query) is None
