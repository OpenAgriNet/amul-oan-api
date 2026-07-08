"""
Tests for helpers.gujarati_numbers – Gujarati number-to-words converter and TTS normalizer.
"""
import pytest
from helpers.gujarati_numbers import (
    number_to_gujarati,
    tag_to_gujarati,
    normalize_numbers_for_tts,
    mask_tag_identifier,
    _int_to_gujarati,
)


# ---------------------------------------------------------------------------
# number_to_gujarati – integers
# ---------------------------------------------------------------------------

class TestIntegerConversion:
    """Core integer-to-Gujarati-words tests."""

    @pytest.mark.parametrize("n, expected", [
        (0, "શૂન્ય"),
        (1, "એક"),
        (9, "નવ"),
    ])
    def test_single_digits(self, n, expected):
        assert number_to_gujarati(n) == expected

    @pytest.mark.parametrize("n, expected", [
        (10, "દસ"),
        (15, "પંદર"),
        (19, "ઓગણીસ"),
        (20, "વીસ"),
        (25, "પચ્ચીસ"),
        (29, "ઓગણત્રીસ"),
        (35, "પાંત્રીસ"),
        (45, "પિસ્તાલીસ"),
        (54, "ચોપન"),
        (69, "અગણોતેર"),
        (70, "સિત્તેર"),
        (82, "બ્યાસી"),
        (99, "નવ્વાણું"),
    ])
    def test_teens_and_tens(self, n, expected):
        """These are the compound numbers that TTS was garbling."""
        assert number_to_gujarati(n) == expected

    @pytest.mark.parametrize("n, expected", [
        (100, "એકસો"),
        (136, "એકસો છત્રીસ"),
        (200, "બસો"),
        (367, "ત્રણસો સડસઠ"),
        (369, "ત્રણસો અગણોતેર"),
        (500, "પાંચસો"),
        (999, "નવસો નવ્વાણું"),
    ])
    def test_hundreds(self, n, expected):
        assert number_to_gujarati(n) == expected

    @pytest.mark.parametrize("n, expected", [
        (1000, "એક હજાર"),
        (1500, "એક હજાર પાંચસો"),
        (5000, "પાંચ હજાર"),
        (10000, "દસ હજાર"),
        (99999, "નવ્વાણું હજાર નવસો નવ્વાણું"),
    ])
    def test_thousands(self, n, expected):
        assert number_to_gujarati(n) == expected

    @pytest.mark.parametrize("n, expected", [
        (100000, "એક લાખ"),
        (250000, "બે લાખ પચાસ હજાર"),
    ])
    def test_lakhs(self, n, expected):
        assert number_to_gujarati(n) == expected

    def test_crore(self):
        assert number_to_gujarati(10000000) == "એક કરોડ"

    def test_float_whole_number_treated_as_int(self):
        """136.0 should produce the same output as 136."""
        assert number_to_gujarati(136.0) == "એકસો છત્રીસ"


# ---------------------------------------------------------------------------
# number_to_gujarati – decimals
# ---------------------------------------------------------------------------

class TestDecimalConversion:
    """Decimal numbers should read as 'integer પોઈન્ટ digit digit ...'"""

    @pytest.mark.parametrize("value, expected", [
        (3.5, "ત્રણ પોઈન્ટ પાંચ"),
        (6.69, "છ પોઈન્ટ છ નવ"),
        (8.86, "આઠ પોઈન્ટ આઠ છ"),
        (3.56, "ત્રણ પોઈન્ટ પાંચ છ"),
        (4.34, "ચાર પોઈન્ટ ત્રણ ચાર"),
        (54.25, "ચોપન પોઈન્ટ બે પાંચ"),
        (100.05, "એકસો પોઈન્ટ શૂન્ય પાંચ"),
        (0.5, "શૂન્ય પોઈન્ટ પાંચ"),
        (0.05, "શૂન્ય પોઈન્ટ શૂન્ય પાંચ"),
        (0.125, "શૂન્ય પોઈન્ટ એક બે પાંચ"),
        (0.425, "શૂન્ય પોઈન્ટ ચાર બે પાંચ"),
        (0.625, "શૂન્ય પોઈન્ટ છ બે પાંચ"),
        (0.875, "શૂન્ય પોઈન્ટ આઠ સાત પાંચ"),
        (1.01, "એક પોઈન્ટ શૂન્ય એક"),
        (1.1, "એક પોઈન્ટ એક"),
        (2.75, "બે પોઈન્ટ સાત પાંચ"),
        (10.01, "દસ પોઈન્ટ શૂન્ય એક"),
        (20.2, "વીસ પોઈન્ટ બે"),
        (99.99, "નવ્વાણું પોઈન્ટ નવ નવ"),
    ])
    def test_decimal_values(self, value, expected):
        assert number_to_gujarati(value) == expected


# ---------------------------------------------------------------------------
# tag_to_gujarati – digit-by-digit reading
# ---------------------------------------------------------------------------

class TestTagConversion:
    def test_mask_tag_identifier_uses_last_four_digits(self):
        assert mask_tag_identifier("106285318721") == "eight seven two one"

    def test_mask_tag_identifier_ignores_non_digits(self):
        assert mask_tag_identifier("tag-10/62") == "one zero six two"

    def test_standard_12_digit_tag(self):
        result = tag_to_gujarati("106285318721")
        assert result == "એક શૂન્ય છ બે આઠ પાંચ ત્રણ એક આઠ સાત બે એક"

    def test_tag_with_zeros(self):
        result = tag_to_gujarati("100066235408")
        assert result == "એક શૂન્ય શૂન્ય શૂન્ય છ છ બે ત્રણ પાંચ ચાર શૂન્ય આઠ"

    def test_tag_ignores_non_digits(self):
        result = tag_to_gujarati("10-62")
        assert result == "એક શૂન્ય છ બે"


# ---------------------------------------------------------------------------
# normalize_numbers_for_tts – full text normalization
# ---------------------------------------------------------------------------

class TestTextNormalization:

    def test_integer_in_sentence(self):
        text = "તમારી પાસે 35 ગાયો છે"
        result = normalize_numbers_for_tts(text)
        assert "પાંત્રીસ" in result
        assert "35" not in result

    def test_decimal_in_sentence(self):
        text = "ફેટ 6.69 છે"
        result = normalize_numbers_for_tts(text)
        assert "છ પોઈન્ટ છ નવ" in result
        assert "6.69" not in result

    def test_long_tag_number_digit_by_digit(self):
        text = "ટેગ નંબર 1062853187210 ની ગાય"
        result = normalize_numbers_for_tts(text)
        # Should be digit-by-digit (13 digits → long sequence)
        assert "એક શૂન્ય છ બે" in result
        # Should NOT be read as a compound number
        assert "કરોડ" not in result

    def test_mixed_numbers(self):
        text = "દૈનિક ઉત્પાદન 367 લિટર, ફેટ 3.56 છે"
        result = normalize_numbers_for_tts(text)
        assert "ત્રણસો સડસઠ" in result
        assert "ત્રણ પોઈન્ટ પાંચ છ" in result
        assert "367" not in result
        assert "3.56" not in result

    def test_empty_string(self):
        assert normalize_numbers_for_tts("") == ""

    def test_no_numbers(self):
        text = "નમસ્તે, હું સરલાબેન છું"
        assert normalize_numbers_for_tts(text) == text

    def test_phone_number_digit_by_digit(self):
        text = "મોબાઇલ નંબર 9979138134"
        result = normalize_numbers_for_tts(text)
        assert "નવ નવ સાત નવ" in result
        assert "9979138134" not in result

    def test_phone_number_sentence_matches_voice_expectation(self):
        text = "તમારો રજિસ્ટર્ડ મોબાઇલ નંબર 9265991405 છે."
        result = normalize_numbers_for_tts(text)
        assert result == "તમારો રજિસ્ટર્ડ મોબાઇલ નંબર નવ બે છ પાંચ નવ નવ એક ચાર શૂન્ય પાંચ છે."

    def test_spaced_digits_read_digit_by_digit(self):
        text = "પશુ 1 7 5 4 માટે તપાસ કરો"
        result = normalize_numbers_for_tts(text)
        assert result == "પશુ એક સાત પાંચ ચાર માટે તપાસ કરો"

    def test_preserves_surrounding_text(self):
        text = "કુલ 4 પશુ છે"
        result = normalize_numbers_for_tts(text)
        assert result == "કુલ ચાર પશુ છે"

    def test_multiple_separate_numbers(self):
        text = "બે ખાતા: 136 લિટર અને 367 લિટર"
        result = normalize_numbers_for_tts(text)
        assert "એકસો છત્રીસ" in result
        assert "ત્રણસો સડસઠ" in result

    def test_range_numbers(self):
        """Numeric ranges should be spoken naturally as a Gujarati range."""
        text = "2-3 દિવસ"
        result = normalize_numbers_for_tts(text)
        assert result == "બે થી ત્રણ દિવસ"

    def test_large_range_numbers(self):
        text = "350-400 કિલો"
        result = normalize_numbers_for_tts(text)
        assert result == "ત્રણસો પચાસ થી ચારસો કિલો"

    def test_en_dash_range_numbers(self):
        text = "20–25 દિવસ"
        result = normalize_numbers_for_tts(text)
        assert result == "વીસ થી પચ્ચીસ દિવસ"

    def test_plain_four_digit_number_stays_quantitative(self):
        text = "વજન 1754 કિલો છે"
        result = normalize_numbers_for_tts(text)
        assert "એક હજાર" in result
        assert "એક સાત પાંચ ચાર" not in result


# ---------------------------------------------------------------------------
# Edge cases from Shridhar's feedback
# ---------------------------------------------------------------------------

class TestFeedbackCases:
    """Numbers that were specifically garbled in the March 25-26 feedback."""

    def test_136_liters(self):
        """[qh] 'એકસો છત્રીસ લિટર' was being garbled."""
        assert number_to_gujarati(136) == "એકસો છત્રીસ"

    def test_369(self):
        """Was garbled as 'ઓગણું સિત્તેર'."""
        assert number_to_gujarati(369) == "ત્રણસો અગણોતેર"

    def test_367(self):
        """Was garbled as 'ઓંસઠ'."""
        assert number_to_gujarati(367) == "ત્રણસો સડસઠ"

    def test_35_animals(self):
        """[rl] Was garbled as 'પિસ્તાઈસ'."""
        assert number_to_gujarati(35) == "પાંત્રીસ"

    def test_54_point_25(self):
        """[qx] Was garbled as 'ચોંયાવન પોઇન્ટ બે પાંચ'."""
        assert number_to_gujarati(54.25) == "ચોપન પોઈન્ટ બે પાંચ"

    def test_fat_3_point_56(self):
        """[qi] Was garbled as 'ત્રણ પોઈન્ટ પાંચ છટ્ઠું'."""
        assert number_to_gujarati(3.56) == "ત્રણ પોઈન્ટ પાંચ છ"

    def test_fat_6_point_69(self):
        assert number_to_gujarati(6.69) == "છ પોઈન્ટ છ નવ"

    def test_snf_8_point_86(self):
        assert number_to_gujarati(8.86) == "આઠ પોઈન્ટ આઠ છ"


class TestExpandedIntegerCoverage:
    """Extra coverage for awkward but common quantitative values."""

    @pytest.mark.parametrize("n, expected", [
        (101, "એકસો એક"),
        (110, "એકસો દસ"),
        (115, "એકસો પંદર"),
        (198, "એકસો અઠ્ઠાણું"),
        (205, "બસો પાંચ"),
        (250, "બસો પચાસ"),
        (300, "ત્રણસો"),
        (398, "ત્રણસો અઠ્ઠાણું"),
        (400, "ચારસો"),
        (505, "પાંચસો પાંચ"),
        (808, "આઠસો આઠ"),
        (909, "નવસો નવ"),
        (1001, "એક હજાર એક"),
        (1010, "એક હજાર દસ"),
        (1100, "એક હજાર એકસો"),
        (1111, "એક હજાર એકસો અગિયાર"),
        (2000, "બે હજાર"),
        (200000, "બે લાખ"),
        (999999, "નવ લાખ નવ્વાણું હજાર નવસો નવ્વાણું"),
    ])
    def test_integer_values(self, n, expected):
        assert number_to_gujarati(n) == expected
