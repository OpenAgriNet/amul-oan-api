"""The farmer context must not tell the agent to ask an unanswerable question.

A mobile is shared by a household. Of 364 real name pairs the agent offered,
250 share a name token and 19 are byte identical, so the caller answers and the
answer fits both records. 71 of the 265 calls in "AI booking not done" died in
that loop — the largest single remaining source of failed bookings (#282).

Whether the question is worth a turn depends on what differs: technicians come
from (unionCode, societyCode) alone, and 90% of multi-account mobiles have every
record in one society, where the visit is identical whichever is used.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

from agents.tools.models.farmer_transport import FarmerDataEnvelope
from app.voice.farmer import _build_compact_farmer_summary


def _env(records):
    return FarmerDataEnvelope.from_records(records, source="api", lookup_status="found")





def _rec(farmer_code, name, society_code="00731", society_name="RAMOS", union_code="159"):
    return {
        "unionCode": union_code, "societyCode": society_code,
        "farmerCode": farmer_code, "farmerName": name, "societyName": society_name,
    }


def test_same_village_does_not_ask_which_farmer():
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Patel Asvinbhai Amrutbhai"),
        _rec("0192", "Patel Nuruben Ashvinbhai A"),
    ]))
    assert "Do NOT ask which farmer name" in out
    assert "Use Farmer option 1." in out
    assert "ask which farmer name the caller wants to use" not in out


def test_duplicate_records_of_one_person_are_named_as_such():
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Mukeshbhai Jivrajbhai"),
        _rec("0192", "Mukeshbhai  Jivrajbhai.."),
    ]))
    assert "duplicate records of one farmer" in out
    assert "Do NOT ask which farmer name" in out


def test_different_villages_ask_by_village_not_by_name():
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Patel Asvinbhai", society_code="00731", society_name="RAMOS"),
        _rec("0192", "Patel Nuruben", society_code="00262", society_name="DHANSURA"),
    ]))
    assert "Ask which village" in out
    assert "RAMOS" in out and "DHANSURA" in out
    assert "Do NOT ask which farmer name" in out


def test_single_record_is_unchanged():
    out = _build_compact_farmer_summary(_env([_rec("0554", "Ramesh")]))
    assert "Multiple farmer records" not in out
    assert "Do NOT ask" not in out


def test_every_record_is_still_listed_for_the_technician_groups():
    """Options are not trimmed: _build_ai_technician_summary keys its groups by
    farmer, so a record dropped here would leave an unreachable group."""
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Patel Asvinbhai"), _rec("0192", "Patel Nuruben"),
    ]))
    assert "Farmer option 1" in out and "Farmer option 2" in out


def test_records_without_codes_are_never_asserted_to_share_a_village():
    """unionCode/societyCode are `extra` fields and can be absent. They must not
    collapse to one empty key and be declared "the same village". These records
    also cannot be booked at all — _fetch_ai_technicians returns None without
    both codes — so the village question would spend a turn and still fail."""
    out = _build_compact_farmer_summary(_env([
        {"farmerCode": "0554", "farmerName": "Ramesh", "societyName": "ANAND"},
        {"farmerCode": "0192", "farmerName": "Suresh", "societyName": "VIDYA"},
    ]))
    assert "all in the same village" not in out
    assert "Ask which village" not in out
    assert "cannot be made from them" in out


def test_union_only_records_are_not_merged_into_one_village():
    """A record with a union code but no society code cannot be booked at all
    (the technician lookup needs both), so it is excluded before grouping — and
    must never be declared to share a village with another society."""
    out = _build_compact_farmer_summary(_env([
        {"unionCode": "159", "farmerCode": "0554", "farmerName": "Ramesh", "societyName": "ANAND"},
        {"unionCode": "159", "farmerCode": "0192", "farmerName": "Suresh", "societyName": "VIDYA"},
    ]))
    assert "all in the same village" not in out


def test_gujarati_matras_are_not_stripped_when_comparing():
    """`\\w` excludes Unicode Mn/Mc, so a naive [^\\w\\s] strip collapsed કડી and
    કડા to કડ — two real villages read as one, and two household members read as
    duplicates of one farmer."""
    from app.voice.farmer import _normalize_farmer_name
    assert _normalize_farmer_name("કડી") != _normalize_farmer_name("કડા")
    assert _normalize_farmer_name("રમેશભાઈ") == "રમેશભાઈ"
    assert _normalize_farmer_name("RAMOS ") == _normalize_farmer_name("Ramos")


def test_society_code_without_union_code_is_not_bookable():
    """_fetch_ai_technicians needs BOTH codes and create_ai_call needs all three,
    so a society code alone must not be advertised as bookable."""
    out = _build_compact_farmer_summary(_env([
        {"societyCode": "00731", "farmerCode": "0554", "farmerName": "A", "societyName": "RAMOS"},
        {"societyCode": "00262", "farmerCode": "0192", "farmerName": "B", "societyName": "DHANSURA"},
    ]))
    assert "Ask which village" not in out
    assert "cannot be made from them" in out


def test_same_village_name_but_no_codes_is_not_declared_identical():
    """Both key to ("name", …) so they group as one village — but neither can be
    booked, so the "visit is identical" claim must not be made."""
    out = _build_compact_farmer_summary(_env([
        {"farmerCode": "0554", "farmerName": "A", "societyName": "RAMOS"},
        {"farmerCode": "0192", "farmerName": "B", "societyName": "RAMOS"},
    ]))
    assert "all in the same village" not in out
    assert "cannot be made from them" in out


def test_one_codeless_record_does_not_poison_the_rest():
    """Two real, distinct villages plus one unbookable record: the village
    question must still be asked over the bookable remainder."""
    out = _build_compact_farmer_summary(_env([
        {"farmerCode": "0001", "farmerName": "No codes"},
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0554",
         "farmerName": "A", "societyName": "RAMOS"},
        {"unionCode": "159", "societyCode": "00262", "farmerCode": "0192",
         "farmerName": "B", "societyName": "DHANSURA"},
    ]))
    assert "Ask which village" in out
    assert "RAMOS (Farmer option 2)" in out and "DHANSURA (Farmer option 3)" in out
    assert "Only these options can be used" in out


def test_village_choice_names_the_option_number():
    """The label may come from a record other than the lowest-numbered one in
    that society, so the mapping must be stated rather than inferred from the
    option lines' society names."""
    out = _build_compact_farmer_summary(_env([
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0554", "farmerName": "A"},
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0555",
         "farmerName": "B", "societyName": "RAMOS"},
        {"unionCode": "159", "societyCode": "00262", "farmerCode": "0192",
         "farmerName": "C", "societyName": "DHANSURA"},
    ]))
    assert "RAMOS (Farmer option 1)" in out
    assert "DHANSURA (Farmer option 3)" in out


def test_village_labels_that_sound_identical_are_not_offered_as_a_choice():
    out = _build_compact_farmer_summary(_env([
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0554",
         "farmerName": "Ramesh", "societyName": "RAMOS"},
        {"unionCode": "159", "societyCode": "00262", "farmerCode": "0192",
         "farmerName": "Suresh", "societyName": "Ramos "},
    ]))
    assert "Ask which village" not in out
    assert "Use Farmer option 1." in out


def test_first_non_empty_label_wins_within_a_society():
    """Record 1 of a society may lack societyName while record 2 carries it."""
    out = _build_compact_farmer_summary(_env([
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0554", "farmerName": "A"},
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0555",
         "farmerName": "B", "societyName": "RAMOS"},
        {"unionCode": "159", "societyCode": "00262", "farmerCode": "0192",
         "farmerName": "C", "societyName": "DHANSURA"},
    ]))
    assert "Ask which village" in out
    assert "RAMOS (Farmer option 1)" in out and "DHANSURA (Farmer option 3)" in out


def test_the_do_not_ask_line_is_scoped_to_ai_booking():
    """The farmer-selection instruction must remain explicitly scoped to AI booking."""
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Patel Asvinbhai"), _rec("0192", "Patel Nuruben"),
    ]))
    assert "for the AI booking" in out
    assert "Do NOT ask which farmer name to use for booking" not in out


def test_never_asks_the_village_question_with_an_unspeakable_label():
    """Different societies but no societyName. Falling back to the society code
    would ask the caller to read out "00731"; falling back to farmerName would
    re-ask the byte-similar-name question this block exists to remove. Neither
    is acceptable — book option 1 and say so."""
    out = _build_compact_farmer_summary(_env([
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0554", "farmerName": "Ramesh"},
        {"unionCode": "159", "societyCode": "00262", "farmerCode": "0192", "farmerName": "Suresh"},
    ]))
    assert "Ask which village" not in out
    # the society CODE must never appear as a spoken choice
    assert "Ask which village the animal is in (00731" not in out
    assert "Use Farmer option 1." in out
    assert "cannot be told apart by village name" in out


def test_same_village_name_across_different_societies_is_not_collapsed():
    """Labels are deduped per village key, not by display text: two societies
    sharing a name must not silently become one choice."""
    out = _build_compact_farmer_summary(_env([
        {"unionCode": "159", "societyCode": "00731", "farmerCode": "0554",
         "farmerName": "Ramesh", "societyName": "RAMOS"},
        {"unionCode": "159", "societyCode": "00262", "farmerCode": "0192",
         "farmerName": "Suresh", "societyName": "RAMOS"},
    ]))
    assert "Ask which village" not in out
    assert "Use Farmer option 1." in out


def test_duplicate_claim_needs_every_record_to_carry_that_name():
    """One named record and one unnamed are not "duplicate records of one
    farmer" — they may be two household members, one missing a name upstream."""
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Ramesh Patel"), _rec("0192", None),
    ]))
    assert "duplicate records of one farmer" not in out


def test_all_names_empty_is_not_called_a_duplicate():
    out = _build_compact_farmer_summary(_env([
        _rec("0554", None), _rec("0192", None),
    ]))
    assert "duplicate records of one farmer" not in out


def test_name_normalisation_handles_any_whitespace():
    from app.voice.farmer import _normalize_farmer_name
    assert _normalize_farmer_name("Patel   Asvin") == _normalize_farmer_name("Patel Asvin")
    assert _normalize_farmer_name("Patel\tAsvin") == _normalize_farmer_name("Patel Asvin")
    assert _normalize_farmer_name("PATEL ASVIN..") == _normalize_farmer_name("patel asvin")


def test_the_instruction_is_scoped_to_booking():
    """The removed line was booking-scoped. An unqualified "do not ask which
    farmer" would also suppress the legitimate question for herd, milk and
    treatment-history lookups."""
    out = _build_compact_farmer_summary(_env([
        _rec("0554", "Patel Asvinbhai"), _rec("0192", "Patel Nuruben"),
    ]))
    assert "for ai booking" in out.lower()
