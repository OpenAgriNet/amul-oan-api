"""Tests for Amul Gujarati name preference in farmer/AIT context."""

from __future__ import annotations

from agents.farmer_context import _append_farmer_markdown
from agents.tools.beckn.amul import AITechnicianRecord
from agents.tools.bonus import _format_bonus_markdown
from agents.tools.farmer_envelope import collect_farmer_accounts
from agents.tools.models.bonus import FarmerBonusAmountRecordModel
from agents.tools.models.farmer import FarmerModel
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from agents.tools.models.local_names import prefer_local_name


class TestPreferLocalName:
    def test_prefers_nonempty_local(self):
        assert prefer_local_name("મૌછા", "MAUCHHA") == "મૌછા"

    def test_falls_back_to_english(self):
        assert prefer_local_name(None, "MAUCHHA") == "MAUCHHA"
        assert prefer_local_name("", "MAUCHHA") == "MAUCHHA"
        assert prefer_local_name("   ", "MAUCHHA") == "MAUCHHA"

    def test_strips_whitespace(self):
        assert prefer_local_name("  રોશનાઈ  ", "Roshan") == "રોશનાઈ"

    def test_both_missing(self):
        assert prefer_local_name(None, None) is None
        assert prefer_local_name("", "  ") is None


class TestFarmerModelGujaratiFields:
    def test_parses_pashugpt_gujarati_fields_and_preserves_script(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerName": "MAKVANA RAJDEEPSINH DILIPSINH",
                "farmerGujaratiName": "મકવાણા રાજદીપસિંહ દિલીપસિંહ",
                "societyName": "MAUCHHA",
                "societyGujaratiName": "મૌછા",
                "unionName": "SABAR",
                "unionCode": "169",
                "societyCode": "12345",
                "farmerCode": "999",
            }
        )
        assert farmer.farmer_name == "makvana rajdeepsinh dilipsinh"
        assert farmer.society_name == "mauchha"
        assert farmer.farmer_gujarati_name == "મકવાણા રાજદીપસિંહ દિલીપસિંહ"
        assert farmer.society_gujarati_name == "મૌછા"
        assert farmer.display_farmer_name == "મકવાણા રાજદીપસિંહ દિલીપસિંહ"
        assert farmer.display_society_name == "મૌછા"

    def test_display_falls_back_when_gujarati_missing(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerName": "MAKVANA RAJDEEPSINH DILIPSINH",
                "societyName": "MAUCHHA",
            }
        )
        assert farmer.farmer_gujarati_name is None
        assert farmer.display_farmer_name == "makvana rajdeepsinh dilipsinh"
        assert farmer.display_society_name == "mauchha"

    def test_display_falls_back_when_gujarati_blank(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerName": "RAMESH BHAI",
                "farmerGujaratiName": "   ",
                "societyName": "DEMO",
                "societyGujaratiName": "",
            }
        )
        assert farmer.farmer_gujarati_name is None
        assert farmer.society_gujarati_name is None
        assert farmer.display_farmer_name == "ramesh bhai"
        assert farmer.display_society_name == "demo"

    def test_accepts_alternate_alias_spellings(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerFullNamesGuj": "પર્વતભાઈ",
                "societyFullNamesGuj": "ભૈસરા",
                "farmerName": "PARVATBHAI",
                "societyName": "BHAISARA",
            }
        )
        assert farmer.display_farmer_name == "પર્વતભાઈ"
        assert farmer.display_society_name == "ભૈસરા"

    def test_accepts_bonus_style_local_aliases(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerLocalName": "ફર્મર",
                "societyNameLocal": "મંડળી",
                "farmerName": "FARMER",
                "societyName": "SOCIETY",
            }
        )
        assert farmer.display_farmer_name == "ફર્મર"
        assert farmer.display_society_name == "મંડળી"


class TestFarmerContextMarkdown:
    def test_markdown_uses_gujarati_when_present(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerName": "MAKVANA RAJDEEPSINH DILIPSINH",
                "farmerGujaratiName": "મકવાણા રાજદીપસિંહ દિલીપસિંહ",
                "societyName": "MAUCHHA",
                "societyGujaratiName": "મૌછા",
                "farmerCode": "999",
                "societyCode": "12345",
            }
        )
        lines: list[str] = []
        _append_farmer_markdown(lines, farmer, 1)
        text = "\n".join(lines)
        assert "મકવાણા રાજદીપસિંહ દિલીપસિંહ" in text
        assert "મૌછા" in text
        assert "makvana rajdeepsinh dilipsinh" not in text
        assert "mauchha" not in text

    def test_markdown_uses_english_when_gujarati_absent(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerName": "MAKVANA RAJDEEPSINH DILIPSINH",
                "societyName": "MAUCHHA",
            }
        )
        lines: list[str] = []
        _append_farmer_markdown(lines, farmer, 1)
        text = "\n".join(lines)
        assert "makvana rajdeepsinh dilipsinh" in text
        assert "mauchha" in text

    def test_record_to_model_keeps_gujarati_fields(self):
        record = FarmerRecord.model_validate(
            {
                "farmerName": "MAKVANA RAJDEEPSINH DILIPSINH",
                "farmerGujaratiName": "મકવાણા રાજદીપસિંહ દિલીપસિંહ",
                "societyName": "MAUCHHA",
                "societyGujaratiName": "મૌછા",
                "farmerCode": "999",
            }
        )
        farmer = FarmerModel.model_validate(record.model_dump())
        assert farmer.display_farmer_name == "મકવાણા રાજદીપસિંહ દિલીપસિંહ"
        assert farmer.display_society_name == "મૌછા"

    def test_model_dump_roundtrip_preserves_gujarati(self):
        farmer = FarmerModel.model_validate(
            {
                "farmerName": "OLD CACHE",
                "societyName": "OLD SOCIETY",
                "farmerGujaratiName": "જૂનું",
                "societyGujaratiName": "મંડળી",
            }
        )
        record = FarmerRecord.model_validate(farmer.model_dump())
        assert record.farmerGujaratiName == "જૂનું"
        restored = FarmerModel.model_validate(record.model_dump())
        assert restored.display_farmer_name == "જૂનું"


class TestAITechnicianGujaratiFields:
    def test_parses_amul_typo_gujrati_fullname(self):
        tech = AITechnicianRecord.model_validate(
            {
                "userId": "abc==",
                "fullName": "Roshan-MPP-Ptest",
                "gujratiFullName": "રોશનાઈ ",
                "mobileNumber": "9216541600",
            }
        )
        assert tech.gujaratiFullName == "રોશનાઈ "
        assert tech.display_full_name == "રોશનાઈ"

    def test_parses_correct_gujarati_spelling(self):
        tech = AITechnicianRecord.model_validate(
            {
                "userId": "abc==",
                "fullName": "SHAILESHBHAI",
                "gujaratiFullName": "શૈલેશભાઈ",
                "mobileNumber": "9000000000",
            }
        )
        assert tech.display_full_name == "શૈલેશભાઈ"

    def test_falls_back_to_english_full_name(self):
        tech = AITechnicianRecord.model_validate(
            {
                "userId": "abc==",
                "fullName": "Switi-Ait-Ait",
                "mobileNumber": "9000000001",
            }
        )
        assert tech.display_full_name == "Switi-Ait-Ait"

    def test_model_dump_roundtrip_keeps_gujarati_for_cache(self):
        tech = AITechnicianRecord.model_validate(
            {
                "userId": "abc==",
                "fullName": "Roshan-MPP-Ptest",
                "gujratiFullName": "રોશનાઈ",
                "mobileNumber": "9216541600",
            }
        )
        dumped = tech.model_dump()
        restored = AITechnicianRecord.model_validate(dumped)
        assert restored.display_full_name == "રોશનાઈ"


class TestBonusAndEnvelopePreference:
    def test_bonus_markdown_prefers_local_names(self):
        record = FarmerBonusAmountRecordModel.model_validate(
            {
                "societyCode": "2004",
                "societyName": "DEMO_2004",
                "societyNameLocal": "ડૅમૉ_૨૦૦૪",
                "farmerCode": "0001",
                "farmerName": "FARMER1",
                "farmerLocalName": "ફર્મૅર્૧",
                "bonusAmount": 1273.1,
                "fromDate": "2026-04-01T00:00:00",
                "toDate": "2026-04-01T00:00:00",
            }
        )
        text = _format_bonus_markdown([record])
        assert "ડૅમૉ_૨૦૦૪" in text
        assert "ફર્મૅર્૧" in text
        assert "DEMO_2004" not in text
        assert "FARMER1" not in text

    def test_bonus_markdown_falls_back_to_english(self):
        record = FarmerBonusAmountRecordModel.model_validate(
            {
                "societyCode": "2004",
                "societyName": "DEMO_2004",
                "farmerCode": "0001",
                "farmerName": "FARMER1",
                "bonusAmount": 10,
                "fromDate": "2026-04-01T00:00:00",
                "toDate": "2026-04-01T00:00:00",
            }
        )
        text = _format_bonus_markdown([record])
        assert "DEMO_2004" in text
        assert "FARMER1" in text

    def test_collect_accounts_prefers_gujarati_names(self):
        env = FarmerDataEnvelope(
            farmers=[
                FarmerRecord.model_validate(
                    {
                        "unionCode": "169",
                        "societyCode": "12345",
                        "farmerCode": "999",
                        "farmerName": "MAKVANA",
                        "farmerGujaratiName": "મકવાણા",
                        "societyName": "MAUCHHA",
                        "societyGujaratiName": "મૌછા",
                    }
                )
            ],
            source="test",
        )
        accounts = collect_farmer_accounts(env)
        assert len(accounts) == 1
        assert accounts[0].farmer_name == "મકવાણા"
        assert accounts[0].society_name == "મૌછા"
