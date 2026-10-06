import pytest

from agents.voice.models.ai_call import AICallResponseModel, strip_ait_name_codes


@pytest.mark.parametrize("raw,expected", [
    ("1712 NARENDRAKUMAR-NARAYANDAS-PANDOR", "Narendrakumar Narayandas Pandor"),
    ("ULPESHPURI-KODARPURI-GOSHVAMI", "Ulpeshpuri Kodarpuri Goshvami"),
    ("1712-NARENDRAKUMAR-99", "Narendrakumar"),
    ("  204   TUSHARKUMAR 809 ", "Tusharkumar"),
    ("1730-Sanjaykumar Laxmanbhai Vanjara", "Sanjaykumar Laxmanbhai Vanjara"),
    ("Niraj AIT Patel", "Niraj AIT Patel"),
    ("1712", "1712"),
    ("", ""),
    (None, None),
])
def test_names_are_speakable(raw, expected):
    assert strip_ait_name_codes(raw) == expected


def test_booking_response_keeps_phone_and_cleans_name():
    model = AICallResponseModel.model_validate(
        {"aitName": "919876543210(918 MITULKUMAR-RAMESHBHAI)", "ticketNumber": "T1"}
    )
    assert model.ait_name == "9876543210(Mitulkumar Rameshbhai)"


def test_summary_cleans_names_written_raw_by_chat_into_the_shared_cache():
    from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
    from app.voice.farmer import _build_ai_technician_summary

    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord(farmerName="Rameshbhai", societyName="Anand", farmerCode="F1")],
        aiTechnicians=[{
            "farmerName": "Rameshbhai", "farmerCode": "F1", "societyName": "Anand",
            "societyCode": "1066", "unionCode": "2021",
            "technicians": [{"fullName": "1712 ULPESHPURI-KODARPURI-GOSHVAMI", "userId": "t1"}],
        }],
        source="cache",
    )
    assert "full_name=Ulpeshpuri Kodarpuri Goshvami" in _build_ai_technician_summary(envelope)
