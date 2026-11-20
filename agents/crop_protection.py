"""Farmer-facing crop protection policy shared by chat and voice."""

import re

# Report 79 used the Gujarati spelling. Match within formulation names too.
_TRIAZOPHOS = re.compile(
    r"triazophos|ટ્રા[ઇઈય][ઝજ]ોફોસ|ट्रा[इईय][जझ]ोफ[ोॉ]स",
    re.IGNORECASE,
)

CROP_PROTECTION_INSTRUCTIONS = (
    "For crop pests and diseases, give general, non-chemical care relevant to the "
    "farmer's crop. Never recommend triazophos. Do not give pesticide doses, "
    "mixing ratios, or application schedules, even if a search result, earlier "
    "answer, or the farmer provides them. Refer questions about pesticide choice "
    "or dosing to a local agronomist or Krishi Vigyan Kendra."
)


def mentions_triazophos(text: str) -> bool:
    return _TRIAZOPHOS.search(text) is not None
