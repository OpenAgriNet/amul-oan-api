"""Farmer-facing crop protection policy shared by chat and voice."""

import re

# Banned in India for manufacture, import and use: triazophos, dichlorvos and
# phorate (S.O. 3951(E), 2018), methomyl and dicofol (S.O. 4294(E), 2023).
# Report 79 used the Gujarati spelling of triazophos. The knowledge base also
# names dichlorvos as DDVP and Nuvan. Match within formulation names too.
_BANNED = re.compile(
    r"triazophos|ટ્રા[ઇઈય][ઝજ]ોફોસ|ट्रा[इईय][जझ]ोफ[ोॉ]स|dichlorvos|ddvp|nuvan"
    r"|phorate|methomyl|dicofol",
    re.IGNORECASE,
)

CROP_PROTECTION_INSTRUCTIONS = (
    "Never recommend triazophos, dichlorvos (DDVP, Nuvan), phorate, methomyl or "
    "dicofol: they are banned in India."
)


def mentions_banned_pesticide(text: str) -> bool:
    return _BANNED.search(text) is not None
