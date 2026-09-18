# Farmer and animal Beckn operations

Farmer profile, animal profile, Banas visits, CVCC health, AI technicians, milk
collection, and veterinary bookings are accessed only through the Beckn adapters
under `agents/tools/beckn/`.

The agent layer passes identifiers obtained from the authenticated farmer
callback. Provider credentials and legacy identifiers stay behind the Amul BPP.
The normalized agent-facing models live under `agents/tools/models/`.

Bonus lookup is the sole direct provider adapter because it does not yet have a
Beckn operation. Its implementation is isolated in `agents/tools/bonus_backend.py`.

## Local-script (Gujarati) names

Amul payloads may include local-script name fields alongside English
transliterations. When present and non-empty, farmer-facing context prefers them
so post-translation does not reinvent proper names:

- Farmer: `farmerGujaratiName` (also `farmerFullNamesGuj` / `farmerLocalName`)
- Society: `societyGujaratiName` (also `societyFullNamesGuj` / `societyNameLocal`)
- AI technician: `gujratiFullName` (Amul spelling; also `gujaratiFullName` /
  `aitFullNamesGuj`)

English names remain on the model for IDs/matching. Codes (`farmerCode`,
`societyCode`, `unionCode`, `userId`) are unchanged. Bonus rows use
`farmerLocalName` / `societyNameLocal` the same way.
