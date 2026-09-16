# Farmer and animal Beckn operations

Farmer profile, animal profile, Banas visits, CVCC health, AI technicians, milk
collection, and veterinary bookings are accessed only through the Beckn adapters
under `agents/tools/beckn/`.

The agent layer passes identifiers obtained from the authenticated farmer
callback. Provider credentials and legacy identifiers stay behind the Amul BPP.
The normalized agent-facing models live under `agents/tools/models/`.

Bonus lookup is the sole direct provider adapter because it does not yet have a
Beckn operation. Its implementation is isolated in `agents/tools/bonus_backend.py`.
