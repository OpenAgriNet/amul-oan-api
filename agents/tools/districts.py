"""Gujarat district → mandi/weather search coordinates.

A **static** table. There is deliberately no runtime geocoder here, and there
must never be one: Bharat Vistaar's price-discovery BPP does a **radius search
(~50 km) around the GPS point** and coordinate changes below ~0.05° (≈5 km) make
no difference at all, so district-HQ-town precision is already more than the
upstream can resolve. A network geocode would add a hop, a failure mode and a
latency tail to buy nothing.

Three things about this table are load-bearing:

1. **Values are district-HQ *town* coordinates, not geometric centroids.**
   APMC market yards sit in the HQ towns; a centroid can easily land in farmland
   40 km from any market.

2. **Each district maps to an *ordered list* of candidates, not one point.**
   A single HQ does not cover a large district under a ~50 km catchment. Measured
   haversines: from Bhuj, Kutch leaves Bhachau at 69 km, Naliya 86, **Rapar 106**
   — most of Sarhad union's geography, including its largest city. From Deesa,
   Banaskantha leaves Tharad at 58 km and Vav at 69 — the north-Banaskantha milk
   heartland, in Amul's *largest* union. The caller walks the list on zero rows
   (see `agents/tools/vistaar.py`), which costs ~2.2 s on the failure path only
   and doubles as drift protection if a stored coordinate goes bad upstream.

3. **District strings must be normalised before lookup.** They arrive lowercased
   from two different backends with two different spellings — `kachchh` vs
   `kutch`, `banas kantha`, `the dangs`, `chhotaudepur`, `devbhumi dwarka`.
   Without `normalize_place`, a lookup silently falls through to the Anand
   default for *exactly* the districts this table was extended to cover.

⚠️ **VERIFICATION STATUS — read before trusting any coordinate.**

`verified=True` means only this: that coordinate was observed returning mandi
rows from the live BV BPP during the 2026-08-12 probe sweep. Everything else is
`verified=False` — geocoded offline from OpenStreetMap (Nominatim, one pass,
rate-limited, literals committed) and **never exercised against BV**, because BV
has been down since 2026-08-13 (`responses: []` from their BAP, both legs
timing out). Run `scripts/validate_district_coords.py` when BV recovers and
promote rows on evidence, not on plausibility.

Two further cautions from the same sweep:
  - **Bharuch/Narmada resolve to `Sendhwa APMC, Madhya Pradesh`** (3/3). The
    BPP's *stored* coordinate for that market is wrong by ~2° of longitude. Our
    coordinate is fine; the upstream record is not. Nothing here can fix it,
    which is exactly why the tool prints the district and state it actually got.
  - **South Gujarat is grain-empty.** Surat, Navsari, Valsad, Tapi and Dang
    return zero rows for Wheat/Bajra/Castor/Groundnut at *any* coordinate. A
    zero-row answer there is correct, not a bad coordinate — do not "fix" it by
    moving the point.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

__all__ = [
    "Candidate",
    "DistrictLocation",
    "DEFAULT_LOCATION",
    "DISTRICTS",
    "GUJARAT_BBOX",
    "covered_district_names",
    "normalize_place",
    "resolve_place",
    "unknown_place_message",
]

# Gujarat's approximate bounding box. Every shipped coordinate must sit inside
# it; a typo'd sign or a transposed lat/lon lands outside and is caught by
# tests/test_districts.py rather than by a farmer getting Madhya Pradesh prices.
GUJARAT_BBOX = (20.1, 24.7, 68.1, 74.5)  # (min_lat, max_lat, min_lon, max_lon)


@dataclass(frozen=True)
class Candidate:
    """One GPS point to try for a district, in order.

    `verified` is evidence, not confidence: True only where the coordinate was
    observed returning live BV mandi rows.
    """

    town: str
    lat: float
    lon: float
    verified: bool = False


@dataclass(frozen=True)
class DistrictLocation:
    key: str
    display: str
    candidates: tuple[Candidate, ...]

    @property
    def primary(self) -> Candidate:
        return self.candidates[0]


def _d(key: str, display: str, *candidates: Candidate) -> tuple[str, DistrictLocation]:
    return key, DistrictLocation(key=key, display=display, candidates=tuple(candidates))


# ── The table ────────────────────────────────────────────────────────────────
# All 33 Gujarat districts. The earlier 28-district draft omitted Narmada, Tapi,
# Dang, Chhota Udaipur and Devbhoomi Dwarka — which is precisely where the
# Dudhdhara, Sumul and Vasudhara farmers are, i.e. the ones a missing row would
# strand 55–67 km from any anchor.
DISTRICTS: dict[str, DistrictLocation] = dict(
    [
        _d("ahmedabad", "Ahmedabad",
           Candidate("Ahmedabad", 23.022, 72.580, verified=True),
           Candidate("Viramgam", 23.122, 72.048)),
        _d("amreli", "Amreli",
           Candidate("Amreli", 21.418, 71.250, verified=True),
           Candidate("Savarkundla", 21.339, 71.308)),
        _d("anand", "Anand",
           Candidate("Anand", 22.474, 72.736, verified=True)),
        _d("aravalli", "Aravalli",
           Candidate("Modasa", 23.463, 73.299, verified=True)),
        # Banaskantha leads with Deesa, not the administrative HQ Palanpur:
        # Deesa + Onion returned 10 rows from Deesa Veg Yard where Palanpur
        # returned 3 rows from Abu Road APMC, *Rajasthan*. ⚠️ That evidence is
        # onion-only and Deesa's known yard is a vegetable yard, so the grain
        # case is untested — which is why Palanpur stays as candidate 2 rather
        # than being dropped. Tharad covers the north-Banaskantha milk belt.
        _d("banaskantha", "Banaskantha",
           Candidate("Deesa", 24.260, 72.180, verified=True),
           Candidate("Palanpur", 24.171, 72.437),
           Candidate("Tharad", 24.387, 71.625)),
        # ⚠️ Bharuch resolves upstream to Sendhwa APMC, Madhya Pradesh (3/3) —
        # a wrong stored coordinate in the BPP, not a wrong coordinate here.
        _d("bharuch", "Bharuch",
           Candidate("Bharuch", 21.708, 72.996),
           Candidate("Jambusar", 22.051, 72.807)),
        _d("bhavnagar", "Bhavnagar",
           Candidate("Bhavnagar", 21.772, 72.142, verified=True),
           Candidate("Mahuva", 21.091, 71.762)),
        _d("botad", "Botad",
           Candidate("Botad", 22.047, 71.669, verified=True)),
        _d("chhotaudaipur", "Chhota Udaipur",
           Candidate("Chhota Udaipur", 22.315, 74.014),
           Candidate("Bodeli", 22.275, 73.717)),
        _d("dahod", "Dahod",
           Candidate("Dahod", 22.919, 74.134, verified=True),
           Candidate("Devgadh Baria", 22.701, 73.909)),
        _d("dang", "Dang",
           Candidate("Ahwa", 20.759, 73.687)),
        _d("devbhoomidwarka", "Devbhoomi Dwarka",
           Candidate("Khambhalia", 22.210, 69.650),
           Candidate("Dwarka", 22.243, 68.961)),
        _d("gandhinagar", "Gandhinagar",
           Candidate("Gandhinagar", 23.223, 72.649, verified=True)),
        _d("girsomnath", "Gir Somnath",
           Candidate("Veraval", 20.910, 70.365, verified=True),
           Candidate("Una", 20.820, 71.039)),
        _d("jamnagar", "Jamnagar",
           Candidate("Jamnagar", 22.473, 70.055, verified=True)),
        _d("junagadh", "Junagadh",
           Candidate("Junagadh", 21.522, 70.458, verified=True),
           Candidate("Visavadar", 21.342, 70.753)),
        # Kheda farmers are Amul (Kaira) — the union is not called "Nadiad".
        _d("kheda", "Kheda",
           Candidate("Nadiad", 22.690, 72.871, verified=True),
           Candidate("Kapadvanj", 23.023, 73.073)),
        _d("kutch", "Kutch",
           Candidate("Bhuj", 23.247, 69.668, verified=True),
           Candidate("Bhachau", 23.298, 70.346),
           Candidate("Rapar", 23.571, 70.645)),
        _d("mahisagar", "Mahisagar",
           Candidate("Lunawada", 23.129, 73.610, verified=True)),
        _d("mehsana", "Mehsana",
           Candidate("Mehsana", 23.601, 72.374, verified=True)),
        _d("morbi", "Morbi",
           Candidate("Morbi", 22.800, 70.886, verified=True)),
        _d("narmada", "Narmada",
           Candidate("Rajpipla", 21.870, 73.505)),
        _d("navsari", "Navsari",
           Candidate("Navsari", 20.952, 72.932, verified=True)),
        _d("panchmahal", "Panchmahal",
           Candidate("Godhra", 22.779, 73.625, verified=True),
           Candidate("Halol", 22.506, 73.472)),
        _d("patan", "Patan",
           Candidate("Patan", 23.774, 71.680, verified=True),
           Candidate("Radhanpur", 23.832, 71.610)),
        _d("porbandar", "Porbandar",
           Candidate("Porbandar", 21.603, 69.854, verified=True)),
        _d("rajkot", "Rajkot",
           Candidate("Rajkot", 22.305, 70.803, verified=True),
           Candidate("Jetpur", 21.754, 70.619)),
        _d("sabarkantha", "Sabarkantha",
           Candidate("Himatnagar", 23.597, 72.959, verified=True),
           Candidate("Idar", 23.841, 73.000)),
        _d("surat", "Surat",
           Candidate("Surat", 21.209, 72.832, verified=True),
           Candidate("Bardoli", 21.122, 73.114)),
        _d("surendranagar", "Surendranagar",
           Candidate("Surendranagar", 22.825, 71.621, verified=True),
           Candidate("Dhrangadhra", 22.991, 71.466)),
        _d("tapi", "Tapi",
           Candidate("Vyara", 21.112, 73.396)),
        _d("vadodara", "Vadodara",
           Candidate("Vadodara", 22.297, 73.194, verified=True)),
        _d("valsad", "Valsad",
           Candidate("Valsad", 20.432, 73.141, verified=True),
           Candidate("Vapi", 20.372, 72.917)),
    ]
)

DEFAULT_LOCATION = DISTRICTS["anand"]


# ── Normalisation ────────────────────────────────────────────────────────────
# District strings arrive lowercased from two backends in two spellings. Every
# key below is already normalised (`normalize_place` applied), so the alias map
# only has to carry genuine *spelling* variants, not case/spacing variants.
_DISTRICT_ALIASES: dict[str, str] = {
    "kachchh": "kutch", "kutchh": "kutch", "kachh": "kutch", "kachchhbhuj": "kutch",
    "banaskantha": "banaskantha", "banaskanta": "banaskantha", "banas": "banaskantha",
    "banaskatha": "banaskantha",
    "thedangs": "dang", "thedang": "dang", "dangs": "dang",
    "chhotaudepur": "chhotaudaipur", "chotaudepur": "chhotaudaipur",
    "chotaudaipur": "chhotaudaipur", "chhotaudepur1": "chhotaudaipur",
    "devbhumidwarka": "devbhoomidwarka", "devbhoomidwaraka": "devbhoomidwarka",
    "devbhumidwaraka": "devbhoomidwarka",
    "mahesana": "mehsana", "mehasana": "mehsana",
    "sabarkanta": "sabarkantha", "sabarkatha": "sabarkantha",
    "panchmahals": "panchmahal", "panchmahaal": "panchmahal",
    "kaira": "kheda", "khera": "kheda",
    "amdavad": "ahmedabad", "ahmadabad": "ahmedabad",
    "baroda": "vadodara",
    "aravali": "aravalli", "arvalli": "aravalli", "aravalii": "aravalli",
    "morvi": "morbi",
    "junagarh": "junagadh",
    "dohad": "dahod",
    "broach": "bharuch",
    "bulsar": "valsad",
    "somnath": "girsomnath", "girsomanath": "girsomnath",
    "mahisagr": "mahisagar",
}

# Towns a farmer plausibly names that are not themselves districts. They resolve
# to their district's candidate list, NOT to their own coordinate — the BPP does
# a radius search and picks markets that trade the queried commodity, so pinning
# a small town's point would defeat the thing the BPP is already good at. The
# tool always prints the market/district/state actually returned, so a farmer is
# never told a distant market is theirs.
_TOWN_ALIASES: dict[str, str] = {
    "khambhat": "anand", "borsad": "anand", "petlad": "anand", "umreth": "anand",
    "vallabhvidyanagar": "anand", "karamsad": "anand",
    "matar": "kheda", "dakor": "kheda", "mahemdavad": "kheda", "nadiad": "kheda",
    "kapadvanj": "kheda",
    "gandhidham": "kutch", "anjar": "kutch", "mandvi": "kutch", "naliya": "kutch",
    "nakhatrana": "kutch", "mundra": "kutch", "bhuj": "kutch", "bhachau": "kutch",
    "rapar": "kutch",
    "palanpur": "banaskantha", "deesa": "banaskantha", "disa": "banaskantha",
    "tharad": "banaskantha", "vav": "banaskantha", "dhanera": "banaskantha",
    "danta": "banaskantha",
    "siddhpur": "patan", "radhanpur": "patan", "chanasma": "patan",
    "unjha": "mehsana", "visnagar": "mehsana", "kadi": "mehsana", "vijapur": "mehsana",
    "kalol": "gandhinagar", "mansa": "gandhinagar", "dehgam": "gandhinagar",
    "himatnagar": "sabarkantha", "idar": "sabarkantha", "prantij": "sabarkantha",
    "modasa": "aravalli", "bayad": "aravalli",
    "santrampur": "mahisagar", "lunawada": "mahisagar", "balasinor": "mahisagar",
    "godhra": "panchmahal", "halol": "panchmahal", "shehera": "panchmahal",
    "devgadhbaria": "dahod", "jhalod": "dahod", "limkheda": "dahod",
    "bodeli": "chhotaudaipur", "kavant": "chhotaudaipur",
    "rajpipla": "narmada", "dediapada": "narmada", "nandod": "narmada",
    "ankleshwar": "bharuch", "jambusar": "bharuch", "amod": "bharuch",
    "bardoli": "surat", "olpad": "surat", "kamrej": "surat",
    "vyara": "tapi", "songadh": "tapi", "valod": "tapi",
    "ahwa": "dang", "waghai": "dang", "saputara": "dang",
    "bilimora": "navsari", "gandevi": "navsari", "chikhli": "navsari",
    "vapi": "valsad", "pardi": "valsad", "dharampur": "valsad",
    "dholka": "ahmedabad", "sanand": "ahmedabad", "bavla": "ahmedabad",
    "viramgam": "ahmedabad", "dhandhuka": "ahmedabad",
    "limbdi": "surendranagar", "wadhwan": "surendranagar",
    "dhrangadhra": "surendranagar", "chotila": "surendranagar",
    "wankaner": "morbi", "halvad": "morbi",
    "gondal": "rajkot", "jetpur": "rajkot", "dhoraji": "rajkot", "jasdan": "rajkot",
    "upleta": "rajkot",
    "keshod": "junagadh", "mangrol": "junagadh", "visavadar": "junagadh",
    "manavadar": "junagadh", "vanthali": "junagadh",
    "veraval": "girsomnath", "una": "girsomnath", "kodinar": "girsomnath",
    "talala": "girsomnath",
    "dhari": "amreli", "savarkundla": "amreli", "babra": "amreli", "rajula": "amreli",
    "sihor": "bhavnagar", "palitana": "bhavnagar", "talaja": "bhavnagar",
    "mahuva": "bhavnagar", "gariadhar": "bhavnagar",
    "khambhalia": "devbhoomidwarka", "dwarka": "devbhoomidwarka",
    "okha": "devbhoomidwarka", "bhanvad": "devbhoomidwarka",
    "dhrol": "jamnagar", "jamjodhpur": "jamnagar", "kalavad": "jamnagar",
    "ranavav": "porbandar", "kutiyana": "porbandar",
    "gadhada": "botad", "barwala": "botad",
    "padra": "vadodara", "dabhoi": "vadodara", "karjan": "vadodara",
    "savli": "vadodara", "waghodia": "vadodara",
}

_ALIASES: dict[str, str] = {**_TOWN_ALIASES, **_DISTRICT_ALIASES}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# "anand district", "anand dist.", "anand जिल्ला" → "anand". Suffixes are stripped
# AFTER the alnum squeeze, so "Banas Kantha District" → "banaskanthadistrict" →
# "banaskantha".
_SUFFIXES = ("district", "dist", "jilla", "jillo", "taluka", "tehsil", "apmc", "mandi")


def normalize_place(text: str | None) -> str:
    """Collapse a district/town string to a lookup key.

    Lowercases, drops every non-alphanumeric character (so `banas kantha`,
    `Banas-Kantha` and `BANASKANTHA` agree) and strips administrative suffixes.
    Returns "" for None/blank.
    """
    if not text:
        return ""
    key = _NON_ALNUM.sub("", str(text).casefold())
    for suffix in _SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            key = key[: -len(suffix)]
    return key


def resolve_place(text: str | None) -> DistrictLocation | None:
    """Resolve a district or town name to its ordered candidate list.

    Returns None when the name is not covered — the caller must then TELL the
    farmer, never silently substitute another location. A silent substitution is
    the failure mode this whole module exists to prevent.
    """
    key = normalize_place(text)
    if not key:
        return None
    key = _ALIASES.get(key, key)
    return DISTRICTS.get(key)


def covered_district_names() -> list[str]:
    """Display names of every covered district, alphabetical."""
    return sorted(loc.display for loc in DISTRICTS.values())


def unknown_place_message(text: str | None) -> str:
    """Farmer-facing text for a place we cannot resolve.

    Names the closest covered districts rather than dumping all 33, and never
    implies that some other location's prices were used instead.
    """
    asked = (text or "").strip() or "that location"
    key = normalize_place(text)
    pool = {normalize_place(loc.display): loc.display for loc in DISTRICTS.values()}
    for town_key, district_key in _TOWN_ALIASES.items():
        pool.setdefault(town_key, DISTRICTS[district_key].display)
    close = difflib.get_close_matches(key, list(pool), n=3, cutoff=0.6)
    suggestions = []
    for match in close:
        display = pool[match]
        if display not in suggestions:
            suggestions.append(display)
    if suggestions:
        hint = "Did you mean " + " or ".join(suggestions) + "?"
    else:
        hint = "I cover districts across Gujarat, for example " + ", ".join(
            ["Anand", "Banaskantha", "Kutch", "Rajkot", "Mehsana", "Junagadh"]
        ) + "."
    return f"I do not have market coverage for '{asked}'. {hint}"
