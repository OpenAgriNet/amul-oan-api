"""What the test case screen offers in its dropdowns (GET /api/test-cases/options/).

The screen hardcoded the case types and tags (src/lib/test-cases/taxonomy.ts in
amul-ui), and testers and test accounts had no list yet. Keeping them here means
a new tester, test account or tag is a backend change, not a UI build. Ids are what the screen sends; labels and hints are for display.
Case types and tags are hints for the labeller (#488), and the lists are still
being reviewed (#486).
"""
from pydantic import BaseModel


class Option(BaseModel):
    id: str
    label: str


class HintOption(Option):
    hint: str


class CaseOptions(BaseModel):
    testers: list[Option]
    test_accounts: list[Option]
    case_types: list[HintOption]
    tags: list[HintOption]


TESTERS = [
    Option(id="suraj", label="Suraj"),
    Option(id="kanav", label="Kanav"),
    Option(id="yash", label="Yash"),
    Option(id="aayush", label="Aayush"),
    Option(id="vishal", label="Vishal"),
]

# The test farmers of eval set v0, all made up. The mock serves only F1 today;
# F2 to F6 need it to serve more farmers (#513).
TEST_ACCOUNTS = [
    Option(id="f1", label="Banas member, one cow and one buffalo, every lookup succeeds"),
    Option(id="f2", label="Dudhsagar member, buffaloes only"),
    Option(id="f3", label="Member of a union where app booking of insemination is switched off"),
    Option(id="f4", label="Lookups fail or come back empty"),
    Option(id="f5", label="Mobile not registered with any society"),
    Option(id="f6", label="One mobile with two member records in two unions"),
]

CASE_TYPES = [
    HintOption(
        id="new_flow",
        label="New flow",
        hint="Something the bot was never tested on: a new feature or a new kind of question.",
    ),
    HintOption(
        id="edge_case",
        label="Edge case",
        hint="A known flow that breaks in an unusual situation (missing data, odd wording, mixed language).",
    ),
    HintOption(
        id="wrong_answer",
        label="Wrong answer",
        hint="The bot answered wrongly. Fix the answer above, or leave it and explain in the description what is wrong.",
    ),
    HintOption(
        id="good_answer",
        label="Good answer to keep",
        hint="The bot got it right and must keep getting it right.",
    ),
    HintOption(
        id="should_not_happen",
        label="Must never happen",
        hint="Advice or behaviour the bot must avoid, e.g. a medicine that is unsafe in this situation.",
    ),
]

TAGS = [
    HintOption(id="veterinary_advisory", label="Veterinary advisory", hint="Animal health, disease, treatment, heat, pregnancy."),
    HintOption(id="ait_booking", label="AIT booking", hint="Booking an artificial insemination technician, and its status."),
    HintOption(id="milk_collection", label="Milk collection", hint="Farmer's milk pouring records, fat/SNF, quantities."),
    HintOption(id="milk_earnings", label="Milk earnings & bonus", hint="Payments, earnings, bonus calculation."),
    HintOption(id="union_schemes", label="Union schemes", hint="Schemes of the milk union (e.g. a specific district union)."),
    HintOption(id="government_schemes", label="Government schemes", hint="State and central government schemes and subsidies."),
    HintOption(id="animal_nutrition", label="Feed & nutrition", hint="Cattle feed, fodder, mineral mixture."),
    HintOption(id="weather", label="Weather", hint="Forecasts and weather-based advice."),
    HintOption(id="market_prices", label="Market prices (APMC)", hint="Mandi / APMC prices for crops."),
    HintOption(id="crop_advisory", label="Crop advisory", hint="Farming, crops, soil, pests."),
    HintOption(id="farmer_profile", label="Farmer profile & sign-in", hint="Farmer ID, society, account details."),
    # On the feature list (#486) but not on the screen before.
    HintOption(id="animal_records", label="Animal records", hint="The farmer's animals and their health and visit records."),
    HintOption(id="cattle_trade", label="Cattle trade", hint="Cattle rates, buying and selling animals."),
    HintOption(id="soil_health_card", label="Soil health card", hint="The farmer's soil health card and what it says."),
    HintOption(id="loan", label="Micro-loan", hint="Micro-loan eligibility and loan codes."),
    HintOption(id="vet_office", label="Nearest vet office", hint="Finding the nearest government veterinary office."),
    HintOption(id="general", label="General / other", hint="Greetings, small talk, out-of-scope questions."),
]

CASE_OPTIONS = CaseOptions(testers=TESTERS, test_accounts=TEST_ACCOUNTS, case_types=CASE_TYPES, tags=TAGS)
