"""AMUL-76: the chat prompt must follow the union-scheme tool's visibility.

`get_union_scheme_data` is hidden for every union without an ingested scheme
catalog (Kaira, Mehsana, Baroda, ... 14 of 21), yet the prompt advertised it to
every farmer and routed union-scheme questions to it. A Kaira farmer asking for
"all milk union, Gujarat and central schemes" got a different answer each turn:
the union part was refused ("not available in this session"), replaced with a
clarifying question, or occasionally found through document search.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agrinet import get_agrinet_instructions  # noqa: E402
from agents.deps import FarmerContext  # noqa: E402
from agents.tools import union_schemes  # noqa: E402
from helpers.utils import get_prompt  # noqa: E402

PROMPT = "agrinet_system_translation_pipeline.md"
QUERY = (
    "Give detailed information on all milk union, Gujarat government and "
    "Government of India schemes for animal husbandry and farming support"
)


def _deps(unions):
    return FarmerContext(query=QUERY, farmer_unions=unions)


def _render(union_schemes_available, network=True):
    return get_prompt(PROMPT, context={
        "today_date": "26-10-2026",
        "farmer_context": None, "ambiguity_hints": None,
        "response_max_chars": None, "loan_max_amount": "5,000",
        "loan_interest_rate_pct": "7", "network_tools_enabled": network,
        "union_schemes_available": union_schemes_available,
    })


@pytest.mark.parametrize("unions, expected", [
    (["kaira"], False),
    (["dudhsagar"], False),
    ([], False),
    (["banas"], True),
    (["sarhad"], True),
    (["kaira", "sabar"], True),
])
def test_catalog_gate(unions, expected):
    assert union_schemes.has_union_scheme_catalog(_deps(unions)) is expected


@pytest.mark.asyncio
async def test_prepare_hook_uses_the_same_gate():
    sentinel = object()
    hidden = await union_schemes.prepare_get_union_scheme_data(SimpleNamespace(deps=_deps(["kaira"])), sentinel)
    shown = await union_schemes.prepare_get_union_scheme_data(SimpleNamespace(deps=_deps(["banas"])), sentinel)
    assert hidden is None
    assert shown is sentinel


def test_hidden_tool_is_not_advertised_and_search_takes_over():
    rendered = _render(False)
    assert "get_union_scheme_data" not in rendered
    assert "no union scheme catalogue is available for this farmer" in rendered
    assert "dairy cooperative union scheme subsidy milk producers" in rendered
    assert "milk society secretary" in rendered


def test_supported_union_keeps_the_union_tool_first():
    rendered = _render(True)
    assert "`get_union_scheme_data(scheme_name=None)`" in rendered
    assert "use `get_union_scheme_data()` before `search_documents`" in rendered
    assert "no union scheme catalogue is available" not in rendered


@pytest.mark.parametrize("available", [True, False])
def test_multi_source_questions_get_every_part_answered(available):
    rendered = _render(available)
    assert "cover every source they named in one answer" in rendered
    assert "using `get_vistaar_scheme_info` for central schemes" in rendered
    assert "Do not reply with a question asking which kind of help or scheme they want" in rendered
    # Gujarat government schemes were dropped from both screenshots' answers.
    assert "Gujarat government animal-husbandry or dairy schemes → `search_documents`" in rendered


def test_central_tool_is_not_named_when_network_tools_are_off():
    assert "get_vistaar_scheme_info" not in _render(False, network=False)


def test_stale_profile_scheme_index_rule_is_gone():
    # Scheme titles stopped being preloaded into the farmer profile on
    # 2026-08-27; the rule sent the model looking for a list that is never there.
    assert "union scheme titles listed in the Farmer Profile" not in _render(True)


@pytest.mark.parametrize("unions, advertised", [(["kaira"], False), (["banas"], True)])
def test_agent_instructions_follow_the_farmers_union(unions, advertised):
    instructions = get_agrinet_instructions(SimpleNamespace(deps=_deps(unions)))
    assert ("get_union_scheme_data" in instructions) is advertised


def test_central_scheme_tool_schema_does_not_point_at_the_hidden_tool():
    # Tool docstrings reach the model in the schema regardless of the prompt.
    from agents.tools.vistaar import get_vistaar_scheme_info

    assert "get_union_scheme_data" not in get_vistaar_scheme_info.__doc__
