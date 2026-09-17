from helpers.utils import get_prompt


def test_moderation_allows_bengali_landing_card_milk_earnings():
    prompt = get_prompt("moderation_system")

    assert "milk-sale earnings" in prompt
    assert "How much have I earned in total this month and last month?" in prompt
    assert "এই মাসে ও গত মাসে দুধ বিক্রি করে আমার মোট আয় কত?" in prompt
    assert "those queries are `valid_agricultural`" in prompt


def test_agent_routes_landing_card_earnings_to_milk_collection():
    prompt = get_prompt("agrinet_system_translation_pipeline")

    assert "equivalent localized landing-card text" in prompt
    assert "make two tool calls (one per month)" in prompt
    assert "Period | Gross Milk Amount | Deductions | Net Amount" in prompt
