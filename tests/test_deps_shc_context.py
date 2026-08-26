from agents.deps import FarmerContext


def test_private_shc_context_is_given_to_agent_with_current_question():
    deps = FarmerContext(
        query="What are the nutrient levels in my soil?",
        soil_health_card_context="Available Nitrogen (N): 175.53 kg/ha",
    )

    message = deps.get_user_message()

    assert "Private Soil Health Card context" in message
    assert "Available Nitrogen (N): 175.53 kg/ha" in message
    assert '"What are the nutrient levels in my soil?"' in message
