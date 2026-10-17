"""The LLM config channel comes from the deployment, not from the Step enum.

It used to be inferred: "voice" if ``Step`` had ``NON_MEANINGFUL``, else "chat".
Once voice's steps join this repo's enum, that inference would move every chat
deployment without ``PIPELINE_CHANNEL`` onto voice's live pipeline config, with
nothing failing. These pin that the default stays "chat" whatever the enum holds,
and that a voice deployment selects its config by saying so.
"""
import enum
import importlib

from app.llm_core import config_model, config_source


def _no_env(_name):
    return None


def test_default_channel_is_chat(monkeypatch):
    monkeypatch.setattr(config_source, "get_config_value", _no_env)

    assert config_source.channel() == "chat"
    assert config_source.key() == "llm_pipeline_config:chat"


def test_voice_steps_in_the_enum_do_not_move_chat_onto_voice_config(monkeypatch):
    class StepWithVoiceSteps(str, enum.Enum):
        AGENT = "agent"
        SUGGESTIONS = "suggestions"
        NON_MEANINGFUL = "non_meaningful"

    monkeypatch.setattr(config_model, "Step", StepWithVoiceSteps)
    try:
        reloaded = importlib.reload(config_source)
        monkeypatch.setattr(reloaded, "get_config_value", _no_env)
        assert reloaded.channel() == "chat"
        assert reloaded.key() == "llm_pipeline_config:chat"
    finally:
        monkeypatch.undo()
        importlib.reload(config_source)


def test_a_voice_deployment_selects_its_config_explicitly(monkeypatch):
    monkeypatch.setattr(
        config_source,
        "get_config_value",
        lambda name: "voice" if name == config_source.CHANNEL_ENV else None,
    )

    assert config_source.channel() == "voice"
    assert config_source.key() == "llm_pipeline_config:voice"
