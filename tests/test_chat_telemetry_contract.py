"""Chat turns send turn.v1 as telemetry/contracts/turn.v1/chat.json says, step by step.

tests/test_chat_telemetry_schema_stamps.py checks the root's key names in chat.py;
this one runs chat turns through the real run_turn with a recording Langfuse
client, as tests/test_voice_telemetry_contract.py does for voice, and compares
the steps Langfuse was handed with the contract.
"""

import asyncio
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from pydantic_ai.messages import ModelResponse, TextPart

from app.channels.chat import WEB
from app.services import chat as chat_service
from app.services import translation as tr
from app.tasks import suggestions as suggestions_task
from app.turn.types import Turn
from tests.test_chat_turn_contract import _Run, patch_turn
from tests.test_voice_telemetry import _Langfuse

CONTRACT = Path(__file__).resolve().parents[1] / "telemetry" / "contracts" / "turn.v1" / "chat.json"


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="chat-contract",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="chat-contract",
        channel=WEB,
        persona="farmer",
    )
    values.update(overrides)
    return Turn(**values)


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


def _recorded_turn(monkeypatch, turn=None, *, answer="Give clean water daily."):
    """One chat turn with every outside stage stubbed, recorded as Langfuse would get it.
    The agent streams ``answer`` and its run's new messages hold it, as pydantic-ai's do."""
    patch_turn(monkeypatch)

    def _agent(**kw):
        run = _Run([answer], deps=kw["deps"])
        run.result = SimpleNamespace(new_messages=lambda: [ModelResponse(parts=[TextPart(content=answer)])])
        return run

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _agent)
    client = _Langfuse()
    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(chat_service, "propagate_attributes", client.propagate_attributes)

    async def _go():
        stream = chat_service.run_turn(turn or _turn(), chat_service.CHAT_SURFACE, scheduler=_Scheduler())
        return [emission async for emission in stream]

    asyncio.run(_go())
    return client


def test_the_moderation_step_sends_the_contracts_output(monkeypatch):
    """Chat's model gives a category and an action, no reason: the action goes under its own name,
    not as turn.v1's reason, which is a model's reason (voice's)."""
    client = _recorded_turn(monkeypatch)

    moderation = client.step("moderation").sent
    assert moderation["output"] == {"rejected": False, "category": "valid_agricultural", "action": "allow"}
    assert set(moderation["output"]) == set(_contract()["steps"]["moderation"]["output"])


def test_the_agent_step_is_the_question_in_and_the_agents_own_answer_out(monkeypatch):
    """As voice's: the question the agent is asked (the pretranslated English) in and its
    English answer out, not the model, persona and moderation action in and the translated
    answer the farmer got out."""
    client = _recorded_turn(monkeypatch)

    agent = client.step("agent").sent
    assert agent["as_type"] == "agent"
    assert agent["input"] == "How much water should I give my cow?"
    assert agent["output"] == "Give clean water daily."
    assert client.io["output"] == "[Give clean water daily.]"  # the root: the answer as delivered, translated


def test_the_roots_input_is_the_whole_question(monkeypatch):
    """turn.v1's input is the question itself, whole: the session and user ids are capped,
    never the text."""
    question = "મારી ગાયને કેટલું પાણી આપવું? " * 40
    client = _recorded_turn(monkeypatch, _turn(query=question))

    assert len(question) > 1000 and client.io["input"] == question


# ── every step chat opens, against the contract ──────────────────────────────


def _module_steps(monkeypatch):
    """The steps chat's modules open on their own, each call recorded: pre_translation,
    post_translation and stream_translation through TranslateGemma and through an LLM
    tier, and the suggestions after the response. Only the network is stubbed."""
    client = _Langfuse()
    monkeypatch.setattr(tr, "_get_langfuse", lambda: client)

    async def _stream(*_a, **_k):
        yield "ગાયને "
        yield "પાણી આપો."

    async def _unary(*_a, **_k):
        return "ગાયને પાણી આપો."

    monkeypatch.setattr(tr, "_raw_translategemma_stream", _stream)
    monkeypatch.setattr(tr, "_raw_llm_translation_stream", _stream)
    monkeypatch.setattr(tr, "_raw_translategemma_unary", _unary)
    monkeypatch.setattr(tr, "_raw_llm_translation_unary", _unary)
    descriptor = SimpleNamespace(model_id="translategemma-27b-base")

    async def _completions(**_kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="How much water?"))])

    tier = SimpleNamespace(provider="openai", model_name="gpt", handle=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_completions))))

    monkeypatch.setattr(suggestions_task, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(suggestions_task, "propagate_attributes", client.propagate_attributes)

    async def _history(_session):
        return []

    async def _noop(*_a, **_k):
        return None

    async def _suggest(*_a, **_k):
        return SimpleNamespace(output=["How often should I water?"])

    monkeypatch.setattr(suggestions_task, "_get_message_history", _history)
    monkeypatch.setattr(suggestions_task, "set_cache", _noop)
    monkeypatch.setattr(suggestions_task.cache, "delete", _noop)
    monkeypatch.setattr(suggestions_task, "_step_metadata", lambda _e: {"amul.served_tier": "vllm:gemma[0]"})
    execution = SimpleNamespace(info=lambda _step: SimpleNamespace(model_name="gemma"), run=_suggest)

    async def _go():
        args = ("Give the cow water.", "english", "gu", 0.0, 64)
        [_ async for _ in tr._translategemma_stream(descriptor, "prompt", *args)]
        [_ async for _ in tr._llm_translation_stream(object(), "gemma", "instruction", *args, provider="vllm")]
        await tr._translategemma_unary(descriptor, "prompt", *args)
        await tr._llm_translation_unary(object(), "gemma", "instruction", *args, provider="openai")
        await tr.pretranslate_with_tier(tier, text="મારી ગાયને કેટલું પાણી?", source_lang="gujarati")
        await suggestions_task.create_suggestions("chat-contract", "gu", execution, None)

    asyncio.run(_go())
    return client.observations


def _steps_sent(observations):
    """Per step name: the types it was opened as, the keys of its output and metadata. The
    metadata is every key the step was opened or updated with: Langfuse keeps them all."""
    steps = {}
    for observation in observations:
        sent = observation.sent
        step = steps.setdefault(sent["name"], {"types": set(), "output": set(), "metadata_keys": set()})
        step["types"].add(sent["as_type"].upper())
        if isinstance(sent.get("output"), dict):
            step["output"].update(sent["output"])
        for metadata in [observation.kwargs.get("metadata"), *(update.get("metadata") for update in observation.updates)]:
            step["metadata_keys"].update(metadata or {})
    return steps


def test_the_steps_chat_opens_match_the_contract(monkeypatch):
    """What check_trace and the root-key test can't see: every step chat opens, its type, the
    keys of its structured output and its metadata, as Langfuse was handed them."""
    observations = [o for o in _recorded_turn(monkeypatch).observations if o.kwargs["name"] != "turn"]
    observations += _module_steps(monkeypatch)
    sent = _steps_sent(observations)
    listed = _contract()["steps"]

    assert set(sent) == set(listed), f"chat opens {sorted(sent)}, the contract lists {sorted(listed)}"
    for name, step in sent.items():
        assert step["types"] == {listed[name]["type"]}, f"step {name} is opened as {sorted(step['types'])}"
        assert step["output"] == set(listed[name].get("output", ())), f"step {name}'s output fields {sorted(step['output'])}"
        assert all(key.startswith("amul.") for key in step["metadata_keys"]), f"step {name} sends {sorted(step['metadata_keys'])}"
        assert step["metadata_keys"] <= set(listed[name]["metadata_keys"]), f"step {name} sends {sorted(step['metadata_keys'])}"
    # Each translation call names the provider that served it, as turn.v1 names it.
    assert [(o.sent["name"], o.sent["metadata"]["amul.served_provider"]) for o in observations if "translation" in o.sent["name"]] == [
        ("stream_translation", "translategemma"), ("stream_translation", "vllm"),
        ("post_translation", "translategemma"), ("post_translation", "openai"), ("pre_translation", "openai"),
    ]
