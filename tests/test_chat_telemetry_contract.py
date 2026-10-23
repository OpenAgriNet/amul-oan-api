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

import pytest
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


def _recorded_turn(monkeypatch, turn=None, *, answer="Give clean water daily.", translate=None):
    """One chat turn with every outside stage stubbed, recorded as Langfuse would get it.
    The agent streams ``answer`` and its run's new messages hold it, as pydantic-ai's do.
    ``translate`` replaces the stubbed translation stream (which echoes ``[text]``)."""
    patch_turn(monkeypatch)
    if translate is not None:
        monkeypatch.setattr(chat_service, "translate_text_stream_fast", translate)

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


def test_post_translation_is_one_span_under_the_root_for_the_whole_answer(monkeypatch):
    """One step for the turn's translation, a direct child of the root (not of the agent
    step, which is current while the answer streams): the English given in, what
    translation delivered out, closed when the answer is out."""
    client = _recorded_turn(monkeypatch, answer="Give clean water daily. Twice a day.")

    (step,) = client.steps("post_translation")
    assert (step.parent, step.sent["as_type"]) == ("turn", "span")
    assert step.sent["input"] == "Give clean water daily. Twice a day."
    # Two batches, each translated as the stubbed stream echoes it.
    assert step.sent["output"] == "[Give clean water daily. ][Twice a day.]" == client.io["output"]
    assert step.ended == 1 and "level" not in step.sent


def test_a_batch_that_fails_translation_marks_the_step_an_error_and_goes_out_in_english(monkeypatch):
    async def _down(text, *_a, **_k):
        raise RuntimeError("translategemma down")
        yield  # pragma: no cover - an async generator

    client = _recorded_turn(monkeypatch, translate=_down)

    step = client.step("post_translation").sent
    assert (step["level"], step["status_message"]) == ("ERROR", "translategemma down")
    assert step["input"] == step["output"] == "Give clean water daily." == client.io["output"]


def test_the_roots_input_is_the_whole_question(monkeypatch):
    """turn.v1's input is the question itself, whole: the session and user ids are capped,
    never the text."""
    question = "મારી ગાયને કેટલું પાણી આપવું? " * 40
    client = _recorded_turn(monkeypatch, _turn(query=question))

    assert len(question) > 1000 and client.io["input"] == question


# ── every step chat opens, against the contract ──────────────────────────────


def _module_steps(monkeypatch):
    """The steps chat's modules open on their own, each call recorded: pre_translation,
    stream_translation and text_translation through TranslateGemma and through an LLM
    tier, under a turn's post_translation span and under the current span, and the
    suggestions after the response. Only the network is stubbed."""
    client = _Langfuse()
    monkeypatch.setattr(tr, "_get_langfuse", lambda: client)
    # llm_core targets as the chain hands them to a call: the provider and the tier that served.
    translategemma = SimpleNamespace(provider="translategemma", served_tier="translategemma:tg[0]")
    vllm = SimpleNamespace(provider="vllm", served_tier="vllm:gemma[1]")

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

    tier = SimpleNamespace(provider="openai", model_name="gpt", served_tier="openai:gpt[0]", handle=SimpleNamespace(
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
        with client.start_as_current_observation(name="post_translation", as_type="span") as span:
            [_ async for _ in tr._translategemma_stream(descriptor, "prompt", *args, served=translategemma, parent=span)]
            [_ async for _ in tr._llm_translation_stream(object(), "gemma", "instruction", *args, provider="vllm", served=vllm, parent=span)]
        await tr._translategemma_unary(descriptor, "prompt", *args, served=translategemma)
        await tr._llm_translation_unary(object(), "gemma", "instruction", *args, provider="openai")
        await tr.pretranslate_with_tier(tier, text="મારી ગાયને કેટલું પાણી?", source_lang="gujarati")
        await suggestions_task.create_suggestions("chat-contract", "gu", execution, None)

    asyncio.run(_go())
    return client.observations


def test_a_translation_call_that_fails_marks_its_generation_an_error(monkeypatch):
    """Under a turn's span and under the current span alike: level ERROR and the reason."""
    client = _Langfuse()
    monkeypatch.setattr(tr, "_get_langfuse", lambda: client)

    async def _down(*_a, **_k):
        raise RuntimeError("translategemma down")
        yield  # pragma: no cover - an async generator

    monkeypatch.setattr(tr, "_raw_translategemma_stream", _down)
    descriptor = SimpleNamespace(model_id="translategemma-27b-base")
    args = ("Give the cow water.", "english", "gu", 0.0, 64)

    async def _go():
        with client.start_as_current_observation(name="post_translation", as_type="span") as span:
            with pytest.raises(RuntimeError):
                [_ async for _ in tr._translategemma_stream(descriptor, "prompt", *args, parent=span)]
        with pytest.raises(RuntimeError):
            [_ async for _ in tr._translategemma_stream(descriptor, "prompt", *args)]

    asyncio.run(_go())
    under_span, current = client.steps("stream_translation")
    assert (under_span.parent, under_span.ended) == ("post_translation", 1)
    assert current.parent is None
    for call in (under_span, current):
        assert (call.sent["level"], call.sent["status_message"]) == ("ERROR", "translategemma down")


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
    # Each translation call names the provider and, from an llm_core target, the tier that
    # served it, as turn.v1 names them; a call made under a turn's post_translation span
    # nests there, one made outside under the current span.
    calls = [o for o in observations if o.sent["name"].endswith("_translation") and o.sent["as_type"] == "generation"]
    assert [(o.sent["name"], o.parent, o.sent["metadata"]) for o in calls] == [
        ("stream_translation", "post_translation", {"amul.served_provider": "translategemma", "amul.served_tier": "translategemma:tg[0]"}),
        ("stream_translation", "post_translation", {"amul.served_provider": "vllm", "amul.served_tier": "vllm:gemma[1]"}),
        ("text_translation", None, {"amul.served_provider": "translategemma", "amul.served_tier": "translategemma:tg[0]"}),
        ("text_translation", None, {"amul.served_provider": "openai"}),
        ("pre_translation", None, {"amul.served_provider": "openai", "amul.served_tier": "openai:gpt[0]"}),
    ]
    assert all(call.ended == 1 for call in calls if call.parent is not None)
