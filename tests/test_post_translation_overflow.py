"""Post-translation overflow: config-driven [TranslateGemma(LB), managed-LLM] chain.

Covers (no network — the aiohttp SSE + AsyncOpenAI stream are mocked):
  * guards short-circuit WITHOUT any model call (untranslatable / same-lang / empty);
  * the per-chunk transform pipeline (_fix_dandas -> _post_normalize_gu_translation)
    is applied identically on the TranslateGemma tier and the LLM overflow tier;
  * the built instruction carries the glossary Rules + GU style rules + length rule
    (shared verbatim by both tiers);
  * chain walk order = TranslateGemma first, LLM overflow second, with first-chunk
    commit (pre-commit fallbackable failure swaps to LLM; post-commit propagates);
  * classify-based fallback fires on TIMEOUT / CONNECTION / 5xx, NOT on BAD_OUTPUT;
  * translate_text (non-stream) falls TranslateGemma -> LLM.

"""
import asyncio
import os
import types

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.llm_core import runtime
from app.llm_core.config_model import NamedProfile, PipelineConfig, Provider, Step, StepConfig, Tier
from app.llm_core.execution import ExecutionContext
import app.services.translation as tr


# ══════════════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════════════
def _sse(*texts) -> list[bytes]:
    """Encode text-completion deltas as TranslateGemma SSE line chunks + [DONE]."""
    import json as _json
    out = [
        f"data: {_json.dumps({'choices': [{'text': t}]})}\n".encode("utf-8")
        for t in texts
    ]
    out.append(b"data: [DONE]\n")
    return out


class _FakeContent:
    def __init__(self, chunks, raise_after=None):
        self._chunks = chunks
        self._raise_after = raise_after

    async def iter_chunked(self, _n):
        for i, c in enumerate(self._chunks):
            yield c
            if self._raise_after is not None and i == self._raise_after:
                raise ConnectionError("stream died mid-flight")


class _FakeResp:
    def __init__(self, *, status=200, sse=None, body=None, raise_after=None):
        self.status = status
        self.content = _FakeContent(sse or [], raise_after=raise_after)
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return "boom"

    async def json(self):
        return self._body


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, *a, **k):
        return self._resp


def _patch_aiohttp(monkeypatch, resp):
    monkeypatch.setattr(tr.aiohttp, "ClientSession", lambda *a, **k: _FakeSession(resp))


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, *, message=False):
        if message:
            self.message = _FakeDelta(content)
        else:
            self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeAsyncStream:
    def __init__(self, contents):
        self._contents = contents

    def __aiter__(self):
        async def _gen():
            for c in self._contents:
                yield _FakeStreamChunk(c)
        return _gen()


class _FakeCompletions:
    def __init__(self, *, stream_contents=None, message_content=None):
        self._stream_contents = stream_contents
        self._message_content = message_content

    async def create(self, *, stream=False, **kw):
        if stream:
            return _FakeAsyncStream(self._stream_contents or [])
        return types.SimpleNamespace(
            choices=[_FakeChoice(self._message_content, message=True)]
        )


class _FakeOpenAIClient:
    def __init__(self, *, stream_contents=None, message_content=None):
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(
                stream_contents=stream_contents, message_content=message_content
            )
        )


class _FakeTGDescriptor:
    completions_url = "http://lb/v1/completions"
    model_id = "translategemma-27b-base"
    endpoint = "http://lb/v1"


def _post_context(*, ttft_ms=10):
    return ExecutionContext(
        session_id="s1",
        profile_name="managed",
        config=PipelineConfig(
            fallback_enabled=True,
            profiles=[NamedProfile(name="managed", weight=100)],
            defaults={
                Step.POST_TRANSLATION: StepConfig(
                    tiers=[
                        Tier(
                            provider=Provider.TRANSLATEGEMMA,
                            model="tg",
                            endpoint="http://lb/v1",
                            ttft_ms=ttft_ms,
                        ),
                        Tier(provider=Provider.OPENAI, model="gpt"),
                    ]
                )
            },
        ),
    )


@pytest.mark.asyncio
async def test_pretranslation_uses_provider_protocol_and_trace(monkeypatch):
    calls = {}

    class Messages:
        async def create(self, **kwargs):
            calls["request"] = kwargs
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="translated")]
            )

    class Observation:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, **kwargs):
            calls["output"] = kwargs["output"]

    class Langfuse:
        def start_as_current_observation(self, **kwargs):
            calls["observation"] = kwargs
            return Observation()

    monkeypatch.setattr(tr, "_get_langfuse", lambda: Langfuse())
    target = types.SimpleNamespace(
        provider="anthropic",
        model_name="claude-haiku",
        handle=types.SimpleNamespace(messages=Messages()),
    )

    result = await tr.pretranslate_with_tier(
        target, text="નમસ્તે", source_lang="gujarati"
    )
    assert result == "translated"
    assert calls["request"]["model"] == "claude-haiku"
    assert calls["observation"]["name"] == "query_pretranslation"
    assert calls["observation"]["metadata"]["translation_provider"] == "anthropic"
    assert calls["output"] == "translated"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Guards short-circuit WITHOUT a model call
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_stream_untranslatable_yields_verbatim_no_model_call(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("resolve_chain must not be called for a guarded input")

    monkeypatch.setattr(tr.llm_core, "context", _boom)
    chunks = [c async for c in tr.translate_text_stream_fast("**", "english", "gujarati")]
    assert chunks == ["**"]


@pytest.mark.asyncio
async def test_stream_same_lang_yields_verbatim_no_model_call(monkeypatch):
    monkeypatch.setattr(
        tr.llm_core, "context",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no chain for same-lang")),
    )
    chunks = [c async for c in tr.translate_text_stream_fast("hi", "english", "english")]
    assert chunks == ["hi"]


@pytest.mark.asyncio
async def test_stream_empty_returns_nothing(monkeypatch):
    monkeypatch.setattr(
        tr.llm_core, "context",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no chain for empty")),
    )
    chunks = [c async for c in tr.translate_text_stream_fast("   ", "english", "gujarati")]
    assert chunks == []


@pytest.mark.asyncio
async def test_unary_guards_short_circuit(monkeypatch):
    async def _boom(*args, **kwargs):
        raise AssertionError("no model call for guard")

    monkeypatch.setattr(tr.llm_core, "context", _boom)
    assert await tr.translate_text("**", "english", "gujarati") == "**"
    assert await tr.translate_text("hi", "gujarati", "gujarati") == "hi"
    assert await tr.translate_text("", "english", "gujarati") == ""


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-chunk transforms applied identically on TG and LLM tiers
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_tg_stream_applies_per_chunk_transforms(monkeypatch):
    # "।" -> "." (dandas) and "ગર્ભવતી" -> "ગાભણ" (GU post-normalize) per chunk.
    _patch_aiohttp(monkeypatch, _FakeResp(sse=_sse("ગર્ભવતી।", " બીજું")))
    out = [
        c async for c in tr._translategemma_stream(
            _FakeTGDescriptor(), "prompt", "english", "gujarati", "src", 0.0, 2048
        )
    ]
    assert out == ["ગાભણ.", " બીજું"]


@pytest.mark.asyncio
async def test_llm_stream_applies_identical_per_chunk_transforms():
    client = _FakeOpenAIClient(stream_contents=["ગર્ભવતી।", " બીજું"])
    out = [
        c async for c in tr._llm_translation_stream(
            client, "gpt-4.1", "instruction", "english", "gujarati", "src", 0.0, 2048
        )
    ]
    # Byte-identical to the TG tier's transformed output.
    assert out == ["ગાભણ.", " બીજું"]


@pytest.mark.asyncio
async def test_unary_transforms_tg_and_llm_match(monkeypatch):
    _patch_aiohttp(monkeypatch, _FakeResp(body={"choices": [{"text": "ગર્ભવતી।"}]}))
    tg = await tr._translategemma_unary(
        _FakeTGDescriptor(), "prompt", "english", "gujarati", "src", 0.0, 2048
    )
    llm = await tr._llm_translation_unary(
        _FakeOpenAIClient(message_content="ગર્ભવતી।"),
        "gpt-4.1", "instruction", "english", "gujarati", "src", 0.0, 2048,
    )
    assert tg == llm == "ગાભણ."


@pytest.mark.asyncio
async def test_posttranslation_uses_anthropic_and_gemini_protocols(monkeypatch):
    monkeypatch.setattr(tr, "_get_langfuse", lambda: None)

    class AnthropicMessages:
        async def create(self, **_kwargs):
            return types.SimpleNamespace(content=[
                types.SimpleNamespace(type="text", text="ગર્ભવતી।")
            ])

    class GeminiModels:
        async def generate_content(self, **_kwargs):
            return types.SimpleNamespace(text="ગર્ભવતી।")

    anthropic = types.SimpleNamespace(messages=AnthropicMessages())
    gemini = types.SimpleNamespace(models=GeminiModels())
    args = ("model", "instruction", "english", "gujarati", "src", 0.0, 2048)

    assert await tr._llm_translation_unary(anthropic, *args, provider="anthropic") == "ગાભણ."
    assert await tr._llm_translation_unary(gemini, *args, provider="gemini") == "ગાભણ."


# ══════════════════════════════════════════════════════════════════════════════
# 3. Instruction carries glossary + GU rules + length rule (shared by both tiers)
# ══════════════════════════════════════════════════════════════════════════════
def test_instruction_has_glossary_gu_rules_and_length_rule():
    instruction, tg_prompt = tr._prepare_translation_inputs(
        "Keep the animal hydrated.", "english", "gujarati", 1600
    )
    # Same instruction text is fed to the LLM tier and (wrapped) to TranslateGemma.
    assert instruction in tg_prompt
    assert tg_prompt.startswith("<bos><start_of_turn>user")
    assert "farmer-preferred Gujarati livestock terms" in instruction  # GU style rules
    assert "no more than 1600 characters" in instruction               # length rule


def test_instruction_injects_glossary_rules_when_present():
    instruction = tr._build_translation_instruction(
        "Society info", "english", "gujarati",
        mini_glossary="Society -> સોસાયટી",
    )
    assert "'Society' must be translated as 'સોસાયટી'" in instruction


# ══════════════════════════════════════════════════════════════════════════════
# 7. End-to-end through the REAL resolved chain (dispatch + wiring)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def _managed_pipeline(monkeypatch):
    for k in ("OSS_INFERENCE_ENDPOINT_URL", "OSS_PIPELINE_PCT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4.1")
    monkeypatch.setenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://lb/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    runtime.configure()
    yield


@pytest.mark.asyncio
async def test_e2e_tg_serves(monkeypatch, _managed_pipeline):
    _patch_aiohttp(monkeypatch, _FakeResp(sse=_sse("ગર્ભવતી।")))
    out = "".join([
        c async for c in tr.translate_text_stream_fast("hydrate", "english", "gujarati")
    ])
    assert out == "ગાભણ."


@pytest.mark.asyncio
async def test_e2e_tg_fails_pre_commit_llm_serves(monkeypatch, _managed_pipeline):
    # TranslateGemma returns 5xx (raises before any chunk) -> managed LLM overflow serves.
    _patch_aiohttp(monkeypatch, _FakeResp(status=500))

    async def _fake_llm_stream(client, model_name, instruction, *a, **k):
        assert "farmer-preferred Gujarati livestock terms" in instruction  # same rules prompt
        yield "llm-served"

    monkeypatch.setattr(tr, "_llm_translation_stream", _fake_llm_stream)
    out = "".join([
        c async for c in tr.translate_text_stream_fast("hydrate", "english", "gujarati")
    ])
    assert out == "llm-served"


@pytest.mark.asyncio
async def test_healthy_tg_does_not_build_invalid_overflow(monkeypatch):
    from app.llm_core import execution

    built = []

    def build(tier, kind):
        built.append(tier.provider)
        if tier.provider is Provider.TRANSLATEGEMMA:
            return _FakeTGDescriptor()
        raise ValueError("unused overflow is intentionally invalid")

    monkeypatch.setattr(execution, "build_handle", build)

    def unexpected_admission():
        raise AssertionError("TranslateGemma must not use managed admission")

    monkeypatch.setattr(execution, "_get_managed_sem", unexpected_admission)
    _patch_aiohttp(monkeypatch, _FakeResp(body={"choices": [{"text": "ગાભણ।"}]}))

    result = await tr.translate_text(
        "pregnant", "english", "gujarati", execution=_post_context()
    )
    assert result == "ગાભણ."
    assert built == [Provider.TRANSLATEGEMMA]


@pytest.mark.asyncio
async def test_post_translation_common_stream_enforces_ttft(monkeypatch):
    from app.llm_core import execution

    monkeypatch.setattr(execution, "build_handle", lambda tier, kind: object())

    async def silent_tg(*args, **kwargs):
        await asyncio.sleep(0.05)
        yield "late"

    async def managed_overflow(*args, **kwargs):
        yield "overflow"

    monkeypatch.setattr(tr, "_translategemma_stream", silent_tg)
    monkeypatch.setattr(tr, "_llm_translation_stream", managed_overflow)

    result = "".join([
        chunk
        async for chunk in tr.translate_text_stream_fast(
            "hydrate", "english", "gujarati", execution=_post_context(ttft_ms=10)
        )
    ])
    assert result == "overflow"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Faithful HTTP status classification
# ══════════════════════════════════════════════════════════════════════════════
from app.llm_core.execution import classify, FallbackReason


def test_tg_http_error_carries_status_for_classify():
    """A TG non-200 must classify by real status, not collapse to UNKNOWN."""
    assert classify(tr._TranslationHTTPError(503, "upstream down")) is FallbackReason.HTTP_5XX
    assert classify(tr._TranslationHTTPError(429, "slow down")) is FallbackReason.RATE_LIMITED
    assert classify(tr._TranslationHTTPError(500, "CUDA out of memory")) is FallbackReason.OOM
