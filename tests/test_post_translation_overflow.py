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

This module stubs ``agents.tools`` before importing ``app.services.translation``
because the locally-installed pydantic-ai (0.2.4, vs the repo-pinned 1.50) fails to
build ``agents/tools/__init__``'s Tool schemas — a pre-existing env mismatch unrelated
to translation. The stub provides only the four glossary symbols translation imports.
"""
import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test-key")

# ── stub agents.tools.terms (see module docstring) ────────────────────────────
_fake_terms = types.ModuleType("agents.tools.terms")
_fake_terms.get_mini_glossary_for_text = lambda *a, **k: ""
_fake_terms.get_ambiguity_hints_for_query = lambda *a, **k: ""
_fake_terms.TERM_PAIRS = []


class TermPair:  # minimal stand-in
    def __init__(self, **kw):
        self.__dict__.update(kw)


_fake_terms.TermPair = TermPair
_fake_agents = types.ModuleType("agents"); _fake_agents.__path__ = []
_fake_agents_tools = types.ModuleType("agents.tools"); _fake_agents_tools.__path__ = []
sys.modules.setdefault("agents", _fake_agents)
sys.modules.setdefault("agents.tools", _fake_agents_tools)
sys.modules.setdefault("agents.tools.terms", _fake_terms)

import pytest

from app.llm_core import runtime
from app.llm_core.config_model import Step
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


class _FakeTier:
    """Duck-types MaterializedTier for the walkers + with_first_token_deadline."""

    def __init__(self, provider, *, kind, endpoint, handle, timeout=None, model_name="m"):
        self.provider = provider
        self.kind = kind
        self.endpoint = endpoint
        self.handle = handle
        self.timeout = timeout
        self.model_name = model_name


# ══════════════════════════════════════════════════════════════════════════════
# 1. Guards short-circuit WITHOUT a model call
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_stream_untranslatable_yields_verbatim_no_model_call(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("resolve_chain must not be called for a guarded input")

    monkeypatch.setattr(tr._llm_resolver, "resolve_chain", _boom)
    chunks = [c async for c in tr.translate_text_stream_fast("**", "english", "gujarati")]
    assert chunks == ["**"]


@pytest.mark.asyncio
async def test_stream_same_lang_yields_verbatim_no_model_call(monkeypatch):
    monkeypatch.setattr(
        tr._llm_resolver, "resolve_chain",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no chain for same-lang")),
    )
    chunks = [c async for c in tr.translate_text_stream_fast("hi", "english", "english")]
    assert chunks == ["hi"]


@pytest.mark.asyncio
async def test_stream_empty_returns_nothing(monkeypatch):
    monkeypatch.setattr(
        tr._llm_resolver, "resolve_chain",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no chain for empty")),
    )
    chunks = [c async for c in tr.translate_text_stream_fast("   ", "english", "gujarati")]
    assert chunks == []


@pytest.mark.asyncio
async def test_unary_guards_short_circuit(monkeypatch):
    monkeypatch.setattr(
        tr._llm_resolver, "resolve_chain",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no chain for guard")),
    )
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
# 4. Chain walk order + first-chunk commit
# ══════════════════════════════════════════════════════════════════════════════
def _two_tier_chain():
    return [
        _FakeTier("translategemma", kind="managed", endpoint="tg", handle=_FakeTGDescriptor()),
        _FakeTier("openai", kind="managed", endpoint="llm", handle=object()),
    ]


async def _agen(*items, raise_before=None, raise_after=None):
    if raise_before is not None:
        raise raise_before
    for i, it in enumerate(items):
        yield it
        if raise_after is not None and i == len(items) - 1:
            raise raise_after


@pytest.mark.asyncio
async def test_stream_walk_tg_serves_first_llm_never_called():
    calls = []

    def make_stream(tier):
        calls.append(tier.endpoint)
        if tier.endpoint == "tg":
            return _agen("a", "b")
        raise AssertionError("LLM overflow must not be reached when TG succeeds")

    out = [c async for c in tr._stream_post_translation_chain(
        _two_tier_chain(), make_stream, source_lang="english", target_lang="gujarati")]
    assert out == ["a", "b"]
    assert calls == ["tg"]


@pytest.mark.asyncio
async def test_stream_walk_tg_fails_pre_commit_llm_serves():
    calls = []

    def make_stream(tier):
        calls.append(tier.endpoint)
        if tier.endpoint == "tg":
            return _agen(raise_before=ConnectionError("connect refused"))
        return _agen("x", "y")

    out = [c async for c in tr._stream_post_translation_chain(
        _two_tier_chain(), make_stream, source_lang="english", target_lang="gujarati")]
    assert out == ["x", "y"]
    assert calls == ["tg", "llm"]  # order preserved, TG then LLM


@pytest.mark.asyncio
async def test_stream_walk_tg_fails_post_commit_propagates_no_llm():
    calls = []

    def make_stream(tier):
        calls.append(tier.endpoint)
        if tier.endpoint == "tg":
            return _agen("a", raise_after=ConnectionError("mid-stream reset"))
        raise AssertionError("post-commit failure must NOT swap tiers")

    got = []
    with pytest.raises(ConnectionError):
        async for c in tr._stream_post_translation_chain(
            _two_tier_chain(), make_stream, source_lang="english", target_lang="gujarati"):
            got.append(c)
    assert got == ["a"]        # the committed chunk reached the caller
    assert calls == ["tg"]     # LLM never reached


# ══════════════════════════════════════════════════════════════════════════════
# 5. classify-based fallback: fires on infra errors, NOT on BAD_OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
class _ValidationError(Exception):
    """Type name contains 'Validation' -> classify -> BAD_OUTPUT (not fallbackable)."""


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    TimeoutError("ttft exceeded"),
    ConnectionError("connect refused"),
])
async def test_stream_fallbackable_infra_errors_swap_to_llm(exc):
    def make_stream(tier):
        if tier.endpoint == "tg":
            return _agen(raise_before=exc)
        return _agen("served-by-llm")

    out = [c async for c in tr._stream_post_translation_chain(
        _two_tier_chain(), make_stream, source_lang="english", target_lang="gujarati")]
    assert out == ["served-by-llm"]


@pytest.mark.asyncio
async def test_stream_bad_output_does_not_swap():
    calls = []

    def make_stream(tier):
        calls.append(tier.endpoint)
        if tier.endpoint == "tg":
            return _agen(raise_before=_ValidationError("schema exhausted"))
        raise AssertionError("BAD_OUTPUT must not trigger the overflow tier")

    with pytest.raises(_ValidationError):
        async for _ in tr._stream_post_translation_chain(
            _two_tier_chain(), make_stream, source_lang="english", target_lang="gujarati"):
            pass
    assert calls == ["tg"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. Non-stream translate_text falls TG -> LLM
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_unary_walk_falls_tg_to_llm():
    calls = []

    async def run(tier):
        calls.append(tier.endpoint)
        if tier.endpoint == "tg":
            raise ConnectionError("tg down")
        return "llm-result"

    result = await tr._run_post_translation_chain(_two_tier_chain(), run)
    assert result == "llm-result"
    assert calls == ["tg", "llm"]


@pytest.mark.asyncio
async def test_unary_walk_tg_success_short_circuits():
    calls = []

    async def run(tier):
        calls.append(tier.endpoint)
        return "tg-result"

    result = await tr._run_post_translation_chain(_two_tier_chain(), run)
    assert result == "tg-result"
    assert calls == ["tg"]


@pytest.mark.asyncio
async def test_unary_walk_bad_output_propagates():
    calls = []

    async def run(tier):
        calls.append(tier.endpoint)
        raise _ValidationError("schema exhausted")

    with pytest.raises(_ValidationError):
        await tr._run_post_translation_chain(_two_tier_chain(), run)
    assert calls == ["tg"]  # not fallbackable -> no LLM attempt


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
    runtime.configure(run_self_check=False)
    # The resolved POST_TRANSLATION chain is [TranslateGemma, managed-LLM].
    chain = tr._llm_resolver.resolve_chain(Step.POST_TRANSLATION, "legacy")
    assert [t.provider for t in chain] == ["translategemma", "openai"]
    assert isinstance(chain[0].handle, tr._TGDescriptor)
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


# ══════════════════════════════════════════════════════════════════════════════
# 8. Review fixes: faithful HTTP status classify + LAZY per-tier handle build
# ══════════════════════════════════════════════════════════════════════════════
from app.services.fallback import classify, FallbackReason
from app.llm_core.config_model import Provider, Tier, ApiStyle


def test_tg_http_error_carries_status_for_classify():
    """A TG non-200 must classify by real status, not collapse to UNKNOWN."""
    assert classify(tr._TranslationHTTPError(503, "upstream down")) is FallbackReason.HTTP_5XX
    assert classify(tr._TranslationHTTPError(429, "slow down")) is FallbackReason.RATE_LIMITED
    assert classify(tr._TranslationHTTPError(500, "CUDA out of memory")) is FallbackReason.OOM


def _tg_inert():
    return Tier(provider=Provider.TRANSLATEGEMMA, model="tg", endpoint="http://lb/v1",
                api_style=ApiStyle.TEXT_COMPLETION, timeout_ms=60000, label="translategemma")


def _unbuildable_llm_inert():
    # openai provider, no api_key_env -> AsyncOpenAI() raises when OPENAI_API_KEY
    # is absent, i.e. an overflow tier that cannot be built in this env.
    return Tier(provider=Provider.OPENAI, model="gpt-x", endpoint=None,
                api_style=ApiStyle.CHAT, timeout_ms=30000, label="llm-fallback")


def test_lazy_tier_metadata_available_without_building_handle():
    """The walker's metadata is readable without constructing any client."""
    llm = tr._PostTranslationTier(_unbuildable_llm_inert())
    assert (llm.kind, llm.provider, llm.model_name, llm.timeout) == ("managed", "openai", "gpt-x", 30.0)
    assert llm._memo == []  # reading metadata never triggered a build


def test_lazy_overflow_build_is_isolated_from_healthy_primary(monkeypatch):
    """Constructing the chain builds NO handles; a broken overflow tier can't take
    the primary down — TG builds fine, and the overflow only errors if its handle
    is actually accessed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tg = tr._PostTranslationTier(_tg_inert())
    llm = tr._PostTranslationTier(_unbuildable_llm_inert())
    assert tg._memo == [] and llm._memo == []           # nothing built yet
    assert isinstance(tg.handle, tr._TGDescriptor)       # primary builds independently
    with pytest.raises(Exception):                       # overflow only raises when reached
        llm.handle


@pytest.mark.asyncio
async def test_walk_tg_serves_never_builds_overflow(monkeypatch):
    """The blocker, proven end-to-end: when TG serves, the overflow client is
    never constructed — even if it is unbuildable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tg = tr._PostTranslationTier(_tg_inert())
    llm = tr._PostTranslationTier(_unbuildable_llm_inert())

    def make_stream(tier):
        assert tier is tg  # walker only reaches the serving tier
        return _agen("ok")

    out = [c async for c in tr._stream_post_translation_chain(
        [tg, llm], make_stream, source_lang="english", target_lang="gujarati")]
    assert out == ["ok"]
    assert llm._memo == []  # overflow handle never built
