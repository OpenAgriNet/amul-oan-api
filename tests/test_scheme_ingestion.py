import asyncio
import base64
from types import SimpleNamespace

import pytest

import app.services.scheme_ingestion as si


class _FakePixmap:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self, image_type: str) -> bytes:
        assert image_type == "png"
        return self._payload


class _FakePage:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def get_pixmap(self, matrix, alpha: bool) -> _FakePixmap:
        assert alpha is False
        assert matrix is not None
        return _FakePixmap(self._payload)


class _FakeDoc:
    def __init__(self, page_payloads: list[bytes]) -> None:
        self.page_payloads = page_payloads
        self.page_count = len(page_payloads)
        self.closed = False

    def load_page(self, index: int) -> _FakePage:
        return _FakePage(self.page_payloads[index])

    def close(self) -> None:
        self.closed = True


def test_render_pdf_to_base64_images_respects_max_pages(monkeypatch):
    fake_doc = _FakeDoc([b"page1", b"page2", b"page3"])

    def fake_open(stream, filetype):
        assert stream == b"%PDF-sample%"
        assert filetype == "pdf"
        return fake_doc

    fake_fitz = SimpleNamespace(
        open=fake_open,
        Matrix=lambda x, y: (x, y),
    )
    monkeypatch.setattr(si, "_get_pymupdf_module", lambda: fake_fitz)

    images = si.render_pdf_to_base64_images(b"%PDF-sample%", dpi=200, max_pages=2)

    assert images == [
        base64.b64encode(b"page1").decode("ascii"),
        base64.b64encode(b"page2").decode("ascii"),
    ]
    assert fake_doc.closed is True


def test_extract_text_from_pdf_bytes_requires_endpoint(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "")

    with pytest.raises(si.SchemeDependencyError):
        asyncio.run(si.extract_text_from_pdf_bytes(SimpleNamespace(), b"pdf"))


def test_extract_text_from_pdf_bytes_calls_ocr_and_merges_pages(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8010")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b", "img-c"])

    captured_calls = []

    class _FakeResponse:
        def __init__(self, markdown: str, error: bool = False) -> None:
            self.markdown = markdown
            self.error = error

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "pages": [
                    {"markdown": self.markdown, "error": self.error},
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            captured_calls.append({"url": url, "json": json, "timeout": timeout})
            image = json["images"][0]
            if image == "img-a":
                return _FakeResponse("First page text")
            if image == "img-b":
                return _FakeResponse("Second page text")
            return _FakeResponse("Third page text")

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))

    assert len(captured_calls) == 3
    for call in captured_calls:
        assert call["url"] == "http://ocr-host:8010/v1/ocr/pages"
        assert call["timeout"] == 45.0
        assert len(call["json"]["images"]) == 1
        assert call["json"]["prompt_type"] == si.SCHEME_OCR_PROMPT_TYPE
        assert call["json"]["max_output_tokens"] == si.SCHEME_OCR_MAX_OUTPUT_TOKENS
    assert combined == "First page text\n\nSecond page text\n\nThird page text"


def test_build_banas_record_returns_expected_schema(monkeypatch):
    async def fake_fetch_bytes(_client, _url):
        return b"pdf"

    monkeypatch.setattr(si, "fetch_bytes", fake_fetch_bytes)

    async def fake_extract(_client, _pdf_bytes):
        return "OCR text"

    monkeypatch.setattr(si, "extract_text_from_pdf_bytes", fake_extract)

    record = asyncio.run(
        si._build_banas_record(
            client=SimpleNamespace(),
            source=si.BANAS_SOURCE,
            scheme_title="Test Scheme",
            scheme_url="https://example.com/scheme.pdf",
            last_refreshed_at="2026-07-01T00:00:00Z",
        )
    )

    assert record is not None
    assert set(record.keys()) == {
        "union_name",
        "source_url",
        "scheme_title",
        "scheme_url",
        "content",
        "content_type",
        "source_name",
        "last_refreshed_at",
    }
    assert record["content"] == "OCR text"
    assert record["content_type"] == "pdf"


def test_extract_text_from_pdf_bytes_raises_when_all_pages_fail(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8010")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b"])

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"pages": [{"markdown": "", "error": True}]}

    class _FakeClient:
        async def post(self, url, json, timeout):
            return _FakeResponse()

    with pytest.raises(si.SchemeParseError, match="failed for all pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_extract_text_from_pdf_bytes_raises_when_ocr_returns_empty_pages(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8010")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b"])

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"pages": []}

    class _FakeClient:
        async def post(self, url, json, timeout):
            return _FakeResponse()

    with pytest.raises(si.SchemeParseError, match="failed for all pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_extract_text_from_pdf_bytes_raises_when_failed_page_ratio_too_high(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8010")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b", "img-c", "img-d"])

    class _FakeResponse:
        def __init__(self, markdown: str, error: bool = False) -> None:
            self.markdown = markdown
            self.error = error

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"pages": [{"markdown": self.markdown, "error": self.error}]}

    class _FakeClient:
        async def post(self, url, json, timeout):
            image = json["images"][0]
            if image == "img-a":
                return _FakeResponse("only one page survives")
            return _FakeResponse("", error=True)

    with pytest.raises(si.SchemeParseError, match="too many pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_build_banas_record_returns_none_on_parse_error(monkeypatch):
    async def fake_fetch_bytes(_client, _url):
        return b"pdf"

    monkeypatch.setattr(si, "fetch_bytes", fake_fetch_bytes)

    async def fake_extract(_client, _pdf_bytes):
        raise si.SchemeParseError("ocr failure")

    monkeypatch.setattr(si, "extract_text_from_pdf_bytes", fake_extract)

    record = asyncio.run(
        si._build_banas_record(
            client=SimpleNamespace(),
            source=si.BANAS_SOURCE,
            scheme_title="Test Scheme",
            scheme_url="https://example.com/scheme.pdf",
            last_refreshed_at="2026-07-01T00:00:00Z",
        )
    )
    assert record is None


def test_ingest_banas_source_heartbeats_lock_per_pdf(monkeypatch):
    links = [
        {"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"},
        {"scheme_title": "Scheme B", "scheme_url": "https://example.com/b.pdf"},
    ]

    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(si, "parse_banas_scheme_links", lambda _html: links)

    async def fake_build(**kwargs):
        return {"scheme_title": kwargs["scheme_title"]}

    monkeypatch.setattr(si, "_build_banas_record", fake_build)

    extend_calls = []

    async def fake_extend(source_key, lock_token, redis_client=None):
        extend_calls.append((source_key, lock_token, redis_client))
        return True

    monkeypatch.setattr(si, "extend_refresh_lock", fake_extend)

    records = asyncio.run(
        si._ingest_banas_source(
            si.BANAS_SOURCE,
            SimpleNamespace(),
            lock_token="tok-123",
            redis_client="redis-stub",
        )
    )

    assert len(records) == 2
    # One heartbeat per processed PDF, always with our own token.
    assert len(extend_calls) == 2
    assert all(call[1] == "tok-123" and call[2] == "redis-stub" for call in extend_calls)


def test_ingest_banas_source_skips_heartbeat_without_token(monkeypatch):
    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(
        si,
        "parse_banas_scheme_links",
        lambda _html: [{"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"}],
    )

    async def fake_build(**kwargs):
        return {"scheme_title": kwargs["scheme_title"]}

    monkeypatch.setattr(si, "_build_banas_record", fake_build)

    called = []

    async def fake_extend(*args, **kwargs):
        called.append(args)
        return True

    monkeypatch.setattr(si, "extend_refresh_lock", fake_extend)

    asyncio.run(si._ingest_banas_source(si.BANAS_SOURCE, SimpleNamespace()))

    assert called == []


def test_ingest_banas_source_raises_when_batch_coverage_too_low(monkeypatch):
    links = [
        {"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"},
        {"scheme_title": "Scheme B", "scheme_url": "https://example.com/b.pdf"},
        {"scheme_title": "Scheme C", "scheme_url": "https://example.com/c.pdf"},
        {"scheme_title": "Scheme D", "scheme_url": "https://example.com/d.pdf"},
        {"scheme_title": "Scheme E", "scheme_url": "https://example.com/e.pdf"},
    ]

    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(si, "parse_banas_scheme_links", lambda _html: links)

    async def fake_build(**kwargs):
        if kwargs["scheme_title"] == "Scheme A":
            return {"scheme_title": kwargs["scheme_title"]}
        return None

    monkeypatch.setattr(si, "_build_banas_record", fake_build)

    with pytest.raises(si.SchemeParseError, match="insufficient Banas ingestion coverage"):
        asyncio.run(si._ingest_banas_source(si.BANAS_SOURCE, SimpleNamespace()))
