import asyncio
import base64
from types import SimpleNamespace

import httpx
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


def test_chandra_chat_completions_payload_shape():
    payload = si._chandra_chat_completions_payload("img-a")
    assert payload["model"] == si.SCHEME_OCR_MODEL_NAME
    assert payload["temperature"] == 0
    assert payload["top_p"] == 0.1
    assert payload["max_tokens"] == si.SCHEME_OCR_MAX_OUTPUT_TOKENS
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == si.SCHEME_OCR_LAYOUT_PROMPT
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,img-a"


def test_page_result_from_chat_completion_strips_html():
    parsed = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '<div data-label="Text"><p>First   page</p></div>',
                },
            }
        ]
    }
    result = si._page_result_from_chat_completion(parsed)
    assert result == {"markdown": "First page", "error": False}


def test_page_result_from_chat_completion_keeps_plain_text():
    parsed = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "Plain OCR text"},
            }
        ]
    }
    result = si._page_result_from_chat_completion(parsed)
    assert result == {"markdown": "Plain OCR text", "error": False}


def test_page_result_from_chat_completion_empty_is_error():
    parsed = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "   "},
            }
        ]
    }
    result = si._page_result_from_chat_completion(parsed)
    assert result == {"markdown": "", "error": True}


def test_page_result_from_chat_completion_length_truncation_is_error():
    parsed = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "<p>Partial page text</p>"},
            }
        ]
    }
    result = si._page_result_from_chat_completion(parsed)
    assert result == {"markdown": "Partial page text", "error": True}


def test_normalize_ocr_endpoint_strips_trailing_v1():
    assert si._normalize_ocr_endpoint("http://ocr-host:8011/v1/") == "http://ocr-host:8011"
    assert si._normalize_ocr_endpoint("http://ocr-host:8011") == "http://ocr-host:8011"


def test_page_result_from_chat_completion_multimodal_content_list():
    parsed = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {"type": "text", "text": "<p>Part A</p>"},
                        {"type": "text", "text": "<p>Part B</p>"},
                    ]
                },
            }
        ]
    }
    result = si._page_result_from_chat_completion(parsed)
    assert result == {"markdown": "Part A Part B", "error": False}


def test_page_result_from_chat_completion_chandra_layout_fixture():
    parsed = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": (
                        '<div data-bbox="0 0 100 40" data-label="Section-Header">'
                        "<p>Animal Cooling Scheme</p></div>"
                        '<div data-bbox="0 50 100 200" data-label="Text">'
                        "<p>Subsidy up to 50% for cattle cooling systems.</p></div>"
                        '<div data-bbox="0 210 100 300" data-label="Table">'
                        "<table><tr><td>Item</td><td>Amount</td></tr>"
                        "<tr><td>Cooler</td><td>10000</td></tr></table></div>"
                    ),
                },
            }
        ]
    }
    result = si._page_result_from_chat_completion(parsed)
    assert result is not None
    assert result["error"] is False
    assert "Animal Cooling Scheme" in result["markdown"]
    assert "Subsidy up to 50%" in result["markdown"]
    assert "Cooler" in result["markdown"]
    assert "10000" in result["markdown"]
    assert "<" not in result["markdown"]


def test_extract_text_from_pdf_bytes_accepts_endpoint_with_v1_suffix(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011/v1")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_ocr_concurrency", 2)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a"])

    captured_urls = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "ok"},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            captured_urls.append(url)
            return _FakeResponse()

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))
    assert combined == "ok"
    assert captured_urls == ["http://ocr-host:8011/v1/chat/completions"]


def test_extract_text_from_pdf_bytes_calls_ocr_and_merges_pages(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b", "img-c"])

    captured_calls = []
    responses = {
        "img-a": "First page text",
        "img-b": "Second page text",
        "img-c": "Third page text",
    }

    class _FakeResponse:
        def __init__(self, markdown: str) -> None:
            self.markdown = markdown

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.markdown},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            captured_calls.append({"url": url, "json": json, "timeout": timeout})
            image_url = json["messages"][0]["content"][1]["image_url"]["url"]
            image_b64 = image_url.split(",", 1)[1]
            return _FakeResponse(responses[image_b64])

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))

    assert len(captured_calls) == 3
    for call in captured_calls:
        assert call["url"] == "http://ocr-host:8011/v1/chat/completions"
        assert call["timeout"] == 45.0
        assert call["json"]["model"] == si.SCHEME_OCR_MODEL_NAME
        assert call["json"]["temperature"] == 0
        assert call["json"]["top_p"] == 0.1
        assert call["json"]["max_tokens"] == si.SCHEME_OCR_MAX_OUTPUT_TOKENS
        assert call["json"]["messages"][0]["content"][0]["text"] == si.SCHEME_OCR_LAYOUT_PROMPT
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
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b"])

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": ""},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            return _FakeResponse()

    with pytest.raises(si.SchemeParseError, match="failed for all pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_extract_text_from_pdf_bytes_raises_when_ocr_returns_empty_choices(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b"])

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": []}

    class _FakeClient:
        async def post(self, url, json, timeout):
            return _FakeResponse()

    with pytest.raises(si.SchemeParseError, match="failed for all pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_extract_text_from_pdf_bytes_raises_when_failed_page_ratio_too_high(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b", "img-c", "img-d"])

    class _FakeResponse:
        def __init__(self, markdown: str) -> None:
            self.markdown = markdown

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.markdown},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            image_url = json["messages"][0]["content"][1]["image_url"]["url"]
            image_b64 = image_url.split(",", 1)[1]
            if image_b64 == "img-a":
                return _FakeResponse("only one page survives")
            return _FakeResponse("")

    with pytest.raises(si.SchemeParseError, match="too many pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_extract_text_from_pdf_bytes_raises_when_ocr_request_fails(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b"])

    class _FakeClient:
        async def post(self, url, json, timeout):
            raise httpx.ConnectError("connection refused")

    with pytest.raises(si.SchemeParseError, match="failed for all pages"):
        asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))


def test_extract_text_from_pdf_bytes_posts_one_image_per_request(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_ocr_concurrency", 4)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(
        si,
        "render_pdf_to_base64_images",
        lambda *_args, **_kwargs: ["img-a", "img-b", "img-c", "img-d", "img-e"],
    )

    captured_calls = []

    class _FakeResponse:
        def __init__(self, markdown: str) -> None:
            self.markdown = markdown

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.markdown},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            image_url = json["messages"][0]["content"][1]["image_url"]["url"]
            image_b64 = image_url.split(",", 1)[1]
            captured_calls.append({"image": image_b64, "timeout": timeout})
            return _FakeResponse(f"text-{image_b64}")

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))

    assert sorted(call["image"] for call in captured_calls) == ["img-a", "img-b", "img-c", "img-d", "img-e"]
    assert all(call["timeout"] == 45.0 for call in captured_calls)
    assert combined == "text-img-a\n\ntext-img-b\n\ntext-img-c\n\ntext-img-d\n\ntext-img-e"


def test_extract_text_from_pdf_bytes_limits_concurrency(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_ocr_concurrency", 2)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(
        si,
        "render_pdf_to_base64_images",
        lambda *_args, **_kwargs: ["img-a", "img-b", "img-c", "img-d"],
    )

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    class _FakeResponse:
        def __init__(self, markdown: str) -> None:
            self.markdown = markdown

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.markdown},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            image_url = json["messages"][0]["content"][1]["image_url"]["url"]
            image_b64 = image_url.split(",", 1)[1]
            async with lock:
                in_flight -= 1
            return _FakeResponse(f"text-{image_b64}")

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))

    assert max_in_flight <= 2
    assert max_in_flight >= 2  # with 4 pages and concurrency=2, we should hit the cap
    assert combined == "text-img-a\n\ntext-img-b\n\ntext-img-c\n\ntext-img-d"


def test_extract_text_from_pdf_bytes_preserves_order_under_concurrency(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_ocr_concurrency", 4)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b", "img-c"])

    delays = {"img-a": 0.08, "img-b": 0.01, "img-c": 0.04}

    class _FakeResponse:
        def __init__(self, markdown: str) -> None:
            self.markdown = markdown

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.markdown},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            image_url = json["messages"][0]["content"][1]["image_url"]["url"]
            image_b64 = image_url.split(",", 1)[1]
            await asyncio.sleep(delays[image_b64])
            return _FakeResponse(f"text-{image_b64}")

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))
    assert combined == "text-img-a\n\ntext-img-b\n\ntext-img-c"


def test_extract_text_from_pdf_bytes_partial_failure_under_concurrency(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_ocr_concurrency", 3)
    monkeypatch.setattr(si.settings, "scheme_ocr_max_failed_page_ratio", 0.5)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a", "img-b", "img-c"])
    # Module constant is captured at import; patch the name used by extract_text thresholds.
    monkeypatch.setattr(si, "SCHEME_OCR_MAX_FAILED_PAGE_RATIO", 0.5)

    class _FakeResponse:
        def __init__(self, markdown: str) -> None:
            self.markdown = markdown

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.markdown},
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            image_url = json["messages"][0]["content"][1]["image_url"]["url"]
            image_b64 = image_url.split(",", 1)[1]
            if image_b64 == "img-b":
                raise httpx.ConnectError("connection refused")
            return _FakeResponse(f"text-{image_b64}")

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))
    assert combined == "text-img-a\n\ntext-img-c"


def test_extract_text_from_pdf_bytes_maps_html_page_content(monkeypatch):
    monkeypatch.setattr(si.settings, "scheme_ocr_endpoint_url", "http://ocr-host:8011")
    monkeypatch.setattr(si.settings, "scheme_ocr_timeout_seconds", 45.0)
    monkeypatch.setattr(si.settings, "scheme_pdf_render_dpi", 150)
    monkeypatch.setattr(si, "render_pdf_to_base64_images", lambda *_args, **_kwargs: ["img-a"])

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '<div data-label="Section-Header"><p>Scheme Title</p></div>'
                                '<div data-label="Text"><p>Eligibility details</p></div>'
                            ),
                        },
                    }
                ]
            }

    class _FakeClient:
        async def post(self, url, json, timeout):
            return _FakeResponse()

    combined = asyncio.run(si.extract_text_from_pdf_bytes(_FakeClient(), b"pdf-bytes"))
    assert combined == "Scheme Title Eligibility details"


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


def test_parse_banas_scheme_links_keeps_published_scheme_pdfs_only():
    payload = [
        {
            "section": "vendor_registration",
            "title": "Web Portal Vendor User Guide",
            "status": "published",
            "sort_order": 0,
            "file": {"file_path": "documents/b2b/vendor-web-portal-user-guide.pdf"},
        },
        {
            "section": "schemes",
            "title": "MILKING MACHINE ASSISTANCE SCHEME",
            "status": "draft",
            "sort_order": 1,
            "file": {"file_path": "documents/schemes/milking-machine.pdf"},
        },
        {
            "section": "schemes",
            "title": "IRON STALL ASSISTANCE SCHEME",
            "status": "published",
            "sort_order": 10,
            "file": {"file_path": "documents/schemes/iron-stall.pdf"},
        },
        {
            "section": "schemes",
            "title": "ANIMAL COOLING SYSTEM ASSISTANCE SCHEME",
            "status": "published",
            "sort_order": 1,
            "file": {"file_path": "documents/schemes/animal-cooling.pdf"},
        },
        {
            "section": "schemes",
            "title": "MISSING FILE SCHEME",
            "status": "published",
            "sort_order": 2,
            "file": {},
        },
        {
            "section": "schemes",
            "title": "ANIMAL COOLING SYSTEM ASSISTANCE SCHEME",
            "status": "published",
            "sort_order": 99,
            "file": {"file_path": "documents/schemes/animal-cooling.pdf"},
        },
    ]

    records = si.parse_banas_scheme_links(payload)

    assert records == [
        {
            "scheme_title": "ANIMAL COOLING SYSTEM ASSISTANCE SCHEME",
            "scheme_url": "https://www.banasdairy.coop/media/documents/schemes/animal-cooling.pdf",
        },
        {
            "scheme_title": "IRON STALL ASSISTANCE SCHEME",
            "scheme_url": "https://www.banasdairy.coop/media/documents/schemes/iron-stall.pdf",
        },
    ]


def test_parse_banas_scheme_links_accepts_wrapped_document_list():
    records = si.parse_banas_scheme_links(
        {
            "data": [
                {
                    "section": "schemes",
                    "title": "PAKI ASSISTANCE SCHEME",
                    "file": {"file_path": "/media/documents/schemes/paki-assistance.pdf"},
                }
            ]
        }
    )

    assert records == [
        {
            "scheme_title": "PAKI ASSISTANCE SCHEME",
            "scheme_url": "https://www.banasdairy.coop/media/documents/schemes/paki-assistance.pdf",
        }
    ]


def test_parse_banas_scheme_links_returns_empty_for_invalid_payload():
    assert si.parse_banas_scheme_links(None) == []
    assert si.parse_banas_scheme_links("not-json-list") == []
    assert si.parse_banas_scheme_links({"ok": True}) == []


def test_ingest_banas_source_heartbeats_lock_per_pdf(monkeypatch):
    links = [
        {"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"},
        {"scheme_title": "Scheme B", "scheme_url": "https://example.com/b.pdf"},
    ]

    async def fake_fetch_json(_client, _url):
        return []

    monkeypatch.setattr(si, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(si, "parse_banas_scheme_links", lambda _payload: links)

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
    async def fake_fetch_json(_client, _url):
        return []

    monkeypatch.setattr(si, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        si,
        "parse_banas_scheme_links",
        lambda _payload: [{"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"}],
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

    async def fake_fetch_json(_client, _url):
        return []

    monkeypatch.setattr(si, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(si, "parse_banas_scheme_links", lambda _payload: links)

    async def fake_build(**kwargs):
        if kwargs["scheme_title"] == "Scheme A":
            return {"scheme_title": kwargs["scheme_title"]}
        return None

    monkeypatch.setattr(si, "_build_banas_record", fake_build)

    with pytest.raises(si.SchemeParseError, match="insufficient Banas ingestion coverage"):
        asyncio.run(si._ingest_banas_source(si.BANAS_SOURCE, SimpleNamespace()))


# ---------------------------------------------------------------------------
# Sumul parser tests
# ---------------------------------------------------------------------------

_SUMUL_HTML_SINGLE = """
<div class="sumul-farmer-short__item is-open">
  <span class="sumul-farmer-short__titles">
    <strong>પશુ સારવાર અને કૃત્રિમ બીજદાન બાબત</strong>
    <small>Animal Treatment</small>
  </span>
  <a href="images/pdf/pasusrvar-paripatra.pdf" class="sim-button" target="_blank">Download</a>
</div></div></div>
"""

_SUMUL_HTML_MULTI = """
<div class="sumul-farmer-short__item is-open">
  <span class="sumul-farmer-short__titles">
    <strong>Scheme A</strong>
  </span>
  <a href="images/pdf/a.pdf">Download</a>
</div></div></div>
<div class="sumul-farmer-short__item">
  <span class="sumul-farmer-short__titles">
    <strong>Scheme B</strong>
  </span>
  <a href="images/pdf/b.pdf">Download</a>
</div></div></div>
"""


def test_parse_sumul_scheme_links_single_item():
    records = si.parse_sumul_scheme_links(_SUMUL_HTML_SINGLE)
    assert len(records) == 1
    assert records[0]["scheme_title"] == "પશુ સારવાર અને કૃત્રિમ બીજદાન બાબત"
    assert records[0]["scheme_url"] == "https://www.sumul.com/images/pdf/pasusrvar-paripatra.pdf"


def test_parse_sumul_scheme_links_multiple_items():
    records = si.parse_sumul_scheme_links(_SUMUL_HTML_MULTI)
    assert len(records) == 2
    assert records[0]["scheme_title"] == "Scheme A"
    assert records[1]["scheme_title"] == "Scheme B"
    assert records[0]["scheme_url"] == "https://www.sumul.com/images/pdf/a.pdf"
    assert records[1]["scheme_url"] == "https://www.sumul.com/images/pdf/b.pdf"


def test_parse_sumul_scheme_links_deduplication():
    html = """
    <div class="sumul-farmer-short__item">
      <span class="sumul-farmer-short__titles"><strong>Dup</strong></span>
      <a href="images/pdf/dup.pdf">Download</a>
      <a href="images/pdf/dup.pdf">Download Again</a>
    </div></div></div>
    """
    records = si.parse_sumul_scheme_links(html)
    assert len(records) == 1


def test_parse_sumul_scheme_links_fallback_no_accordion():
    html = '<a href="images/pdf/fallback.pdf">Download</a>'
    records = si.parse_sumul_scheme_links(html)
    assert len(records) == 1
    assert records[0]["scheme_url"] == "https://www.sumul.com/images/pdf/fallback.pdf"
    assert records[0]["scheme_title"] == "fallback.pdf"


# ---------------------------------------------------------------------------
# Sursagar parser tests
# ---------------------------------------------------------------------------

_SURSAGAR_HTML = """
<h6 class="fw-bold producer-title" title="યોજના A">યોજના A</h6>
<a href="/Farmer/DownloadMilkProducerFile?file=aaa.pdf" class="btn">view</a>
<h6 class="fw-bold producer-title" title="યોજના B">યોજના B</h6>
<a href="/Farmer/DownloadMilkProducerFile?file=bbb.pdf" class="btn">view</a>
"""


def test_parse_sursagar_scheme_links():
    records = si.parse_sursagar_scheme_links(_SURSAGAR_HTML)
    assert len(records) == 2
    assert records[0]["scheme_title"] == "યોજના A"
    assert records[0]["scheme_url"] == "https://sursagardairy.com/Farmer/DownloadMilkProducerFile?file=aaa.pdf"
    assert records[1]["scheme_title"] == "યોજના B"
    assert records[1]["scheme_url"] == "https://sursagardairy.com/Farmer/DownloadMilkProducerFile?file=bbb.pdf"


def test_parse_sursagar_scheme_links_deduplication():
    html = """
    <h6 class="fw-bold producer-title" title="Scheme X">Scheme X</h6>
    <a href="/Farmer/DownloadMilkProducerFile?file=x.pdf" class="btn">view</a>
    <a href="/Farmer/DownloadMilkProducerFile?file=x.pdf" class="btn">download</a>
    """
    records = si.parse_sursagar_scheme_links(html)
    assert len(records) == 1


def test_parse_sursagar_scheme_links_fallback_no_cards():
    html = '<a href="/Farmer/DownloadMilkProducerFile?file=orphan.pdf" class="btn">view</a>'
    records = si.parse_sursagar_scheme_links(html)
    assert len(records) == 1
    assert records[0]["scheme_url"] == "https://sursagardairy.com/Farmer/DownloadMilkProducerFile?file=orphan.pdf"
    assert records[0]["scheme_title"] == "orphan.pdf"


# ---------------------------------------------------------------------------
# Sumul ingestion tests
# ---------------------------------------------------------------------------

def test_ingest_sumul_source_heartbeats_lock(monkeypatch):
    links = [
        {"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"},
        {"scheme_title": "Scheme B", "scheme_url": "https://example.com/b.pdf"},
    ]

    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(si, "parse_sumul_scheme_links", lambda _html: links)

    async def fake_build(**kwargs):
        return {"scheme_title": kwargs["scheme_title"]}

    monkeypatch.setattr(si, "_build_pdf_record", fake_build)

    extend_calls = []

    async def fake_extend(source_key, lock_token, redis_client=None):
        extend_calls.append((source_key, lock_token, redis_client))
        return True

    monkeypatch.setattr(si, "extend_refresh_lock", fake_extend)

    records = asyncio.run(
        si._ingest_sumul_source(
            si.SUMUL_SOURCE,
            SimpleNamespace(),
            lock_token="tok-sumul",
            redis_client="redis-stub",
        )
    )

    assert len(records) == 2
    assert len(extend_calls) == 2
    assert all(call[1] == "tok-sumul" and call[2] == "redis-stub" for call in extend_calls)


def test_ingest_sumul_source_skips_heartbeat_without_token(monkeypatch):
    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(
        si,
        "parse_sumul_scheme_links",
        lambda _html: [{"scheme_title": "A", "scheme_url": "https://example.com/a.pdf"}],
    )

    async def fake_build(**kwargs):
        return {"scheme_title": kwargs["scheme_title"]}

    monkeypatch.setattr(si, "_build_pdf_record", fake_build)

    called = []

    async def fake_extend(*args, **kwargs):
        called.append(args)
        return True

    monkeypatch.setattr(si, "extend_refresh_lock", fake_extend)

    asyncio.run(si._ingest_sumul_source(si.SUMUL_SOURCE, SimpleNamespace()))

    assert called == []


def test_ingest_sumul_source_raises_when_coverage_too_low(monkeypatch):
    links = [
        {"scheme_title": f"Scheme {c}", "scheme_url": f"https://example.com/{c}.pdf"}
        for c in "ABCDE"
    ]

    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(si, "parse_sumul_scheme_links", lambda _html: links)

    async def fake_build(**kwargs):
        if kwargs["scheme_title"] == "Scheme A":
            return {"scheme_title": kwargs["scheme_title"]}
        return None

    monkeypatch.setattr(si, "_build_pdf_record", fake_build)

    with pytest.raises(si.SchemeParseError, match="insufficient sumul ingestion coverage"):
        asyncio.run(si._ingest_sumul_source(si.SUMUL_SOURCE, SimpleNamespace()))


# ---------------------------------------------------------------------------
# Sursagar ingestion tests
# ---------------------------------------------------------------------------

def test_ingest_sursagar_source_heartbeats_lock(monkeypatch):
    links = [
        {"scheme_title": "Scheme A", "scheme_url": "https://example.com/a.pdf"},
        {"scheme_title": "Scheme B", "scheme_url": "https://example.com/b.pdf"},
    ]

    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(si, "parse_sursagar_scheme_links", lambda _html: links)

    async def fake_build(**kwargs):
        return {"scheme_title": kwargs["scheme_title"]}

    monkeypatch.setattr(si, "_build_pdf_record", fake_build)

    extend_calls = []

    async def fake_extend(source_key, lock_token, redis_client=None):
        extend_calls.append((source_key, lock_token, redis_client))
        return True

    monkeypatch.setattr(si, "extend_refresh_lock", fake_extend)

    records = asyncio.run(
        si._ingest_sursagar_source(
            si.SURSAGAR_SOURCE,
            SimpleNamespace(),
            lock_token="tok-sursagar",
            redis_client="redis-stub",
        )
    )

    assert len(records) == 2
    assert len(extend_calls) == 2
    assert all(call[1] == "tok-sursagar" and call[2] == "redis-stub" for call in extend_calls)


def test_ingest_sursagar_source_raises_when_coverage_too_low(monkeypatch):
    links = [
        {"scheme_title": f"Scheme {c}", "scheme_url": f"https://example.com/{c}.pdf"}
        for c in "ABCDE"
    ]

    async def fake_fetch_html(_client, _url):
        return "<html></html>"

    monkeypatch.setattr(si, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(si, "parse_sursagar_scheme_links", lambda _html: links)

    async def fake_build(**kwargs):
        if kwargs["scheme_title"] == "Scheme A":
            return {"scheme_title": kwargs["scheme_title"]}
        return None

    monkeypatch.setattr(si, "_build_pdf_record", fake_build)

    with pytest.raises(si.SchemeParseError, match="insufficient sursagar ingestion coverage"):
        asyncio.run(si._ingest_sursagar_source(si.SURSAGAR_SOURCE, SimpleNamespace()))
