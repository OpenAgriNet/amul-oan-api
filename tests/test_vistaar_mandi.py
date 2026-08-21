"""Tests for the Bharat Vistaar mandi tool (agents/tools/vistaar.py).

The BV `price-discovery` BPP is a *range* API. With no from_date/to_date it
answers with a single fallback row from whichever market it considers nearest
(Padra APMC), not the farmer's market (Anand APMC). These tests pin down:

  1. the from_date/to_date window resolution (DD-MM-YYYY, to_date never null,
     clamped to 30 days), and
  2. the wire shape of the tags — FLAT `{"code","value"}`, NOT the Beckn
     tag-group form. The BPP silently ignores the group form, so only a test
     that asserts the flat shape catches a regression back to it.

Mirrors the spec in bharat-oan-api's tests/test_mandi.py (Bharat Vistaar's own
AI layer, branch OD-2787-fix-date-range).
"""
import importlib.util
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load vistaar directly by path so we don't trigger agents/tools/__init__ (which
# imports pydantic_ai + every heavy tool). The module only needs httpx and
# helpers.utils.
_spec = importlib.util.spec_from_file_location(
    "vistaar", ROOT / "agents" / "tools" / "vistaar.py"
)
vistaar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vistaar)

MAX_DAYS = vistaar.MANDI_MAX_RANGE_DAYS
# The DEFAULT lookback (7 d) and the CAP on an explicit range (30 d) are
# different numbers. The BPP caps every response at 10 rows, so a default wider
# than ~7 days returns byte-identical data — but an explicit "last 20 days"
# must still be honoured in full, which is why one constant could not do both.
DEFAULT_DAYS = vistaar.MANDI_DEFAULT_RANGE_DAYS
_DDMMYYYY = re.compile(r"^\d{2}-\d{2}-\d{4}$")


# IST is declared HERE, independently of the module under test. Deriving the
# expected dates from vistaar._IST would make every date assertion vacuous: swap
# _IST for UTC and implementation + expectation move together, so nothing fails.
_TEST_IST = timezone(timedelta(hours=5, minutes=30))

# Freeze the clock inside the 18:30–23:59 UTC window, where IST has already
# rolled over to the next calendar day. Every date assertion below therefore
# distinguishes IST from UTC: 2026-07-25 20:00 UTC is 2026-07-26 01:30 IST.
FROZEN_UTC = datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    """datetime with a pinned now(); strptime etc. inherit unchanged."""
    _instant = FROZEN_UTC

    @classmethod
    def now(cls, tz=None):
        return cls._instant.astimezone(tz) if tz is not None else cls._instant.replace(tzinfo=None)


@contextmanager
def frozen_clock(instant: datetime):
    class _Pinned(_FrozenDatetime):
        _instant = instant

    previous = vistaar.datetime
    vistaar.datetime = _Pinned
    try:
        yield
    finally:
        vistaar.datetime = previous


@pytest.fixture(autouse=True)
def _freeze_clock():
    with frozen_clock(FROZEN_UTC):
        yield


def _today_ist_str() -> str:
    return _days_ago_str(0)


def _days_ago_str(days: int) -> str:
    """Expected date, computed from the locally-declared IST — never vistaar._IST."""
    return (FROZEN_UTC.astimezone(_TEST_IST).date() - timedelta(days=days)).strftime("%d-%m-%Y")


class TestISTClock:
    """The window is resolved in IST, not UTC or local time.

    Mutate `_IST` in agents/tools/vistaar.py to `timezone.utc` and these must go
    red — they assert literal dates, so the expectation cannot drift with the
    implementation.
    """

    def test_after_1830_utc_the_window_ends_on_the_ist_date(self):
        # 20:00 UTC on the 25th is 01:30 IST on the 26th.
        with frozen_clock(datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc)):
            from_date, to_date = vistaar._resolve_date_range(None, None)
        assert to_date == "26-07-2026", "to_date must be today in IST, not the UTC date"
        assert from_date == "19-07-2026"

    def test_the_ist_date_flips_exactly_at_1830_utc(self):
        with frozen_clock(datetime(2026, 7, 25, 18, 29, tzinfo=timezone.utc)):
            assert vistaar._resolve_date_range(None, None)[1] == "25-07-2026"
        with frozen_clock(datetime(2026, 7, 25, 18, 31, tzinfo=timezone.utc)):
            assert vistaar._resolve_date_range(None, None)[1] == "26-07-2026"

    def test_a_mid_day_instant_resolves_to_that_same_date(self):
        # 09:00 UTC is 14:30 IST — same calendar day, the ordinary case.
        with frozen_clock(datetime(2026, 7, 25, 9, 0, tzinfo=timezone.utc)):
            from_date, to_date = vistaar._resolve_date_range(None, None)
        assert (from_date, to_date) == ("18-07-2026", "25-07-2026")

    def test_todays_ist_date_is_not_capped_away_as_a_future_date(self):
        # In UTC terms the 26th is "tomorrow", so a UTC clock would cap it back
        # to the 25th and silently drop the freshest arrivals.
        with frozen_clock(datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc)):
            assert vistaar._resolve_date_range("26-07-2026", "26-07-2026") == (
                "26-07-2026", "26-07-2026",
            )

    def test_the_ist_rollover_can_cross_the_year_boundary(self):
        with frozen_clock(datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)):
            from_date, to_date = vistaar._resolve_date_range(None, None)
        assert to_date == "01-01-2027"
        assert from_date == "25-12-2026"


class TestParseDate:
    def test_parses_ddmmyyyy(self):
        assert vistaar._parse_date("25-07-2026") == datetime(2026, 7, 25).date()

    def test_parses_iso_and_slashed_variants(self):
        assert vistaar._parse_date("2026-07-25") == datetime(2026, 7, 25).date()
        assert vistaar._parse_date("25/07/2026") == datetime(2026, 7, 25).date()

    def test_returns_none_for_empty_or_blank(self):
        assert vistaar._parse_date(None) is None
        assert vistaar._parse_date("") is None
        assert vistaar._parse_date("   ") is None

    def test_returns_none_for_unparseable(self):
        assert vistaar._parse_date("not-a-date") is None


class TestResolveDateRange:
    def test_both_ends_are_ddmmyyyy_never_iso(self):
        # Guards the wire format: the BPP wants DD-MM-YYYY, not ISO.
        for args in ((None, None), ("25-07-2026", None), (_days_ago_str(9), _days_ago_str(2))):
            from_date, to_date = vistaar._resolve_date_range(*args)
            assert _DDMMYYYY.match(from_date), from_date
            assert _DDMMYYYY.match(to_date), to_date

    def test_to_date_is_never_empty(self):
        for args in ((None, None), ("25-07-2026", None), ("garbage", "garbage")):
            _, to_date = vistaar._resolve_date_range(*args)
            assert to_date, f"to_date must never be null/blank, got {to_date!r} for {args}"

    def test_to_date_defaults_to_today_for_a_single_date_ask(self):
        _, to_date = vistaar._resolve_date_range(_days_ago_str(5))
        assert to_date == _today_ist_str()

    def test_no_date_looks_back_the_default_window_not_the_cap(self):
        # 30 days was wasted: the BPP's 10-row cap makes 7, 15 and 30 identical.
        from_date, to_date = vistaar._resolve_date_range(None, None)
        assert from_date == _days_ago_str(DEFAULT_DAYS)
        assert to_date == _today_ist_str()
        assert DEFAULT_DAYS < MAX_DAYS, "the default must not BE the cap"

    def test_an_explicit_range_is_still_honoured_out_to_the_full_cap(self):
        # The regression this guards: clamping explicit dates to the new 7-day
        # default would silently trim a farmer's "prices for the last 20 days".
        for days in (10, 20, MAX_DAYS):
            start = _days_ago_str(days)
            from_date, to_date = vistaar._resolve_date_range(start, _today_ist_str())
            assert from_date == start, f"{days}-day explicit range was trimmed"
            assert to_date == _today_ist_str()

    def test_recent_price_date_is_used_as_is(self):
        recent = _days_ago_str(5)
        from_date, to_date = vistaar._resolve_date_range(recent)
        assert from_date == recent
        assert to_date == _today_ist_str()

    def test_price_date_beyond_max_range_is_clamped(self):
        from_date, to_date = vistaar._resolve_date_range(_days_ago_str(MAX_DAYS + 100))
        assert from_date == _days_ago_str(MAX_DAYS)
        assert to_date == _today_ist_str()

    def test_price_date_exactly_at_boundary_is_not_clamped(self):
        boundary = _days_ago_str(MAX_DAYS)
        from_date, _ = vistaar._resolve_date_range(boundary)
        assert from_date == boundary

    def test_unparseable_date_falls_back_to_latest_window(self):
        from_date, to_date = vistaar._resolve_date_range("not-a-real-date")
        assert from_date == _days_ago_str(DEFAULT_DAYS)
        assert to_date == _today_ist_str()

    def test_explicit_range_uses_both_ends(self):
        start, end = _days_ago_str(20), _days_ago_str(10)
        assert vistaar._resolve_date_range(start, end) == (start, end)

    def test_reversed_range_is_swapped(self):
        start, end = _days_ago_str(10), _days_ago_str(20)
        assert vistaar._resolve_date_range(start, end) == (end, start)

    def test_range_end_in_the_future_is_capped_at_today(self):
        start = _days_ago_str(5)
        assert vistaar._resolve_date_range(start, _days_ago_str(-10)) == (start, _today_ist_str())

    def test_range_wider_than_max_is_clamped_from_the_end(self):
        start, end = _days_ago_str(MAX_DAYS + 40), _days_ago_str(5)
        from_date, to_date = vistaar._resolve_date_range(start, end)
        assert from_date == _days_ago_str(MAX_DAYS + 5)
        assert to_date == end

    def test_resolved_window_is_never_wider_than_max(self):
        from_date, to_date = vistaar._resolve_date_range(_days_ago_str(365), _days_ago_str(1))
        span = (vistaar._parse_date(to_date) - vistaar._parse_date(from_date)).days
        assert span <= MAX_DAYS, f"window of {span} days exceeds the API limit of {MAX_DAYS}"

    def test_range_end_without_start_searches_back_from_the_end(self):
        end = _days_ago_str(10)
        from_date, to_date = vistaar._resolve_date_range(None, end)
        assert from_date == _days_ago_str(DEFAULT_DAYS + 10)
        assert to_date == end


def _mandi_item(arrival: str, market: str = "Anand APMC", modal: str = "2000",
                lo: str = "1800", hi: str = "2200", variety: str = "Local") -> dict:
    def _t(code, value):
        return {"descriptor": {"code": code}, "value": value}
    return {
        "descriptor": {"name": "Onion"},
        "tags": [{"descriptor": {"code": "attributes"}, "list": [
            _t("Arrival Date", arrival), _t("Market", market), _t("Modal Price", modal),
            _t("Min Price", lo), _t("Max Price", hi), _t("Variety", variety),
            _t("Price Unit", "Rs./Qtl"),
        ]}],
    }


class TestTagValues:
    def test_flattens_tag_groups(self):
        flat = vistaar._tag_values(_mandi_item("25-07-2026"))
        assert flat["Market"] == "Anand APMC"
        assert flat["Arrival Date"] == "25-07-2026"

    def test_item_without_tags_is_empty(self):
        assert vistaar._tag_values({"descriptor": {"name": "Onion"}}) == {}


class TestFormatMandiItems:
    def test_one_line_per_arrival_date_newest_first(self):
        items = [_mandi_item("20-07-2026"), _mandi_item("25-07-2026"), _mandi_item("22-07-2026")]
        lines = vistaar._format_mandi_items(items).splitlines()
        assert len(lines) == 3
        assert [l.split(" — ")[0] for l in lines] == ["- 25-07-2026", "- 22-07-2026", "- 20-07-2026"]

    def test_renders_market_and_price_band(self):
        out = vistaar._format_mandi_items([_mandi_item("25-07-2026")])
        assert "Anand APMC" in out
        assert "modal 2000" in out
        assert "min 1800" in out and "max 2200" in out
        assert "Rs./Qtl" in out

    def test_a_full_window_is_not_truncated_at_twenty_rows(self):
        # The generic _format_items caps at 20, which would eat most of a
        # 30-day window; _format_mandi_items exists precisely to avoid that.
        items = [_mandi_item(_days_ago_str(d)) for d in range(24)]
        lines = vistaar._format_mandi_items(items).splitlines()
        assert len(lines) == 24

    def test_undated_rows_sort_last_and_do_not_crash(self):
        items = [_mandi_item(""), _mandi_item("25-07-2026")]
        lines = vistaar._format_mandi_items(items).splitlines()
        assert lines[0].startswith("- 25-07-2026")
        assert "date n/a" in lines[-1]

    @pytest.mark.parametrize("junk", ["not-a-date", "32-13-2026", "2026-07-25T00:00:00Z"])
    def test_malformed_arrival_dates_render_as_unknown_not_raw(self, junk):
        # The raw value must not reach the farmer: an unparseable arrival date is
        # an unknown date, and rendering "32-13-2026" reads as a real one.
        items = [_mandi_item(junk), _mandi_item("25-07-2026")]
        lines = vistaar._format_mandi_items(items).splitlines()
        assert lines[0].startswith("- 25-07-2026")
        assert "date n/a" in lines[-1]
        assert junk not in vistaar._format_mandi_items(items)

    def test_rows_beyond_the_cap_are_summarised(self):
        items = [_mandi_item(_days_ago_str(d)) for d in range(45)]
        lines = vistaar._format_mandi_items(items, max_rows=40).splitlines()
        assert len(lines) == 41
        assert "5 older rows omitted" in lines[-1]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        self.calls.append((url, json))
        return _FakeResponse(self._payload)


def _seeker_payload(items):
    leg = vistaar.VISTAAR_LEG
    return {"results": {leg: {"message": {"catalog": {"providers": [{"id": leg, "items": items}]}}}}}


class TestMandiIntent:
    @pytest.mark.asyncio
    async def test_intent_carries_flat_from_and_to_date_tags(self):
        fake = _FakeAsyncClient(_seeker_payload([_mandi_item(_days_ago_str(1))]))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            await vistaar.get_vistaar_mandi_prices(
                None, "Onion", None, _days_ago_str(9), _days_ago_str(2)
            )
        tags = fake.calls[0][1]["intent"]["tags"]
        # FLAT shape — the BPP silently ignores the Beckn tag-group form, so a
        # regression to {"descriptor":..., "list":[...]} would look like a
        # working call while quietly returning the wrong market.
        assert tags == [
            {"code": "from_date", "value": _days_ago_str(9)},
            {"code": "to_date", "value": _days_ago_str(2)},
        ]
        for tag in tags:
            assert set(tag) == {"code", "value"}
            assert "descriptor" not in tag and "list" not in tag

    @pytest.mark.asyncio
    async def test_intent_always_sends_a_window_even_with_no_dates(self):
        # Sending no window is the original bug: one row from Padra APMC.
        fake = _FakeAsyncClient(_seeker_payload([_mandi_item(_days_ago_str(1))]))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            await vistaar.get_vistaar_mandi_prices(None, "Onion")
        codes = {t["code"]: t["value"] for t in fake.calls[0][1]["intent"]["tags"]}
        assert codes["from_date"] == _days_ago_str(DEFAULT_DAYS)
        assert codes["to_date"] == _today_ist_str()

    @pytest.mark.asyncio
    async def test_intent_keeps_commodity_category_and_anand_location(self):
        fake = _FakeAsyncClient(_seeker_payload([]))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            await vistaar.get_vistaar_mandi_prices(None, "Tomato")
        intent = fake.calls[0][1]["intent"]
        assert intent["category"]["descriptor"]["code"] == "price-discovery"
        assert intent["item"]["descriptor"]["name"] == "Tomato"
        assert intent["fulfillment"]["end"]["location"]["descriptor"]["name"] == "Anand"

    @pytest.mark.asyncio
    async def test_intent_sends_gps_because_the_name_is_decorative(self):
        # Measured: name=Junagadh + Anand gps returns nothing; name=Anand +
        # Junagadh gps returns Junagadh APMC. The gps is the only thing that
        # selects a market, so dropping it would silently answer from the
        # BPP's fallback market with no error anywhere.
        fake = _FakeAsyncClient(_seeker_payload([]))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            await vistaar.get_vistaar_mandi_prices(None, "Tomato")
        gps = fake.calls[0][1]["intent"]["fulfillment"]["end"]["location"]["gps"]
        lat, lon = (float(v) for v in gps.split(","))
        assert (lat, lon) == (22.474, 72.736), "Anand's default coordinate"

    @pytest.mark.asyncio
    async def test_returns_the_series_with_the_window_in_the_header(self):
        items = [_mandi_item(_days_ago_str(d)) for d in (1, 3, 5)]
        fake = _FakeAsyncClient(_seeker_payload(items))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            out = await vistaar.get_vistaar_mandi_prices(None, "Onion", None, _days_ago_str(9))
        assert out.startswith(
            f"Mandi prices for Onion near Anand ({_days_ago_str(9)} to {_today_ist_str()}):"
        )
        assert out.count("Anand APMC") == 3

    @pytest.mark.asyncio
    async def test_a_thirty_day_window_reaches_the_agent_whole(self):
        # End-to-end guard on the formatter choice: the generic _format_items
        # caps at 20 items, which would silently eat a third of the window.
        items = [_mandi_item(_days_ago_str(d)) for d in range(28)]
        fake = _FakeAsyncClient(_seeker_payload(items))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            out = await vistaar.get_vistaar_mandi_prices(None, "Onion")
        rows = [l for l in out.splitlines() if l.startswith("- ")]
        assert len(rows) == 28, f"expected all 28 dated rows, got {len(rows)}"
        assert all(_DDMMYYYY.match(l.split(" — ")[0][2:]) for l in rows)

    @pytest.mark.asyncio
    async def test_empty_result_names_the_window_searched(self):
        fake = _FakeAsyncClient(_seeker_payload([]))
        with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
            out = await vistaar.get_vistaar_mandi_prices(None, "Onion")
        assert "No mandi prices were found" in out
        assert _days_ago_str(DEFAULT_DAYS) in out and _today_ist_str() in out

    @pytest.mark.asyncio
    async def test_transport_failure_degrades_gracefully(self):
        class _Boom(_FakeAsyncClient):
            async def post(self, url, json=None):
                raise RuntimeError("connection reset")

        with patch.object(vistaar.httpx, "AsyncClient", return_value=_Boom(None)):
            out = await vistaar.get_vistaar_mandi_prices(None, "Onion")
        assert "temporarily unavailable" in out


class TestToolDocstring:
    def test_every_argument_is_documented(self):
        # pydantic-ai registers this tool with require_parameter_descriptions=True,
        # so a missing Args entry is a hard failure at tool registration.
        doc = vistaar.get_vistaar_mandi_prices.__doc__ or ""
        for arg in ("commodity_name:", "location:", "price_date:", "price_date_to:"):
            assert arg in doc, f"{arg} missing from the tool docstring"
