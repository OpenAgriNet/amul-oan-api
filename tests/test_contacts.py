"""Chat's phone number check, on the shape of the 11-18 prod answer (numbers replaced)."""
import asyncio

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from agents.tools.contacts import DROPPED_NOTE, Contacts, note_tool_result, track

LISTINGS = {"status": "ok", "listings": [
    {"listing_id": 42, "seller": {"name": "Ramesh Vaghela", "mobile": "9811111111"}},
    {"listing_id": 77, "seller": {"name": "Suresh Solanki", "mobile": "9822222222"}},
]}
HISTORY = [
    ModelRequest(parts=[UserPromptPart(content="Where can I find fodder in Anand rural?")]),
    ModelResponse(parts=[ToolCallPart(tool_name="search_fodder_availability", args={"subdistrict": "Anand rural"}, tool_call_id="c1")]),
    ModelRequest(parts=[ToolReturnPart(tool_name="search_fodder_availability", content=LISTINGS, tool_call_id="c1")]),
    ModelResponse(parts=[TextPart(content="Green maize in Navli...")]),
]
ANSWER = (
    "Here are some of the available options:\n"
    "*   **Green Maize:** ₹60 per Maund — Navli. Seller: Ramesh Vaghela, Mobile: 9811111111\n"
    "*   **Green Bajra:** ₹500 per Trolley — Mehsana. Seller: Rajeshbhai, Mobile: 98250 00000\n"
    "*   **Groundnut haulms:** ₹450 per Bag — Chikhodra. Seller: Suresh Solanki, Mobile: +91 98222 22222\n"
    "*   **Cotton Seed Cake:** ₹2,800 per Quintal — Surat. Seller: Feed Store, Mobile: 9974000000\n"
    "Please contact the sellers directly."
)


async def _run(contacts, chunks):
    async def source():
        for chunk in chunks:
            yield chunk
    return "".join([out async for out in contacts.filter(source())])


def _chunks(text, size=7):
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_drops_lines_with_numbers_no_tool_returned():
    out = asyncio.run(_run(Contacts(HISTORY), _chunks(ANSWER)))
    assert "9811111111" in out and "98222 22222" in out
    assert "Mehsana" not in out and "Surat" not in out
    assert out.endswith(DROPPED_NOTE)


def test_this_turns_tool_results_count():
    contacts = track([])
    note_tool_result("get_fodder_listing", {"listing": {"seller": {"mobile": "9974000000"}}})
    assert contacts.fodder and not contacts.unconfirmed("Mobile: 9974000000")


def test_session_without_fodder_is_untouched():
    text = "Call the vet office at 9876500000.\nDone."
    assert asyncio.run(_run(Contacts([]), _chunks(text))) == text
