"""The test case labeller (#488) and its dry-run script.

The model is a fake: a reply that isn't JSON, or a group or answer type off the
lists, must come out as an error, not a label. Personal data is masked before the
prompt. The script reads the seed export, eval set v0 and raw submissions into one
case per session and writes a CSV. No network.
"""
import csv
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from app.services.personal_data import mask_codes, mask_personal_data
from app.services.test_case_labeller import (
    DEFAULT_MODEL,
    Case,
    LabelError,
    LiteLLM,
    Turn,
    build_prompt,
    label_case,
    parse_label,
)

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("label_test_cases", REPO / "scripts" / "label_test_cases.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

GOOD = {"group": "ai", "answer_type": "ask:technician"}


# --- masking ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, masked",
    [
        ("My farmer code is 1234", "My farmer code is [code]"),
        ("society code: NA4192", "society code: [code]"),
        ("Your OTP is ૪૮૨૧.", "Your OTP is [code]."),
        ("મારો સભાસદ કોડ ૧૨૩૪ છે", "મારો સભાસદ કોડ [code] છે"),
        ("ઓટીપી 4821", "ઓટીપી [code]"),
        ("member no. 56/2", "member no. [code]"),
        ("Call 98765 43210, code 77", "Call ******3210, code [code]"),
        ("Aadhaar ૧૨૩૪ ૫૬૭૮ ૯૦૧૨", "Aadhaar ******9012"),
    ],
)
def test_phone_aadhaar_otp_and_farmer_codes_are_masked(text, masked):
    assert mask_personal_data(text) == masked


@pytest.mark.parametrize(
    "text",
    ["2025-26 cycle", "₹1,25,000", "On 02-11-2026", "14 tarikh nu dudh", "code of conduct", "code for 2 animals",
     "postcode 380001", "fat 6.5 snf 8.5", "૧૨૦ લિટર"],
)
def test_dates_amounts_cycles_and_ordinary_words_stay(text):
    assert mask_personal_data(text) == text
    assert mask_codes(text) == text


# --- labels ----------------------------------------------------------------------


def test_a_good_reply_is_a_label():
    assert parse_label(json.dumps(GOOD)).model_dump() == GOOD
    assert parse_label("```json\n" + json.dumps(GOOD) + "\n```").group == "ai"
    assert parse_label("Here is the label: " + json.dumps(GOOD) + " Hope it helps.").group == "ai"


def test_small_slips_in_a_good_reply_are_not_errors():
    label = parse_label(json.dumps({"group": " Vet_Health ", "answer_type": "ASK:Species"}))

    assert (label.group, label.answer_type) == ("vet-health", "ask:species")
    assert parse_label(json.dumps({**GOOD, "answer_type": "Answer"})).answer_type == "answer"
    assert parse_label(json.dumps({**GOOD, "answer_type": "Out-Of-Scope"})).answer_type == "out_of_scope"


@pytest.mark.parametrize(
    "reply, error",
    [
        ("not json", "the reply is not JSON"),
        ("[1, 2]", "reply: Input should be"),
        (json.dumps({**GOOD, "group": "ait_booking"}), "group: 'ait_booking' is not a group"),
        (json.dumps({**GOOD, "answer_type": "maybe"}), "answer_type: 'maybe' is not an answer type"),
        (json.dumps({**GOOD, "answer_type": "refer_vet"}), "answer_type: 'refer_vet' is not an answer type"),
        (json.dumps({**GOOD, "answer_type": "ask:"}), "answer_type: 'ask:' is not an answer type"),
        (json.dumps({**GOOD, "sub_tags": ["cow"]}), "sub_tags: Extra inputs are not permitted"),
        (json.dumps({"group": "ai"}), "answer_type: Field required"),
    ],
    ids=["not json", "not an object", "group off the list", "unknown answer type", "refer_vet is vet now",
         "empty ask", "extra key", "missing key"],
)
def test_a_bad_reply_is_an_error_not_a_label(reply, error):
    with pytest.raises(LabelError) as caught:
        parse_label(reply)
    assert str(caught.value).startswith(error)


def test_every_group_and_answer_type_passes():
    for answer_type in ["answer", "book", "vet", "say_unavailable", "out_of_scope", "ask:species"]:
        assert parse_label(json.dumps({**GOOD, "answer_type": answer_type})).answer_type == answer_type
    for group in ["ai", "vet-health", "milk", "scheme", "bonus", "other"]:
        assert parse_label(json.dumps({**GOOD, "group": group})).group == group


def test_the_prompt_masks_personal_data_and_lists_the_groups_and_answer_types():
    case = Case(
        "amul-test-case-submissions",
        "c1",
        (Turn("user", "My number is 9876543210, farmer code 1234"), Turn("assistant", "Thanks.")),
        description="Farmer on 98765 43210 asks to book.",
        hints={"tester_tags": "ait_booking", "other_topic": ""},
    )

    prompt = build_prompt(case)

    assert "9876543210" not in prompt and "98765 43210" not in prompt and "code 1234" not in prompt
    assert "Farmer: My number is ******3210, farmer code [code]" in prompt
    assert "What the tester says the case checks: Farmer on ******3210 asks to book." in prompt
    assert "Bot: Thanks." in prompt
    assert "Hints: tester_tags: ait_booking" in prompt and "other_topic" not in prompt
    assert "- ai: anything about artificial insemination (AI)" in prompt
    assert "- vet-health: animal health" in prompt and "- other: anything else" in prompt
    assert "- vet: the main advice is to get a vet" in prompt and "refer_vet" not in prompt
    assert "- ask:<what it should ask for>" in prompt
    assert "after the farmer's last message" in prompt and "ask:species first, then ask:technician" in prompt
    assert "If the hints say persona: doctor, the user is a vet" in prompt


def test_label_case_sends_the_prompt_and_parses_the_reply():
    seen = []
    case = Case("eval-set-v0", "ai-1", (Turn("user", "I want AI for my cow"),))

    label = label_case(case, lambda prompt: seen.append(prompt) or json.dumps(GOOD))

    assert label.group == "ai" and seen == [build_prompt(case)]


# --- LiteLLM ---------------------------------------------------------------------


def _litellm(monkeypatch, handler, model=None):
    monkeypatch.setenv("SCENARIO_GEMMA_BASE_URL", "https://litellm.example/v1/")
    monkeypatch.setenv("LITELLM_API_KEY", "key-1")
    if model:
        monkeypatch.setenv("SCENARIO_GEMMA_MODEL", model)
    else:
        monkeypatch.delenv("SCENARIO_GEMMA_MODEL", raising=False)
    return LiteLLM(httpx.Client(transport=httpx.MockTransport(handler)))


def _reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_litellm_is_called_the_way_the_judge_calls_it(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return _reply(json.dumps(GOOD))

    assert _litellm(monkeypatch, handler).complete("hello") == json.dumps(GOOD)

    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://litellm.example/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer key-1"
    assert body == {
        "model": DEFAULT_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_object"},
    }


def test_litellm_takes_the_model_from_the_env_without_the_openai_prefix(monkeypatch):
    requests = []
    client = _litellm(monkeypatch, lambda request: requests.append(request) or _reply("{}"), model="openai/gemma-4-e4b")

    client.complete("hello")

    assert json.loads(requests[0].content)["model"] == "gemma-4-e4b"


def test_litellm_asks_again_without_json_mode_after_a_400(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"error": "no json mode"}) if len(bodies) == 1 else _reply("{}")

    assert _litellm(monkeypatch, handler).complete("hello") == "{}"
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


def test_litellm_errors_raise(monkeypatch):
    with pytest.raises(httpx.HTTPStatusError):
        _litellm(monkeypatch, lambda request: httpx.Response(500)).complete("hello")
    with pytest.raises(LabelError, match="no text"):
        _litellm(monkeypatch, lambda request: _reply(None)).complete("hello")


def test_litellm_needs_its_url_and_key(monkeypatch):
    monkeypatch.delenv("SCENARIO_GEMMA_BASE_URL", raising=False)
    monkeypatch.setenv("LITELLM_API_KEY", "key-1")

    with pytest.raises(LabelError, match="SCENARIO_GEMMA_BASE_URL and LITELLM_API_KEY are required"):
        LiteLLM()


# --- the dry-run script ----------------------------------------------------------


def _seed_item(item_id, session, question_id, gu, en, category="AI_technician_booking", tool="create_ai_call", status="ACTIVE"):
    return {
        "id": item_id,
        "status": status,
        "input": {"session_id": session, "question_id": question_id, "question_english": en, "question_gujarati": gu},
        "metadata": {"category": category, "tool_hint": tool},
        "expectedOutput": None,
    }


def test_the_seed_export_becomes_one_case_per_session_in_question_order(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({"data": [
        _seed_item("ai-cow-2", "ai-cow", "ai-cow-10", "પહેલા.", "The first one."),
        _seed_item("ai-cow-1", "ai-cow", "ai-cow-2", "ગાય માટે.", "For the cow."),
        _seed_item("x", "ai-cow", "ai-cow-1", "મારી ગાયને બીજદાન કરવાનું છે", "My cow needs AI."),
        _seed_item("old", "ai-cow", "ai-cow-3", "જૂનું", "Old", status="ARCHIVED"),
        _seed_item("milk-1", "milk", "milk-1", "Ha", "Ha", category="Milk_collection", tool="get_farmer_milk_collection_details"),
    ]}), encoding="utf-8")

    cases = script.load_seed(path)

    assert [case.case_id for case in cases] == ["ai-cow", "milk"]
    assert [turn.text for turn in cases[0].turns] == [
        "મારી ગાયને બીજદાન કરવાનું છે (English: My cow needs AI.)",
        "ગાય માટે. (English: For the cow.)",
        "પહેલા. (English: The first one.)",
    ]
    assert cases[0].hints == {"category": "AI_technician_booking", "tool_hint": "create_ai_call"}
    assert cases[1].turns == (Turn("user", "Ha"),)


def test_v0_becomes_cases_with_its_expected_action_kept_out_of_the_prompt(tmp_path):
    rows = [
        ["session_id", "question_id", "turn", "category", "fixture", "input_style", "question_gujarati",
         "question_english", "expected_tools", "must_not_call", "expected_behaviour"],
        ["ai-1", "ai-1-2", "2", "ai_booking", "F1", "gu_script", "પહેલા.", "The first one.", "create_ai_call", "", "Books."],
        ["ai-1", "ai-1-1", "1", "ai_booking", "F1", "gu_script", "બીજદાન કરવું છે", "I want AI.", "", "create_ai_call", "Asks."],
        ["adv-1", "adv-1-1", "1", "vet_advice", "F1", "gu_script", "પેટ ફૂલી ગયું છે", "The belly is swollen.", "search_documents", "", "Vet now."],
    ]
    csv_path, meta_path = tmp_path / "v0.csv", tmp_path / "meta.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    meta_path.write_text(json.dumps({
        "_meta": {},
        "ai-1-1": {"expected_action": "ask:technician", "acceptable_actions": ["ask:technician", "ask:species"]},
        "ai-1-2": {"expected_action": "book"},
        "adv-1-1": {"expected_action": "urgent_vet"},
    }))

    cases, actions = script.load_v0(csv_path, meta_path)

    assert [turn.text for turn in cases[0].turns] == ["બીજદાન કરવું છે (English: I want AI.)", "પહેલા. (English: The first one.)"]
    assert cases[0].hints == {"category": "ai_booking", "expected_tools": "create_ai_call", "must_not_call": "create_ai_call"}
    assert actions == {"ai-1": "ask:technician (or ask:species) | book", "adv-1": "vet"}
    assert actions["ai-1"] not in build_prompt(cases[0])


def test_submissions_keep_both_roles_and_the_bots_first_answer(tmp_path):
    body = {
        "source": "chat", "submitted_by": "kanav", "persona": "farmer", "language": "en",
        "description": "Pregnant buffalo: no painkillers without a vet.", "case_types": ["wrong_answer"],
        "tags": ["veterinary_advisory"],
        "turns": [
            {"role": "user", "content": "Can I give painkillers?"},
            {"role": "assistant", "content": "Ask your vet first.", "original_content": "Give paracetamol."},
        ],
    }
    path = tmp_path / "subs.json"
    path.write_text(json.dumps({"data": [{"id": "c1", "status": "ACTIVE", "input": body}]}), encoding="utf-8")

    case = script.load_submissions(path)[0]

    assert case.turns == (
        Turn("user", "Can I give painkillers?"),
        Turn("assistant", "Ask your vet first. [the tester corrected this; the bot first said: Give paracetamol.]"),
    )
    assert case.description == body["description"]
    assert case.hints["tester_tags"] == "veterinary_advisory" and case.hints["tester_case_types"] == "wrong_answer"


def test_the_table_has_a_row_per_case_with_errors_in_their_own_column(tmp_path):
    cases = [
        Case("eval-set-v0", "ai-1", (Turn("user", "Call me on 9876543210"),), hints={"category": "ai_booking"}),
        Case("amul-seed-questions", "milk", (Turn("user", "How much milk?"),)),
    ]
    replies = iter([json.dumps(GOOD), "not json"])

    rows = script.label_rows(cases, lambda prompt: next(replies), {"ai-1": "ask:technician"})
    out = tmp_path / "labels.csv"
    script.write_table(rows, out)

    assert out.read_bytes().startswith(b"\xef\xbb\xbf")
    written = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert list(written[0].keys()) == script.COLUMNS
    assert written[0]["turns"] == "Farmer: Call me on ******3210"
    assert (written[0]["group"], written[0]["answer_type"], written[0]["set_expected_action"]) == ("ai", "ask:technician", "ask:technician")
    assert (written[1]["group"], written[1]["error"], written[1]["set_expected_action"]) == ("", "the reply is not JSON", "")


def test_a_failed_call_is_an_error_row_and_the_run_goes_on():
    def boom(prompt):
        raise httpx.ConnectError("no route")

    rows = script.label_rows([Case("eval-set-v0", "a", (Turn("user", "Hi"),))], boom, {})

    assert rows[0]["error"] == "ConnectError: no route"


def test_the_script_needs_a_set_and_the_litellm_settings(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        script.main(["--out", str(tmp_path / "x.csv")])
    assert "give at least one of --seed, --v0 or --submissions" in capsys.readouterr().err

    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"data": [_seed_item("a", "s", "s-1", "હા", "Yes")]}), encoding="utf-8")
    monkeypatch.delenv("SCENARIO_GEMMA_BASE_URL", raising=False)
    with pytest.raises(SystemExit):
        script.main(["--seed", str(seed), "--out", str(tmp_path / "x.csv")])
    assert "SCENARIO_GEMMA_BASE_URL and LITELLM_API_KEY are required" in capsys.readouterr().err


def test_the_script_labels_and_writes_the_table(tmp_path, monkeypatch, capsys):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"data": [_seed_item("a", "s1", "s1-1", "હા", "Yes"), _seed_item("b", "s2", "s2-1", "ના", "No")]}), encoding="utf-8")

    class _FakeLiteLLM:
        def complete(self, prompt):
            return json.dumps(GOOD)

    monkeypatch.setattr(script, "LiteLLM", _FakeLiteLLM)
    out = tmp_path / "labels.csv"

    assert script.main(["--seed", str(seed), "--out", str(out), "--limit", "1"]) == 0

    assert len(list(csv.DictReader(out.open(encoding="utf-8-sig")))) == 1
    assert "labelled 1 of 1 cases, 0 errors" in capsys.readouterr().out
