"""Evidence-quality checks for the live Bharat Vistaar coordinate validator."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_district_coords", ROOT / "scripts" / "validate_district_coords.py"
)
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


def test_default_commodities_cover_the_banaskantha_decision_set():
    required = {"Onion", "Wheat", "Bajra", "Castor", "Groundnut"}
    assert set(validator.DEFAULT_COMMODITIES) == required


def test_repeat_signature_ignores_latency_but_not_returned_markets():
    first = {"rows": 10, "markets": ["Deesa APMC"], "elapsed_s": 2.1}
    slower = {"rows": 10, "markets": ["Deesa APMC"], "elapsed_s": 3.9}
    drifted = {"rows": 10, "markets": ["Abu Road APMC"], "elapsed_s": 2.1}

    assert validator._result_signature(first) == validator._result_signature(slower)
    assert validator._result_signature(first) != validator._result_signature(drifted)


def test_repeat_signature_distinguishes_errors_from_empty_catalogues():
    error = {"error": "leg_unavailable: timeout", "elapsed_s": 30.0}
    empty = {"rows": 0, "markets": [], "elapsed_s": 2.2}
    assert validator._result_signature(error) != validator._result_signature(empty)


def test_checkpoint_writes_complete_json(tmp_path):
    target = tmp_path / "nested" / "results.json"
    evidence = {"anand": {"Anand": {"rows": [10, 10]}}}

    validator._checkpoint(target, evidence)

    assert json.loads(target.read_text()) == evidence
    assert not target.with_name(f".{target.name}.pending").exists()
