import pytest
from pydantic import ValidationError

from app.config import Settings


def _enabled_settings(**overrides):
    values = {
        "beckn_callback_transactions_enabled": True,
        "beckn_bap_caller_url": "http://10.5.25.173:16241/transactions",
        "beckn_bap_uri": "https://beckn-bap.prod.amulai.in/bap/receiver",
        "beckn_transaction_bridge_token": "transaction-secret",
        "beckn_callback_token": "callback-secret",
    }
    values.update(overrides)
    return Settings(**values)


def test_enabled_beckn_transport_requires_both_credentials():
    with pytest.raises(ValidationError, match="BECKN_TRANSACTION_BRIDGE_TOKEN"):
        _enabled_settings(beckn_transaction_bridge_token=None)
    with pytest.raises(ValidationError, match="BECKN_CALLBACK_TOKEN"):
        _enabled_settings(beckn_callback_token=None)


def test_enabled_beckn_transport_accepts_complete_private_bridge_config():
    configured = _enabled_settings()
    assert configured.beckn_bap_caller_url == "http://10.5.25.173:16241/transactions"
