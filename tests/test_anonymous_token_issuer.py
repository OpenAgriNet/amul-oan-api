import httpx
import pytest
from fastapi import HTTPException

from app.routers import auth


@pytest.mark.asyncio
async def test_verification_only_service_delegates_without_loading_private_key(monkeypatch):
    monkeypatch.setattr(auth.settings, "anonymous_token_issuer_url", "http://chat/api/auth/anonymous")
    def forbidden(**kwargs):
        raise AssertionError("delegated service must not read a signing key")
    monkeypatch.setattr(auth, "create_anonymous_jwt_token", forbidden)
    client = httpx.AsyncClient
    def handle(request):
        assert request.method == "POST"
        assert str(request.url) == "http://chat/api/auth/anonymous"
        return httpx.Response(200, json={"access_token": "signed-test-token", "token_type": "bearer", "expires_in": 86400})
    monkeypatch.setattr(auth.httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    result = await auth.anonymous_token()
    assert result.access_token == "signed-test-token"
    assert result.expires_in == 86400


@pytest.mark.asyncio
@pytest.mark.parametrize("code,body", [(503, {}), (200, {}),
    (200, {"access_token": "", "token_type": "bearer", "expires_in": 86400})])
async def test_bad_issuer_response_returns_clean_unavailable_error(monkeypatch, code, body):
    monkeypatch.setattr(auth.settings, "anonymous_token_issuer_url", "http://chat/api/auth/anonymous")
    client = httpx.AsyncClient
    monkeypatch.setattr(auth.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(lambda request: httpx.Response(code, json=body)), **kwargs))
    with pytest.raises(HTTPException) as exc:
        await auth.anonymous_token()
    assert exc.value.status_code == 503
    assert "private" not in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_unconfigured_issuer_retains_local_signing(monkeypatch):
    monkeypatch.setattr(auth.settings, "anonymous_token_issuer_url", None)
    monkeypatch.setattr(auth, "create_anonymous_jwt_token", lambda **kwargs: "local-signed-test-token")
    assert (await auth.anonymous_token()).access_token == "local-signed-test-token"
