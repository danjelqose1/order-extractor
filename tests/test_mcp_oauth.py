from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import httpx
import jwt
import pytest

ROOT = Path(__file__).resolve().parents[1]
ISSUER = "https://test-factory.eu.auth0.com/"
RESOURCE = "https://factory.example.test/mcp"
USER = "auth0|approved-operator"
SCOPES = "orders:read orders:write"


@pytest.fixture(scope="module")
def signing_keys():
    result = []
    for kid in ("current", "rotated"):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
        public.update(kid=kid, alg="RS256", use="sig")
        result.append((key, public))
    return result


def configure(monkeypatch):
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_ENABLED", "true")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_AUTH_MODE", "oauth")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_OAUTH_ISSUER", ISSUER)
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_RESOURCE_URL", RESOURCE)
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS", USER + ",auth0|second-operator")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_OAUTH_CLIENT_IDS", "chatgpt-test")
    monkeypatch.delenv("ORDER_EXTRACTOR_MCP_READ_ONLY", raising=False)


@pytest.fixture
def oauth_api(tmp_path, monkeypatch, signing_keys):
    configure(monkeypatch)
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    monkeypatch.setenv("ORDER_EXTRACTOR_LOAD_DOTENV", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-no-network")
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
    monkeypatch.syspath_prepend(str(ROOT / "backend"))
    for name in list(sys.modules):
        if name in {"app", "db", "llm", "workspace_service", "workspace_agent"} or name.startswith(("workspace_agents.", "mcp_server", "services.platform_")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    import mcp_auth
    original_fetch = mcp_auth.fetch_public_json
    calls = []
    async def fetch(url):
        calls.append(url)
        assert url == ISSUER + ".well-known/jwks.json"
        return {"keys": [signing_keys[0][1]]}
    monkeypatch.setattr(mcp_auth, "fetch_public_json", fetch)
    module = importlib.import_module("app")
    def token(changes=None, *, key_index=0, header=None, remove=()):
        now = int(time.time())
        claims = dict(iss=ISSUER, aud=RESOURCE, sub=USER, iat=now, exp=now+900,
                      scope=SCOPES, permissions=SCOPES.split(), azp="chatgpt-test")
        claims.update(changes or {})
        for name in remove:
            claims.pop(name, None)
        return jwt.encode(claims, signing_keys[key_index][0], algorithm="RS256",
                          headers={"kid": signing_keys[key_index][1]["kid"], **(header or {})})
    with TestClient(module.app) as client:
        def rpc(method, params=None, access_token=None):
            headers = {"Accept": "application/json, text/event-stream"}
            if access_token is not None:
                headers["Authorization"] = "Bearer " + access_token
            return client.post("/mcp", headers=headers, json={"jsonrpc":"2.0","id":1,"method":method,"params":params or {}})
        yield SimpleNamespace(module=module, client=client, rpc=rpc, token=token, calls=calls,
                              auth=mcp_auth, original_fetch=original_fetch)


def call(api, name, args=None, token=None):
    response = api.rpc("tools/call", {"name": name, "arguments": args or {}}, token or api.token())
    assert response.status_code == 200, response.text
    return response.json()["result"]


def test_public_discovery_and_authentication_challenge(oauth_api):
    api = oauth_api
    for path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
        response = api.client.get(path)
        assert response.status_code == 200
        assert response.json() == dict(resource=RESOURCE, resource_name="Order Extractor", authorization_servers=[ISSUER],
                                      scopes_supported=SCOPES.split(), bearer_methods_supported=["header"])
        assert USER not in response.text
    for method in ("GET", "POST", "DELETE"):
        response = api.client.request(method, "/mcp")
        assert response.status_code == 401
        assert 'resource_metadata="https://factory.example.test/.well-known/oauth-protected-resource/mcp"' in response.headers["WWW-Authenticate"]
    assert api.calls == []


def test_signed_oauth_initialize_discovery_and_read(oauth_api):
    api = oauth_api
    token = api.token({"scope":"orders:read", "permissions":["orders:read"]})
    initialized = api.rpc("initialize", dict(protocolVersion="2025-06-18", capabilities={}, clientInfo={"name":"oauth-test","version":"1"}), token)
    assert initialized.status_code == 200
    tools = api.rpc("tools/list", access_token=token).json()["result"]["tools"]
    assert len(tools) == 18
    for tool in tools:
        expected = ["orders:read"] if tool["annotations"]["readOnlyHint"] else SCOPES.split()
        assert tool["securitySchemes"] == [{"type":"oauth2","scopes":expected}]
        assert tool["_meta"]["securitySchemes"] == tool["securitySchemes"]
    assert call(api,"platform_health",token=token)["structuredContent"]["data"]["database"] == "ready"
    assert len(api.calls) == 1


@pytest.mark.parametrize("changes,remove", [
    ({"iss":"https://other.example/"},()), ({"aud":"another-api"},()),
    ({"exp":1},()), ({"nbf":int(time.time())+600},()), ({"iat":int(time.time())+600},()),
    ({},("exp",)), ({},("iat",)), ({},("sub",)), ({},("permissions",)),
    ({"permissions":"orders:read"},()), ({"scope":["orders:read"]},()),
    ({"exp":int(time.time())+7200},()), ({"iat":True},()),
    ({"azp":"chatgpt-test","client_id":"different"},()), ({"sub":"machine@clients"},()),
])
def test_invalid_registered_claims_are_rejected(oauth_api, changes, remove):
    api = oauth_api
    token = api.token(changes, remove=remove)
    response = api.rpc("tools/list", access_token=token)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_TOKEN"
    assert token not in response.text and USER not in response.text


@pytest.mark.parametrize("changes", [{"sub":"auth0|unapproved"},{"azp":"unapproved-client"}])
def test_signed_but_unapproved_identity_is_forbidden(oauth_api, changes):
    response = oauth_api.rpc("tools/list", access_token=oauth_api.token(changes))
    assert response.status_code == 403 and response.json()["error"]["code"] == "FORBIDDEN"


def test_invalid_signatures_algorithms_and_untrusted_key_urls(oauth_api):
    api = oauth_api
    forged = api.token(key_index=1, header={"kid":"current"})
    assert api.rpc("tools/list", access_token=forged).status_code == 401
    hs = jwt.encode(dict(sub=USER),"only-a-local-test-key-that-is-not-secret",algorithm="HS256",headers={"kid":"current"})
    for token in (hs, "invalid", "x"*16_385, api.token(header={"jku":"https://untrusted.invalid/keys"})):
        assert api.rpc("tools/list", access_token=token).status_code == 401
    assert api.calls == [ISSUER + ".well-known/jwks.json"]


def test_oauth_mode_does_not_accept_private_bearer(oauth_api, monkeypatch):
    key = "private-test-only-bearer-token-0123456789"
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_TOKEN", key)
    assert oauth_api.rpc("tools/list", access_token=key).status_code == 401


@pytest.mark.parametrize("claims", [
    {"scope":"orders:read", "permissions":SCOPES.split()},
    {"scope":SCOPES, "permissions":["orders:read"]},
])
def test_scope_and_role_permission_both_required_for_writes(oauth_api, claims):
    api = oauth_api
    result = call(api,"create_manual_order_draft",token=api.token(claims))
    assert result["isError"] and result["structuredContent"]["error"]["code"] == "INSUFFICIENT_SCOPE"
    challenge = result["_meta"]["mcp/www_authenticate"][0]
    assert 'error="insufficient_scope"' in challenge and 'scope="orders:read orders:write"' in challenge
    assert call(api,"list_manual_orders")["structuredContent"]["data"]["total"] == 0


def test_oauth_write_audit_identity_idempotency_and_readonly(oauth_api, monkeypatch):
    api = oauth_api
    draft = dict(client_name="OAuth test",order_number="OAUTH-1",order_date="2026-09-01",idempotency_key="oauth-create-001",
                 rows=[dict(width_mm=1000,height_mm=600,quantity=1,glass_type="4F")])
    first = call(api,"create_manual_order_draft",draft)["structuredContent"]["data"]
    assert call(api,"create_manual_order_draft",draft)["structuredContent"]["data"] == first
    second = call(api,"create_manual_order_draft",{**draft,"order_number":"OAUTH-2"},api.token({"sub":"auth0|second-operator"}))["structuredContent"]["data"]
    assert second["order_id"] != first["order_id"]
    from sqlalchemy import select
    db = api.module.db_module
    with db.read_session() as session:
        actors = set(session.scalars(select(db.McpAudit.actor)))
    assert len(actors) == 2
    assert "oauth:"+hashlib.sha256((ISSUER+"\0"+USER).encode()).hexdigest() in actors
    assert all(USER not in actor for actor in actors)
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_READ_ONLY","true")
    assert call(api,"create_manual_order_draft",draft)["structuredContent"]["error"]["code"] == "FORBIDDEN"


def test_artifact_download_requires_read_permission(oauth_api):
    api = oauth_api
    artifact = api.module.app.state.mcp_service.repo.add_artifact("manual:1","test",b"%PDF-local-test","application/pdf")
    assert api.client.get(artifact["download_path"]).status_code == 401
    no_scope = api.token({"scope":"", "permissions":[]})
    response = api.client.get(artifact["download_path"],headers={"Authorization":"Bearer "+no_scope})
    assert response.status_code == 403 and 'error="insufficient_scope"' in response.headers["WWW-Authenticate"]
    response = api.client.get(artifact["download_path"],headers={"Authorization":"Bearer "+api.token()})
    assert response.status_code == 200 and response.content == b"%PDF-local-test"


def test_key_rotation_unknown_kid_throttle_and_provider_failure(oauth_api, signing_keys, monkeypatch):
    api, clock = oauth_api, [100.0]
    auth = api.auth
    monkeypatch.setattr(auth,"time",SimpleNamespace(monotonic=lambda:clock[0]))
    verifier = auth.OAuthVerifier(auth.OAuthSettings.from_env())
    fetches = []
    async def rotated(url):
        fetches.append(url)
        return {"keys":[item[1] for item in signing_keys]}
    async def scenario():
        await verifier.verify(api.token())
        with pytest.raises(auth.AuthenticationError):
            await verifier.verify(api.token(key_index=1))
        clock[0] += 11
        monkeypatch.setattr(auth,"fetch_public_json",rotated)
        await verifier.verify(api.token(key_index=1))
        assert len(fetches) == 1
        async def unavailable(url):
            raise httpx.ConnectError("Must never appear in a response")
        monkeypatch.setattr(auth,"fetch_public_json",unavailable)
        await verifier.verify(api.token())
        clock[0] += 301
        with pytest.raises(auth.AuthenticationError,match="temporarily unavailable") as error:
            await verifier.verify(api.token())
        assert error.value.status == 503
        with pytest.raises(auth.AuthenticationError) as retry_error:
            await verifier.verify(api.token())
        assert retry_error.value.status == 503
    asyncio.run(scenario())


@pytest.mark.parametrize("setting,value", [
    ("ORDER_EXTRACTOR_MCP_OAUTH_ISSUER","http://issuer.test/"),
    ("ORDER_EXTRACTOR_MCP_OAUTH_ISSUER",ISSUER.rstrip("/")),
    ("ORDER_EXTRACTOR_MCP_RESOURCE_URL",RESOURCE+"?token=invalid"),
    ("ORDER_EXTRACTOR_MCP_RESOURCE_URL","https://user:password@factory.test/mcp"),
    ("ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS",""),
    ("ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS","*"),
    ("ORDER_EXTRACTOR_MCP_AUTH_MODE","invalid"),
])
def test_incomplete_configuration_fails_closed(oauth_api, monkeypatch, setting, value):
    monkeypatch.setenv(setting,value)
    authentication = oauth_api.auth.Authentication()
    with pytest.raises(oauth_api.auth.AuthenticationError) as error:
        authentication.ensure_ready()
    assert error.value.status == 503


def test_public_fetch_is_bounded_and_does_not_follow_redirects(oauth_api, monkeypatch):
    real_client = httpx.AsyncClient
    def client_for(handler):
        def factory(**kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
            return real_client(transport=httpx.MockTransport(handler),**kwargs)
        return factory
    original = oauth_api.original_fetch
    monkeypatch.setattr(httpx,"AsyncClient",client_for(lambda request: httpx.Response(302,headers={"Location":"https://other.test"})))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(original(ISSUER+".well-known/jwks.json"))
    monkeypatch.setattr(httpx,"AsyncClient",client_for(lambda request: httpx.Response(200,content=b"x"*256_001)))
    with pytest.raises(ValueError,match="size limit"):
        asyncio.run(original(ISSUER+".well-known/jwks.json"))


def test_official_sdk_completes_discovery_pkce_and_token_exchange(oauth_api):
    """Real MCP SDK OAuth client against our HTTP boundary and a local fake issuer."""
    import base64
    from urllib.parse import parse_qs, urlsplit
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
    api = oauth_api
    state = {}
    metadata = OAuthClientMetadata(redirect_uris=["http://localhost:49123/callback"],
        client_name="Local OAuth integration check", token_endpoint_auth_method="client_secret_post")
    class Storage:
        tokens = None
        client = OAuthClientInformationFull(**metadata.model_dump(),client_id="chatgpt-test",
                                            client_secret="local-test-client-secret")
        async def get_tokens(self): return self.tokens
        async def set_tokens(self,value): self.tokens = value
        async def get_client_info(self): return self.client
        async def set_client_info(self,value): self.client = value
    storage = Storage()
    async def redirect(url):
        args = parse_qs(urlsplit(url).query)
        assert args["client_id"] == ["chatgpt-test"]
        assert args["resource"] == [RESOURCE]
        assert args["code_challenge_method"] == ["S256"]
        assert args["scope"] == ["orders:read"]
        state.update(args)
    async def callback():
        return "local-one-use-code", state["state"][0]
    async def transport(request):
        if request.url.host == "factory.example.test":
            forwarded = api.client.request(request.method,request.url.raw_path.decode(),
                                           headers=dict(request.headers),content=request.content)
            return httpx.Response(forwarded.status_code,headers=dict(forwarded.headers),content=forwarded.content)
        if request.url.path.startswith("/.well-known/"):
            return httpx.Response(200,json=dict(issuer=ISSUER,authorization_endpoint=ISSUER+"authorize",
                token_endpoint=ISSUER+"oauth/token",jwks_uri=ISSUER+".well-known/jwks.json",
                response_types_supported=["code"],grant_types_supported=["authorization_code"],
                token_endpoint_auth_methods_supported=["client_secret_post"],code_challenge_methods_supported=["S256"]))
        assert str(request.url) == ISSUER+"oauth/token"
        form = parse_qs(request.content.decode())
        assert form["code"] == ["local-one-use-code"]
        assert form["grant_type"] == ["authorization_code"]
        assert form["client_secret"] == ["local-test-client-secret"]
        assert form["resource"] == [RESOURCE]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"][0].encode()).digest()).rstrip(b"=").decode()
        assert challenge == state["code_challenge"][0]
        return httpx.Response(200,json=dict(access_token=api.token({"scope":"orders:read"}),
                                           token_type="Bearer",expires_in=900,scope="orders:read"))
    async def scenario():
        auth = OAuthClientProvider(RESOURCE,metadata,storage,redirect,callback)
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport),auth=auth) as client:
            response = await client.post(RESOURCE,headers={"Accept":"application/json, text/event-stream",
                "MCP-Protocol-Version":"2025-06-18"},json={"jsonrpc":"2.0","id":1,"method":"tools/call",
                "params":{"name":"platform_health","arguments":{}}})
            assert response.status_code == 200
            assert response.json()["result"]["structuredContent"]["data"]["database"] == "ready"
        assert storage.tokens is not None
    asyncio.run(scenario())


def test_provider_preflight_checks_metadata(oauth_api, signing_keys, monkeypatch):
    import mcp_oauth_check
    async def metadata(url):
        if url.endswith("jwks.json"):
            return {"keys":[signing_keys[0][1]]}
        return dict(issuer=ISSUER,authorization_endpoint=ISSUER+"authorize",token_endpoint=ISSUER+"oauth/token",
                    jwks_uri=ISSUER+".well-known/jwks.json",code_challenge_methods_supported=["S256"],
                    response_types_supported=["code"],authorization_response_iss_parameter_supported=True)
    monkeypatch.setattr(mcp_oauth_check,"fetch_public_json",metadata)
    result = asyncio.run(mcp_oauth_check.check_provider(oauth_api.auth.OAuthSettings.from_env()))
    assert result["ok"] and result["issuer_identification"]
