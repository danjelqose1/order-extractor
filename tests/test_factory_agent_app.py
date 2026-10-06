"""Integration through the existing FastAPI app; factory persistence is replaced
by the established smoke-test fake DB, and OpenAI by an HTTP transport fixture.
"""
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import uuid

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
from test_smoke import _load_app


def test_real_app_startup_without_api_credentials_uses_setup_state(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, ORDER_EXTRACTOR_LOAD_DOTENV="false", DB_DIR=str(tmp_path),
                       ENABLE_FACTORY_AGENT="true", OPENAI_API_KEY="", APP_KEY="", FACTORY_AGENT_ACCESS_KEY="",
                       ORDER_EXTRACTOR_MCP_ENABLED="false")
    script = '''
import sys
sys.path.insert(0, 'backend')
from fastapi.testclient import TestClient
from app import app
import llm
with TestClient(app) as client:
    assert client.get('/healthz').json() == {'ok': True}
    response = client.get('/api/factory-agent/config')
    assert response.status_code == 503
    assert response.json()['detail']['missing'] == ['FACTORY_AGENT_ACCESS_KEY']
    assert app.state.factory_agent.store is None
try:
    llm.get_client()
except RuntimeError as exc:
    assert 'OPENAI_API_KEY is not set' in str(exc)
else:
    raise AssertionError('Extraction must still fail without credentials')
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=environment,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_default_off_adds_no_resources_and_preserves_health(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDER_EXTRACTOR_LOAD_DOTENV", "false")
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    monkeypatch.delenv("ENABLE_FACTORY_AGENT", raising=False)
    app_module, calls = _load_app(monkeypatch)
    # The established fake DB omits unrelated workspace/Telegram background APIs.
    # Keep Factory Agent's real startup/shutdown while excluding that worker only.
    app_module.app.router.on_startup.remove(app_module.load_workspace_agent_modules)
    with TestClient(app_module.app) as client:
        assert client.get("/healthz").json() == {"ok": True}
        assert client.get("/api/features").json()["factory_agent"] is False
        assert client.get("/api/features").json()["living_dashboard"] is False
        assert client.get("/api/factory-agent/config").status_code == 404
        assert app_module.app.state.factory_agent.store is None
        assert not (tmp_path / "factory-agent").exists()
    assert not any(calls.values())


def test_dedicated_beta_key_preserves_legacy_unauthenticated_routes(monkeypatch, tmp_path):
    dedicated = "isolated-beta-operator-test-secret-1234"
    monkeypatch.setenv("ORDER_EXTRACTOR_LOAD_DOTENV", "false")
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("FACTORY_AGENT_ACCESS_KEY", dedicated)
    monkeypatch.delenv("APP_KEY", raising=False)
    app_module, calls = _load_app(monkeypatch)
    app_module.app.router.on_startup.remove(app_module.load_workspace_agent_modules)
    source = {"text": "Mother Sheet\n4F\n1 – 400 × 1200 × 2", "glass_headers": ["4F"],
              "row_count": 1, "piece_count": 2}
    with TestClient(app_module.app) as client:
        assert app_module.APP_KEY is None
        assert client.get("/api/factory-agent/config").status_code == 401
        response = client.get("/api/factory-agent/config", headers={"X-App-Key": dedicated})
        assert response.status_code == 200 and response.json()["ready"] is True
        # The dedicated key must not accidentally enable the legacy global gate.
        assert client.post("/api/production-sheets/preview", json={"source": source}).status_code == 200
    assert not any(calls.values())


def test_existing_app_auth_to_real_provider_adapter_read_report_cleanup(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDER_EXTRACTOR_LOAD_DOTENV", "false")
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("APP_KEY", "isolated-application-test-key")
    app_module, calls = _load_app(monkeypatch)
    app_module.app.router.on_startup.remove(app_module.load_workspace_agent_modules)
    from factory_agent_provider import FactoryAgentProvider
    from factory_agent_fixture import load_order, FIXTURE_ORDER_ID
    service = app_module.app.state.factory_agent
    service.poll_seconds = 0.01
    state = {"input": 0, "create": 0, "delete": 0, "events": [], "metadata": {}, "read": False, "origin": False}
    source = load_order(FIXTURE_ORDER_ID)
    report = "Synthetic fixture only; no customer PDF checked. Client: " + source["client"] + "\n"
    report += "\n".join(" | ".join("Missing" if item[field] is None else item[field] for field in ("index_number", "position", "glass_type", "width", "height", "quantity")) for item in source["items"])
    report += "\nAmbiguity: repeated A-01; alternative width 975/995; missing height, position and glass; uncertain quantity 2?."

    def transport(request):
        assert request.url.host == "api.openai.com"
        assert request.headers["OpenAI-Beta"] == "agents=v1"
        path = request.url.path
        if request.method == "POST" and path == "/v1/agents/sessions":
            body = json.loads(request.content)
            assert body["environment"]["network"] == {"access": "disabled"}
            assert "test-key" not in request.content.decode()
            assert "isolated-application-test-key" not in request.content.decode()
            state["create"] += 1
            state["metadata"] = body["metadata"]
            return httpx.Response(200, json={"id": "asess_fixture", "status": "idle", "environment": {"id": "aenv_fixture"}})
        if path == "/v1/agents/environments/aenv_fixture":
            return httpx.Response(200, json={"id": "aenv_fixture", "status": "connected"})
        if request.method == "POST" and path.endswith("/events"):
            event = json.loads(request.content)["events"][0]
            state["events"].append(event)
            if event["type"] == "agent.session.input.message":
                state["input"] += 1
                assert request.headers.get("Idempotency-Key")
            elif event["type"] == "agent.session.input.tool_result":
                assert event["success"] is True
                assert json.loads(event["output"]) == source
                state["read"] = True
            elif event["type"] == "agent.session.input.computer_use_approval_request_result":
                assert event["response"] == {"type": "browser_origin_access", "decision": "approve"}
                state["origin"] = True
            return httpx.Response(202, json={})
        done = state["read"] and state["origin"]
        if path.endswith("/turns"):
            return httpx.Response(200, json={"data": [{"id": "aturn_fixture", "subagent_id": None, "status": "completed" if done else "waiting"}], "has_more": False})
        if path.endswith("/items"):
            items = [{"id": "browser_fixture", "turn_id": "aturn_fixture", "type": "computer_use_call", "title": "Read fixture page (transport fixture)", "status": "completed" if done else "in_progress", "output": None}]
            if done:
                items.append({"id": "message_fixture", "turn_id": "aturn_fixture", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": report}]})
            return httpx.Response(200, json={"data": items, "has_more": False})
        if request.method == "DELETE":
            state["delete"] += 1
            return httpx.Response(200, json={"id": "asess_fixture", "deleted": True, "object": "agent.session.deleted"})
        if path == "/v1/agents/sessions/asess_fixture":
            actions = [] if not state["input"] or done else [
                {"type": "function_call", "turn_id": "aturn_fixture", "call_id": "call_read", "name": "get_selected_order", "arguments": {}},
                {"type": "computer_use_approval_request", "request_id": "approval_origin", "request": {"type": "browser_origin_access", "origin": "http://127.0.0.1:8765"}},
            ]
            return httpx.Response(200, json={"id": "asess_fixture", "status": "requires_action" if actions else "idle", "environment": {"id": "aenv_fixture"}, "required_actions": actions, "metadata": state["metadata"]})
        raise AssertionError(f"Unexpected provider request: {request.method} {path}")

    service.provider_factory = lambda: FactoryAgentProvider("test-key", transport=httpx.MockTransport(transport))
    with TestClient(app_module.app) as client:
        assert client.get("/api/features").json()["factory_agent"] is True
        assert client.get("/api/factory-agent/config").status_code == 401
        headers = {"X-App-Key": "isolated-application-test-key", "Origin": "http://127.0.0.1:5500"}
        assert client.get("/api/factory-agent/config", headers=headers).json()["ready"] is True
        payload = {"request_id": str(uuid.uuid4()), "order_id": FIXTURE_ORDER_ID, "message": "Read only the fixture."}
        result = client.post("/api/factory-agent/sessions", json=payload, headers=headers)
        assert result.status_code == 202
        identity = result.json()["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            detail = client.get(f"/api/factory-agent/sessions/{identity}", headers=headers).json()
            if detail["cleanup_status"] == "deleted":
                break
            time.sleep(0.01)
        assert detail["status"] == "completed" and detail["cleanup_status"] == "deleted"
        assert detail["result_text"] == report
        assert detail["screenshot"] is None
        assert state["create"] == state["input"] == state["delete"] == 1
        recovered = client.post("/api/factory-agent/sessions", json=payload, headers=headers).json()
        assert recovered["id"] == identity and state["input"] == 1
        listing = client.get("/api/factory-agent/sessions", headers=headers)
        assert "result_text" not in listing.json()["sessions"][0]
        assert "screenshot" not in listing.json()["sessions"][0]
        assert listing.headers["Cache-Control"] == "no-store"
        assert client.get("/healthz").json() == {"ok": True}
        assert client.delete(f"/api/factory-agent/sessions/{identity}", headers=headers).status_code == 200
        assert client.get("/api/factory-agent/sessions", headers=headers).json()["sessions"] == []
    assert not any(calls.values())
