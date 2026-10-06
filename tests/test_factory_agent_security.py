"""Security boundaries using an isolated FastAPI app and no production modules."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from factory_agent import FactoryAgentService, MAX_SCREENSHOT, StartTask, StateStore, install_factory_agent  # noqa: E402
from factory_agent_fixture import FIXTURE_ORDER_ID  # noqa: E402
from factory_agent_provider import ProviderError  # noqa: E402


KEY = "factory-agent-security-test-key"
SESSION_ID = "f0c466c4-0ba2-4c4e-b758-d4d5bfa00597"


@pytest.fixture
def protected_app(monkeypatch):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = FastAPI()
    service = install_factory_agent(app, lambda: KEY, lambda: {"https://factory.example.test"})
    with TestClient(app) as client:
        yield client, service


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/factory-agent/config"),
    ("GET", "/api/factory-agent/sessions"),
    ("POST", "/api/factory-agent/sessions"),
    ("GET", f"/api/factory-agent/sessions/{SESSION_ID}"),
    ("POST", f"/api/factory-agent/sessions/{SESSION_ID}/stop"),
    ("DELETE", f"/api/factory-agent/sessions/{SESSION_ID}"),
])
def test_every_session_route_requires_existing_authentication(protected_app, method, path):
    client, _ = protected_app
    body = {"request_id": str(uuid.uuid4()), "order_id": FIXTURE_ORDER_ID, "message": "Read the test order"}
    assert client.request(method, path, json=body).status_code == 401
    assert client.request(method, path, json=body, headers={"X-App-Key": "incorrect"}).status_code == 401


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_FACTORY_AGENT", raising=False)
    app = FastAPI()
    service = install_factory_agent(app, lambda: KEY, lambda: set())
    with TestClient(app) as client:
        assert client.get("/api/factory-agent/config", headers={"X-App-Key": KEY}).status_code == 404
    assert service.store is None and service.provider is None


def test_missing_application_key_cannot_make_beta_public(monkeypatch):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FACTORY_AGENT_ACCESS_KEY", raising=False)
    app = FastAPI()
    install_factory_agent(app, lambda: "", lambda: set())
    with TestClient(app) as client:
        response = client.get("/api/factory-agent/config")
        assert response.status_code == 503
        assert response.json()["detail"]["missing"] == ["FACTORY_AGENT_ACCESS_KEY"]


@pytest.mark.parametrize("dedicated", ["", "too-short", "a" * 4097, "x" * 31 + " ", "x" * 31 + "é", "x" * 31 + "\n"])
def test_invalid_dedicated_access_key_fails_closed(monkeypatch, dedicated):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("FACTORY_AGENT_ACCESS_KEY", dedicated)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = FastAPI()
    service = install_factory_agent(app, lambda: "", lambda: set())
    with TestClient(app) as client:
        response = client.get("/api/factory-agent/config")
        assert response.status_code == 503
        assert response.json()["detail"]["missing"] == ["FACTORY_AGENT_ACCESS_KEY"]
        assert service.configuration()["ready"] is False
        assert service.store is None


def test_dedicated_access_key_uses_existing_header_without_changing_global_auth(monkeypatch):
    dedicated = "isolated-beta-operator-test-secret-1234"
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("FACTORY_AGENT_ACCESS_KEY", dedicated)
    monkeypatch.delenv("APP_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = FastAPI()
    service = install_factory_agent(app, lambda: None, lambda: set())
    with TestClient(app) as client:
        assert client.get("/api/factory-agent/config").status_code == 401
        assert client.get("/api/factory-agent/config", headers={"Authorization": "Bearer " + dedicated}).status_code == 401
        assert client.get("/api/factory-agent/config", params={"app_key": dedicated}).status_code == 401
        response = client.get("/api/factory-agent/config", headers={"X-App-Key": dedicated})
        assert response.status_code == 200
        assert response.json()["missing"] == ["OPENAI_API_KEY"]
        assert dedicated not in response.text
        assert service.app_key_getter() == dedicated
    import os
    assert os.getenv("APP_KEY") is None


def test_existing_global_access_key_takes_precedence(monkeypatch):
    dedicated = "isolated-beta-operator-test-secret-1234"
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("FACTORY_AGENT_ACCESS_KEY", dedicated)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = FastAPI()
    install_factory_agent(app, lambda: KEY, lambda: set())
    with TestClient(app) as client:
        assert client.get("/api/factory-agent/config", headers={"X-App-Key": dedicated}).status_code == 401
        assert client.get("/api/factory-agent/config", headers={"X-App-Key": KEY}).status_code == 200


def test_openai_api_key_is_never_used_as_dedicated_operator_credential(monkeypatch):
    provider_key = "sk-provider-secret-must-stay-on-server-1234"
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("FACTORY_AGENT_ACCESS_KEY", provider_key)
    monkeypatch.setenv("OPENAI_API_KEY", provider_key)
    app = FastAPI()
    service = install_factory_agent(app, lambda: None, lambda: set())
    with TestClient(app) as client:
        assert client.get("/api/factory-agent/config", headers={"X-App-Key": provider_key}).status_code == 503
        assert service.store is None and service.provider is None


def test_configuration_reports_missing_credential_without_exposing_auth_key(protected_app):
    client, _ = protected_app
    response = client.get("/api/factory-agent/config", headers={"X-App-Key": KEY})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    body = response.json()
    assert body["state"] == "setup_required"
    assert body["ready"] is False
    assert body["missing"] == ["OPENAI_API_KEY"]
    assert body["mode"] == "fixture"
    assert [order["id"] for order in body["orders"]] == [FIXTURE_ORDER_ID]
    assert KEY not in response.text


def test_request_body_is_bounded_before_validation_and_errors_are_not_cached(protected_app):
    client, service = protected_app
    response = client.post("/api/factory-agent/sessions", content=b"x" * 16385, headers={"X-App-Key": KEY})
    assert response.status_code == 413
    assert response.headers["Cache-Control"] == "no-store"
    assert service.records == {}
    unauthorized = client.get("/api/factory-agent/config")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["Cache-Control"] == "no-store"
    malformed = client.post("/api/factory-agent/sessions", content=b"{", headers={"X-App-Key": KEY, "Content-Type": "application/json"})
    assert malformed.status_code == 422
    assert malformed.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("origin", ["null", "https://evil.invalid", "https://factory.example.test.evil.invalid", "https://localhost.evil.invalid", "http://127.0.0.1.evil.invalid:8765"])
def test_foreign_platform_origins_are_rejected(protected_app, origin):
    client, _ = protected_app
    response = client.get("/api/factory-agent/config", headers={"X-App-Key": KEY, "Origin": origin})
    assert response.status_code == 403


@pytest.mark.parametrize("origin", ["https://factory.example.test", "http://localhost:8000", "http://127.0.0.1:8765"])
def test_platform_and_existing_local_development_origins_are_supported(protected_app, origin):
    client, _ = protected_app
    assert client.get("/api/factory-agent/config", headers={"X-App-Key": KEY, "Origin": origin}).status_code == 200


@pytest.mark.parametrize("method,suffix", [("GET", ""), ("POST", "/stop"), ("DELETE", "")])
def test_session_owner_boundary_covers_reads_stop_and_deletion(protected_app, method, suffix):
    client, service = protected_app
    service.records[SESSION_ID] = {"id": SESSION_ID, "owner": hashlib.sha256(b"another application key").hexdigest()}
    assert client.request(method, f"/api/factory-agent/sessions/{SESSION_ID}{suffix}", headers={"X-App-Key": KEY}).status_code == 404


def test_session_list_omits_large_details_while_owned_detail_remains_available(protected_app):
    client, service = protected_app
    owner = hashlib.sha256(("factory-agent\0" + KEY).encode()).hexdigest()
    service.records[SESSION_ID] = {
        "id": SESSION_ID, "owner": owner, "request_id": str(uuid.uuid4()),
        "order_id": FIXTURE_ORDER_ID, "created_ts": 1, "created_at": "2026-10-06T00:00:00Z",
        "deadline_at": "2026-10-06T00:03:00Z", "status": "completed", "cleanup_status": "deleted",
        "retired": False, "message": "private operator request", "result_text": "saved detailed result",
        "screenshot": "data:image/png;base64,iVBORw0KGgo=", "activity": [{"title": "private activity"}],
    }
    listing = client.get("/api/factory-agent/sessions", headers={"X-App-Key": KEY})
    assert listing.status_code == 200
    summary = listing.json()["sessions"][0]
    assert summary["id"] == SESSION_ID and summary["status"] == "completed"
    assert all(key not in summary for key in ("message", "result_text", "screenshot", "activity", "owner", "input_hash"))
    detail = client.get(f"/api/factory-agent/sessions/{SESSION_ID}", headers={"X-App-Key": KEY}).json()
    assert detail["result_text"] == "saved detailed result"
    assert detail["screenshot"] == "data:image/png;base64,iVBORw0KGgo="
    assert "owner" not in detail and "input_hash" not in detail


@pytest.mark.parametrize("extra", [
    {"order_id": "R-26-0884"},
    {"order_id": 1},
    {"environment": {"network": {"access": "enabled"}}},
    {"agent": {"tools": [{"type": "shell"}]}},
    {"message": "x" * 4001},
])
def test_task_payload_cannot_select_production_or_override_security(protected_app, extra):
    client, service = protected_app
    body = {"request_id": str(uuid.uuid4()), "order_id": FIXTURE_ORDER_ID, "message": "Read the fixture"}
    body.update(extra)
    response = client.post("/api/factory-agent/sessions", json=body, headers={"X-App-Key": KEY})
    assert response.status_code == 422
    assert service.records == {}


class EventSink:
    def __init__(self):
        self.events = []

    async def send_events(self, session_id, events, idempotency_key=None):
        self.events.extend(events)


def _action_service():
    service = FactoryAgentService(lambda: KEY)
    service.provider = EventSink()
    service.save = lambda row: None
    row = {
        "id": SESSION_ID, "remote_session_id": "asess_test", "order_id": FIXTURE_ORDER_ID,
        "handled_actions": [], "activity": [], "turn_id": "aturn_test", "screenshot": None,
        "stop_requested": False, "deadline": time.time() + 180,
    }
    return service, row


def test_only_selected_fixture_tool_executes_and_denials_return_no_order_data():
    service, row = _action_service()
    actions = [
        {"type": "function_call", "turn_id": "aturn_test", "call_id": "read", "name": "get_selected_order", "arguments": {}},
        {"type": "function_call", "turn_id": "aturn_test", "call_id": "edit", "name": "update_order", "arguments": {"width": 1}},
        {"type": "function_call", "turn_id": "aturn_test", "call_id": "other", "name": "get_selected_order", "arguments": {"order_id": "1"}},
    ]
    asyncio.run(service.handle_actions(row, actions))
    events = service.provider.events
    assert len(events) == 3
    assert events[0]["success"] is True
    assert json.loads(events[0]["output"])["order_id"] == FIXTURE_ORDER_ID
    assert all(event["success"] is False and "output" not in event for event in events[1:])
    asyncio.run(service.handle_actions(row, actions))
    assert len(service.provider.events) == 3, "Re-reading pending actions must not resubmit handled results"


def test_stop_interrupts_action_batch_before_any_further_approvals_or_results():
    service, row = _action_service()

    class StopAfterFirstResult(EventSink):
        async def send_events(self, session_id, events, idempotency_key=None):
            await super().send_events(session_id, events, idempotency_key)
            row["stop_requested"] = True

    service.provider = StopAfterFirstResult()
    actions = [
        {"type": "function_call", "turn_id": "aturn_test", "call_id": "read", "name": "get_selected_order", "arguments": {}},
        {"type": "computer_use_approval_request", "request_id": "approval_after_stop", "request": {"type": "browser_origin_access", "origin": "http://127.0.0.1:8765"}},
    ]
    asyncio.run(service.handle_actions(row, actions))
    assert len(service.provider.events) == 1
    assert row["handled_actions"] == ["read"]


def test_runtime_expiry_prevents_action_servicing():
    service, row = _action_service()
    row["deadline"] = time.time() - 1
    asyncio.run(service.handle_actions(row, [{"type": "function_call", "turn_id": "aturn_test", "call_id": "read", "name": "get_selected_order", "arguments": {}}]))
    assert service.provider.events == []
    assert row["handled_actions"] == []


@pytest.mark.parametrize("origin,decision", [
    ("http://127.0.0.1:8765", "approve"),
    ("http://127.0.0.1:8765/", "deny"),
    ("http://127.0.0.1:80", "deny"),
    ("https://127.0.0.1:8765", "deny"),
    ("http://127.0.0.1.evil.invalid:8765", "deny"),
    ("https://factory.example.test", "deny"),
    ("file:///etc/passwd", "deny"),
])
def test_browser_origin_approval_is_exact_loopback_only(origin, decision):
    service, row = _action_service()
    action = {"type": "computer_use_approval_request", "request_id": "approval_test", "request": {"type": "browser_origin_access", "origin": origin}}
    asyncio.run(service.handle_actions(row, [action]))
    assert service.provider.events[0]["response"] == {"type": "browser_origin_access", "decision": decision}


def test_authentication_requests_cancel_without_collecting_or_sending_credentials():
    service, row = _action_service()
    action = {"type": "computer_use_approval_request", "request_id": "auth_test", "request": {"type": "browser_authentication", "origin": "https://factory.example.test"}}
    asyncio.run(service.handle_actions(row, [action]))
    assert service.provider.events[0]["response"] == {"type": "browser_authentication", "action": "cancel"}


@pytest.mark.parametrize("action", [
    {"type": "computer_use_approval_request", "request_id": "unknown", "request": {"type": "new_sensitive_action"}},
    {"type": "connect_environment", "request_id": "unknown"},
])
def test_unknown_approval_types_fail_closed(action):
    service, row = _action_service()
    with pytest.raises(RuntimeError):
        asyncio.run(service.handle_actions(row, [action]))
    assert service.provider.events == []


@pytest.mark.parametrize("image", [
    "https://evil.invalid/screenshot.png",
    "data:text/html;base64,PHNjcmlwdD4=",
    "data:image/svg+xml;base64,PHN2Zz4=",
    "data:image/png;base64,<script>",
    "data:image/png;base64," + "A" * MAX_SCREENSHOT,
])
def test_screenshot_cannot_trigger_remote_fetch_or_active_content(image):
    service, row = _action_service()
    service.collect_items(row, [{"id": "browser1", "type": "computer_use_call", "turn_id": row["turn_id"], "output": [{"type": "computer_screenshot", "image_url": image}]}])
    assert row["screenshot"] is None


def test_only_current_turn_saved_output_is_displayed():
    service, row = _action_service()
    image = "data:image/png;base64,iVBORw0KGgo="
    items = [
        {"id": "browser1", "type": "computer_use_call", "turn_id": row["turn_id"], "status": "completed", "output": [{"type": "computer_screenshot", "image_url": image}]},
        {"type": "message", "turn_id": "old_turn", "role": "assistant", "content": [{"type": "output_text", "text": "Unrelated result"}]},
        {"type": "message", "turn_id": row["turn_id"], "role": "user", "content": [{"type": "output_text", "text": "User input is not an agent result"}]},
        {"type": "message", "turn_id": row["turn_id"], "role": "assistant", "content": [{"type": "output_text", "text": "Saved agent result"}]},
    ]
    service.collect_items(row, items)
    assert row["screenshot"] == image
    assert row["result_text"] == "Saved agent result"


def test_hosted_environment_security_configuration_is_not_user_controlled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-never-in-sandbox")
    monkeypatch.setenv("APP_KEY", KEY)
    service, row = _action_service()
    payload = service.session_payload(row)
    environment = payload["environment"]
    assert environment["type"] == "openai_hosted"
    assert environment["desktop"] == {"enabled": True}
    assert environment["network"] == {"access": "disabled"}
    assert "env" not in environment and "environment_template_id" not in environment
    assert payload["agent"]["multi_agent"] == {"enabled": False}
    assert [tool.get("name") for tool in payload["agent"]["tools"] if tool["type"] == "function"] == ["get_selected_order"]
    assert "secret-never-in-sandbox" not in json.dumps(payload)
    assert KEY not in json.dumps(payload)


@pytest.mark.parametrize("invalid_key", [" ", "invalid-é", "invalid\nsecret"])
def test_invalid_provider_credentials_leave_safe_setup_required_state(tmp_path, monkeypatch, invalid_key):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("OPENAI_API_KEY", invalid_key)

    def reject_provider():
        raise ProviderError(503, "setup_required", "Set OPENAI_API_KEY on the backend to enable Factory Agent.")

    service = FactoryAgentService(lambda: KEY, provider_factory=reject_provider, directory=tmp_path)
    try:
        asyncio.run(service.start())
        assert service.configuration()["ready"] is False
        assert service.configuration()["state"] == "setup_required"
        assert service.store is None and service.provider is None
    finally:
        if service.store is not None:
            service.store.close()


def _journal_row(timestamp, *, active=False):
    message = "Read this selected fixture and flag ambiguity"
    digest = hashlib.sha256(json.dumps({"order_id": FIXTURE_ORDER_ID, "message": message}, sort_keys=True).encode()).hexdigest()
    return {
        "id": str(uuid.uuid4()), "owner": hashlib.sha256(("factory-agent\0" + KEY).encode()).hexdigest(),
        "request_id": str(uuid.uuid4()), "input_hash": digest, "order_id": FIXTURE_ORDER_ID,
        "message": message, "created_ts": timestamp, "created_at": "2026-10-06T00:00:00Z",
        "deadline": timestamp + 180, "deadline_at": "2026-10-06T00:03:00Z",
        "status": "running" if active else "completed", "phase": "input_attempted",
        "remote_session_id": "asess_test", "turn_id": "aturn_test", "activity": [{"title": "private activity"}],
        "result_text": "private result", "screenshot": "data:image/png;base64,iVBORw0KGgo=", "error": None,
        "cleanup_status": "pending" if active else "deleted", "stop_requested": False,
        "timeout_requested": False, "input_attempted": True, "handled_actions": [], "retired": False,
    }


def test_retention_caps_results_preserves_active_tasks_and_persistent_request_tombstones(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    now = time.time()
    service = FactoryAgentService(lambda: KEY)
    service.store = StateStore(tmp_path)
    finished = [_journal_row(now - index) for index in range(25)]
    expired = _journal_row(now - 86401)
    active = _journal_row(now - 86402, active=True)
    original_message = expired["message"]
    for row in [*finished, expired, active]:
        service.records[row["id"]] = row
        service.save(row)
    try:
        service.prune()
        retained = [row for row in service.records.values() if row["cleanup_status"] == "deleted" and not row["retired"]]
        assert len(retained) == 20
        assert {row["id"] for row in retained} == {row["id"] for row in finished[:20]}
        assert active["retired"] is False
        assert active["screenshot"] is not None and active["result_text"] == "private result"
        reloaded = service.store.load()
        for row in [*finished[20:], expired]:
            tombstone = reloaded[row["id"]]
            assert tombstone["retired"] is True
            assert tombstone["message"] == tombstone["result_text"] == ""
            assert tombstone["screenshot"] is None and tombstone["activity"] == []
            assert tombstone["request_id"] == row["request_id"]
            assert tombstone["input_hash"] == row["input_hash"]
            assert tombstone["owner"] == row["owner"]
        replay = asyncio.run(service.create(expired["owner"], StartTask(request_id=expired["request_id"], order_id=FIXTURE_ORDER_ID, message=original_message)))
        assert replay["id"] == expired["id"] and replay["retired"] is True
        assert service.tasks == {}, "A retired request must never create a duplicate remote session"
    finally:
        service.store.close()


@pytest.mark.parametrize("payload", ["{broken", json.dumps({"id": SESSION_ID}), "null"])
def test_corrupt_journal_fails_as_setup_required_without_destroying_recovery_evidence(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    store = StateStore(tmp_path)
    with store.db:
        store.db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (SESSION_ID, "owner", "request", payload))
    store.close()
    service = FactoryAgentService(lambda: KEY, directory=tmp_path)
    asyncio.run(service.start())
    assert service.configuration()["ready"] is False
    assert service.configuration()["state"] == "setup_required"
    assert service.store is None and service.provider is None and service.tasks == {}
    # Corruption never justifies dropping data that may identify a remote session.
    recovered = StateStore(tmp_path)
    try:
        assert recovered.db.execute("SELECT payload FROM sessions WHERE id=?", (SESSION_ID,)).fetchone()[0] == payload
    finally:
        recovered.close()
