"""Retired-agent application integration; no credentials or production data."""
import os
from pathlib import Path
import sys
import subprocess

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
from test_smoke import _load_app


def test_retired_agent_startup_without_credentials_preserves_health(tmp_path):
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
    assert response.status_code == 404
    assert client.get('/api/features').json()['factory_agent'] is False
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
    # Keep retired Factory Agent cleanup while excluding that worker only.
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
        assert client.get("/api/factory-agent/config").status_code == 404
        response = client.get("/api/factory-agent/config", headers={"X-App-Key": dedicated})
        assert response.status_code == 404
        assert client.get("/api/features").json()["factory_agent"] is False
        assert client.post("/api/factory-agent/sessions", json={}, headers={"X-App-Key": dedicated}).status_code == 404
        # The dedicated key must not accidentally enable the legacy global gate.
        assert client.post("/api/production-sheets/preview", json={"source": source}).status_code == 200
    assert not any(calls.values())



def test_retirement_cleans_existing_remote_without_starting_work(monkeypatch, tmp_path):
    import asyncio
    from fastapi import FastAPI
    from factory_agent import install_factory_agent_cleanup
    from test_factory_agent_lifecycle import FakeProvider, service, reserve

    monkeypatch.setenv('ENABLE_FACTORY_AGENT', 'true')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-key')
    directory = tmp_path / 'factory-agent'
    monkeypatch.setenv('FACTORY_AGENT_STATE_DIR', str(directory))
    fake = FakeProvider()

    async def seed():
        original = service(directory, fake)
        row = await reserve(original)
        row.update(remote_session_id='sess_fixture', phase='created', status='running')
        original.save(row)
        original.janitor.cancel()
        await asyncio.gather(original.janitor, return_exceptions=True)
        original.store.close()
        return row['id']

    identity = asyncio.run(seed())
    app = FastAPI()
    retired = install_factory_agent_cleanup(app, lambda: '')
    retired.provider_factory = lambda: fake
    # Cleanup does not require the obsolete feature flag or operator access key.
    monkeypatch.setenv('ENABLE_FACTORY_AGENT', 'false')
    with TestClient(app) as client:
        assert client.post('/api/factory-agent/sessions', json={}).status_code == 404
        import time
        deadline = time.monotonic() + 2
        while retired.records[identity]['cleanup_status'] != 'deleted' and time.monotonic() < deadline:
            time.sleep(.01)
        assert retired.records[identity]['cleanup_status'] == 'deleted'
        assert retired.records[identity]['stop_requested'] is True
        assert fake.events('agent.session.input.cancel')
        assert not fake.events('agent.session.input.message')
        assert not fake.events('agent.session.input.tool_result')
        assert not any(call[0] == 'create' for call in fake.calls)
    assert (directory / 'sessions.sqlite3').is_file()
