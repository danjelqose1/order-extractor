from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import production_sheet_voice as voice


@pytest.fixture(autouse=True)
def clean_registry():
    voice._sessions.clear()
    voice._starts.clear()
    voice._pending = 0
    yield
    voice._sessions.clear()
    voice._starts.clear()


def context(proposal=""):
    return voice.VoiceContext(title="Mother Sheet – Ëldi", source_digest="a" * 64,
                              row_count=16, piece_count=27, settings={}, proposal=proposal)


def offer():
    return voice.VoiceOffer(sdp="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", context=context())


def fake_client(decision=None):
    calls, options = [], []
    client = SimpleNamespace()
    client.with_options = lambda **kwargs: (options.append(kwargs) or client)
    client.post = lambda path, **kwargs: (calls.append((path, kwargs)) or
                                        {"session": {"id": "live_test"}, "transport": {"sdp": "answer"}})
    client.responses = SimpleNamespace(create=lambda **kwargs: (calls.append(kwargs) or
        SimpleNamespace(status="completed", output_text=json.dumps(decision, ensure_ascii=False))))
    return client, calls, options


def test_installed_sdk_can_create_and_hangup_with_exact_live_rest_contract():
    requests = []
    def transport(request):
        requests.append(request)
        if request.url.path.endswith("/hangup"):
            return httpx.Response(200)
        return httpx.Response(201, json={"session": {"id": "live_test"}, "transport": {"type": "webrtc", "sdp": "answer"}})
    client = OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    result = voice.create_session(client, offer(), "test")
    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/live/sessions"
    assert payload["session"]["model"] == "gpt-live-1"
    assert payload["session"]["delegation"] == {"type": "client"}
    assert payload["session"]["store"] is False
    assert "Ëldi" in payload["session"]["input"][0]["content"][0]["text"]
    assert "Albanian" in payload["session"]["instructions"] and "user's language" in payload["session"]["instructions"]
    assert set(result) == {"session_id", "sdp", "token"}
    assert voice.close_session(client, voice.VoiceOwnership(**{k: result[k] for k in ["session_id", "token"]})) == {"ok": True}
    assert requests[1].url.path == "/v1/live/sessions/live_test/hangup"
    assert not requests[1].content and not voice._sessions


def test_only_owner_can_close_or_route_a_conversation():
    client, calls, _ = fake_client()
    session = voice.create_session(client, offer(), "test")
    before = len(calls)
    bad = voice.VoiceOwnership(session_id=session["session_id"], token="x" * 43)
    with pytest.raises(PermissionError): voice.close_session(client, bad)
    with pytest.raises(PermissionError): voice.decide_turn(client, voice.VoiceTurn(**bad.model_dump(), transcript="USER: Po", context=context()))
    assert len(calls) == before


@pytest.mark.parametrize("language,text", [("sq", "USER: Rrite shkrimin dhe hapësirën pas llojit të xhamit."),
                                          ("it", "USER: Ingrandisci il testo."), ("de", "USER: Mach die Schrift größer.")])
def test_multilingual_requests_use_sol_medium_with_presentation_only_schema(language, text, monkeypatch):
    monkeypatch.delenv("PRODUCTION_SHEET_MODEL", raising=False)
    client, calls, options = fake_client({"action": "propose", "instruction": text, "reply": "Po."})
    session = voice.create_session(client, offer(), "test")
    decision = voice.decide_turn(client, voice.VoiceTurn(session_id=session["session_id"], token=session["token"], transcript=text, context=context()))
    call = calls[-1]
    assert decision["instruction"] == text and text in call["input"]
    assert call["model"] == "gpt-6.1-sol" and call["reasoning"] == {"effort": "medium"}
    assert call["store"] is False and options[-1]["max_retries"] == 0
    schema = call["text"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"action", "instruction", "reply"}
    assert "explicit USER approval" in call["instructions"] and "Never change dimensions" in call["instructions"]


@pytest.mark.parametrize("action", ["apply", "discard"])
def test_review_actions_require_a_displayed_proposal(action):
    client, _, _ = fake_client({"action": action, "instruction": "", "reply": "Po."})
    session = voice.create_session(client, offer(), "test")
    request = voice.VoiceTurn(session_id=session["session_id"], token=session["token"], transcript="USER: Po, aplikoje.", context=context())
    with pytest.raises(ValueError, match="no pending proposal"): voice.decide_turn(client, request)
    request.context.proposal = "Displayed layout proposal"
    assert voice.decide_turn(client, request)["action"] == action


def test_invalid_or_oversized_requests_cannot_override_model_or_reach_openai():
    with pytest.raises(ValidationError): voice.VoiceOffer(**offer().model_dump(), api_key="secret")
    with pytest.raises(ValidationError): voice.VoiceOwnership(session_id="../other", token="x" * 43)
    with pytest.raises(ValidationError): voice.VoiceDecision(action="change_dimensions", instruction="", reply="No")
    client, calls, _ = fake_client()
    request = offer(); request.sdp = "not an audio offer at all"
    with pytest.raises(ValueError): voice.create_session(client, request, "test")
    assert calls == []


def test_rate_limit_and_failed_creation_release_reserved_capacity():
    client, _, _ = fake_client()
    client.post = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private upstream"))
    for _ in range(30):
        with pytest.raises(RuntimeError): voice.create_session(client, offer(), "test")
    assert voice._pending == 0 and not voice._sessions
    with pytest.raises(ValueError, match="busy"): voice.create_session(client, offer(), "test")


def test_voice_routes_require_allowed_origin_and_existing_app_key_and_hide_errors(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_smoke import _load_app
    from fastapi.testclient import TestClient
    app_module, writes = _load_app(monkeypatch)
    app_module.APP_KEY = "test-app-key"
    client = TestClient(app_module.app)
    path = "/api/production-sheets/voice/session"
    headers = {"Origin": "https://danjelqose1.github.io", "X-App-Key": "test-app-key"}
    assert client.post(path, json=offer().model_dump()).status_code == 401
    assert client.post(path, json=offer().model_dump(), headers={**headers, "Origin": "https://other.example"}).status_code == 403
    app_module.get_client = lambda: object()
    app_module.create_session = lambda *_: (_ for _ in ()).throw(RuntimeError("private server key"))
    response = client.post(path, json=offer().model_dump(), headers=headers)
    assert response.status_code == 502 and "private" not in response.text
    assert not writes["update_order_rows"] and not writes["update_order_status"] and not writes["insert_extraction_with_rows"]
