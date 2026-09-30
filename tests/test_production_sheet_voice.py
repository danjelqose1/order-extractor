from __future__ import annotations

import json
import sys
from email.parser import BytesParser
from email.policy import default
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


def offer():
    return voice.VoiceOffer(sdp="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n")


def fake_client():
    calls, options = [], []
    client = SimpleNamespace()
    client.with_options = lambda **kwargs: (options.append(kwargs) or client)
    def post(path, **kwargs):
        calls.append((path, kwargs))
        if path.endswith("/hangup"):
            return None
        return httpx.Response(201, headers={"Location": "/v1/realtime/calls/rtc_test"},
                              text="v=0\r\nm=audio 9 answer\r\n")
    client.post = post
    return client, calls, options


def test_installed_sdk_sends_transcription_only_multipart_and_owned_hangup():
    requests = []
    def transport(request):
        requests.append(request)
        if request.url.path.endswith("/hangup"):
            return httpx.Response(200)
        return httpx.Response(201, headers={"Location": "/v1/realtime/calls/rtc_test",
                                           "Content-Type": "application/sdp"},
                              text="v=0\r\nm=audio 9 answer\r\n")
    client = OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    result = voice.create_session(client, offer(), "test")
    assert requests[0].url.path == "/v1/realtime/calls"
    envelope = BytesParser(policy=default).parsebytes(
        ("Content-Type: " + requests[0].headers["Content-Type"] + "\r\n\r\n").encode() + requests[0].content)
    parts = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
             for part in envelope.iter_parts()}
    assert parts["sdp"] == offer().sdp
    payload = json.loads(parts["session"])
    assert payload["type"] == "transcription"
    audio = payload["audio"]
    assert set(audio) == {"input"} and audio["input"]["turn_detection"] is None
    config = audio["input"]["transcription"]
    assert config["model"] == "gpt-live-transcribe" and config["delay"] == "medium"
    assert "switch languages" in config["prompt"]
    assert "languages" not in config  # No English-only or Albanian-only restriction.
    assert not any(key in payload for key in ["instructions", "tools", "delegation", "input", "model"])
    assert set(result) == {"session_id", "sdp", "token", "model"}
    assert result["model"] == "gpt-live-transcribe"
    assert voice.close_session(client, voice.VoiceOwnership(**{key: result[key] for key in ["session_id", "token"]})) == {"ok": True}
    assert requests[1].url.path == "/v1/realtime/calls/rtc_test/hangup"
    assert not requests[1].content and not voice._sessions


def test_only_owner_can_close_dictation():
    client, calls, _ = fake_client()
    session = voice.create_session(client, offer(), "test")
    before = len(calls)
    with pytest.raises(PermissionError):
        voice.close_session(client, voice.VoiceOwnership(session_id=session["session_id"], token="x" * 43))
    assert len(calls) == before


def test_invalid_or_oversized_requests_cannot_override_model_or_reach_openai():
    with pytest.raises(ValidationError):
        voice.VoiceOffer(**offer().model_dump(), model="gpt-live-1")
    with pytest.raises(ValidationError):
        voice.VoiceOffer(**offer().model_dump(), context={"title": "Private order"})
    with pytest.raises(ValidationError):
        voice.VoiceOwnership(session_id="../other", token="x" * 43)
    client, calls, _ = fake_client()
    request = offer(); request.sdp = "not an audio offer at all"
    with pytest.raises(ValueError):
        voice.create_session(client, request, "test")
    assert calls == []


def test_rate_limit_and_failed_creation_release_reserved_capacity():
    client, _, _ = fake_client()
    client.post = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private upstream"))
    for _ in range(30):
        with pytest.raises(RuntimeError):
            voice.create_session(client, offer(), "test")
    assert voice._pending == 0 and not voice._sessions
    with pytest.raises(ValueError, match="busy"):
        voice.create_session(client, offer(), "test")


def test_malformed_answer_is_hung_up_and_does_not_reserve_a_session():
    client, calls, _ = fake_client()
    post = client.post
    def malformed(path, **kwargs):
        result = post(path, **kwargs)
        return httpx.Response(201, headers={"Location": "/v1/realtime/calls/rtc_test"}, text="bad SDP") if result else None
    client.post = malformed
    with pytest.raises(RuntimeError, match="incomplete"):
        voice.create_session(client, offer(), "test")
    assert calls[-1][0].endswith("/rtc_test/hangup")
    assert voice._pending == 0 and not voice._sessions


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
    assert client.post("/api/production-sheets/voice/turn", json={"transcript": "apply"}, headers=headers).status_code == 410
    assert not writes["update_order_rows"] and not writes["update_order_status"] and not writes["insert_extraction_with_rows"]
