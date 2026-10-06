"""HTTP-contract and transport-safety tests; all requests use a local fake."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import factory_agent_provider as provider_module
from factory_agent_provider import FactoryAgentProvider, ProviderError


def run(coroutine):
    return asyncio.run(coroutine)


def test_create_uses_documented_host_header_metadata_and_no_create_retry_key():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(201, json={"id": "sess_test", "status": "idle"})

    async def scenario():
        async with FactoryAgentProvider("test-secret", "proj_example", transport=httpx.MockTransport(handle)) as client:
            return await client.create_session({
                "agent": {"model": "gpt-6-astra", "tools": [{"type": "computer_use", "include_screenshots": True}]},
                "environment": {"type": "openai_hosted", "desktop": {"enabled": True}, "network": {"access": "disabled"}},
                "metadata": {"factory_agent_request_id": "request-123"},
            })

    assert run(scenario())["id"] == "sess_test"
    assert len(calls) == 1
    request = calls[0]
    assert request.url == "https://api.openai.com/v1/agents/sessions"
    assert request.headers["OpenAI-Beta"] == "agents=v1"
    assert request.headers["OpenAI-Project"] == "proj_example"
    assert request.headers["Authorization"] == "Bearer test-secret"
    assert "Idempotency-Key" not in request.headers
    assert "test-secret" not in request.content.decode()
    assert json.loads(request.content)["metadata"]["factory_agent_request_id"] == "request-123"


def test_input_and_real_cancellation_use_events_endpoint():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(202)

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            message = {"type": "agent.session.input.message", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Read the selected fixture."}]}]}
            assert await client.send_events("sess_123", [message], "message-123") == {}
            assert await client.send_events("sess_123", [{"type": "agent.session.input.cancel"}]) == {}

    run(scenario())
    assert [request.method for request in calls] == ["POST", "POST"]
    assert calls[0].headers["Idempotency-Key"] == "message-123"
    assert json.loads(calls[1].content) == {"events": [{"type": "agent.session.input.cancel"}]}
    assert all(request.url.path == "/v1/agents/sessions/sess_123/events" for request in calls)


def test_browser_approval_auth_cancel_and_function_result_contracts():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(202)

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            await client.send_events("sess_123", [
                {"type": "agent.session.input.computer_use_approval_request_result", "request_id": "req_origin", "response": {"type": "browser_origin_access", "decision": "deny"}},
                {"type": "agent.session.input.computer_use_approval_request_result", "request_id": "req_auth", "response": {"type": "browser_authentication", "action": "cancel"}},
                {"type": "agent.session.input.tool_result", "turn_id": "turn_123", "call_id": "call_123", "success": True, "output": "{\"order_id\":\"fixture\"}"},
            ])

    run(scenario())
    events = calls[0]["events"]
    assert "turn_id" not in events[0] and "turn_id" not in events[1]
    assert events[2]["call_id"] == "call_123"


def test_history_paginates_and_keeps_screenshot_and_turn_result():
    calls = []

    def handle(request):
        calls.append(request)
        if "after" not in request.url.params:
            return httpx.Response(200, json={"data": [{"id": "item_1", "type": "computer_use_call", "output": {"type": "computer_screenshot", "image_url": "data:image/jpeg;base64,aW1hZ2U="}}], "last_id": "item_1", "has_more": True})
        return httpx.Response(200, json={"data": [{"id": "item_2", "type": "message", "content": [{"type": "output_text", "text": "Actual result"}]}], "last_id": "item_2", "has_more": False})

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            return await client.list_items("sess_123")

    items = run(scenario())
    assert len(items) == 2
    assert items[0]["output"]["image_url"].startswith("data:image/jpeg;base64,")
    assert calls[0].url.params["order"] == "asc"
    assert calls[1].url.params["after"] == "item_1"


def test_recovery_inspects_same_session_turns_and_environment_without_posting():
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path.endswith("/turns"):
            return httpx.Response(200, json={"data": [{"id": "turn_123", "status": "cancelled", "subagent_id": None}], "has_more": False})
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json={"data": [{"id": "sess_123", "metadata": {"factory_agent_request_id": "request-123"}}], "has_more": False})
        return httpx.Response(200, json={"id": "sess_123", "status": "idle"})

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            assert (await client.get_session("sess_123"))["status"] == "idle"
            assert (await client.list_turns("sess_123"))[0]["status"] == "cancelled"
            assert (await client.list_sessions())[0]["metadata"]["factory_agent_request_id"] == "request-123"
            await client.get_environment("env_123")

    run(scenario())
    assert all(request.method == "GET" for request in calls)
    assert calls[1].url.params["order"] == "desc"
    assert calls[-1].url.path == "/v1/agents/environments/env_123"


@pytest.mark.parametrize("status,code", [(400, "invalid_request"), (401, "authentication_required"), (403, "permission_required"), (404, "resource_unavailable"), (409, "conflict"), (429, "rate_limited"), (500, "provider_unavailable")])
def test_error_body_never_leaks_and_mutations_are_not_retried(status, code):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "SECRET body api-key password", "code": "SECRET"}})

    async def scenario():
        async with FactoryAgentProvider("SECRET key", transport=httpx.MockTransport(handle)) as client:
            await client.create_session({"environment": {"type": "openai_hosted"}})

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.status == status and caught.value.code == code
    assert "SECRET" not in str(caught.value) and "SECRET" not in repr(caught.value)
    assert caught.value.outcome_unknown is (status >= 500)
    assert len(calls) == 1


def test_timeout_marks_create_outcome_unknown_without_retry():
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret from upstream", request=request)

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            await client.create_session({"environment": {"type": "openai_hosted"}})

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "provider_timeout"
    assert caught.value.outcome_unknown is True
    assert "secret from upstream" not in str(caught.value)
    assert len(calls) == 1


def test_redirect_is_never_followed_with_credentials():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://attacker.invalid"})

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            await client.get_session("sess_123")

    with pytest.raises(ProviderError, match="redirect"):
        run(scenario())
    assert len(calls) == 1


@pytest.mark.parametrize("resource_id", ["../secret", "sess_1?next=bad", "//attacker.invalid", "sess_1%2fother", "sess_1\nAuthorization:x"])
def test_untrusted_resource_ids_cannot_change_request_target(resource_id):
    calls = []

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: calls.append(request))) as client:
            await client.get_session(resource_id)

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "invalid_resource_id"
    assert not calls


@pytest.mark.parametrize("body", [b"not JSON SECRET", b"[]", b""])
def test_invalid_success_body_is_safe_and_unknown_for_create(body):
    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(201, content=body))) as client:
            await client.create_session({})

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "invalid_response"
    assert caught.value.outcome_unknown is True
    assert "SECRET" not in str(caught.value)


def test_response_size_limit_is_enforced():
    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 200))) as client:
            await client._request("GET", "/agents/sessions/sess_123", byte_limit=100)

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "response_too_large"


def test_streamed_response_is_bounded_without_content_length():
    class Chunked(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 100
            yield b"x" * 100

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Chunked()))) as client:
            await client._request("GET", "/agents/sessions/sess_123", byte_limit=100)

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "response_too_large"


def test_total_deadline_covers_slow_response_body(monkeypatch):
    monkeypatch.setattr(provider_module, "REQUEST_DEADLINE_SECONDS", 0.001)

    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield b"{}"

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Slow()))) as client:
            await client.get_session("sess_123")

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "provider_timeout" and not caught.value.outcome_unknown


@pytest.mark.parametrize("key", ["", "  ", "secret\nvalue", "secret\x00value", "secret\u0100value"])
def test_invalid_credentials_fail_without_exposing_value(key):
    with pytest.raises(ProviderError) as caught:
        FactoryAgentProvider(key)
    assert caught.value.code == "setup_required"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("page", [{"data": [], "has_more": True, "last_id": "cursor"}, {"data": [1], "has_more": False}, {"data": [{}]}, {"data": [{}], "has_more": True, "last_id": ""}])
def test_incomplete_history_never_reports_success(page):
    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=page))) as client:
            await client.list_items("sess_123")

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "invalid_response"


def test_page_limit_is_bounded(monkeypatch):
    monkeypatch.setattr(provider_module, "MAX_LIST_PAGES", 2)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"id": str(len(calls))}], "last_id": str(len(calls)), "has_more": True})

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            await client.list_items("sess_123")

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "history_limit" and len(calls) == 2


def test_delete_uses_provider_endpoint_and_propagates_conflict_for_retry_by_owner():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(409)

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(handle)) as client:
            await client.delete_session("sess_123")

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "conflict"
    assert calls[0].method == "DELETE"
    assert calls[0].url.path == "/v1/agents/sessions/sess_123"


def test_delete_requires_confirmation_for_same_session():
    expected = {"id": "sess_123", "deleted": True, "object": "agent.session.deleted"}

    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=expected))) as client:
            assert await client.delete_session("sess_123") == expected

    run(scenario())


@pytest.mark.parametrize("body", [
    {},
    {"id": "sess_other", "deleted": True, "object": "agent.session.deleted"},
    {"id": "sess_123", "deleted": False, "object": "agent.session.deleted"},
    {"id": "sess_123", "deleted": "true", "object": "agent.session.deleted"},
    {"id": "sess_123", "deleted": 1, "object": "agent.session.deleted"},
    {"id": "sess_123", "deleted": True, "object": "agent.session"},
    {"id": "sess_123", "deleted": True},
])
def test_ambiguous_delete_acknowledgment_does_not_claim_cleanup(body):
    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
            await client.delete_session("sess_123")

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "invalid_response"
    assert caught.value.outcome_unknown is True


def test_empty_204_is_not_documented_deletion_confirmation():
    async def scenario():
        async with FactoryAgentProvider("secret", transport=httpx.MockTransport(lambda request: httpx.Response(204))) as client:
            await client.delete_session("sess_123")

    with pytest.raises(ProviderError) as caught:
        run(scenario())
    assert caught.value.code == "invalid_response"
    assert caught.value.outcome_unknown is True
