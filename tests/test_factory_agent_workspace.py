"""Conversation and proposal lifecycle, separate from all production storage."""
import asyncio
import copy
import uuid

from fastapi import HTTPException
import pytest

from test_factory_agent_lifecycle import FakeProvider, service, reserve, config
from factory_agent import StartTask, WORKSPACE_ID, StateStore


class Tools:
    version = "a" * 64

    def __init__(self):
        self.calls = []

    def tool_definitions(self):
        return [{"type": "function", "name": "get_order", "parameters": {"type": "object"}}]

    def call_tool(self, name, args):
        self.calls.append((name, copy.deepcopy(args)))
        if name == "get_order":
            return {"order_id": "manual:1", "version": self.version}
        if name == "prepare_change":
            return {"proposal": {"order_id": "manual:1", "source_version": self.version,
                                 "title": "Review fixture change", "summary": "Operator request",
                                 "changes": [{"field": "rows[0].quantity", "before": 1, "after": 2}]}}
        raise ValueError("This mutation tool is unavailable.")


def request(message="Inspect saved orders", context=None, identity=None):
    return StartTask(request_id=identity or uuid.uuid4(), order_id=WORKSPACE_ID, message=message,
                     context_session_id=context)


def test_workspace_reads_and_freeform_task_without_forcing_fixture(tmp_path):
    async def scenario():
        fake, tools = FakeProvider(), Tools()
        svc = service(tmp_path, fake)
        svc.tool_factory = lambda: tools
        try:
            row = await reserve(svc, request("Explain how to compare glass orders"))
            await svc.run(row)
            payload = next(c[1] for c in fake.calls if c[0] == "create")
            assert payload["environment"] == {"type": "openai_hosted", "desktop": {"enabled": True}, "network": {"access": "disabled"}}
            assert "prepare_change" in payload["agent"]["instructions"]
            text = fake.events("agent.session.input.message")[0][2][0]["input"][0]["content"][0]["text"]
            assert "Explain how to compare glass orders" in text
            assert "synthetic order" not in text
            assert row["cleanup_status"] == "deleted"
            assert svc.store.load()[row["id"]]["order_id"] == WORKSPACE_ID
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_continuation_is_explicit_owned_bounded_and_idempotent(tmp_path):
    async def scenario():
        svc = service(tmp_path, FakeProvider())
        svc.tool_factory = Tools
        try:
            first = await reserve(svc, request())
            first.update(status="completed", cleanup_status="deleted", result_text="Earlier answer" * 2000)
            svc.save(first)
            payload = request("And compare the quantities", context=first["id"])
            second = await reserve(svc, payload)
            assert len(second["conversation"]) == 2
            assert len(second["conversation"][1]["text"]) == 12000
            assert (await svc.create("owner", payload))["id"] == second["id"]
            with pytest.raises(HTTPException) as conflict:
                await svc.create("owner", request(payload.message, identity=payload.request_id))
            assert conflict.value.status_code == 409
            with pytest.raises(HTTPException) as private:
                await svc.create("different-owner", request(context=first["id"]))
            assert private.value.status_code == 404
            svc.retire(first)
            assert not first["conversation"] and not first["proposals"]
            with pytest.raises(HTTPException) as retired:
                await svc.create("owner", request(context=first["id"]))
            assert retired.value.status_code == 409
            second.update(status="cancelled", cleanup_status="deleted")
            plain = await reserve(svc, request("Start independently"))
            assert plain["conversation"] == []
            plain.update(status="cancelled", cleanup_status="deleted")
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_proposals_persist_before_acknowledgement_review_without_apply(tmp_path):
    async def scenario():
        fake, tools = FakeProvider(), Tools()
        svc = service(tmp_path, fake)
        svc.tool_factory = lambda: tools
        try:
            row = await reserve(svc, request("Prepare a quantity correction"))
            row.update(remote_session_id="sess_fixture", turn_id="turn_fixture")
            action = {"type": "function_call", "turn_id": "turn_fixture", "call_id": "proposal-call", "name": "prepare_change", "arguments": {}}
            original = fake.send_events
            async def lost_ack(*args, **kwargs):
                raise RuntimeError("lost tool acknowledgement")
            fake.send_events = lost_ack
            with pytest.raises(RuntimeError):
                await svc.handle_actions(row, [action])
            assert len(svc.store.load()[row["id"]]["proposals"]) == 1
            fake.send_events = original
            await svc.handle_actions(row, [action])
            assert len(row["proposals"]) == 1
            assert len([c for c in tools.calls if c[0] == "prepare_change"]) == 1
            proposal = row["proposals"][0]
            with pytest.raises(HTTPException):
                await svc.review_proposal(row, proposal["id"], "accepted")
            row.update(status="completed", cleanup_status="deleted")
            tools.version = "b" * 64
            with pytest.raises(HTTPException) as stale:
                await svc.review_proposal(row, proposal["id"], "accepted")
            assert stale.value.status_code == 409 and proposal["status"] == "pending"
            tools.version = "a" * 64
            await svc.review_proposal(row, proposal["id"], "accepted")
            await svc.review_proposal(row, proposal["id"], "accepted")
            assert proposal["status"] == "accepted" and proposal["applied"] is False
            assert {c[0] for c in tools.calls} == {"prepare_change", "get_order"}
            with pytest.raises(HTTPException):
                await svc.review_proposal(row, proposal["id"], "rejected")
            svc.retire(row)
            assert svc.store.load()[row["id"]]["proposals"] == []
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_workspace_denies_production_browser_and_unknown_mutation(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        svc.tool_factory = Tools
        try:
            row = await reserve(svc, request())
            row.update(remote_session_id="sess_fixture", turn_id="turn_fixture")
            await svc.handle_actions(row, [
                {"type": "function_call", "turn_id": "turn_fixture", "call_id": "mutate", "name": "approve_order", "arguments": {}},
                {"type": "computer_use_approval_request", "request_id": "browser", "request": {"type": "browser_origin_access", "origin": "https://order-extractor-kdih.onrender.com"}},
                {"type": "computer_use_approval_request", "request_id": "signin", "request": {"type": "browser_authentication"}},
            ])
            assert fake.events("agent.session.input.tool_result")[0][2][0]["success"] is False
            responses = fake.events("agent.session.input.computer_use_approval_request_result")
            assert responses[0][2][0]["response"]["decision"] == "deny"
            assert responses[1][2][0]["response"]["action"] == "cancel"
            row.update(status="cancelled", cleanup_status="deleted")
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_saved_shell_activity_exposes_failure_without_credentials(tmp_path):
    svc = service(tmp_path, FakeProvider())
    row = {"turn_id": "turn_fixture", "activity": [], "result_text": ""}
    svc.collect_items(row, [{"id": "cmd", "type": "command_execution", "turn_id": "turn_fixture",
                            "status": "failed", "exit_code": 1,
                            "output": "Fixture listener failed synthetic-test-key local-test-auth"}])
    assert row["activity"][0]["status"] == "failed"
    assert "exit 1" in row["activity"][0]["title"]
    assert "Fixture listener failed" in row["activity"][0]["title"]
    assert "synthetic-test-key" not in str(row)
    assert "local-test-auth" not in str(row)


def test_fixture_connection_success_does_not_hide_failed_page_visit(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.items.extend([
            {"id": "connect", "type": "computer_use_call", "turn_id": "turn_fixture", "status": "completed", "title": "Connect browser"},
            {"id": "visit", "type": "computer_use_call", "turn_id": "turn_fixture", "status": "failed", "title": "Read fixture page"},
        ])
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            svc.activity(row, "read", "Read-only tool: get_selected_order returned the fixture.")
            await svc.run(row)
            assert row["status"] == "completed"
            assert "browser verification is not confirmed" in row["error"]
        finally:
            await svc.close()
    asyncio.run(scenario())
