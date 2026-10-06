"""Control-plane tests with synthetic provider responses and a private journal.

These exercise failures and replay boundaries; they never contact OpenAI or
read/write the production order database.
"""
from __future__ import annotations

import asyncio
import copy
import sys
import time
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from factory_agent import FactoryAgentService, StartTask, StateStore
from factory_agent_fixture import FIXTURE_ORDER_ID
from factory_agent_provider import ProviderError


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.session = {"id": "sess_fixture", "status": "idle", "environment": {"id": "env_fixture"}, "required_actions": []}
        self.environment = {"id": "env_fixture", "status": "connected"}
        self.sessions = []
        self.turns = [{"id": "turn_fixture", "subagent_id": None, "status": "completed"}]
        self.items = [{"id": "message_fixture", "turn_id": "turn_fixture", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Synthetic provider result for lifecycle verification."}]}]
        self.create_error = None
        self.input_error = None
        self.delete_error = None
        self.on_input = None
        self.on_cancel = None
        self.on_turns = None
        self.diagnostics = []

    async def read_failure_diagnostics(self, session_id):
        self.calls.append(("diagnostics", session_id))
        return self.diagnostics

    async def create_session(self, payload):
        self.calls.append(("create", copy.deepcopy(payload)))
        self.session["metadata"] = copy.deepcopy(payload["metadata"])
        self.sessions = [self.session]
        if self.create_error:
            raise self.create_error
        return copy.deepcopy(self.session)

    async def get_session(self, session_id):
        self.calls.append(("get", session_id))
        return copy.deepcopy(self.session)

    async def get_environment(self, environment_id):
        self.calls.append(("environment", environment_id))
        return copy.deepcopy(self.environment)

    async def list_sessions(self):
        self.calls.append(("sessions",))
        return copy.deepcopy(self.sessions)

    async def list_turns(self, session_id):
        self.calls.append(("turns", session_id))
        if self.on_turns:
            self.on_turns()
        return copy.deepcopy(self.turns)

    async def list_items(self, session_id):
        self.calls.append(("items", session_id))
        return copy.deepcopy(self.items)

    async def send_events(self, session_id, events, idempotency_key=None):
        self.calls.append(("events", session_id, copy.deepcopy(events), idempotency_key))
        if events[0]["type"] == "agent.session.input.message":
            if self.on_input:
                self.on_input()
            if self.input_error:
                raise self.input_error
        if events[0]["type"] == "agent.session.input.cancel" and self.on_cancel:
            self.on_cancel()
        return {}

    async def delete_session(self, session_id):
        self.calls.append(("delete", session_id))
        if self.delete_error:
            raise self.delete_error
        return {"id": session_id, "deleted": True}

    async def aclose(self):
        self.calls.append(("close",))

    def events(self, kind):
        return [call for call in self.calls if call[0] == "events" and call[2][0]["type"] == kind]


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv("ENABLE_FACTORY_AGENT", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("FACTORY_AGENT_MAX_CONCURRENCY", "1")


def request(message="Read this fixture", request_id=None):
    return StartTask(request_id=request_id or uuid.uuid4(), order_id=FIXTURE_ORDER_ID, message=message)


def service(tmp_path, fake):
    result = FactoryAgentService(lambda: "local-test-auth", provider_factory=lambda: fake, directory=tmp_path, poll_seconds=0.001)
    # Run the exact worker deterministically from the test after reservation.
    result.spawn = lambda row, recovery=False: None
    return result


async def reserve(svc, payload=None, owner="owner"):
    public = await svc.create(owner, payload or request())
    return svc.records[public["id"]]


def test_complete_reads_canonical_items_and_deletes_hosted_session(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] == "completed"
            assert row["result_text"] == fake.items[0]["content"][0]["text"]
            assert row["cleanup_status"] == "deleted"
            assert len(fake.events("agent.session.input.message")) == 1
            payload = next(call[1] for call in fake.calls if call[0] == "create")
            assert "input" not in payload
            assert payload["environment"]["network"] == {"access": "disabled"}
            assert payload["environment"]["desktop"] == {"enabled": True}
            assert payload["agent"]["multi_agent"] == {"enabled": False}
            assert payload["agent"]["model"] == "gpt-6.1-sol"
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_failed_setup_preserves_redacted_diagnostics_before_deletion(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.session.update(status="failed", error="Setup failed: synthetic-test-key local-test-auth")
        fake.diagnostics = [{"source": "error", "code": "setup_error", "message": "Listener did not start. sk-project-othersecret Bearer opaque-credential"}]
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] == "failed"
            assert row["cleanup_status"] == "deleted"
            assert "setup_error" in row["error"]
            assert "Listener did not start" in row["error"]
            stored = repr(svc.store.load())
            for secret in ("synthetic-test-key", "local-test-auth", "sk-project-othersecret", "opaque-credential"):
                assert secret not in row["error"]
                assert secret not in stored
            operations = [call[0] for call in fake.calls]
            assert operations.index("diagnostics") < operations.index("delete")
            assert not fake.events("agent.session.input.message")
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_failed_session_retains_saved_error_when_no_failure_events_return(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.session.update(status="failed", error="Provisioning service unavailable")
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert "Provisioning service unavailable" in row["error"]
            assert row["cleanup_status"] == "deleted"
            assert len([call for call in fake.calls if call[0] == "diagnostics"]) == 1
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_environment_failure_collects_diagnostics_without_task_submission(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.environment["status"] = "failed"
        fake.diagnostics = [{"source": "agent.session.environment.failed", "code": "setup_exit", "message": "Health check exited nonzero"}]
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert "Health check exited nonzero" in row["error"]
            assert row["cleanup_status"] == "deleted"
            assert not fake.events("agent.session.input.message")
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_failure_text_is_bounded_and_model_override_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTORY_AGENT_MODEL", "gpt-6-luna")
    svc = service(tmp_path, FakeProvider())
    assert len(svc.safe_diagnostic("x" * 20000)) == 2000
    assert svc.safe_diagnostic({"unexpected": "object"}) == ""
    assert svc.session_payload({"id": "local-test"})["agent"]["model"] == "gpt-6-luna"


def test_long_operator_secret_is_removed_before_display_cap(tmp_path, monkeypatch):
    secret = "operator-" + "q" * 4080
    monkeypatch.setenv("FACTORY_AGENT_ACCESS_KEY", secret)
    svc = service(tmp_path, FakeProvider())
    text = svc.safe_diagnostic("Setup error " + secret + " listener failed")
    assert text == "Setup error [redacted] listener failed"


def test_lost_create_ack_reconciles_metadata_without_duplicate_create(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.create_error = ProviderError(504, "provider_timeout", "Creation acknowledgement lost.", outcome_unknown=True)
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] == "completed"
            assert len([call for call in fake.calls if call[0] == "create"]) == 1
            assert len(fake.events("agent.session.input.message")) == 1
            assert row["remote_session_id"] == "sess_fixture"
            assert any(call[0] == "sessions" for call in fake.calls)
        finally:
            await svc.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("matches", [0, 2])
def test_unknown_create_identity_holds_slot_and_never_submits_task(tmp_path, matches):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            row["phase"] = "creating"
            svc.save(row)
            fake.sessions = [{"id": "sess_" + str(i), "metadata": {"factory_agent_request_id": row["id"]}} for i in range(matches)]
            await svc.run(row, recovery=True)
            assert row["status"] == "cleanup_required"
            assert row["cleanup_status"] != "deleted"
            assert not fake.events("agent.session.input.message")
            assert not any(call[0] == "create" for call in fake.calls)
            with pytest.raises(HTTPException) as error:
                await svc.create("owner", request("Another task"))
            assert error.value.status_code == 409
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_lost_input_ack_recovers_completed_turn_without_resubmitting(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.input_error = ProviderError(504, "provider_timeout", "Input acknowledgement lost.", outcome_unknown=True)
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] == "completed"
            assert len(fake.events("agent.session.input.message")) == 1
            assert row["input_attempted"] is True
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_unknown_input_with_no_turn_expires_cancels_and_never_replays(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.turns = []
        fake.input_error = ProviderError(504, "provider_timeout", "Input acknowledgement lost.", outcome_unknown=True)
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            fake.on_input = lambda: row.update(deadline=time.time() - 1, stop_started=time.time() - 31)
            await svc.run(row)
            assert row["status"] == "timed_out"
            assert row["cleanup_status"] == "deleted"
            assert len(fake.events("agent.session.input.message")) == 1
            assert fake.events("agent.session.input.cancel")
            assert not row["result_text"]
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_restart_uses_persisted_input_identity_and_never_resubmits(tmp_path):
    async def scenario():
        first = service(tmp_path, FakeProvider())
        row = await reserve(first)
        row.update(remote_session_id="sess_fixture", phase="input_attempted", input_attempted=True, status="running")
        first.save(row)
        # Model abrupt exit: release local handles without sending cancellation.
        first.janitor.cancel()
        await asyncio.gather(first.janitor, return_exceptions=True)
        first.store.close()
        first.store = None
        fake = FakeProvider()
        restored = service(tmp_path, fake)
        try:
            await restored.start()
            restored_row = restored.records[row["id"]]
            await restored.run(restored_row, recovery=True)
            assert restored_row["status"] == "completed"
            assert not fake.events("agent.session.input.message")
            assert not any(call[0] == "create" for call in fake.calls)
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_stop_cancels_remote_turn_instead_of_only_local_observer(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.turns[0]["status"] = "in_progress"
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            fake.on_turns = lambda: row.update(stop_requested=True)
            fake.on_cancel = lambda: fake.turns[0].update(status="cancelled")
            await svc.run(row)
            assert row["status"] == "cancelled"
            assert fake.events("agent.session.input.cancel")
            assert row["cleanup_status"] == "deleted"
            assert len(fake.events("agent.session.input.message")) == 1
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_stop_before_create_avoids_remote_resource_entirely(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.stop(row)
            await svc.run(row)
            assert row["status"] == "cancelled"
            assert row["cleanup_status"] == "deleted"
            assert not fake.calls
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_idempotent_local_submit_and_concurrency_hold_until_cleanup(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            payload = request()
            row = await reserve(svc, payload)
            same = await svc.create("owner", payload)
            assert same["id"] == row["id"]
            with pytest.raises(HTTPException) as mismatch:
                await svc.create("owner", request("Changed input", payload.request_id))
            assert mismatch.value.status_code == 409
            row["status"] = "completed"
            with pytest.raises(HTTPException) as occupied:
                await svc.create("owner", request("Next task"))
            assert occupied.value.status_code == 409
            assert not fake.calls
        finally:
            # This reservation never reached the provider.
            row["cleanup_status"] = "deleted"
            await svc.close()
    asyncio.run(scenario())


def test_cleanup_failure_reserves_slot_and_preserves_reviewable_result(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.delete_error = ProviderError(409, "conflict", "Still running.")
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] == "cleanup_required"
            assert row["cleanup_status"] == "required"
            assert row["result_text"]
            assert len([call for call in fake.calls if call[0] == "delete"]) == 3
            with pytest.raises(HTTPException) as blocked:
                await svc.remove(row)
            assert blocked.value.status_code == 409
            assert not row["retired"]
            with pytest.raises(HTTPException):
                await svc.create("owner", request("More work"))
            fake.delete_error = None
            await svc.cleanup(row, stopping=True)
            assert row["cleanup_status"] == "deleted"
            assert row["status"] == "completed"
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_retirement_wipes_content_but_keeps_replay_tombstone(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            payload = request()
            row = await reserve(svc, payload)
            await svc.run(row)
            await svc.remove(row)
            assert row["retired"] and not row["result_text"] and row["screenshot"] is None
            assert (await svc.create("owner", payload))["id"] == row["id"]
            assert len([call for call in fake.calls if call[0] == "create"]) == 1
            with pytest.raises(HTTPException) as denied:
                svc.owned(row["id"], "other-owner")
            assert denied.value.status_code == 404
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_browser_and_tool_actions_cannot_grant_production_access(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            row.update(remote_session_id="sess_fixture", turn_id="turn_fixture")
            actions = [
                {"type": "computer_use_approval_request", "request_id": "req_origin", "request": {"type": "browser_origin_access", "origin": "https://factory.invalid"}},
                {"type": "computer_use_approval_request", "request_id": "req_auth", "request": {"type": "browser_authentication", "credential_origin": "https://factory.invalid"}},
                {"type": "function_call", "turn_id": "turn_fixture", "call_id": "call_edit", "name": "approve_order", "arguments": {"order_id": 1}},
                {"type": "function_call", "turn_id": "turn_fixture", "call_id": "call_read", "name": "get_selected_order", "arguments": {}},
            ]
            await svc.handle_actions(row, actions)
            events = [call[2][0] for call in fake.calls if call[0] == "events"]
            assert events[0]["response"] == {"type": "browser_origin_access", "decision": "deny"}
            assert events[1]["response"] == {"type": "browser_authentication", "action": "cancel"}
            assert "turn_id" not in events[0] and "turn_id" not in events[1]
            assert events[2]["success"] is False
            assert events[3]["success"] is True and FIXTURE_ORDER_ID in events[3]["output"]
            await svc.handle_actions(row, actions)
            assert len([call for call in fake.calls if call[0] == "events"]) == 4
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_malformed_turns_cancel_and_cleanup_without_fabricating_result(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.turns = ["malformed provider response"]
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] != "completed"
            assert not row["result_text"]
            assert fake.events("agent.session.input.cancel")
            assert row["cleanup_status"] == "deleted"
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_missing_text_is_never_reported_as_verified_reading_success(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.items = []
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert not row["result_text"]
            assert "No order-reading success" in row["error"]
            assert row["cleanup_status"] == "deleted"
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_second_worker_cannot_own_same_control_journal(tmp_path):
    first = StateStore(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="one application worker"):
            StateStore(tmp_path)
    finally:
        first.close()


def test_provider_setup_error_does_not_crash_platform_startup(tmp_path):
    def invalid_provider():
        raise ProviderError(503, "setup_required", "Set OPENAI_API_KEY on the backend.")

    async def scenario():
        svc = FactoryAgentService(lambda: "local-auth", provider_factory=invalid_provider, directory=tmp_path)
        try:
            await svc.start()
            assert svc.configuration()["ready"] is False
            assert svc.configuration()["state"] == "setup_required"
        finally:
            if svc.store:
                svc.store.close()
                svc.store = None
    asyncio.run(scenario())


def test_wall_clock_watchdog_interrupts_slow_poll_then_cancels_remote_task(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            row.update(remote_session_id="sess_fixture", phase="input_attempted", input_attempted=True, deadline=time.time() + 0.02)

            async def slow_get(session_id):
                fake.calls.append(("slow_get", session_id))
                await asyncio.sleep(2)
                return fake.session

            fake.get_session = slow_get
            began = time.monotonic()
            await svc.run(row)
            assert time.monotonic() - began < 1
            assert row["status"] == "timed_out"
            assert row["cleanup_status"] == "deleted"
            assert fake.events("agent.session.input.cancel")
            assert not fake.events("agent.session.input.message")
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_stop_endpoint_sends_cancellation_before_worker_poll_returns(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            row.update(remote_session_id="sess_fixture", input_attempted=True, status="running")
            public = await svc.stop(row)
            assert public["status"] == "stopping"
            assert len(fake.events("agent.session.input.cancel")) == 1
            # A 202 cancel acknowledgment must not be reported as final cancellation.
            assert row["cleanup_status"] != "deleted"
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_unexpected_second_root_turn_is_cancelled_without_collecting_output(tmp_path):
    async def scenario():
        fake = FakeProvider()
        fake.turns.append({"id": "turn_unexpected", "subagent_id": None, "status": "completed"})
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert row["status"] != "completed"
            assert not row["result_text"]
            assert fake.events("agent.session.input.cancel")
            assert row["cleanup_status"] == "deleted"
        finally:
            await svc.close()
    asyncio.run(scenario())


def test_screenshots_only_accept_safe_inline_images_for_intended_turn(tmp_path):
    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            row["turn_id"] = "turn_fixture"
            def screenshot(url, turn="turn_fixture"):
                return {"id": "browser_item", "turn_id": turn, "type": "computer_use_call", "status": "completed", "output": {"type": "computer_screenshot", "image_url": url}}
            svc.collect_items(row, [screenshot("https://attacker.invalid/tracking.png"), screenshot("data:image/svg+xml;base64,eA==")])
            assert row["screenshot"] is None
            svc.collect_items(row, [screenshot("data:image/png;base64,eA==", "other_turn")])
            assert row["screenshot"] is None
            svc.collect_items(row, [screenshot("data:image/jpeg;base64,eA==")])
            assert row["screenshot"] == "data:image/jpeg;base64,eA=="
        finally:
            row["cleanup_status"] = "deleted"
            await svc.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("corrupt_payload", ["{corrupt SECRET journal data", "[]", '{"status":"running"}', '{"status":"completed","retired":true}'])
def test_corrupt_journal_is_preserved_and_isolated_as_setup_required(tmp_path, corrupt_payload):
    journal = StateStore(tmp_path)
    journal.db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (str(uuid.uuid4()), "owner", str(uuid.uuid4()), corrupt_payload))
    journal.db.commit()
    journal.close()

    async def scenario():
        fake = FakeProvider()
        svc = service(tmp_path, fake)
        try:
            await svc.start()
            config = svc.configuration()
            assert config["ready"] is False
            assert config["state"] == "setup_required"
            assert config["error"] and "SECRET" not in config["error"]
            assert svc.store is None
            assert not fake.calls
            assert not svc.tasks
            with pytest.raises(HTTPException) as rejected:
                await svc.create("owner", request())
            assert rejected.value.status_code == 503
            assert rejected.value.detail["code"] == "setup_required"
            # Failure releases the lock and must not erase potentially live IDs.
            check = StateStore(tmp_path)
            try:
                assert check.db.execute("SELECT payload FROM sessions").fetchone()[0] == corrupt_payload
            finally:
                check.close()
        finally:
            if svc.store:
                svc.store.close()
                svc.store = None
    asyncio.run(scenario())


@pytest.mark.parametrize("remote_status", ["starting", None, "completed", {"unexpected": "object"}])
def test_unknown_session_state_never_submits_initial_input(tmp_path, remote_status):
    async def scenario():
        fake = FakeProvider()
        fake.session["status"] = remote_status
        svc = service(tmp_path, fake)
        try:
            row = await reserve(svc)
            await svc.run(row)
            assert not fake.events("agent.session.input.message")
            assert not row["input_attempted"]
            assert row["status"] != "completed"
            assert not row["result_text"]
            assert row["error"]
            assert row["cleanup_status"] == "deleted"
        finally:
            await svc.close()
    asyncio.run(scenario())
