from __future__ import annotations

import importlib
import inspect
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"


def _load_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-5-mini")
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    for name in [
        "workspace_agent",
        "workspace_service",
        "workspace_agents.sdk_agent",
        "workspace_agents.intent",
        "workspace_agents.smart_chat",
        "workspace_agents.response_format",
        "app",
        "db",
        "llm",
    ]:
        sys.modules.pop(name, None)
    db = importlib.import_module("db")
    db.init_db()
    service = importlib.import_module("workspace_service")
    return db, service


def _row(**overrides):
    row = {
        "order_number": "R-26-0042",
        "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
        "dimension": "500x600",
        "position": "1-1",
        "quantity": 2,
        "area": 0.6,
    }
    row.update(overrides)
    return row


def _insert_order(db, rows=None, status="approved"):
    rows = rows or [_row()]
    inserted = db.insert_extraction_with_rows(
        source="pdf",
        rows=rows,
        raw_input="raw pdf text",
        prepared_text="prepared text",
        llm_output_json='{"rows":[]}',
        model_used="test-model",
        hash_value=f"hash-{status}-{rows[0].get('dimension')}",
        confidence=0.9,
        client_name="Client A",
    )
    if status == "approved":
        db.update_order_rows(inserted["order_id"], rows, status="approved", client_name="Client A")
    return inserted["order_id"]


def _install_fake_smart_chat_sdk(monkeypatch, smart_chat, responses):
    captured = {"calls": []}
    if not isinstance(responses, list):
        responses = [responses]
    queue = list(responses)

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["run_config"] = kwargs
            self.tracing_disabled = kwargs.get("tracing_disabled", False)

    class FakeRunner:
        @staticmethod
        def run_sync(agent, input, *, context=None, max_turns=None, run_config=None):
            captured["calls"].append({"input": input, "context": context, "max_turns": max_turns, "run_config": run_config})
            output = queue.pop(0) if queue else responses[-1]
            if isinstance(output, Exception):
                raise output

            class Result:
                final_output = json.dumps(output)

            return Result()

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WORKSPACE_CHAT_ENGINE", "agents_sdk")
    monkeypatch.setattr(
        smart_chat,
        "_sdk_imports",
        lambda: (FakeAgent, FakeRunConfig, object, FakeRunner, fake_function_tool, None),
    )
    return captured


def test_process_approved_order_delegates_clean_order_to_frontend_modules(tmp_path, monkeypatch):
    db, service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db)
    before = db.get_order_with_extraction(order_id)

    result = service.process_approved_order("R-26-0042", requested_by="test")

    after = db.get_order_with_extraction(order_id)
    assert result["status"] == "frontend_workflow_required"
    assert result["summary"]["total_pieces"] == 2
    assert result["actions"][0]["type"] == "process_via_existing_modules"
    assert before["rows"] == after["rows"]
    assert before["extraction"]["raw_input"] == after["extraction"]["raw_input"]


def test_telegram_reextract_does_not_overwrite_approved_order(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    source_hash = "same-telegram-file"
    approved_id = db.insert_extraction_with_rows(
        source="pdf",
        rows=[_row()],
        raw_input="original raw",
        prepared_text="original prepared",
        llm_output_json='{"rows":[]}',
        model_used="test-model",
        hash_value=source_hash,
        confidence=0.9,
        client_name="Client A",
    )["order_id"]
    db.update_order_rows(approved_id, [_row()], status="approved", client_name="Client A")

    inserted = db.insert_extraction_with_rows(
        source="telegram",
        rows=[_row(dimension="700x800", area=0.56)],
        raw_input="telegram raw",
        prepared_text="telegram prepared",
        llm_output_json='{"rows":[]}',
        model_used="test-model",
        hash_value=source_hash,
        confidence=0.8,
        client_name="Client A",
        source_metadata={"source": "telegram", "telegram_chat_id": 123, "telegram_message_id": 456},
    )

    approved_after = db.get_order_with_extraction(approved_id)
    draft_after = db.get_order_with_extraction(inserted["order_id"])
    assert inserted["created_new_version"] is True
    assert inserted["protected_order_id"] == approved_id
    assert approved_after["status"] == "approved"
    assert approved_after["rows"][0]["dimension"] == "500x600"
    assert draft_after["status"] == "draft"
    assert draft_after["source"] == "telegram"
    assert draft_after["source_metadata"]["telegram_chat_id"] == 123
    assert draft_after["rows"][0]["dimension"] == "700x800"


def test_telegram_file_queue_metadata_and_recovery_query(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    record = db.create_telegram_file_record(
        original_filename="queued.pdf",
        stored_filename="queued.pdf",
        file_path=str(tmp_path / "queued.pdf"),
        mime_type="application/pdf",
        file_size=120,
        telegram_file_id="telegram-file-1",
        telegram_chat_id=123,
        telegram_message_id=456,
        telegram_sender_name="Sender",
        telegram_caption="Caption",
        received_at=now,
        extraction_status="queued",
        queued_at=now,
    )

    assert record["touched"] is False
    assert record["retry_count"] == 0
    assert record["last_error"] is None
    assert record["telegram_file_id"] == "telegram-file-1"
    assert record["telegram_caption"] == "Caption"

    stale_before = now - timedelta(minutes=10)
    assert record["id"] in db.list_unfinished_telegram_file_ids(stale_processing_before=stale_before)

    fresh_processing = db.update_telegram_file_record(
        record["id"],
        extraction_status="processing",
        processing_started_at=now,
        retry_count=1,
    )
    assert fresh_processing["extraction_status"] == "processing"
    assert record["id"] not in db.list_unfinished_telegram_file_ids(stale_processing_before=stale_before)

    old_started = now - timedelta(minutes=20)
    db.update_telegram_file_record(record["id"], processing_started_at=old_started)
    assert record["id"] in db.list_unfinished_telegram_file_ids(stale_processing_before=stale_before)

    failed = db.update_telegram_file_record(
        record["id"],
        extraction_status="failed",
        processed_at=now,
        retry_count=3,
        last_error="boom",
    )
    assert failed["extraction_status"] == "failed"
    assert failed["retry_count"] == 3
    assert failed["last_error"] == "boom"


def test_telegram_files_filter_touched_without_deleting_records(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    active = db.create_telegram_file_record(
        original_filename="active.pdf",
        stored_filename="active.pdf",
        file_path=str(tmp_path / "active.pdf"),
        mime_type="application/pdf",
        file_size=120,
        received_at=now,
        extraction_status="extracted",
    )
    touched = db.create_telegram_file_record(
        original_filename="handled.pdf",
        stored_filename="handled.pdf",
        file_path=str(tmp_path / "handled.pdf"),
        mime_type="application/pdf",
        file_size=120,
        received_at=now - timedelta(minutes=1),
        extraction_status="extracted",
    )

    db.touch_telegram_file_record(touched["id"])

    default_ids = {item["id"] for item in db.list_telegram_files()}
    active_ids = {item["id"] for item in db.list_telegram_files(touched=False)}
    touched_ids = {item["id"] for item in db.list_telegram_files(touched=True)}
    all_ids = {item["id"] for item in db.list_telegram_files(touched=None)}

    assert default_ids == {active["id"]}
    assert active_ids == {active["id"]}
    assert touched_ids == {touched["id"]}
    assert all_ids == {active["id"], touched["id"]}
    assert db.get_telegram_file(touched["id"])["original_filename"] == "handled.pdf"


def test_telegram_file_soft_delete_hides_without_removing_pdf_or_queue_breakage(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    pdf_path = tmp_path / "delete-me.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nkept")
    record = db.create_telegram_file_record(
        original_filename="delete-me.pdf",
        stored_filename="delete-me.pdf",
        file_path=str(pdf_path),
        mime_type="application/pdf",
        file_size=120,
        received_at=now,
        extraction_status="queued",
        queued_at=now,
    )

    deleted = db.soft_delete_telegram_file_record(record["id"])

    assert deleted["deleted"] is True
    assert deleted["deleted_at"]
    assert pdf_path.exists()
    assert record["id"] not in {item["id"] for item in db.list_telegram_files(touched=None)}
    assert record["id"] not in db.list_unfinished_telegram_file_ids(stale_processing_before=now + timedelta(minutes=1))
    assert db.get_telegram_file(record["id"])["deleted"] is True


def test_telegram_delete_linked_draft_archives_only_when_selected(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    order_id = _insert_order(db, status="draft")
    record = db.create_telegram_file_record(
        original_filename="draft-linked.pdf",
        stored_filename="draft-linked.pdf",
        file_path=str(tmp_path / "draft-linked.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="extracted",
    )
    db.update_telegram_file_record(record["id"], linked_order_id=order_id)
    client = TestClient(app_module.app)

    response = client.delete(f"/telegram-files/{record['id']}?also_delete_linked_order=true")

    assert response.status_code == 200
    assert response.json()["file"]["deleted"] is True
    assert response.json()["linked_order_deleted"] is True
    assert db.get_order_with_extraction(order_id)["status"] == "archived"
    assert db.get_telegram_file(record["id"])["deleted"] is True


def test_telegram_delete_linked_draft_preserves_order_when_not_selected(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    order_id = _insert_order(db, status="draft")
    record = db.create_telegram_file_record(
        original_filename="draft-linked-keep-order.pdf",
        stored_filename="draft-linked-keep-order.pdf",
        file_path=str(tmp_path / "draft-linked-keep-order.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="extracted",
    )
    db.update_telegram_file_record(record["id"], linked_order_id=order_id)
    client = TestClient(app_module.app)

    response = client.delete(f"/telegram-files/{record['id']}")

    assert response.status_code == 200
    assert response.json()["file"]["deleted"] is True
    assert response.json()["linked_order_deleted"] is False
    assert db.get_order_with_extraction(order_id)["status"] == "draft"


def test_telegram_delete_preserves_approved_linked_order(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    order_id = _insert_order(db, status="approved")
    record = db.create_telegram_file_record(
        original_filename="approved-linked.pdf",
        stored_filename="approved-linked.pdf",
        file_path=str(tmp_path / "approved-linked.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="extracted",
    )
    db.update_telegram_file_record(record["id"], linked_order_id=order_id)
    client = TestClient(app_module.app)

    response = client.delete(f"/telegram-files/{record['id']}?also_delete_linked_order=true")

    assert response.status_code == 200
    assert response.json()["warning"] == "Linked order is not draft and was not deleted."
    assert response.json()["linked_order_deleted"] is False
    assert db.get_order_with_extraction(order_id)["status"] == "approved"
    assert db.get_telegram_file(record["id"])["deleted"] is True


def test_telegram_file_auto_touches_after_labels_then_order_opened(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, status="draft")
    record = db.create_telegram_file_record(
        original_filename="labels-first.pdf",
        stored_filename="labels-first.pdf",
        file_path=str(tmp_path / "labels-first.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="extracted",
    )
    db.update_telegram_file_record(record["id"], linked_order_id=order_id)

    after_labels = db.mark_telegram_file_labels_printed(record["id"])
    assert after_labels["labels_printed"] is True
    assert after_labels["linked_order_opened"] is False
    assert after_labels["touched"] is False

    before_status = db.get_order_with_extraction(order_id)["status"]
    after_open = db.mark_telegram_file_linked_order_opened(record["id"])

    assert after_open["labels_printed"] is True
    assert after_open["linked_order_opened"] is True
    assert after_open["touched"] is True
    assert after_open["touched_at"]
    assert db.get_order_with_extraction(order_id)["status"] == before_status
    assert db.get_telegram_file(record["id"])["download_url"] == f"/telegram-files/{record['id']}/download"


def test_telegram_file_auto_touches_after_order_opened_then_labels(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, status="draft")
    record = db.create_telegram_file_record(
        original_filename="order-first.pdf",
        stored_filename="order-first.pdf",
        file_path=str(tmp_path / "order-first.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="extracted",
    )
    db.update_telegram_file_record(record["id"], linked_order_id=order_id)

    after_open = db.mark_telegram_file_linked_order_opened(record["id"])
    assert after_open["linked_order_opened"] is True
    assert after_open["labels_printed"] is False
    assert after_open["touched"] is False

    after_labels = db.mark_telegram_file_labels_printed(record["id"])
    assert after_labels["labels_printed"] is True
    assert after_labels["linked_order_opened"] is True
    assert after_labels["touched"] is True


def test_telegram_file_does_not_auto_touch_failed_or_unlinked_records(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, status="draft")
    failed = db.create_telegram_file_record(
        original_filename="failed.pdf",
        stored_filename="failed.pdf",
        file_path=str(tmp_path / "failed.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="failed",
    )
    unlinked = db.create_telegram_file_record(
        original_filename="unlinked.pdf",
        stored_filename="unlinked.pdf",
        file_path=str(tmp_path / "unlinked.pdf"),
        mime_type="application/pdf",
        file_size=120,
        extraction_status="extracted",
    )
    db.update_telegram_file_record(failed["id"], linked_order_id=order_id)

    db.mark_telegram_file_labels_printed(failed["id"])
    failed_after = db.mark_telegram_file_linked_order_opened(failed["id"])
    assert failed_after["labels_printed"] is True
    assert failed_after["linked_order_opened"] is True
    assert failed_after["touched"] is False

    db.mark_telegram_file_labels_printed(unlinked["id"])
    unlinked_after = db.mark_telegram_file_linked_order_opened(unlinked["id"])
    assert unlinked_after["labels_printed"] is True
    assert unlinked_after["linked_order_opened"] is True
    assert unlinked_after["touched"] is False


def test_process_approved_order_blocks_draft(tmp_path, monkeypatch):
    db, service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, status="draft")

    result = service.process_approved_order("R-26-0042", requested_by="test")

    assert result["status"] == "blocked"
    assert result["reason"] == "order_not_approved"


def test_process_approved_order_allows_approved_order_with_warnings(tmp_path, monkeypatch):
    db, service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(dimension="", area=0)])

    result = service.process_approved_order("R-26-0042", requested_by="test")

    assert result["status"] == "frontend_workflow_required"
    assert result["informational_warnings"]
    assert not result.get("pending_action")


def test_process_approved_order_does_not_create_backend_batches(tmp_path, monkeypatch):
    db, service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)

    first = service.process_approved_order("R-26-0042", requested_by="test")
    second = service.process_approved_order("R-26-0042", requested_by="test")

    assert first["status"] == "frontend_workflow_required"
    assert second["status"] == "frontend_workflow_required"


def test_workspace_queue_groups_orders(tmp_path, monkeypatch):
    db, service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)

    queue = service.get_workspace_queue()

    assert queue["counts"]["approved_ready"] == 1
    assert queue["groups"]["approved_ready"][0]["order_number"] == "R-26-0042"


def test_agent_route_delegates_clean_approved_order_to_frontend_modules(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    response = client.post("/api/agent/workspace", json={"message": "Process order R-26-0042"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "frontend_workflow_required"
    assert data["actions"][0]["type"] == "process_via_existing_modules"


def test_factory_assistant_casual_chat_redirects_to_smart_chat_without_tools(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "hi"}).json()

    assert data["status"] == "ok"
    assert "smart chat" in data["message"].lower()
    assert data["action_plan"]["planned_tool"] == "none"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
        tool_names = [row.tool_name for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types
    assert "agent_tool_called" not in action_types
    assert "process_orders_via_existing_modules" not in tool_names


def test_factory_assistant_thanks_redirects_without_tools(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "thanks"}).json()

    assert data["status"] == "ok"
    assert data["action_plan"]["intent"] == "casual_chat"
    assert data["action_plan"]["planned_tool"] == "none"
    assert "smart chat" in data["message"].lower()
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "agent_tool_called" not in action_types


def test_smart_chat_route_calls_openai_for_casual_message(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    captured = _install_fake_smart_chat_sdk(monkeypatch, smart_chat, {
        "message": "Yeah, it is coming together nicely.",
        "status": "ok",
        "actions": [],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
    })

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "nice"}).json()

    assert data["status"] == "ok"
    assert data["message"] == "Yeah, it is coming together nicely."
    assert "Ask me general questions" not in data["message"]
    assert json.loads(captured["calls"][0]["input"])["message"] == "nice"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types
    assert "agent_tool_called" not in action_types


def test_smart_chat_are_you_here_lol_answers_naturally_without_process_tools(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    _install_fake_smart_chat_sdk(monkeypatch, smart_chat, {
        "message": "I am here with you.",
        "status": "ok",
        "actions": [],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
    })

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "are you here with me lol?"}).json()

    assert data["status"] == "ok"
    assert "here" in data["message"].lower()
    assert "process approved orders" not in data["message"].lower()
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types
    assert "agent_tool_called" not in action_types


def test_smart_chat_date_question_answers_current_day_without_factory_tools(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")

    expected = sdk_agent.current_datetime_summary()
    _install_fake_smart_chat_sdk(monkeypatch, smart_chat, {
        "message": f"Today is {expected['day_name']}, {expected['date']}.",
        "status": "ok",
        "current_datetime": expected,
        "actions": [],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
    })
    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "what day is today btw"}).json()

    assert data["status"] == "ok"
    assert expected["day_name"] in data["message"]
    assert expected["date"] in data["current_datetime"]["date"]
    assert "process approved orders" not in data["message"].lower()
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types
    assert "agent_tool_called" not in action_types


def test_factory_assistant_date_question_redirects_to_smart_chat(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "what day is today btw"}).json()

    assert data["status"] == "ok"
    assert "smart chat" in data["message"].lower()
    assert data["action_plan"]["planned_tool"] == "none"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "agent_tool_called" not in action_types
    assert "process_orders_delegated" not in action_types


def test_agent_route_capability_question_explains_without_processing(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "what can you do?"}).json()

    assert data["status"] == "ok"
    assert "production workflows" in data["message"]
    assert "combine selected orders" in data["message"]
    assert data["action_plan"]["planned_tool"] == "none"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types


def test_smart_chat_explain_this_page_explains_workspace_without_process_tools(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    _install_fake_smart_chat_sdk(monkeypatch, smart_chat, {
        "message": "Workspace is the production staging page where approved orders move toward Processing and Labels.",
        "status": "ok",
        "actions": [],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
    })

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "explain this page"}).json()

    assert data["status"] == "ok"
    assert "Workspace" in data["message"]
    assert "production" in data["message"]
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types


def test_smart_chat_explains_processing_workflow(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    _install_fake_smart_chat_sdk(monkeypatch, smart_chat, {
        "message": "Processing uses approved orders, applies Danko rounding and grouping, then Labels are created from that processing result.",
        "status": "ok",
        "actions": [],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
    })

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "how does processing work?"}).json()

    assert data["status"] == "ok"
    assert "Danko" in data["message"]
    assert "Labels" in data["message"]
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types


def test_agent_route_casual_plus_process_request_processes_order(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "hi can you process R-25-1290"}).json()

    assert data["status"] == "frontend_workflow_required"
    assert data["actions"][0]["type"] == "process_via_existing_modules"
    assert data["actions"][0]["identifiers"]


def test_smart_chat_production_command_redirects_to_factory_without_processing(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "process R-25-1290"}).json()

    assert data["status"] == "ok"
    assert "factory assistant" in data["message"].lower()
    assert data["actions"][0]["kind"] == "send_to_factory"
    assert data["actions"][0]["payload"]["message"] == "process R-25-1290"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types
    assert "agent_tool_called" not in action_types


def test_smart_chat_repeated_casual_questions_do_not_return_capability_pitch(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    _install_fake_smart_chat_sdk(monkeypatch, smart_chat, [
        {"message": "Doing well. I am here.", "status": "ok", "actions": [], "cards": [], "refresh": {"queue": False, "recent_files": False}},
        {"message": "Yep, still here with you.", "status": "ok", "actions": [], "cards": [], "refresh": {"queue": False, "recent_files": False}},
    ])

    client = TestClient(app_module.app)
    first = client.post("/api/agent/smart-chat", json={"message": "how are you doing"}).json()
    second = client.post("/api/agent/smart-chat", json={"message": "are you here with me lol?"}).json()

    assert "process approved orders" not in first["message"].lower()
    assert "process approved orders" not in second["message"].lower()
    assert "factory workflows" not in first["message"].lower()
    assert "factory workflows" not in second["message"].lower()


def test_agent_route_show_approved_orders_uses_read_only_queue(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "show approved orders"}).json()

    assert data["status"] == "ok"
    assert data["action_plan"]["planned_tool"] == "get_workspace_queue"
    assert data["queue"]["counts"]["approved_ready"] == 1


def test_agent_route_yes_without_pending_does_not_process(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "yes", "context": {"current_page": "Workspace"}}).json()

    assert data["status"] == "ok"
    assert "nothing pending" in data["message"]
    assert data["action_plan"]["planned_tool"] == "none"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types


def test_agent_route_continue_without_pending_does_not_process(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "continue", "context": {"current_page": "Workspace"}}).json()

    assert data["status"] == "ok"
    assert "nothing pending" in data["message"]
    assert data["action_plan"]["planned_tool"] == "none"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types


def test_workspace_agent_extracts_multiple_order_numbers(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    agent = importlib.import_module("workspace_agent")

    tokens = agent.extract_order_tokens("process order 25-1290 and 25-1190 together")

    assert tokens == ["R-25-1290", "R-25-1190"]
    assert agent.infer_processing_mode("process order 25-1290 and 25-1190 together", len(tokens)) == "combined"


def test_workspace_agent_extracts_comma_separated_order_numbers(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    agent = importlib.import_module("workspace_agent")

    tokens = agent.extract_order_tokens("process orders R-25-1290, R-25-1190")

    assert tokens == ["R-25-1290", "R-25-1190"]


def test_workspace_agent_detects_separate_multi_order_mode(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    agent = importlib.import_module("workspace_agent")

    tokens = agent.extract_order_tokens("process R-25-1290 and R-25-1190 separately")

    assert tokens == ["R-25-1290", "R-25-1190"]
    assert agent.infer_processing_mode("process R-25-1290 and R-25-1190 separately", len(tokens)) == "separate"


def test_workspace_agent_keep_orders_separate_is_combined_without_merge(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    agent = importlib.import_module("workspace_agent")

    message = "process R-25-1290 and R-25-1190 together but keep orders separate"
    tokens = agent.extract_order_tokens(message)

    assert tokens == ["R-25-1290", "R-25-1190"]
    assert agent.infer_processing_mode(message, len(tokens)) == "combined"
    assert agent.infer_merge_across_orders(message) is False


def test_workspace_agent_merge_across_orders_phrase_sets_merge_flag(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    agent = importlib.import_module("workspace_agent")

    message = "process R-25-1290 and R-25-1190 together and merge across orders"

    assert agent.infer_processing_mode(message, 2) == "combined"
    assert agent.infer_merge_across_orders(message) is True


def test_agent_route_returns_combined_multi_order_frontend_action(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="700x800")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/agent/workspace",
        json={"message": "can you process order 25-1290 and 25-1190 at the same time together"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "frontend_workflow_required"
    action = data["actions"][0]
    assert action["type"] == "process_via_existing_modules"
    assert action["identifiers"] == [str(first_id), str(second_id)]
    assert action["mode"] == "combined"
    assert action["mergeAcrossOrders"] is False


def test_agent_route_combined_merge_across_orders_sets_frontend_flag(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="500x600")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post(
        "/api/agent/workspace",
        json={"message": "process order 25-1290 and 25-1190 together and merge across orders"},
    ).json()

    action = data["actions"][0]
    assert action["mode"] == "combined"
    assert action["mergeAcrossOrders"] is True


def test_agent_route_combined_keep_orders_separate_sets_merge_false(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="500x600")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post(
        "/api/agent/workspace",
        json={"message": "process order 25-1290 and 25-1190 together but don't mix orders"},
    ).json()

    action = data["actions"][0]
    assert action["mode"] == "combined"
    assert action["mergeAcrossOrders"] is False


def test_agent_route_uses_context_for_process_this(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/agent/workspace",
        json={
            "message": "process this",
            "context": {
                "current_page": "Workspace",
                "selected_order_id": str(order_id),
                "selected_order_number": "R-26-0042",
                "visible_orders": [],
                "last_agent_action": None,
            },
        },
    )

    data = response.json()
    assert data["status"] == "frontend_workflow_required"
    assert data["actions"][0]["identifiers"] == [str(order_id)]
    assert data["action_plan"]["intent"] == "process_orders"


def test_agent_route_process_this_asks_when_none_selected(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post(
        "/api/agent/workspace",
        json={"message": "process this", "context": {"current_page": "Workspace", "selected_orders": []}},
    ).json()

    assert data["status"] == "needs_choice"
    assert "which order" in data["message"]


def test_agent_route_uses_context_for_selected_orders_together(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="700x800")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/agent/workspace",
        json={
            "message": "process these two together",
            "context": {
                "current_page": "Workspace",
                "selected_order_ids": [str(first_id), str(second_id)],
                "selected_order_numbers": ["R-25-1290", "R-25-1190"],
                "visible_orders": [],
                "last_agent_action": None,
            },
        },
    )

    data = response.json()
    action = data["actions"][0]
    assert data["status"] == "frontend_workflow_required"
    assert action["identifiers"] == [str(first_id), str(second_id)]
    assert action["mode"] == "combined"


def test_agent_route_do_these_two_together_uses_selected_orders(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="700x800")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post(
        "/api/agent/workspace",
        json={
            "message": "do these two together",
            "context": {
                "current_page": "Workspace",
                "selected_orders": [
                    {"order_id": str(first_id), "order_number": "R-25-1290"},
                    {"order_id": str(second_id), "order_number": "R-25-1190"},
                ],
            },
        },
    ).json()

    assert data["status"] == "frontend_workflow_required"
    assert data["actions"][0]["identifiers"] == [str(first_id), str(second_id)]
    assert data["actions"][0]["mode"] == "combined"


def test_agent_route_uses_context_for_second_approved_order(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="700x800")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/agent/workspace",
        json={
            "message": "process the second approved order",
            "context": {
                "current_page": "Workspace",
                "visible_orders": [
                    {"order_id": str(first_id), "order_number": "R-25-1290", "status": "approved", "queue_group": "approved_ready"},
                    {"order_id": str(second_id), "order_number": "R-25-1190", "status": "approved", "queue_group": "approved_ready"},
                ],
            },
        },
    )

    data = response.json()
    assert data["status"] == "frontend_workflow_required"
    assert data["actions"][0]["identifiers"] == [str(second_id)]


def test_agent_route_returns_context_action_for_labels_again(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/agent/workspace",
        json={
            "message": "download the labels again",
            "context": {
                "current_page": "Workspace",
                "last_agent_action": {"type": "process_order", "order_numbers": ["R-26-0042"], "files": None},
            },
        },
    )

    data = response.json()
    assert data["status"] == "context_action"
    assert data["actions"][0]["type"] == "show_last_agent_files"


def test_agent_route_download_labels_again_returns_existing_context_links(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post(
        "/api/agent/workspace",
        json={
            "message": "download the labels again",
            "context": {
                "current_page": "Workspace",
                "last_agent_action": {
                    "type": "process_order",
                    "status": "success",
                    "order_numbers": ["R-26-0042"],
                    "files": {"processing_pdf_url": "blob:processing", "labels_pdf_url": "blob:labels"},
                },
            },
        },
    ).json()

    assert data["status"] == "context_action"
    assert data["files"]["labels_pdf_url"] == "blob:labels"


def test_agent_route_same_as_before_reuses_last_mode_safely(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    first_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    second_id = _insert_order(db, rows=[_row(order_number="R-25-1190", dimension="700x800")])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_module.app)
    data = client.post(
        "/api/agent/workspace",
        json={
            "message": "same as before",
            "context": {
                "current_page": "Workspace",
                "last_agent_action": {
                    "type": "process_orders_combined",
                    "status": "success",
                    "order_numbers": ["R-25-1290", "R-25-1190"],
                    "mode": "combined",
                },
            },
        },
    ).json()

    assert data["status"] == "frontend_workflow_required"
    assert data["actions"][0]["identifiers"] == [str(first_id), str(second_id)]
    assert data["actions"][0]["mode"] == "combined"


def test_combined_approved_orders_with_warnings_process_without_pending_action(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    clean_id = _insert_order(db, rows=[_row(order_number="R-26-0410", dimension="500x600")])
    warning_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="", area=0)])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)
    session_context = {"workspace_session_id": "session-a"}

    data = client.post(
        "/api/agent/workspace",
        json={"message": "process order 26-0410 and 25-1290", "context": session_context},
    ).json()

    assert data["status"] == "frontend_workflow_required"
    action = data["actions"][0]
    assert action["identifiers"] == [str(clean_id), str(warning_id)]
    assert action["mode"] == "combined"
    assert data["informational_warnings"]
    assert "pending_action" not in data


def test_agent_route_why_did_it_stop_without_pending_action_is_clear(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="", area=0)])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)
    context = {"workspace_session_id": "why-stop"}

    processed = client.post("/api/agent/workspace", json={"message": "process order 25-1290", "context": context}).json()
    data = client.post(
        "/api/agent/workspace",
        json={"message": "why did it stop?", "context": {**context}},
    ).json()

    assert processed["status"] == "frontend_workflow_required"
    assert data["status"] == "ok"
    assert "blocked action" in data["message"]


def test_agent_route_why_did_it_stop_explains_last_blocked_action(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)

    data = client.post(
        "/api/agent/workspace",
        json={
            "message": "why did it stop?",
            "context": {
                "current_page": "Workspace",
                "last_agent_action": {
                    "type": "process_order",
                    "status": "blocked",
                    "order_numbers": ["R-25-0527"],
                    "reason": "R-25-0527 is still Draft",
                },
            },
        },
    ).json()

    assert data["status"] == "ok"
    assert "R-25-0527 is still Draft" in data["message"]


def test_agent_route_what_should_i_do_next_uses_queue(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)

    data = client.post("/api/agent/workspace", json={"message": "what should I do next?"}).json()

    assert data["status"] == "ok"
    assert "approved order" in data["message"]
    assert data["action_plan"]["planned_tool"] == "get_workspace_queue"


def test_approved_warnings_separate_mode_remains_separate_without_pending(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    clean_id = _insert_order(db, rows=[_row(order_number="R-26-0410", dimension="500x600")])
    warning_id = _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="", area=0)])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)
    context = {"workspace_session_id": "session-separate"}

    data = client.post(
        "/api/agent/workspace",
        json={"message": "process order 26-0410 and 25-1290 separately", "context": context},
    ).json()

    action = data["actions"][0]
    assert action["identifiers"] == [str(clean_id), str(warning_id)]
    assert action["mode"] == "separate"
    assert data["informational_warnings"]


def test_invalid_pending_action_cannot_be_continued(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)
    context = {"workspace_session_id": "session-expired"}

    continued = client.post(
        "/api/workspace/confirm-action",
        json={"action": "pending_action", "pending_action_id": "missing", "decision": "continue", "context": context},
    ).json()

    assert continued["status"] == "invalid_pending_action"


def test_combined_processing_blocks_if_any_order_is_draft(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(order_number="R-26-0410", dimension="500x600")])
    _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")], status="draft")
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)

    data = client.post("/api/agent/workspace", json={"message": "process order 26-0410 and 25-1290"}).json()

    assert data["status"] == "blocked"
    assert data["reason"] == "order_not_approved"


def test_approved_warning_processing_audit_records_delegated_not_pending(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="", area=0)])
    app_module = importlib.import_module("app")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app_module.app)
    context = {"workspace_session_id": "session-audit"}

    client.post("/api/agent/workspace", json={"message": "process order 25-1290", "context": context})

    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" in action_types
    assert "pending_processing_action_created" not in action_types


def test_agents_sdk_engine_available_and_tracing_not_disabled(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WORKSPACE_AGENT_ENGINE", "agents_sdk")

    assert sdk_agent.agents_sdk_available()
    assert sdk_agent.should_use_agents_sdk()


def test_agent_route_uses_agents_sdk_runner_with_trace_config(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")
    app_module = importlib.import_module("app")
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["run_config"] = kwargs
            self.tracing_disabled = kwargs.get("tracing_disabled", False)

    class FakeRunner:
        @staticmethod
        def run_sync(agent, input, *, context=None, max_turns=None, run_config=None):
            captured["runner_called"] = True
            captured["input"] = input
            captured["context"] = context
            captured["max_turns"] = max_turns
            captured["run_config_object"] = run_config

            class Result:
                final_output = '{"message":"SDK done","status":"ok","cards":[],"last_agent_action":null,"refresh":{"queue":false,"recent_files":false}}'

            return Result()

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WORKSPACE_AGENT_ENGINE", "agents_sdk")
    monkeypatch.setattr(
        sdk_agent,
        "_sdk_imports",
        lambda: (FakeAgent, FakeRunConfig, object, FakeRunner, fake_function_tool, None),
    )

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "show approved orders"}).json()

    assert captured["runner_called"] is True
    assert captured["agent"]["name"] == "Workspace Factory Assistant"
    assert captured["run_config"]["workflow_name"] == "Workspace Factory Assistant"
    assert captured["run_config_object"].tracing_disabled is False
    payload = json.loads(captured["input"])
    assert payload["message"] == "show approved orders"
    assert "classified_intent" not in payload
    assert data["message"] == "SDK done"
    assert "cards" in data


def test_casual_message_reaches_agents_sdk_instead_of_preclassified_fallback(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")
    app_module = importlib.import_module("app")
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.tracing_disabled = kwargs.get("tracing_disabled", False)

    class FakeRunner:
        @staticmethod
        def run_sync(agent, input, *, context=None, max_turns=None, run_config=None):
            captured["runner_called"] = True
            captured["input"] = input

            class Result:
                final_output = '{"message":"Hey, I am here.","status":"ok","cards":[],"last_agent_action":null,"refresh":{"queue":false,"recent_files":false}}'

            return Result()

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WORKSPACE_AGENT_ENGINE", "agents_sdk")
    monkeypatch.setattr(
        sdk_agent,
        "_sdk_imports",
        lambda: (FakeAgent, FakeRunConfig, object, FakeRunner, fake_function_tool, None),
    )

    client = TestClient(app_module.app)
    data = client.post("/api/agent/workspace", json={"message": "hi how are you doing"}).json()

    assert captured["runner_called"] is True
    assert json.loads(captured["input"])["message"] == "hi how are you doing"
    assert data["message"] == "Hey, I am here."


def test_no_giant_casual_phrase_parser_is_used():
    intent_module = importlib.import_module("workspace_agents.intent")
    source = inspect.getsource(intent_module)

    assert "CASUAL_EXACT" not in source
    assert "def classify_workspace_intent" not in source
    assert "def conversational_response" not in source


def test_agents_sdk_tool_catalog_is_high_level_only(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    tools = sdk_agent._build_tools(fake_function_tool, object)
    names = {tool.__name__ for tool in tools}

    assert "get_current_datetime" not in names
    assert "resolve_orders" in names
    assert "process_orders_via_existing_modules" in names
    assert "get_workspace_queue" in names
    assert "write_database_record" not in names
    assert "update_order_row_raw" not in names
    assert "generate_custom_fake_pdf" not in names


def test_smart_chat_tool_catalog_is_read_only(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    tools = {tool.__name__: tool for tool in smart_chat._build_tools(fake_function_tool, object)}
    names = set(tools)

    assert {"get_current_datetime", "get_platform_overview", "get_workspace_help", "get_workspace_queue_summary"} <= names
    assert "process_orders_via_existing_modules" not in names
    assert "write_database_record" not in names

    result = tools["get_current_datetime"]()

    assert set(result) == {"date", "day_name", "time", "timezone"}
    assert result["date"] == importlib.import_module("workspace_agents.sdk_agent").current_datetime_summary()["date"]
    assert result["day_name"]


def test_smart_chat_route_uses_agents_sdk_runner_for_normal_messages(tmp_path, monkeypatch):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["run_config"] = kwargs
            self.tracing_disabled = kwargs.get("tracing_disabled", False)

    class FakeRunner:
        @staticmethod
        def run_sync(agent, input, *, context=None, max_turns=None, run_config=None):
            captured["runner_called"] = True
            captured["input"] = input
            captured["context"] = context
            captured["max_turns"] = max_turns
            captured["run_config_object"] = run_config

            class Result:
                final_output = '{"message":"Yeah, I am smart enough to help with the platform and normal questions.","status":"ok","actions":[],"cards":[],"refresh":{"queue":false,"recent_files":false}}'

            return Result()

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WORKSPACE_CHAT_ENGINE", "agents_sdk")
    monkeypatch.setenv("OPENAI_SMART_CHAT_MODEL", "gpt-smart-test")
    monkeypatch.setattr(
        smart_chat,
        "_sdk_imports",
        lambda: (FakeAgent, FakeRunConfig, object, FakeRunner, fake_function_tool, None),
    )

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "are you smart tho?", "context": {"current_page": "Workspace"}}).json()

    assert captured["runner_called"] is True
    assert captured["agent"]["name"] == "Smart Chat"
    assert captured["agent"]["model"] == "gpt-smart-test"
    assert captured["run_config"]["workflow_name"] == "Smart Chat"
    assert captured["run_config"]["trace_metadata"]["conversation_type"] == "smart_chat"
    assert captured["run_config_object"].tracing_disabled is False
    assert json.loads(captured["input"])["message"] == "are you smart tho?"
    assert "smart enough" in data["message"]
    assert "Ask me general questions" not in data["message"]


def test_smart_chat_api_failures_are_logged_and_visible_in_dev(tmp_path, monkeypatch, caplog):
    _db, _service = _load_modules(tmp_path, monkeypatch)
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")
    _install_fake_smart_chat_sdk(monkeypatch, smart_chat, RuntimeError("model not found"))

    caplog.set_level(logging.INFO, logger="workspace_agents.smart_chat")
    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "nice"}).json()

    assert data["status"] == "error"
    assert data["fallback_used"] is True
    assert "Smart Chat API error: model not found" in data["message"]
    assert "Smart Chat API error" in caplog.text
    assert "fallback used" in caplog.text


def test_smart_chat_sdk_normalizer_blocks_production_action_without_processing(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    _insert_order(db, rows=[_row(order_number="R-25-1290", dimension="500x600")])
    smart_chat = importlib.import_module("workspace_agents.smart_chat")
    app_module = importlib.import_module("app")

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.tracing_disabled = kwargs.get("tracing_disabled", False)

    class FakeRunner:
        @staticmethod
        def run_sync(agent, input, *, context=None, max_turns=None, run_config=None):
            class Result:
                final_output = '{"message":"I will process that now.","status":"ok","actions":[],"cards":[],"refresh":{"queue":false,"recent_files":false}}'

            return Result()

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WORKSPACE_CHAT_ENGINE", "agents_sdk")
    monkeypatch.setattr(
        smart_chat,
        "_sdk_imports",
        lambda: (FakeAgent, FakeRunConfig, object, FakeRunner, fake_function_tool, None),
    )

    client = TestClient(app_module.app)
    data = client.post("/api/agent/smart-chat", json={"message": "process R-25-1290"}).json()

    assert "factory assistant" in data["message"].lower()
    assert data["actions"][0]["kind"] == "send_to_factory"
    with db.SessionLocal() as session:
        action_types = [row.action_type for row in session.query(db.WorkspaceAction).all()]
    assert "process_orders_delegated" not in action_types


def test_agents_sdk_resolve_orders_tool_uses_database(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    class Ctx:
        context = {"workspace_context": {}, "requested_by": "test-agent"}

    tools = {tool.__name__: tool for tool in sdk_agent._build_tools(fake_function_tool, object)}
    result = tools["resolve_orders"](Ctx(), refs=["26-0042"], context={})

    assert result["status"] == "ok"
    assert result["resolved"][0]["order_id"] == str(order_id)
    assert result["resolved"][0]["order_number"] == "R-26-0042"


def test_agents_sdk_process_orders_tool_returns_structured_action(tmp_path, monkeypatch):
    db, _service = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db)
    sdk_agent = importlib.import_module("workspace_agents.sdk_agent")

    def fake_function_tool(*decorator_args, **_decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorator(func):
            return func

        return decorator

    class Ctx:
        context = {"message": "process this", "workspace_context": {}, "requested_by": "test-agent"}

    tools = {tool.__name__: tool for tool in sdk_agent._build_tools(fake_function_tool, object)}
    result = tools["process_orders_via_existing_modules"](
        Ctx(),
        order_ids=[str(order_id)],
        mode="combined",
        source="workspace_agent",
        force_reprocess=False,
    )

    assert result["status"] == "frontend_workflow_required"
    assert result["actions"][0]["type"] == "process_via_existing_modules"
    assert result["actions"][0]["identifiers"] == [str(order_id)]
    assert result["actions"][0]["mode"] == "single"
