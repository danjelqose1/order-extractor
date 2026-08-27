from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest
from sqlalchemy import text


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"


def _load_beta_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    for module_name in ("beta_service", "db"):
        sys.modules.pop(module_name, None)
    db = importlib.import_module("db")
    db.init_db()
    service = importlib.import_module("beta_service")
    return db, service


def _workspace_snapshot() -> Dict[str, Any]:
    return {
        "queue": {
            "groups": {
                "needs_review": [
                    {
                        "id": 11,
                        "order_id": 11,
                        "order_number": "R-26-1011",
                        "client_name": "Client A",
                        "status": "draft",
                        "line_count": 3,
                        "total_pieces": 8,
                        "total_area_m2": 4.25,
                        "warnings_count": 1,
                        "created_at": "2026-08-27T08:00:00+00:00",
                        "raw_input": "SECRET RAW PDF CONTENT",
                        "api_token": "SECRET TOKEN",
                    }
                ],
                "approved_ready": [
                    {
                        "id": 12,
                        "order_id": 12,
                        "order_number": "R-26-1012",
                        "client_name": "Client B",
                        "status": "approved",
                        "total_pieces": 5,
                        "total_area_m2": 2.5,
                        "warnings_count": 0,
                    }
                ],
                "processing_done": [],
                "labels_ready": [],
                "finished": [],
            },
            "counts": {"needs_review": 999},
        },
        "recent_files": {
            "items": [
                {
                    "batch_id": 7,
                    "order_id": 12,
                    "order_number": "R-26-1012",
                    "client_name": "Client B",
                    "generated_at": "2026-08-27T09:00:00+00:00",
                    "batch_status": "ready",
                    "processing_pdf_url": "/api/workspace/files/1/download",
                    "labels_pdf_url": "/api/workspace/files/2/download",
                    "file_path": "/private/factory/order.pdf",
                }
            ]
        },
    }


def _valid_output(*, approval_needed: bool = True, lessons=None) -> Dict[str, Any]:
    return {
        "summary": "Two orders were reviewed in Shadow Mode.",
        "observations": ["One order has a deterministic validation warning."],
        "plan": [
            {
                "step": 1,
                "module": "History",
                "action": "Review the warning on R-26-1011.",
                "reason": "Deterministic validation reported one warning.",
                "risk": "medium",
                "requires_human_approval": approval_needed,
                "would_mutate_data": False,
            }
        ],
        "warnings": ["R-26-1011 remains ambiguous."],
        "approval_needed": approval_needed,
        "lessons": lessons or [],
    }


def _production_snapshot(db) -> Dict[str, Dict[str, Any]]:
    with db.engine.connect() as connection:
        tables = connection.execute(
            text(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'beta_%'"
                " ORDER BY name"
            )
        ).mappings().all()
        return {
            table["name"]: {
                "schema": table["sql"],
                "rows": [
                    dict(row)
                    for row in connection.execute(
                        text(f'SELECT * FROM "{table["name"]}" ORDER BY rowid')
                    ).mappings().all()
                ],
            }
            for table in tables
        }


def test_beta_tables_are_isolated_and_shadow_run_does_not_change_production(tmp_path, monkeypatch):
    db, service = _load_beta_modules(tmp_path, monkeypatch)
    with db.get_session() as session:
        session.add(db.Order(source="pdf", client_name="Factory Client", status="draft"))

    before = _production_snapshot(db)
    result = service.run_shadow_session(
        "Prepare today's new orders for processing.",
        planner=lambda _payload: _valid_output(approval_needed=False),
        context_reader=_workspace_snapshot,
    )
    after = _production_snapshot(db)

    assert result["status"] == "completed"
    assert result["mode"] == "shadow"
    assert before == after
    with db.engine.connect() as connection:
        beta_tables = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'beta_%'")
            ).scalars()
        )
    assert beta_tables == {
        "beta_sessions",
        "beta_journal_entries",
        "beta_hard_rules",
        "beta_learned_notes",
    }


def test_shadow_session_reads_only_sanitized_context_and_saves_journal(tmp_path, monkeypatch):
    _db, service = _load_beta_modules(tmp_path, monkeypatch)
    captured: Dict[str, Any] = {}

    def planner(payload):
        captured.update(payload)
        return _valid_output()

    result = service.run_shadow_session(
        "Prepare today's new orders for processing.",
        planner=planner,
        context_reader=_workspace_snapshot,
    )

    encoded_context = json.dumps(captured)
    assert "SECRET RAW PDF CONTENT" not in encoded_context
    assert "SECRET TOKEN" not in encoded_context
    assert "/private/factory/order.pdf" not in encoded_context
    assert "/api/workspace/files/1/download" not in encoded_context
    assert captured["workspace"]["queue"]["counts"]["needs_review"] == 1
    assert captured["workspace"]["recent_production"][0]["processing_ready"] is True
    assert captured["current_datetime"]["date"]
    assert captured["current_datetime"]["timezone"]
    assert result["status"] == "awaiting_approval"
    assert result["production_action_executed"] is False
    assert result["plan"][0]["would_mutate_data"] is False
    sequences = [entry["sequence"] for entry in result["journal_entries"]]
    assert sequences == sorted(sequences)
    assert any(
        entry["metadata"].get("source") == "deterministic"
        for entry in result["journal_entries"]
        if entry["entry_type"] == "observation"
    )
    assert any(
        entry["message"].startswith("Planning date:")
        for entry in result["journal_entries"]
    )
    assert any(entry["entry_type"] == "approval_request" for entry in result["journal_entries"])


def test_invalid_model_output_is_rejected_and_fails_closed(tmp_path, monkeypatch):
    _db, service = _load_beta_modules(tmp_path, monkeypatch)
    invalid = _valid_output()
    invalid["plan"][0]["would_mutate_data"] = True

    result = service.run_shadow_session(
        "Prepare orders safely.",
        planner=lambda _payload: invalid,
        context_reader=_workspace_snapshot,
    )

    assert result["status"] == "failed"
    assert result["plan"] == []
    errors = [entry for entry in result["journal_entries"] if entry["entry_type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["metadata"]["error_type"] == "invalid_model_output"
    assert errors[0]["metadata"]["production_data_changed"] is False


def test_missing_read_only_context_fails_closed_and_journals_error(tmp_path, monkeypatch):
    _db, service = _load_beta_modules(tmp_path, monkeypatch)
    planner_called = False

    def planner(_payload):
        nonlocal planner_called
        planner_called = True
        return _valid_output()

    result = service.run_shadow_session(
        "Prepare orders safely.",
        planner=planner,
        context_reader=lambda: {},
    )

    assert planner_called is False
    assert result["status"] == "failed"
    assert result["production_action_executed"] is False
    errors = [entry for entry in result["journal_entries"] if entry["entry_type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["metadata"]["error_type"] == "shadow_session_failure"
    assert errors[0]["metadata"]["production_data_changed"] is False


def test_default_planner_reuses_shared_client_with_strict_structured_output(tmp_path, monkeypatch):
    _db, service = _load_beta_modules(tmp_path, monkeypatch)
    captured: Dict[str, Any] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps(_valid_output(approval_needed=False))
                        )
                    )
                ]
            )

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())
    )
    fake_llm = types.ModuleType("llm")
    fake_llm.get_client = lambda: fake_client
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.setenv("BETA_MODEL", "gpt-beta-test")

    result = service.run_shadow_session(
        "Prepare orders safely.",
        context_reader=_workspace_snapshot,
    )

    assert result["status"] == "completed"
    assert captured["model"] == "gpt-beta-test"
    assert captured["messages"][0]["role"] == "developer"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in captured


def test_hard_rules_and_learned_notes_remain_separate_in_planner_context(tmp_path, monkeypatch):
    _db, service = _load_beta_modules(tmp_path, monkeypatch)
    rule = service.add_hard_rule(
        title="Never guess missing quantities",
        rule_text="Missing quantities must always be sent to manual review.",
        enabled=True,
        priority=1,
    )
    service.add_hard_rule(
        title="Disabled rule",
        rule_text="This must not be sent to the planner.",
        enabled=False,
        priority=2,
    )
    note = service.add_learned_note(
        title="Known client layout",
        note_text="Client A usually places totals at the bottom.",
        enabled=True,
    )
    service.add_learned_note(
        title="Pending note",
        note_text="This unreviewed note must not be sent to the planner.",
        enabled=False,
    )
    captured: Dict[str, Any] = {}

    result = service.run_shadow_session(
        "Observe incoming orders.",
        planner=lambda payload: (captured.update(payload) or _valid_output(lessons=["Review recurring warning patterns."])),
        context_reader=_workspace_snapshot,
    )

    assert captured["hard_rules"] == [
        {
            "id": rule["id"],
            "title": rule["title"],
            "rule_text": rule["rule_text"],
            "priority": rule["priority"],
        }
    ]
    assert captured["learned_notes"] == [
        {"id": note["id"], "title": note["title"], "note_text": note["note_text"]}
    ]
    all_notes = service.list_learned_notes()
    generated = [item for item in all_notes if item["source_session_id"] == result["id"]]
    assert len(generated) == 1
    assert generated[0]["enabled"] is False
    assert service.list_hard_rules()[0]["rule_text"] != generated[0]["note_text"]


def test_approval_records_beta_decision_without_triggering_production_action(tmp_path, monkeypatch):
    db, service = _load_beta_modules(tmp_path, monkeypatch)
    before = _production_snapshot(db)
    shadow = service.run_shadow_session(
        "Prepare approved orders.",
        planner=lambda _payload: _valid_output(),
        context_reader=_workspace_snapshot,
    )

    approved = service.record_approval(
        shadow["id"],
        decision="approved",
        approved_by="operator",
        note="Plan reviewed.",
    )

    assert approved["status"] == "completed"
    assert approved["approval_decision"] == "approved"
    assert approved["approved_by"] == "operator"
    assert approved["production_action_executed"] is False
    decision = [
        entry for entry in approved["journal_entries"] if entry["entry_type"] == "approval_decision"
    ][0]
    assert decision["metadata"]["production_action_triggered"] is False
    assert before == _production_snapshot(db)


def test_execute_guard_always_rejects_and_only_journals_warning(tmp_path, monkeypatch):
    db, service = _load_beta_modules(tmp_path, monkeypatch)
    before = _production_snapshot(db)
    shadow = service.run_shadow_session(
        "Prepare approved orders.",
        planner=lambda _payload: _valid_output(),
        context_reader=_workspace_snapshot,
    )

    with pytest.raises(service.BetaProductionMutationBlocked):
        service.reject_production_mutation(shadow["id"], attempted_action="process_order")

    detail = service.get_session_detail(shadow["id"])
    guarded = [
        entry
        for entry in detail["journal_entries"]
        if entry["metadata"].get("guard") == "beta_v1_production_mutation_guard"
    ]
    assert len(guarded) == 1
    assert guarded[0]["metadata"]["production_data_changed"] is False
    assert before == _production_snapshot(db)


def test_execute_api_returns_403_even_for_unknown_session(tmp_path, monkeypatch):
    _db, _service = _load_beta_modules(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sys.modules.pop("app", None)
    sys.modules.pop("llm", None)
    app_module = importlib.import_module("app")

    from fastapi.testclient import TestClient

    response = TestClient(app_module.app).post(
        "/api/beta/sessions/999/execute",
        json={"action": "update_order_status"},
    )

    assert response.status_code == 403
    assert "Shadow Mode only" in response.json()["detail"]


def test_approval_api_owns_the_local_audit_identity(tmp_path, monkeypatch):
    _db, service = _load_beta_modules(tmp_path, monkeypatch)
    shadow = service.run_shadow_session(
        "Prepare approved orders.",
        planner=lambda _payload: _valid_output(),
        context_reader=_workspace_snapshot,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sys.modules.pop("app", None)
    sys.modules.pop("llm", None)
    app_module = importlib.import_module("app")

    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    spoofed = client.post(
        f"/api/beta/sessions/{shadow['id']}/approval",
        json={"decision": "approved", "approved_by": "spoofed-user"},
    )
    assert spoofed.status_code == 422

    approved = client.post(
        f"/api/beta/sessions/{shadow['id']}/approval",
        json={"decision": "approved"},
    )
    assert approved.status_code == 200
    assert approved.json()["approved_by"] == "local_operator"
    assert approved.json()["production_action_executed"] is False
