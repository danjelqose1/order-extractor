from __future__ import annotations

import base64
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


def _load_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    for name in ("beta_teaching_service", "beta_service", "db"):
        sys.modules.pop(name, None)
    db = importlib.import_module("db")
    db.init_db()
    shadow = importlib.import_module("beta_service")
    teaching = importlib.import_module("beta_teaching_service")
    return db, shadow, teaching


def _insert_example_order(db) -> int:
    pdf_data = base64.b64encode(b"%PDF-1.7\nTeach Mode fixture").decode("ascii")
    inserted = db.insert_extraction_with_rows(
        source="telegram",
        rows=[
            {
                "order_number": "R-26-0707",
                "type": "2 VETRI 33.1F +18+ 4LOWE (28MM)",
                "dimension": "698x1038",
                "position": "1-1",
                "quantity": 3,
                "area": 0.720,
            }
        ],
        raw_input=f"data:application/pdf;base64,{pdf_data}",
        prepared_text=(
            "CLIENTE MASSIMILIANO CAPUTO 24.07\n"
            "R-26-0707/1-1 698 x 1038 0,720 3 2,160\n"
            "Totale 3 2,160"
        ),
        llm_output_json=json.dumps({"rows": []}),
        model_used="fixture",
        hash_value="teach-example-order",
        confidence=0.93,
        client_name="MASSIMILIANO CAPUTO 24.07",
    )
    return int(inserted["order_id"])


def _matching_vision() -> Dict[str, Any]:
    return {
        "summary": "The original PDF row matches the extracted row.",
        "comparisons": [
            {
                "row_index": 1,
                "pdf_dimension": "698x1038",
                "extracted_dimension": "698x1038",
                "dimension_match": True,
                "pdf_quantity": 3,
                "extracted_quantity": 3,
                "quantity_match": True,
                "pdf_unit_area": 0.720,
                "extracted_area": 0.720,
                "area_match": True,
                "evidence": "The visible PDF row shows 698 x 1038, 3 pieces, and 0,720 m² per piece.",
            }
        ],
        "document_total_units": 3,
        "document_total_area": 2.160,
        "warnings": [],
        "confidence": 0.99,
        "ambiguous": False,
    }


def _workflow_output() -> Dict[str, Any]:
    return {
        "title": "Review and approve matching extracted orders",
        "summary": "The operator compared the source PDF with the extracted row before approval.",
        "steps": [
            {
                "step": 1,
                "module": "Orders",
                "operator_action": "Open the order and compare its original PDF with the extracted items.",
                "reason": "The source document is the approval evidence.",
                "decision_condition": "Approve only when dimension, quantity, and quantity-aware area match.",
                "evidence_event_sequences": [2, 3],
            }
        ],
        "candidate_hard_rules": [
            {
                "title": "Use quantity-aware PDF totals",
                "rule_text": "When row area is per piece, multiply it by quantity before comparing it with the PDF total area.",
                "confidence": "high",
                "evidence_event_sequences": [3],
            }
        ],
        "candidate_learned_notes": [
            {
                "title": "Approval evidence",
                "note_text": "The operator visually checks dimensions and quantity before approving.",
                "evidence_event_sequences": [2, 3],
            }
        ],
        "uncertainties": [],
    }


def _production_order_snapshot(db, order_id: int) -> Dict[str, Any]:
    order = db.get_order_with_extraction(order_id)
    assert order is not None
    return {
        "status": order["status"],
        "rows": order["rows"],
        "notes": order["notes"],
        "raw_input": order["extraction"]["raw_input"],
    }


def test_teach_mode_records_semantic_events_and_quantity_aware_visual_comparison(tmp_path, monkeypatch):
    db, _shadow, teaching = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_example_order(db)
    before = _production_order_snapshot(db, order_id)
    session = teaching.start_teaching_session("Learn how I review and approve an order.")

    event = teaching.record_teaching_event(
        session["id"],
        event_type="order_opened",
        module="Orders",
        message="Opened R-26-0707.",
        order_id=order_id,
        order_number="R-26-0707",
        metadata={"status": "draft", "api_token": "must-not-be-saved"},
    )
    comparison = teaching.compare_order(
        session["id"],
        order_id=order_id,
        force_vision=True,
        vision_analyzer=lambda payload: (_matching_vision() if payload["pdf_bytes"].startswith(b"%PDF") else {}),
    )

    assert event["event_type"] == "order_opened"
    assert "api_token" not in event["metadata"]
    assert comparison["verdict"] == "matched"
    assert comparison["vision_used"] is True
    assert comparison["extracted_units"] == 3
    assert comparison["quantity_aware_extracted_area"] == pytest.approx(2.160)
    assert comparison["declared_area"] == pytest.approx(2.160)
    assert "0.720 m² per piece × 3 = 2.160 m²" in comparison["suggested_reason"]
    assert _production_order_snapshot(db, order_id) == before


def test_approval_success_event_is_verified_but_never_executes_approval(tmp_path, monkeypatch):
    db, _shadow, teaching = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_example_order(db)
    session = teaching.start_teaching_session("Observe approval.")

    with pytest.raises(ValueError, match="could not be verified"):
        teaching.record_teaching_event(
            session["id"],
            event_type="approval_succeeded",
            module="Orders",
            message="Approved order.",
            order_id=order_id,
            order_number="R-26-0707",
        )

    assert db.get_order_with_extraction(order_id)["status"] == "draft"


def test_full_context_events_cover_all_modules_and_redact_credentials(tmp_path, monkeypatch):
    _db, _shadow, teaching = _load_modules(tmp_path, monkeypatch)
    session = teaching.start_teaching_session("Learn my complete in-platform workflow.")

    interaction = teaching.record_teaching_event(
        session["id"],
        event_type="ui_action",
        module="Manual Orders",
        message="Used Save manual order.",
        metadata={
            "control": {"label": "Save manual order", "role": "button"},
            "context_before": {
                "view": "manual",
                "visible_warnings": ["Quantity is required"],
                "api_key": "sk-this-must-never-be-stored",
            },
        },
    )
    result = teaching.record_teaching_event(
        session["id"],
        event_type="action_result",
        module="Production",
        message="POST /manual-orders returned HTTP 200.",
        metadata={
            "request": {"method": "POST", "endpoint": "/manual-orders"},
            "result": {"ok": True, "status": 200},
        },
    )

    assert interaction["module"] == "Manual Orders"
    assert interaction["metadata"]["context_before"]["visible_warnings"] == ["Quantity is required"]
    assert "api_key" not in interaction["metadata"]["context_before"]
    assert result["module"] == "Production"
    assert result["metadata"]["result"] == {"ok": True, "status": 200}


def test_pdf_comparison_and_teaching_synthesis_share_gpt_5_6_terra(tmp_path, monkeypatch):
    db, _shadow, teaching = _load_modules(tmp_path, monkeypatch)
    monkeypatch.delenv("BETA_MODEL", raising=False)
    monkeypatch.delenv("BETA_REASONING_EFFORT", raising=False)
    order_id = _insert_example_order(db)
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            schema_name = kwargs["response_format"]["json_schema"]["name"]
            payload = _matching_vision() if schema_name == "beta_teach_pdf_comparison" else _workflow_output()
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(payload)))]
            )

    fake_llm = types.ModuleType("llm")
    fake_llm.get_client = lambda: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())
    )
    fake_llm.pdf_to_png_pages = lambda _pdf, dpi=135: [b"rendered-page"]
    monkeypatch.setitem(sys.modules, "llm", fake_llm)

    session = teaching.start_teaching_session("Learn review with visual comparison.")
    comparison = teaching.compare_order(session["id"], order_id=order_id, force_vision=True)
    learned = teaching.finish_teaching_session(session["id"])

    assert comparison["vision_used"] is True
    assert learned["status"] == "awaiting_approval"
    assert [call["model"] for call in calls] == ["gpt-5.6-terra", "gpt-5.6-terra"]
    assert [call["reasoning_effort"] for call in calls] == ["medium", "medium"]
    vision_schema = calls[0]["response_format"]["json_schema"]["schema"]
    workflow_schema = calls[1]["response_format"]["json_schema"]["schema"]
    assert set(vision_schema["required"]) == set(vision_schema["properties"])
    assert set(vision_schema["$defs"]["VisionRowComparison"]["required"]) == set(
        vision_schema["$defs"]["VisionRowComparison"]["properties"]
    )
    assert "pdf_quantity" in vision_schema["$defs"]["VisionRowComparison"]["required"]
    assert set(workflow_schema["required"]) == set(workflow_schema["properties"])
    assert "evidence_event_sequences" in workflow_schema["$defs"]["TeachingWorkflowStep"]["required"]


def test_finish_requires_schema_and_memory_is_saved_only_after_human_review(tmp_path, monkeypatch):
    db, shadow, teaching = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_example_order(db)
    shadow.add_hard_rule(title="Existing safety rule", rule_text="Never approve a missing quantity.")
    session = teaching.start_teaching_session("Learn review workflow.")
    teaching.record_teaching_event(
        session["id"],
        event_type="order_opened",
        module="Orders",
        message="Opened the order.",
        order_id=order_id,
        order_number="R-26-0707",
    )
    teaching.record_teaching_event(
        session["id"],
        event_type="decision_reason",
        module="Orders",
        message="Approval reason: dimensions and quantity match.",
        order_id=order_id,
        order_number="R-26-0707",
        metadata={"reason": "dimensions and quantity match"},
    )
    captured: Dict[str, Any] = {}

    def synthesize(payload):
        captured.update(payload)
        return _workflow_output()

    learned = teaching.finish_teaching_session(session["id"], synthesizer=synthesize)

    assert learned["status"] == "awaiting_approval"
    assert learned["teaching_workflow"]["status"] == "draft"
    assert [rule["title"] for rule in captured["hard_rules"]] == ["Existing safety rule"]
    assert len(shadow.list_hard_rules()) == 1
    assert shadow.list_learned_notes() == []

    reviewed = teaching.review_teaching_workflow(
        learned["teaching_workflow"]["id"],
        decision="accepted",
        accept_hard_rules=True,
        accept_learned_notes=True,
        reviewed_by="local_operator",
    )

    assert reviewed["status"] == "completed"
    assert reviewed["teaching_workflow"]["status"] == "accepted"
    assert len(shadow.list_hard_rules()) == 2
    assert len(shadow.list_learned_notes()) == 1
    assert shadow.list_learned_notes()[0]["source_session_id"] == session["id"]
    assert db.get_order_with_extraction(order_id)["status"] == "draft"


def test_invalid_teaching_workflow_fails_closed_and_preserves_events(tmp_path, monkeypatch):
    _db, _shadow, teaching = _load_modules(tmp_path, monkeypatch)
    session = teaching.start_teaching_session("Learn a task.")
    teaching.record_teaching_event(
        session["id"],
        event_type="queue_viewed",
        module="Orders",
        message="Viewed the review queue.",
    )
    invalid = _workflow_output()
    invalid["steps"][0]["step"] = 2

    result = teaching.finish_teaching_session(
        session["id"],
        synthesizer=lambda _payload: invalid,
    )

    assert result["status"] == "failed"
    assert result["teaching_workflow"] is None
    assert len(result["teaching_events"]) >= 3
    assert any(
        entry["entry_type"] == "error"
        and entry["metadata"].get("error_type") == "invalid_teaching_workflow"
        for entry in result["journal_entries"]
    )


def test_teaching_tables_remain_separate_from_production_tables(tmp_path, monkeypatch):
    db, _shadow, _teaching = _load_modules(tmp_path, monkeypatch)
    with db.engine.connect() as connection:
        beta_tables = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'beta_%'")
            ).scalars()
        )
    assert {"beta_teaching_events", "beta_teaching_workflows"}.issubset(beta_tables)
