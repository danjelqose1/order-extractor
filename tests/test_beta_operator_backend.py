from __future__ import annotations

import base64
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"


def _load_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    for name in (
        "workspace_service",
        "beta_operator_service",
        "beta_teaching_service",
        "beta_service",
        "db",
    ):
        sys.modules.pop(name, None)
    db = importlib.import_module("db")
    db.init_db()
    shadow = importlib.import_module("beta_service")
    teaching = importlib.import_module("beta_teaching_service")
    operator = importlib.import_module("beta_operator_service")
    return db, shadow, teaching, operator


def _insert_order(db, suffix: str = "0754") -> int:
    pdf_data = base64.b64encode(b"%PDF-1.7\nAssisted Review fixture").decode("ascii")
    inserted = db.insert_extraction_with_rows(
        source="telegram",
        rows=[
            {
                "order_number": f"R-26-{suffix}",
                "type": "3 VETRI 4F +16+ 4F +16+ 4LOWE",
                "dimension": "593x2281",
                "position": "1-1",
                "quantity": 1,
                "area": 1.35,
            }
        ],
        raw_input=f"data:application/pdf;base64,{pdf_data}",
        prepared_text=f"R-26-{suffix}/1-1 593 x 2281 1,35 1\nTotale 1 1,35",
        llm_output_json=json.dumps({"rows": []}),
        model_used="fixture",
        hash_value=f"operator-{suffix}",
        confidence=0.93,
        client_name="VINI JURGENI",
    )
    return int(inserted["order_id"])


def _model_review(rule_id: Optional[int] = None, *, mismatch: bool = False) -> Dict[str, Any]:
    checks = []
    if rule_id is not None:
        checks.append(
            {
                "rule_id": rule_id,
                "title": "Approve exact PDF matches only",
                "outcome": "pass" if not mismatch else "fail",
                "evidence": "The visible dimension, quantity, and area were checked.",
            }
        )
    return {
        "summary": "The source PDF was compared with the extracted row.",
        "comparisons": [
            {
                "row_index": 1,
                "pdf_dimension": "593x2281",
                "extracted_dimension": "593x2281",
                "dimension_match": not mismatch,
                "pdf_quantity": 1,
                "extracted_quantity": 1,
                "quantity_match": True,
                "pdf_unit_area": 1.35,
                "extracted_area": 1.35,
                "area_match": True,
                "evidence": "The visible PDF row shows 593 x 2281, quantity 1, and 1.35 m2.",
            }
        ],
        "document_total_units": 1,
        "document_total_area": 1.35,
        "warnings": [],
        "confidence": 0.99,
        "ambiguous": False,
        "hard_rule_checks": checks,
        "recommendation": "reject" if mismatch else "approve",
        "reason": "All required fields match." if not mismatch else "The dimension does not match.",
        "next_actions": [
            {
                "module": "Processing",
                "action": "Prepare the order for Processing after approval.",
                "reason": "Processing is the next supervised workflow stage.",
                "risk": "medium",
                "requires_human_approval": True,
            }
        ],
    }


def _snapshot(db, order_id: int) -> Dict[str, Any]:
    order = db.get_order_with_extraction(order_id)
    return {
        "status": order["status"],
        "rows": order["rows"],
        "raw_input": order["extraction"]["raw_input"],
        "prepared_text": order["extraction"]["prepared_text"],
    }


def test_assisted_review_uses_memory_and_requires_explicit_confirmation(tmp_path, monkeypatch):
    db, shadow, _teaching, operator = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db)
    rule = shadow.add_hard_rule(
        title="Approve exact PDF matches only",
        rule_text="Dimension, quantity, and quantity-aware area must match the original PDF.",
        priority=1,
    )
    shadow.add_learned_note(
        title="Visual comparison",
        note_text="The operator opens Files and compares the original PDF before approval.",
        enabled=True,
    )
    with db.get_session() as session:
        taught = db.BetaSession(
            goal="Teach PDF review.",
            mode="teach",
            status="completed",
            summary="Learned visual review.",
        )
        session.add(taught)
        session.flush()
        session.add(
            db.BetaTeachingWorkflow(
                source_session_id=taught.id,
                title="Review original PDF before approval",
                status="accepted",
                summary="Compare every row before approving.",
                workflow_json=json.dumps({"steps": [{"step": 1, "operator_action": "Compare PDF"}]}),
            )
        )
    before = _snapshot(db, order_id)
    captured = {}

    started = operator.start_review_session("Review Needs Review orders.", limit=5)
    reviewed = operator.review_order(
        started["id"],
        order_id=order_id,
        reviewer=lambda payload, _pdf: (captured.update(payload) or _model_review(rule["id"])),
    )

    assert reviewed["review"]["verdict"] == "safe_to_approve"
    assert reviewed["session"]["status"] == "awaiting_approval"
    assert captured["hard_rules"][0]["id"] == rule["id"]
    assert captured["learned_notes"][0]["title"] == "Visual comparison"
    assert captured["accepted_teaching_workflows"][0]["title"] == "Review original PDF before approval"
    assert _snapshot(db, order_id) == before

    with pytest.raises(ValueError, match="Explicit human confirmation"):
        operator.approve_reviewed_orders(
            started["id"], order_ids=[order_id], confirmed=False
        )
    assert _snapshot(db, order_id) == before

    completed = operator.approve_reviewed_orders(
        started["id"], order_ids=[order_id], confirmed=True
    )
    after = _snapshot(db, order_id)
    assert completed["status"] == "completed"
    assert completed["production_action_executed"] is True
    assert after["status"] == "approved"
    assert after["rows"] == before["rows"]
    assert after["raw_input"] == before["raw_input"]
    assert after["prepared_text"] == before["prepared_text"]


def test_mismatch_is_blocked_and_never_offered_for_approval(tmp_path, monkeypatch):
    db, _shadow, _teaching, operator = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, "0755")
    before = _snapshot(db, order_id)
    started = operator.start_review_session("Review risky orders.", limit=5)

    reviewed = operator.review_order(
        started["id"],
        order_id=order_id,
        reviewer=lambda _payload, _pdf: _model_review(mismatch=True),
    )

    assert reviewed["review"]["verdict"] == "blocked"
    assert reviewed["session"]["status"] == "completed"
    assert reviewed["session"]["approval_requested"] is False
    assert _snapshot(db, order_id) == before
    with pytest.raises(ValueError, match="not awaiting approval"):
        operator.approve_reviewed_orders(
            started["id"], order_ids=[order_id], confirmed=True
        )


def test_model_cannot_approve_by_echoing_a_different_extracted_row(tmp_path, monkeypatch):
    db, _shadow, _teaching, operator = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, "0758")
    response = _model_review()
    response["comparisons"][0]["extracted_dimension"] = "999x999"
    started = operator.start_review_session("Review orders.", limit=5)

    reviewed = operator.review_order(
        started["id"],
        order_id=order_id,
        reviewer=lambda _payload, _pdf: response,
    )

    assert reviewed["review"]["verdict"] == "manual_review"
    assert any("echoed the wrong extracted dimension" in item for item in reviewed["review"]["blockers"])
    assert db.get_order_with_extraction(order_id)["status"] == "draft"


def test_changed_order_invalidates_safe_review_before_atomic_approval(tmp_path, monkeypatch):
    db, _shadow, _teaching, operator = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, "0756")
    started = operator.start_review_session("Review orders.", limit=5)
    operator.review_order(
        started["id"],
        order_id=order_id,
        reviewer=lambda _payload, _pdf: _model_review(),
    )
    current = db.get_order_with_extraction(order_id)
    changed_rows = [dict(current["rows"][0])]
    changed_rows[0]["quantity"] = 2
    db.update_order_rows(order_id, changed_rows)

    with pytest.raises(ValueError, match="changed after review"):
        operator.approve_reviewed_orders(
            started["id"], order_ids=[order_id], confirmed=True
        )
    assert db.get_order_with_extraction(order_id)["status"] == "draft"


def test_declining_safe_matches_closes_session_without_order_changes(tmp_path, monkeypatch):
    db, _shadow, _teaching, operator = _load_modules(tmp_path, monkeypatch)
    order_id = _insert_order(db, "0757")
    before = _snapshot(db, order_id)
    started = operator.start_review_session("Review orders.", limit=5)
    operator.review_order(
        started["id"],
        order_id=order_id,
        reviewer=lambda _payload, _pdf: _model_review(),
    )

    declined = operator.decline_reviewed_orders(started["id"])

    assert declined["status"] == "completed"
    assert declined["approval_decision"] == "rejected"
    assert declined["production_action_executed"] is False
    assert _snapshot(db, order_id) == before


def test_concurrent_teach_events_receive_unique_atomic_sequences(tmp_path, monkeypatch):
    _db, _shadow, teaching, _operator = _load_modules(tmp_path, monkeypatch)
    started = teaching.start_teaching_session("Capture concurrent UI events.")

    def record(index: int):
        return teaching.record_teaching_event(
            started["id"],
            event_type="ui_action",
            module="Orders",
            message=f"Concurrent event {index}",
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        events = list(pool.map(record, range(18)))

    sequences = [event["sequence"] for event in events]
    assert len(sequences) == len(set(sequences)) == 18
    detail = teaching._session_detail(started["id"])
    stored = [event["sequence"] for event in detail["teaching_events"]]
    assert stored == list(range(1, 20))


def test_default_assisted_reviewer_uses_terra_and_strict_structured_output(tmp_path, monkeypatch):
    _db, _shadow, _teaching, operator = _load_modules(tmp_path, monkeypatch)
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Completion",
                (),
                {"choices": [type("Choice", (), {"message": type("Message", (), {"content": json.dumps(_model_review())})()})()]},
            )()

    fake_llm = type(sys)("llm")
    fake_llm.get_client = lambda: type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    fake_llm.pdf_to_png_pages = lambda _pdf, dpi=150: [b"page"]
    monkeypatch.setitem(sys.modules, "llm", fake_llm)
    monkeypatch.delenv("BETA_MODEL", raising=False)

    result = operator._default_reviewer(
        {"goal": "Review orders.", "order": {"rows": []}, "hard_rules": []},
        b"%PDF fixture",
    )

    assert result["recommendation"] == "approve"
    assert captured["model"] == "gpt-5.6-terra"
    assert captured["reasoning_effort"] == "medium"
    assert captured["response_format"]["json_schema"]["strict"] is True
    schema = captured["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
