"""Real isolated SQLite tests for inspect/prepare tools and their write boundary."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from factory_agent_tools import FactoryAgentTools, ToolError, tool_definitions
from mcp_contracts import ManualDraft, OrderRef
from services.platform_service import PlatformService
from services.platform_repository import version_of


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("factory_tools_fixture_db", ROOT / "backend" / "db.py")
    db = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, db)
    spec.loader.exec_module(db)
    db.Base.metadata.create_all(db.engine)
    draft = ManualDraft(client_name="Fixture Client", order_number="TEST-INSPECT-001", order_date="2026-10-06",
        mode="client_positions_red_index", dimension_unit="mm", reference_notes="Source notes",
        rows=[dict(section="A", position="01", client_position="Kitchen", red_index=7,
                   width_mm=1001.5, height_mm=604, quantity=2, glass_type="4F", row_notes="Preserve 01")])
    service = PlatformService(db, None, tmp_path / "invoices.json")
    created = db.create_manual_order(service._manual_payload(draft))
    ref = f"manual:{created['id']}"
    tools = FactoryAgentTools(db.DB_PATH, db_module=db)
    yield db, tools, ref, draft.model_dump(mode="json")
    db.engine.dispose()


def digest(db):
    return hashlib.sha256(Path(db.DB_PATH).read_bytes()).hexdigest()


def proposal_args(tools, ref, replacement):
    current = tools.call_tool("get_order", {"order_id": ref})
    return {"order_id": ref, "expected_version": current["version"], "replacement": replacement,
            "rationale": "Operator requested a dimension correction for review."}


def test_reads_reuse_canonical_saved_values_without_writing_or_auditing(store):
    db, tools, ref, _ = store
    before = digest(db)
    view = tools.call_tool("get_order", {"order_id": ref})
    canonical_view = PlatformService(db, None, "/dev/null").get_order(OrderRef(order_id=ref))
    assert view["version"] == canonical_view["version"]
    assert view["rows"][0]["width_mm"] == 1001.5
    assert view["rows"][0]["position"] == canonical_view["rows"][0]["position"] == "Kitchen 7"
    assert view["rows"][0]["client_position"] == "Kitchen"
    assert view["rows"][0]["red_index"] == 7
    assert view["dimension_unit"] == "mm"
    assert "raw_values" not in view and view["artifacts"] == []
    page = tools.call_tool("list_orders", {"client": "Fixture", "limit": 1})
    assert page["total"] == 1 and page["items"][0]["order_id"] == ref
    summary = tools.call_tool("get_platform_summary", {})
    assert summary["draft_count"] == 1 and summary["pieces"] == 2
    assert before == digest(db)
    with db.read_session() as session:
        assert session.scalar(text("SELECT COUNT(*) FROM mcp_audit")) == 0


def test_prepared_change_is_complete_review_data_with_no_order_mutation(store):
    db, tools, ref, replacement = store
    args = proposal_args(tools, ref, replacement)
    replacement["rows"][0]["width_mm"] = 975
    replacement["reference_notes"] = "Reviewed proposal only"
    before = digest(db)
    result = tools.call_tool("prepare_change", args)
    proposal = result["proposal"]
    assert proposal["status"] == "pending" and proposal["applied"] is False
    assert proposal["order_id"] == ref and proposal["source_version"] == args["expected_version"]
    assert proposal["title"] and proposal["summary"] == args["rationale"]
    assert proposal["current_snapshot"]["rows"][0]["width_mm"] == 1001.5
    assert proposal["replacement"]["rows"][0]["width_mm"] == 975
    assert {"field": "rows[0].width_mm", "before": 1001.5, "after": 975} in proposal["changes"]
    assert digest(db) == before
    assert db.get_manual_order(int(ref.split(":")[1]))["rows"][0]["width_mm"] == 1001.5


@pytest.mark.parametrize("name", ["approve_order", "update_manual_order_draft", "create_invoice_draft", "print", "delete_order", "invoke", "__dict__"])
def test_no_write_or_general_dispatch_tool(store, name):
    db, tools, _, _ = store
    before = digest(db)
    with pytest.raises(ToolError, match="unavailable") as error:
        tools.call_tool(name, {})
    assert error.value.code == "TOOL_DENIED" and digest(db) == before
    assert {item["name"] for item in tool_definitions()} == {
        "list_orders", "get_order", "get_platform_summary", "get_processing_job", "prepare_change"}


@pytest.mark.parametrize("statement", [
    "UPDATE manual_orders SET client_name='Buggy service wrote'",
    "PRAGMA query_only=OFF", "ATTACH DATABASE ':memory:' AS escape",
    "CREATE TABLE forbidden (id INTEGER)", "DELETE FROM manual_order_rows",
])
def test_sqlite_enforces_no_writes_even_if_read_service_is_buggy(store, monkeypatch, statement):
    db, tools, ref, _ = store
    before = digest(db)
    def buggy_get_order(self, _args):
        with self.db.read_session() as session:
            session.execute(text(statement))
            session.commit()
        return {"written": True}
    monkeypatch.setattr(PlatformService, "get_order", buggy_get_order)
    with pytest.raises(ToolError) as error:
        tools.call_tool("get_order", {"order_id": ref})
    assert error.value.code == "READ_FAILED"
    assert statement not in str(error.value)
    assert digest(db) == before


def test_database_facade_exposes_no_legacy_writer_or_engine(store, monkeypatch):
    _, tools, ref, _ = store
    def inspect(self, _args):
        assert not hasattr(self.db, "get_session")
        assert not hasattr(self.db, "update_manual_order")
        assert not hasattr(self.db, "engine")
        assert not hasattr(self.db, "SessionLocal")
        return {"read_only": True}
    monkeypatch.setattr(PlatformService, "get_order", inspect)
    assert tools.call_tool("get_order", {"order_id": ref}) == {"read_only": True}


def test_stale_version_protected_order_and_duplicate_indices_fail_without_proposal(store):
    db, tools, ref, replacement = store
    args = proposal_args(tools, ref, replacement)
    args["expected_version"] = "0" * 64
    with pytest.raises(ToolError) as error:
        tools.call_tool("prepare_change", args)
    assert error.value.code == "VERSION_CONFLICT"
    args["expected_version"] = tools.call_tool("get_order", {"order_id": ref})["version"]
    args["replacement"]["rows"].append(deepcopy(args["replacement"]["rows"][0]))
    with pytest.raises(ToolError) as error:
        tools.call_tool("prepare_change", args)
    assert error.value.code == "VALIDATION_ERROR"
    db.update_manual_order_status(int(ref.split(":")[1]), status="approved")
    args["expected_version"] = tools.call_tool("get_order", {"order_id": ref})["version"]
    with pytest.raises(ToolError) as error:
        tools.call_tool("prepare_change", args)
    assert error.value.code == "ORDER_PROTECTED"


@pytest.mark.parametrize("name,args", [
    ("get_order", {"order_id": "manual:1", "token": "injected"}),
    ("get_order", {"order_id": "manual:1; DROP TABLE orders"}),
    ("list_orders", {"limit": 26}), ("list_orders", {"offset": 10001}),
    ("get_platform_summary", {"write": True}), ("get_processing_job", {"processing_job_id": "../../etc/passwd"}),
])
def test_strict_schema_rejects_extra_fields_and_unbounded_requests(store, name, args):
    _, tools, _, _ = store
    with pytest.raises(ToolError) as error:
        tools.call_tool(name, args)
    assert error.value.code == "INVALID_ARGUMENTS"


def test_result_size_limit_and_credentials_never_leave_backend(store, monkeypatch):
    _, tools, ref, _ = store
    monkeypatch.setattr(PlatformService, "get_order", lambda *_: {"notes": "x" * 256001})
    with pytest.raises(ToolError) as error:
        tools.call_tool("get_order", {"order_id": ref})
    assert error.value.code == "RESULT_LIMIT"
    secret = "server-only-test-secret-123456"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(PlatformService, "get_order", lambda *_: {"notes": "accidental " + secret})
    with pytest.raises(ToolError) as error:
        tools.call_tool("get_order", {"order_id": ref})
    assert error.value.code == "SENSITIVE_DATA" and secret not in str(error.value)


def test_missing_database_is_not_created(store, tmp_path):
    db, _, _, _ = store
    missing = tmp_path / "missing" / "orders.db"
    tools = FactoryAgentTools(missing, db_module=db)
    with pytest.raises(ToolError) as error:
        tools.call_tool("get_platform_summary", {})
    assert error.value.code == "SETUP_REQUIRED" and not missing.parent.exists()


def test_processing_snapshot_read_does_not_regenerate_or_modify_job(store):
    db, tools, ref, _ = store
    snapshot = db.get_manual_order(int(ref.split(":")[1]))
    job_id = "job:" + "a" * 32
    with db.get_session() as session:
        session.add(db.WorkflowJob(id=job_id, order_ref=ref, order_version=tools.call_tool("get_order", {"order_id": ref})["version"],
            snapshot_json=json.dumps(snapshot), result_json=json.dumps({"rows": snapshot["rows"], "preview": {"groups": []}})))
    before = digest(db)
    job = tools.call_tool("get_processing_job", {"processing_job_id": job_id})
    assert job["processing_job_id"] == job_id and job["rows"][0]["width_mm"] == 1001.5
    assert job["artifacts"] == [] and digest(db) == before


def test_pdf_saved_snapshot_version_matches_existing_reader_without_raw_text_export(store):
    db, tools, _, _ = store
    with db.get_session() as session:
        order = db.Order(id=9001, source="pdf", client_name="PDF Fixture", order_numbers_raw='["PDF-001"]',
                         units_total=2, area_total=1.4, status="reviewed")
        order.rows = [db.OrderRow(order_number="PDF-001", type="4F", dimension="975/995 x 800", position="007", quantity=2, area=1.4)]
        order.extraction = db.Extraction(raw_input="Original source text should not be exported", llm_output_json="{}", model_used="test")
        session.add(order)
    before = digest(db)
    view = tools.call_tool("get_order", {"order_id": "pdf:9001"})
    assert view["version"] == version_of(db.get_order_with_extraction(9001))
    assert view["rows"][0]["dimension"] == "975/995 x 800"
    assert view["rows"][0]["position"] == "007"
    assert "Original source text" not in json.dumps(view)
    assert "extraction" not in view and view["artifacts"] == []
    assert before == digest(db)


def test_proposals_reject_duplicate_order_numbers_and_noop(store):
    db, tools, ref, replacement = store
    current = tools.call_tool("get_order", {"order_id": ref})
    replacement["rows"][0]["position"] = current["rows"][0]["position"]
    args = proposal_args(tools, ref, replacement)
    with pytest.raises(ToolError) as error:
        tools.call_tool("prepare_change", args)
    assert error.value.code == "NO_CHANGES"
    second = deepcopy(replacement)
    second["order_number"] = "OTHER-001"
    service = PlatformService(db, None, "/dev/null")
    db.create_manual_order(service._manual_payload(ManualDraft.model_validate(second)))
    args["replacement"]["order_number"] = "OTHER-001"
    with pytest.raises(ToolError) as error:
        tools.call_tool("prepare_change", args)
    assert error.value.code == "DUPLICATE_ORDER_NUMBER"
