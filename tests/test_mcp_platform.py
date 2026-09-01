from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import fitz
import pytest
from fastapi.testclient import TestClient
from jsonschema import validate

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "local-test-only-not-a-secret-0123456789"


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    monkeypatch.setenv("ORDER_EXTRACTOR_LOAD_DOTENV", "false")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_ENABLED", "true")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_AUTH_MODE", "bearer")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_TOKEN", TOKEN)
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_EXPENSIVE_PER_MINUTE", "100")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-no-network")
    monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
    sys.path.insert(0, str(ROOT / "backend"))
    names = [n for n in sys.modules if n in {"app", "db", "llm", "workspace_service", "workspace_agent"} or n.startswith(("workspace_agents.", "mcp_server", "services.platform_"))]
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("app")
    with TestClient(module.app) as client:
        counter = 0
        headers = {"Authorization": "Bearer " + TOKEN, "Accept": "application/json, text/event-stream"}
        def rpc(method, params):
            nonlocal counter
            counter += 1
            response = client.post("/mcp", headers=headers, json={"jsonrpc":"2.0", "id":counter, "method":method, "params":params})
            assert response.status_code == 200, response.text
            body = response.json()
            assert "error" not in body, body
            return body["result"]
        def call(name, args=None, error=None):
            result = rpc("tools/call", {"name":name, "arguments":args or {}})
            structured = result["structuredContent"]
            if error:
                assert result["isError"] and structured["error"]["code"] == error, structured
                return structured
            assert not result.get("isError"), structured
            return structured["data"]
        yield module, client, rpc, call, headers


def draft(**changes):
    data = dict(client_name="Test Client", order_number="MCP-001", order_date="2026-09-01", mode="standard",
                dimension_unit="mm", reference_notes="keep raw", idempotency_key="test-create-001",
                rows=[dict(position="A", width_mm=1001, height_mm=604, quantity=2, glass_type="4F")])
    data.update(changes)
    return data


def approve(call, order):
    return call("approve_order", dict(order_id=order["order_id"], expected_version=order["version"], confirmed=True))


def process(call, order):
    return call("send_order_to_processing", dict(order_id=order["order_id"], expected_version=order["version"], confirmed=True, idempotency_key="test-process-001"))


def test_initialize_discovery_annotations_and_output_schemas(api):
    _, _, rpc, call, _ = api
    init = rpc("initialize", dict(protocolVersion="2025-06-18", capabilities={}, clientInfo={"name":"pytest","version":"1"}))
    assert init["serverInfo"]["name"] == "order-extractor"
    tools = rpc("tools/list", {})["tools"]
    assert len(tools) == 18
    readonly = {"platform_health","list_orders","get_order","list_manual_orders","get_manual_order","get_processing_job","get_platform_summary","list_order_artifacts"}
    for tool in tools:
        assert tool["annotations"]["readOnlyHint"] == (tool["name"] in readonly)
        assert tool["annotations"]["openWorldHint"] is False
        assert tool["inputSchema"]["additionalProperties"] is False
        assert tool["outputSchema"]
        if tool["name"] in readonly:
            assert tool["annotations"]["destructiveHint"] is False
    assert next(t for t in tools if t["name"] == "delete_manual_order_draft")["annotations"]["destructiveHint"]
    response = rpc("tools/call", {"name":"platform_health","arguments":{}})["structuredContent"]
    validate(response, next(t for t in tools if t["name"] == "platform_health")["outputSchema"])
    assert response["data"]["database"] == "ready"


def test_authentication_all_transport_methods_and_artifacts(api, monkeypatch):
    _, client, _, call, headers = api
    for method in ("GET", "POST", "DELETE"):
        assert client.request(method, "/mcp").status_code == 401
    assert client.get("/mcp/artifacts/no-such-id").status_code == 401
    assert client.post("/mcp", headers={**headers,"Authorization":"Bearer wrong"}).status_code == 401
    assert client.post("/mcp?token="+TOKEN).status_code == 401
    assert client.post("/mcp", headers={**headers,"Origin":"https://untrusted.example"}).status_code == 403
    monkeypatch.delenv("ORDER_EXTRACTOR_MCP_TOKEN")
    assert client.post("/mcp", headers=headers).status_code == 503
    assert client.get("/healthz").status_code == 200


def test_standard_draft_exact_area_raw_override_and_high_quantity(api):
    module, _, _, call, _ = api
    payload = draft(rows=[dict(width_mm=1200, height_mm=800, quantity=1234, glass_type="  My RAW Glass  ", area_override_m2=42.12345)])
    order = call("create_manual_order_draft", payload)
    assert order["status"] == "draft"
    assert order["piece_count"] == 1234
    assert order["warnings"]
    assert order["calculated_area_m2"] == module.db_module._manual_area(1200,800,1234)
    assert order["total_area_m2"] == 42.123
    assert order["raw_values"]["rows"][0]["glass_type"] == "  My RAW Glass  "
    assert order["raw_values"]["rows"][0]["area_override_m2"] == 42.12345
    assert "idempotency_key" not in order["raw_values"]
    assert module.db_module.get_orders(year="all") == []


def red_draft():
    return draft(mode="client_positions_red_index", rows=[dict(section="Kitchen", client_position="A", red_index=i, width_mm=1001, height_mm=604, quantity=i, glass_type="4F", row_notes="Keep this") for i in (1,2)])


def test_red_indexes_and_repeated_client_positions(api):
    _, _, _, call, _ = api
    order = call("create_manual_order_draft", red_draft())
    assert [r["client_position"] for r in order["rows"]] == ["A","A"]
    assert [r["red_index"] for r in order["rows"]] == [1,2]
    assert order["piece_count"] == 3


def test_duplicate_red_indexes_report_exact_field(api):
    _, _, _, call, _ = api
    payload = red_draft()
    payload["rows"][1]["red_index"] = 1
    error = call("create_manual_order_draft", payload, error="VALIDATION_ERROR")
    assert error["error"]["issues"][0]["field"] == "rows[1].red_index"


@pytest.mark.parametrize("field,value", [("width_mm",0),("height_mm",-2),("quantity",0),("quantity",1.5),("quantity",True),("glass_type"," ")])
def test_invalid_rows_are_rejected_without_echoing_inputs(api, field, value):
    _, _, _, call, _ = api
    payload = draft()
    payload["rows"][0][field] = value
    result = call("create_manual_order_draft", payload, error="VALIDATION_ERROR")
    assert result["error"]["issues"][0]["field"] == f"rows.0.{field}"
    assert "input" not in json.dumps(result["error"])
    assert call("list_manual_orders")["total"] == 0


def test_duplicate_number_conflict_idempotency_and_concurrent_retry(api):
    _, _, _, call, _ = api
    payload = draft()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: call("create_manual_order_draft", payload), range(2)))
    assert results[0] == results[1]
    assert call("list_manual_orders")["total"] == 1
    changed = {**payload,"client_name":"Different"}
    call("create_manual_order_draft", changed, error="IDEMPOTENCY_CONFLICT")
    call("create_manual_order_draft", {**payload,"idempotency_key":"new-key-001"}, error="DUPLICATE_ORDER_NUMBER")


def test_replace_draft_version_and_protected_later_statuses(api):
    _, _, _, call, _ = api
    order = call("create_manual_order_draft", draft())
    replacement = draft(client_name="Changed")
    replacement.pop("idempotency_key")
    args = dict(order_id=order["order_id"], expected_version=order["version"], replacement=replacement)
    changed = call("update_manual_order_draft", args)
    assert changed["client_name"] == "Changed" and changed["version"] != order["version"]
    call("update_manual_order_draft", args, error="VERSION_CONFLICT")
    approved = approve(call, changed)
    call("update_manual_order_draft", {**args,"expected_version":approved["version"]}, error="ORDER_PROTECTED")
    call("delete_manual_order_draft", dict(order_id=approved["order_id"],expected_version=approved["version"],confirmed=True), error="ORDER_PROTECTED")


def test_confirmation_and_readonly_authorization(api, monkeypatch):
    _, _, _, call, _ = api
    order = call("create_manual_order_draft", draft())
    call("approve_order",dict(order_id=order["order_id"],expected_version=order["version"]),error="VALIDATION_ERROR")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_READ_ONLY","true")
    call("create_manual_order_draft",draft(order_number="other"),error="FORBIDDEN")
    assert call("get_order",{"order_id":order["order_id"]})["status"] == "draft"


@pytest.mark.parametrize("mode",["standard","client_positions_red_index"])
def test_real_processing_rounding_grouping_and_persistent_documents(api, mode, tmp_path):
    module, client, _, call, headers = api
    order = approve(call, call("create_manual_order_draft", red_draft() if mode != "standard" else draft()))
    job = process(call, order)
    assert not job["rounding_applied"] and not job["grouped"]
    job = call("apply_danko_rounding",dict(processing_job_id=job["processing_job_id"],expected_version=job["version"]))
    assert job["rows"][0]["width"] == 1000 and job["rows"][0]["height"] == 605
    assert job["original_rows"][0]["width_mm"] == 1001
    assert job["rows"][0]["area"] == order["rows"][0]["final_area_m2"]
    job = call("group_processing_dimensions",dict(processing_job_id=job["processing_job_id"],expected_version=job["version"]))
    origins = [origin for group in job["groups"] for line in group["lines"] for origin in line["originRows"]]
    assert sum(origin["quantity"] for origin in origins) == order["piece_count"]
    if mode != "standard":
        assert {origin["red_index"] for origin in origins} == {1,2}
        assert all(origin["section"] == "Kitchen" for origin in origins)
    for tool in ("generate_processing_sheet","generate_labels"):
        args=dict(processing_job_id=job["processing_job_id"],expected_version=job["version"],idempotency_key=tool)
        artifact = call(tool,args)["artifacts"][0]
        assert call(tool,args)["artifacts"][0] == artifact
        response = client.get(artifact["download_path"],headers=headers)
        assert response.status_code == 200
        pdf = fitz.open(stream=response.content,filetype="pdf")
        assert pdf.page_count > 0
        text = " ".join(page.get_text() for page in pdf)
        assert "MCP-001" in text and "4F" in text
        if tool == "generate_labels":
            assert pdf.page_count == order["piece_count"]
            assert pdf[0].rect.width == pytest.approx(100 * 72 / 25.4,abs=.02)
            assert pdf[0].rect.height == pytest.approx(40 * 72 / 25.4,abs=.02)
        # A new repository instance reads the persisted bytes, not a temporary file.
        from services.platform_repository import Repository
        assert Repository(module.db_module).artifact_content(artifact["artifact_id"])[0] == response.content
    assert len(call("list_order_artifacts",{"order_id":order["order_id"]})["artifacts"]) == 2
    assert module.db_module.get_orders(year="all") == []


def test_processing_rollback_on_failure_and_source_change(api, monkeypatch):
    module, _, _, call, _ = api
    order = approve(call,call("create_manual_order_draft",draft()))
    import services.platform_service as service
    original=service.run_workflow
    monkeypatch.setattr(service,"run_workflow",lambda *a,**kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("worker",1)))
    args=dict(order_id=order["order_id"],expected_version=order["version"],confirmed=True,idempotency_key="timeout-test")
    call("send_order_to_processing",args,error="GENERATION_TIMEOUT")
    assert call("get_order",{"order_id":order["order_id"]})["status"] == "approved"
    monkeypatch.setattr(service,"run_workflow",original)
    job=call("send_order_to_processing",args)
    legacy=module.db_module.get_manual_order(1)
    legacy["notes"]="changed by website"
    module.db_module.update_manual_order(1,legacy)
    call("generate_labels",dict(processing_job_id=job["processing_job_id"],expected_version=job["version"],idempotency_key="stale-test"),error="ORDER_CHANGED")


def test_invoice_shared_engine_and_api_store(api):
    module, client, _, call, _ = api
    order=call("create_manual_order_draft",draft(rows=[dict(width_mm=1000,height_mm=1000,quantity=2,glass_type="4F+12+4 LowE")]))
    args=dict(order_id=order["order_id"],expected_version=order["version"],idempotency_key="invoice-001")
    invoice=call("create_invoice_draft",args)
    assert invoice["status"] == "draft" and invoice["safe_to_price"]
    assert invoice["invoice"]["calculated"]["area"] == 2
    assert invoice["invoice"]["calculated"]["total"] == 7000
    assert call("create_invoice_draft",args)["invoice_id"] == invoice["invoice_id"]
    jobs=client.get("/api/invoices").json()["jobs"]
    assert len(jobs)==1 and jobs[0]["id"]==invoice["invoice_id"]


def test_delete_draft_only_and_no_cascade(api):
    _, _, _, call, _ = api
    order=call("create_manual_order_draft",draft())
    assert call("delete_manual_order_draft",dict(order_id=order["order_id"],expected_version=order["version"],confirmed=True))["deleted"]
    assert call("list_manual_orders")["total"]==0


def test_pdf_existing_validation_and_processing(api):
    module, _, _, call, _ = api
    rows=[dict(order_number="PDF-001",type="4F",dimension="1001x604",position="1",quantity=2,area=1.209)]
    created=module.db_module.insert_extraction_with_rows(source="pdf",rows=rows,raw_input="test",prepared_text="test",llm_output_json="{}",model_used="test",hash_value="test-pdf",confidence=1,client_name="PDF Client")
    order=call("get_order",{"order_id":f"pdf:{created['order_id']}"})
    approved=approve(call,order)
    job=process(call,approved)
    assert job["order_id"].startswith("pdf:") and job["rows"][0]["qty"]==2
    assert module.db_module.get_manual_order(1) is None
    call("create_manual_order_draft",draft(order_number="PDF-001"),error="DUPLICATE_ORDER_NUMBER")


def test_filters_pagination_summary_and_audit_no_contents(api):
    module, _, _, call, _ = api
    for i in range(3):
        call("create_manual_order_draft",draft(order_number=f"M-{i}",idempotency_key=f"create-{i:04d}",order_date=f"202{4+i}-09-01"))
    page=call("list_orders",dict(source="manual",year=2025,client="Test",date_from="2025-01-01",date_to="2025-12-31",limit=1))
    assert page["total"]==1 and page["items"][0]["order_number"]=="M-1"
    page=call("list_manual_orders",dict(limit=1,offset=1))
    assert page["total"]==3 and page["has_more"]
    summary=call("get_platform_summary")
    assert summary["draft_count"]==3 and summary["pieces"]==6
    from sqlalchemy import select
    with module.db_module.read_session() as session:
        audits=session.scalars(select(module.db_module.McpAudit)).all()
        assert audits and all(a.actor=="private-operator" and a.request_id for a in audits)
        assert all("Test Client" not in a.outcome and TOKEN not in a.outcome for a in audits)


def test_artifact_failure_rolls_back_bytes_invoice_and_idempotency(api, monkeypatch):
    module, _, _, call, _ = api
    order=call("create_manual_order_draft",draft())
    service=module.app.state.mcp_service
    monkeypatch.setattr(service.repo,"add_artifact",lambda *a,**kw: (_ for _ in ()).throw(RuntimeError("test")))
    call("create_invoice_draft",dict(order_id=order["order_id"],expected_version=order["version"],idempotency_key="rollback-invoice"),error="INTERNAL_ERROR")
    assert module._load_invoices()=={"jobs":[]}
    assert service.repo.replay("private-operator","create_invoice_draft","rollback-invoice") is None


def test_deleted_ids_are_never_reused(api):
    _, _, _, call, _ = api
    first=call("create_manual_order_draft",draft())
    call("delete_manual_order_draft",dict(order_id=first["order_id"],expected_version=first["version"],confirmed=True))
    second=call("create_manual_order_draft",draft(order_number="MCP-002",idempotency_key="different-create"))
    assert first["order_id"] != second["order_id"]
    call("get_order",{"order_id":first["order_id"]},error="NOT_FOUND")


def test_danko_all_digits_and_fractional_rules_are_unchanged(api):
    _, _, _, call, _ = api
    quantities=list(range(1,12))
    widths=[1000+i for i in range(10)]+[1001.5]
    payload=draft(rows=[dict(width_mm=w,height_mm=800,quantity=q,glass_type="4F") for w,q in zip(widths,quantities)])
    order=approve(call,call("create_manual_order_draft",payload))
    job=process(call,order)
    rounded=call("apply_danko_rounding",dict(processing_job_id=job["processing_job_id"],expected_version=job["version"]))
    assert [r["width"] for r in rounded["rows"]]==[1000,1000,1000,1003,1005,1005,1005,1005,1008,1010,1001.5]
    assert [r["qty"] for r in rounded["rows"]]==quantities
    assert [r["area"] for r in rounded["rows"]]==[r["final_area_m2"] for r in order["rows"]]


def test_generation_limits_and_rate_limits_preserve_order(api, monkeypatch):
    module, _, _, call, _ = api
    order=approve(call,call("create_manual_order_draft",draft()))
    job=process(call,order)
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_MAX_LABELS","1")
    call("generate_labels",dict(processing_job_id=job["processing_job_id"],expected_version=job["version"],idempotency_key="labels-limit"),error="GENERATION_LIMIT")
    monkeypatch.setenv("ORDER_EXTRACTOR_MCP_EXPENSIVE_PER_MINUTE","1")
    call("apply_danko_rounding",dict(processing_job_id=job["processing_job_id"],expected_version=job["version"]),error="RATE_LIMITED")
    assert call("get_processing_job",{"processing_job_id":job["processing_job_id"]})["original_rows"] == job["original_rows"]
    from sqlalchemy import select
    with module.db_module.read_session() as session:
        audit = session.scalars(select(module.db_module.McpAudit).where(
            module.db_module.McpAudit.tool == "generate_labels")).all()
        assert len(audit) == 1
        assert audit[0].order_ref == order["order_id"]
        assert audit[0].outcome == "GENERATION_LIMIT"


def test_legacy_invoice_migration_preserves_existing_jobs(api):
    module, _, _, call, _ = api
    module.INVOICES_PATH.write_text(json.dumps({"jobs":[{"id":"legacy","status":"draft"}]}))
    order=call("create_manual_order_draft",draft())
    call("create_invoice_draft",dict(order_id=order["order_id"],expected_version=order["version"],idempotency_key="migrate-invoice"))
    assert [j["id"] for j in module._load_invoices()["jobs"]][0] == "legacy"
    assert json.loads(module.INVOICES_PATH.read_text())["jobs"] == [{"id":"legacy","status":"draft"}]


def test_legacy_artifacts_are_authenticated_confined_and_content_addressed(api, tmp_path):
    module, client, _, call, headers = api
    db = module.db_module
    created = db.insert_extraction_with_rows(source="pdf", rows=[dict(order_number="OLD-1",type="4F",dimension="1000x600",position="1",quantity=1,area=.6)],
        raw_input="test",prepared_text="test",llm_output_json="{}",model_used="test",hash_value="legacy-test",confidence=1,client_name="Legacy")
    production = tmp_path / "production-files"
    production.mkdir()
    path = production / "old-sheet.pdf"
    pdf = fitz.open()
    pdf.new_page().insert_text((30,30),"Existing production sheet")
    content = pdf.tobytes()
    pdf.close()
    path.write_bytes(content)
    outside = tmp_path / "private.pdf"
    outside.write_bytes(content)
    with db.get_session() as session:
        batch = db.ProcessingBatch(order_id=created["order_id"],order_number="OLD-1")
        session.add(batch)
        session.flush()
        for file in (path, outside):
            session.add(db.ProductionFile(order_id=created["order_id"],processing_batch_id=batch.id,
                order_number="OLD-1",file_type="processing_pdf",file_path=str(file),download_url="",status="ready"))
    artifacts = call("list_order_artifacts",{"order_id":f"pdf:{created['order_id']}"})["artifacts"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["artifact_id"].startswith("legacy:")
    assert "file_path" not in artifact
    assert client.get(artifact["download_path"]).status_code == 401
    assert client.get(artifact["download_path"],headers=headers).content == content
    path.write_bytes(content+b"\n% changed")
    assert client.get(artifact["download_path"],headers=headers).status_code == 404


@pytest.mark.parametrize("status",["approved","processing","finished","cancelled"])
def test_every_later_manual_status_blocks_edit_and_delete(api, status):
    module, _, _, call, _ = api
    order = call("create_manual_order_draft",draft())
    saved = module.db_module.get_manual_order(int(order["order_id"].split(":")[1]))
    saved["status"] = status
    module.db_module.update_manual_order(saved["id"],saved)
    current = call("get_manual_order",{"order_id":order["order_id"]})
    replacement = draft()
    replacement.pop("idempotency_key")
    call("update_manual_order_draft",dict(order_id=order["order_id"],expected_version=current["version"],replacement=replacement),error="ORDER_PROTECTED")
    call("delete_manual_order_draft",dict(order_id=order["order_id"],expected_version=current["version"],confirmed=True),error="ORDER_PROTECTED")
