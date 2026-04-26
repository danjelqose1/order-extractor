from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"


def _install_fake_db(monkeypatch) -> Dict[str, List[Dict[str, Any]]]:
    calls: Dict[str, List[Dict[str, Any]]] = {
        "insert_extraction_with_rows": [],
        "update_order_rows": [],
    }

    fake_db = types.ModuleType("db")
    fake_db.init_db = lambda: None
    fake_db.find_similar_corrections = lambda *args, **kwargs: []
    fake_db.bump_correction_hit = lambda *args, **kwargs: None

    def _insert_extraction_with_rows(**kwargs):
        calls["insert_extraction_with_rows"].append(kwargs)
        return {"order_id": 1, "status": "draft"}

    def _update_order_rows(*args, **kwargs):
        calls["update_order_rows"].append({"args": args, "kwargs": kwargs})
        return {}

    fake_db.insert_extraction_with_rows = _insert_extraction_with_rows
    fake_db.update_order_rows = _update_order_rows
    fake_db.update_order_status = lambda *args, **kwargs: {}
    fake_db.get_orders = lambda *args, **kwargs: []
    fake_db.get_orders_by_identifiers = lambda *args, **kwargs: []
    fake_db.get_order_with_extraction = lambda *args, **kwargs: None
    fake_db.delete_order = lambda *args, **kwargs: False
    fake_db.get_all_rows_for_export = lambda *args, **kwargs: []
    fake_db.save_correction = lambda *args, **kwargs: {}
    fake_db.list_corrections = lambda *args, **kwargs: []
    fake_db.delete_correction = lambda *args, **kwargs: False
    fake_db.normalize_order_status = lambda status, default="draft": (str(status or "").strip().lower() or default)
    fake_db.is_processing_eligible_status = lambda status: str(status or "").strip().lower() in {"approved", "in_production", "completed"}
    fake_db.ORDER_STATUS_SEQUENCE = ("draft", "reviewed", "approved", "in_production", "completed", "archived")
    fake_db.APPROVABLE_STATUSES = {"draft", "reviewed"}
    monkeypatch.setitem(sys.modules, "db", fake_db)
    return calls


def _load_app(monkeypatch, legacy_enabled: str = "false"):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-5-mini")
    monkeypatch.setenv("LEGACY_OCR_ENABLED", legacy_enabled)

    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    calls = _install_fake_db(monkeypatch)

    for module_name in [
        "app",
        "llm",
        "prompts",
        "schema",
        "validators",
        "dimension_repair",
        "area_dimension_validator",
        "utils_text",
    ]:
        sys.modules.pop(module_name, None)

    app_module = importlib.import_module("app")
    return app_module, calls


def _load_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EXTRACTION_MODEL", "gpt-5-mini")
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    _install_fake_db(monkeypatch)
    for module_name in ["llm", "prompts", "utils_text"]:
        sys.modules.pop(module_name, None)
    return importlib.import_module("llm")


def _bundle(rows: List[Dict[str, Any]], warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    payload = {
        "order_number": rows[0]["order_number"] if rows else "",
        "client_name": "Client A",
        "rows": rows,
        "warnings": warnings or [],
        "confidence": 0.92,
    }
    return {
        "data": payload,
        "raw": payload,
        "raw_response": {"id": "resp_test", "model": "gpt-5-mini"},
        "model_used": "gpt-5-mini",
        "prepared_text": "",
        "declared_units": None,
        "declared_area": None,
        "applied_corrections": [],
        "output_text": "",
    }


def test_text_pdf_extracts_rows(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")

    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "R-26-1001",
                "type": "2 VETRI 33.1F + 14 + 33.1 LOWE C.CALDO 28mm",
                "dimension": "520x1168",
                "position": "1-1",
                "quantity": 1,
                "area": 0.607,
            }
        ]
    )
    client = TestClient(app_module.app)
    response = client.post(
        "/extract_pdf",
        files={"file": ("text-order.pdf", b"%PDF-1.7\ntext-pdf", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["order_number"] == "R-26-1001"
    assert len(data["rows"]) == 1
    assert data["rows"][0]["dimension"] == "520x1168"
    assert data["extraction_method"] == "base64_pdf_visual"
    assert data["model_used"] == "gpt-5-mini"

    stored = calls["insert_extraction_with_rows"][0]
    assert stored["raw_input"].startswith("data:application/pdf;base64,")
    assert stored["model_used"] == "gpt-5-mini"


def test_pdf_visual_llm_uses_input_file_payload(monkeypatch):
    llm_module = _load_llm(monkeypatch)
    captured: Dict[str, Any] = {}

    class _Response:
        status_code = 200

        def json(self):
            return {
                "id": "resp_1",
                "model": "gpt-5-mini",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"order_number":"R-26-9901","client_name":"A",'
                                    '"rows":[{"order_number":"R-26-9901","type":"X","dimension":"500x500",'
                                    '"position":"1-1","quantity":1,"area":0.25}],"warnings":[],"confidence":0.9}'
                                ),
                            }
                        ],
                    }
                ],
            }

    def _fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _Response()

    llm_module.httpx.post = _fake_post
    bundle = llm_module.call_llm_for_pdf_base64_visual(b"%PDF-1.7\nabc", filename="sample.pdf")

    user_content = captured["json"]["input"][1]["content"]
    file_item = [item for item in user_content if item.get("type") == "input_file"][0]
    assert captured["url"].endswith("/v1/responses")
    assert file_item["filename"] == "sample.pdf"
    assert file_item["file_data"].startswith("data:application/pdf;base64,")
    assert bundle["data"]["order_number"] == "R-26-9901"


def test_scanned_pdf_extracts_rows(monkeypatch):
    app_module, _ = _load_app(monkeypatch, legacy_enabled="false")

    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "R-26-1002",
                "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
                "dimension": "704x2301",
                "position": "12-3",
                "quantity": 2,
                "area": 3.24,
            }
        ],
        warnings=["visual_parse_note: scanned image style PDF"],
    )
    client = TestClient(app_module.app)
    response = client.post(
        "/extract_pdf",
        files={"file": ("scan-order.pdf", b"%PDF-1.7\nscan-pdf", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 1
    assert any("scanned image style PDF" in w for w in data["warnings"])


def test_invalid_rows_are_flagged(monkeypatch):
    app_module, _ = _load_app(monkeypatch, legacy_enabled="false")

    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "",
                "type": "",
                "dimension": "",
                "position": "",
                "quantity": 1,
                "area": 0.0,
            }
        ]
    )
    client = TestClient(app_module.app)
    response = client.post(
        "/extract_pdf",
        files={"file": ("invalid-order.pdf", b"%PDF-1.7\ninvalid", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    flattened = [msg for row in body["row_warnings"].values() for msg in row]
    assert any("missing_required_field:dimension" in msg for msg in flattened)
    assert any("missing_required_field:position" in msg for msg in flattened)
    assert any("missing_required_field:type" in msg for msg in flattened)


def test_approved_orders_not_overwritten(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")

    app_module.insert_extraction_with_rows = lambda **kwargs: {"order_id": 77, "status": "approved"}
    app_module.update_order_rows = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("update_order_rows must not be called during extraction")
    )
    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "R-26-1003",
                "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
                "dimension": "600x1200",
                "position": "2-1",
                "quantity": 1,
                "area": 0.72,
            }
        ]
    )

    client = TestClient(app_module.app)
    response = client.post(
        "/extract_pdf",
        files={"file": ("approved-check.pdf", b"%PDF-1.7\napproved", "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "approved"
    assert data["saved_order_id"] == 77
    assert calls["update_order_rows"] == []


def test_legacy_ocr_not_called_when_disabled(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")

    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: (_ for _ in ()).throw(
        RuntimeError("visual failure")
    )
    legacy_calls = {"count": 0}

    def _legacy(raw_bytes):
        legacy_calls["count"] += 1
        return {}

    app_module._extract_pdf_via_legacy_ocr = _legacy
    client = TestClient(app_module.app)
    response = client.post(
        "/extract_pdf",
        files={"file": ("failure.pdf", b"%PDF-1.7\nfailure", "application/pdf")},
    )

    assert response.status_code == 502
    assert "Base64 PDF extraction failed" in response.json()["detail"]
    assert legacy_calls["count"] == 0
    assert calls["insert_extraction_with_rows"] == []
