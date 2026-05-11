from __future__ import annotations

import asyncio
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
        "update_order_status": [],
        "create_telegram_file_record": [],
        "update_telegram_file_record": [],
        "touch_telegram_file_record": [],
        "soft_delete_telegram_file_record": [],
        "mark_telegram_file_labels_printed": [],
        "mark_telegram_file_linked_order_opened": [],
        "mark_telegram_file_pdf_printed": [],
        "find_telegram_file_by_sha256": [],
        "find_possible_duplicate_order": [],
        "create_whatsapp_file_record": [],
        "update_whatsapp_file_record": [],
        "mark_whatsapp_file_deleted": [],
    }
    whatsapp_records: Dict[int, Dict[str, Any]] = {}

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

    def _update_order_status(*args, **kwargs):
        calls["update_order_status"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, "status": kwargs.get("status")}

    def _create_telegram_file_record(**kwargs):
        calls["create_telegram_file_record"].append(kwargs)
        return {"id": len(calls["create_telegram_file_record"]), "touched": False, "touched_at": None, **kwargs}

    def _update_telegram_file_record(*args, **kwargs):
        calls["update_telegram_file_record"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, **kwargs}

    def _touch_telegram_file_record(*args, **kwargs):
        calls["touch_telegram_file_record"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, "touched": True, "touched_at": "2026-05-02T12:00:00+00:00"}

    def _soft_delete_telegram_file_record(*args, **kwargs):
        calls["soft_delete_telegram_file_record"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, "deleted": True, "deleted_at": "2026-05-02T12:00:00+00:00"}

    def _mark_telegram_file_labels_printed(*args, **kwargs):
        calls["mark_telegram_file_labels_printed"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, "linked_order_id": 9, "labels_printed": True, "touched": False}

    def _mark_telegram_file_linked_order_opened(*args, **kwargs):
        calls["mark_telegram_file_linked_order_opened"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, "linked_order_id": 9, "linked_order_opened": True, "touched": False}

    def _mark_telegram_file_pdf_printed(*args, **kwargs):
        calls["mark_telegram_file_pdf_printed"].append({"args": args, "kwargs": kwargs})
        return {
            "id": args[0] if args else 1,
            "linked_order_id": 9,
            "pdf_printed": True,
            "pdf_printed_at": "2026-05-02T12:00:00+00:00",
            "labels_printed": False,
            "linked_order_opened": False,
            "touched": False,
        }

    fake_db.insert_extraction_with_rows = _insert_extraction_with_rows
    fake_db.update_order_rows = _update_order_rows
    fake_db.create_telegram_file_record = _create_telegram_file_record
    fake_db.update_telegram_file_record = _update_telegram_file_record
    fake_db.touch_telegram_file_record = _touch_telegram_file_record
    fake_db.soft_delete_telegram_file_record = _soft_delete_telegram_file_record
    fake_db.mark_telegram_file_labels_printed = _mark_telegram_file_labels_printed
    fake_db.mark_telegram_file_linked_order_opened = _mark_telegram_file_linked_order_opened
    fake_db.mark_telegram_file_pdf_printed = _mark_telegram_file_pdf_printed
    fake_db.list_telegram_files = lambda *args, **kwargs: []
    fake_db.count_untouched_telegram_files = lambda *args, **kwargs: 0
    fake_db.get_telegram_file_counts = lambda *args, **kwargs: {
        "untouched_count": 0,
        "queued_count": 0,
        "processing_count": 0,
        "failed_count": 0,
    }
    fake_db.find_telegram_file_record = lambda *args, **kwargs: None
    fake_db.find_telegram_file_by_sha256 = lambda *args, **kwargs: None
    fake_db.find_possible_duplicate_order = lambda *args, **kwargs: None
    def _create_whatsapp_file_record(**kwargs):
        calls["create_whatsapp_file_record"].append(kwargs)
        for record in whatsapp_records.values():
            if record.get("wa_message_id") == kwargs.get("wa_message_id"):
                return dict(record)
        record_id = len(whatsapp_records) + 1
        record = {
            "id": record_id,
            "created_at": "2026-05-10T12:00:00+00:00",
            "updated_at": "2026-05-10T12:00:00+00:00",
            "deleted": False,
            "linked_order_id": None,
            "local_path": None,
            "error_message": None,
            **kwargs,
        }
        whatsapp_records[record_id] = record
        return dict(record)

    def _update_whatsapp_file_record(file_id, **kwargs):
        calls["update_whatsapp_file_record"].append({"args": (file_id,), "kwargs": kwargs})
        record = whatsapp_records.get(int(file_id))
        if not record:
            return None
        if kwargs.pop("clear_error", False):
            record["error_message"] = None
        for key, value in kwargs.items():
            if value is not None:
                record[key] = value
        return dict(record)

    def _mark_whatsapp_file_deleted(file_id):
        calls["mark_whatsapp_file_deleted"].append({"args": (file_id,), "kwargs": {}})
        record = whatsapp_records.get(int(file_id))
        if not record:
            return None
        record["deleted"] = True
        return dict(record)

    fake_db.create_whatsapp_file_record = _create_whatsapp_file_record
    fake_db.update_whatsapp_file_record = _update_whatsapp_file_record
    fake_db.list_whatsapp_files = lambda *args, **kwargs: [dict(record) for record in whatsapp_records.values() if not record.get("deleted")]
    fake_db.get_whatsapp_file = lambda file_id: dict(whatsapp_records[int(file_id)]) if int(file_id) in whatsapp_records else None
    fake_db.mark_whatsapp_file_deleted = _mark_whatsapp_file_deleted
    fake_db.list_unfinished_telegram_file_ids = lambda *args, **kwargs: []
    fake_db.get_telegram_file = lambda *args, **kwargs: None
    fake_db.update_order_status = _update_order_status
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
        "extraction_normalizer",
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
    app_module._store_telegram_pdf = lambda raw_bytes, original_filename: {
        "original_filename": original_filename,
        "stored_filename": "stored.pdf",
        "file_path": "/tmp/stored.pdf",
        "file_size": len(raw_bytes),
    }
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
    assert stored["client_name"] == "Client A"


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


def test_extraction_removes_dimension_from_type_once(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")

    bad_type = "2 VETRI 33.1 SATINAT +14+33.1 LOWE 522 x 1262 C.CALDO 28mm"
    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "R-26-0379",
                "type": bad_type,
                "dimension": "522x1262",
                "position": "97-1",
                "quantity": 1,
                "area": 0.66,
            },
            {
                "order_number": "R-26-0379",
                "type": bad_type,
                "dimension": "522x1262",
                "position": "98-1",
                "quantity": 1,
                "area": 0.66,
            },
        ],
        warnings=["duplicate warning", "duplicate warning"],
    )
    client = TestClient(app_module.app)
    response = client.post(
        "/extract_pdf",
        files={"file": ("dimension-type.pdf", b"%PDF-1.7\ndimension-type", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    expected_type = "2 VETRI 33.1 SATINAT +14+33.1 LOWE C.CALDO 28mm"
    assert [row["type"] for row in data["rows"]] == [expected_type, expected_type]
    assert data["warnings"].count("Dimension-like text was removed from type field") == 1
    assert data["warnings"].count("duplicate warning") == 1
    stored_rows = calls["insert_extraction_with_rows"][0]["rows"]
    assert [row["type"] for row in stored_rows] == [expected_type, expected_type]


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


def test_telegram_pdf_document_extracts_as_draft_with_metadata(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    replies: List[Dict[str, Any]] = []
    enqueued: List[int] = []

    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "R-26-2001",
                "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
                "dimension": "600x1200",
                "position": "1-1",
                "quantity": 1,
                "area": 0.72,
            }
        ]
    )

    async def _download(token, file_id):
        return b"%PDF-1.7\ntelegram", {"file_path": "documents/order.pdf"}

    async def _reply(token, chat_id, message_id, text):
        replies.append({"chat_id": chat_id, "message_id": message_id, "text": text})

    app_module._telegram_download_file = _download
    app_module._telegram_reply = _reply
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True

    client = TestClient(app_module.app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 1,
            "message": {
                "message_id": 42,
                "chat": {"id": -100123, "type": "group"},
                "from": {"first_name": "Ada", "last_name": "Lovelace"},
                "caption": "Please process",
                "document": {
                    "file_id": "file_pdf",
                    "file_name": "order.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert [reply["text"] for reply in replies] == ["Order received ✅ Queued for extraction."]
    assert calls["insert_extraction_with_rows"] == []
    assert enqueued == [1]
    assert calls["create_telegram_file_record"][0]["original_filename"] == "order.pdf"
    assert calls["create_telegram_file_record"][0]["extraction_status"] == "queued"
    assert calls["create_telegram_file_record"][0]["telegram_file_id"] == "file_pdf"
    assert calls["create_telegram_file_record"][0]["telegram_caption"] == "Please process"

    record = {
        "id": 1,
        "received_at": "2026-05-02T12:00:00+00:00",
        **calls["create_telegram_file_record"][0],
    }
    app_module.get_telegram_file = lambda file_id: record
    asyncio.run(app_module._process_telegram_queue_job(1))

    stored = calls["insert_extraction_with_rows"][0]
    assert stored["source"] == "telegram"
    assert stored["source_metadata"]["telegram_chat_id"] == -100123
    assert stored["source_metadata"]["telegram_message_id"] == 42
    assert stored["source_metadata"]["telegram_sender_name"] == "Ada Lovelace"
    assert stored["source_metadata"]["original_filename"] == "order.pdf"
    assert stored["source_metadata"]["caption"] == "Please process"
    assert stored["raw_input"].startswith("data:application/pdf;base64,")
    assert calls["update_order_rows"] == []
    assert calls["update_telegram_file_record"][0]["kwargs"]["extraction_status"] == "processing"
    assert calls["update_telegram_file_record"][-1]["kwargs"]["linked_order_id"] == 1
    assert calls["update_telegram_file_record"][-1]["kwargs"]["extraction_status"] == "extracted"
    assert replies[-1]["text"] == "Extraction finished ✅ Please review in platform."


def test_telegram_photo_uses_largest_image(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    seen: Dict[str, Any] = {}
    enqueued: List[int] = []

    app_module.call_llm_for_image_visual = lambda image_bytes, filename, mime_type: _bundle(
        [
            {
                "order_number": "R-26-2002",
                "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
                "dimension": "500x1000",
                "position": "2-1",
                "quantity": 2,
                "area": 1.0,
            }
        ]
    )

    async def _download(token, file_id):
        seen["file_id"] = file_id
        return b"\xff\xd8image", {"file_path": "photos/order.jpg"}

    async def _reply(*args, **kwargs):
        return None

    app_module._telegram_download_file = _download
    app_module._telegram_reply = _reply
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True

    client = TestClient(app_module.app)
    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 43,
                "chat": {"id": 321, "type": "private"},
                "photo": [
                    {"file_id": "small", "file_unique_id": "s", "width": 100, "height": 100, "file_size": 1000},
                    {"file_id": "large", "file_unique_id": "l", "width": 1000, "height": 1000, "file_size": 3000},
                ],
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert seen["file_id"] == "large"
    assert enqueued == [1]
    assert calls["create_telegram_file_record"][0]["mime_type"] == "image/jpeg"
    assert len(calls["create_telegram_file_record"][0]["file_sha256"]) == 64

    record = {
        "id": 1,
        "received_at": "2026-05-02T12:00:00+00:00",
        **calls["create_telegram_file_record"][0],
    }
    app_module.get_telegram_file = lambda file_id: record
    asyncio.run(app_module._process_telegram_queue_job(1))

    stored = calls["insert_extraction_with_rows"][0]
    assert stored["source"] == "telegram"
    assert stored["raw_input"].startswith("data:image/jpeg;base64,")


def test_telegram_text_message_is_ignored(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        json={"message": {"message_id": 44, "chat": {"id": 1}, "text": "hello"}},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert calls["insert_extraction_with_rows"] == []


def test_telegram_missing_token_fails_clearly(monkeypatch):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 45,
                "chat": {"id": 1},
                "document": {
                    "file_id": "file_pdf",
                    "file_name": "order.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            }
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "TELEGRAM_BOT_TOKEN is not set"


def test_telegram_webhook_secret_is_validated(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "expected-secret")
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        json={
            "message": {
                "message_id": 45,
                "chat": {"id": 1},
                "document": {
                    "file_id": "file_pdf",
                    "file_name": "order.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            }
        },
    )

    assert response.status_code == 401
    assert calls["insert_extraction_with_rows"] == []


def test_whatsapp_webhook_verification_returns_plain_challenge(monkeypatch):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "elonia_verify_token")
    client = TestClient(app_module.app)

    response = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "elonia_verify_token", "hub.challenge": "challenge-text"},
    )

    assert response.status_code == 200
    assert response.text == "challenge-text"
    assert response.headers["content-type"].startswith("text/plain")


def test_whatsapp_webhook_verification_wrong_token_returns_403(monkeypatch):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "elonia_verify_token")
    client = TestClient(app_module.app)

    response = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "challenge-text"},
    )

    assert response.status_code == 403


def test_whatsapp_document_webhook_creates_queue_record(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.download_whatsapp_media = lambda *args, **kwargs: {
        "filename": "order.pdf",
        "mime_type": "application/pdf",
        "file_size": 12,
        "local_path": "/tmp/order.pdf",
    }
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook",
        json={
            "entry": [{
                "id": "waba",
                "changes": [{
                    "field": "messages",
                    "value": {"messages": [{
                        "id": "wamid.1",
                        "from": "355691234567",
                        "timestamp": "1770000000",
                        "type": "document",
                        "document": {
                            "id": "media-1",
                            "filename": "order.pdf",
                            "mime_type": "application/pdf",
                            "sha256": "abc",
                            "file_size": 1024,
                        },
                    }]},
                }],
            }]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    created = calls["create_whatsapp_file_record"][0]
    assert created["wa_message_id"] == "wamid.1"
    assert created["sender"] == "355691234567"
    assert created["media_id"] == "media-1"
    assert created["filename"] == "order.pdf"
    assert created["mime_type"] == "application/pdf"
    assert calls["update_whatsapp_file_record"][-1]["kwargs"]["status"] == "downloaded"


def test_whatsapp_duplicate_message_id_does_not_create_duplicate_row(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.download_whatsapp_media = lambda *args, **kwargs: {
        "filename": "order.pdf",
        "mime_type": "application/pdf",
        "file_size": 12,
        "local_path": "/tmp/order.pdf",
    }
    client = TestClient(app_module.app)
    payload = {
        "entry": [{"changes": [{"value": {"messages": [{
            "id": "wamid.dup",
            "from": "3556",
            "timestamp": "1",
            "type": "document",
            "document": {"id": "media-dup", "filename": "order.pdf", "mime_type": "application/pdf"},
        }]}}]}]
    }

    assert client.post("/webhook", json=payload).status_code == 200
    assert client.post("/webhook", json=payload).status_code == 200
    listed = client.get("/api/whatsapp/files").json()["items"]

    assert len(listed) == 1
    assert listed[0]["wa_message_id"] == "wamid.dup"
    assert len(calls["create_whatsapp_file_record"]) == 2


def test_whatsapp_bad_mime_type_is_rejected_safely(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.download_whatsapp_media = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not download"))
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook",
        json={"entry": [{"changes": [{"value": {"messages": [{
            "id": "wamid.bad",
            "from": "3556",
            "timestamp": "1",
            "type": "document",
            "document": {"id": "media-bad", "filename": "bad.exe", "mime_type": "application/x-msdownload"},
        }]}}]}]},
    )

    assert response.status_code == 200
    created = calls["create_whatsapp_file_record"][0]
    assert created["status"] == "failed"
    assert "Unsupported" in created["error_message"]
    assert calls["update_whatsapp_file_record"] == []


def test_whatsapp_oversized_file_is_rejected_safely(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.download_whatsapp_media = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not download"))
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook",
        json={"entry": [{"changes": [{"value": {"messages": [{
            "id": "wamid.large",
            "from": "3556",
            "timestamp": "1",
            "type": "document",
            "document": {
                "id": "media-large",
                "filename": "large.pdf",
                "mime_type": "application/pdf",
                "file_size": 26 * 1024 * 1024,
            },
        }]}}]}]},
    )

    assert response.status_code == 200
    created = calls["create_whatsapp_file_record"][0]
    assert created["status"] == "failed"
    assert "larger than 25MB" in created["error_message"]
    assert calls["update_whatsapp_file_record"] == []


def test_telegram_pdf_extraction_failure_still_stores_file(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    replies: List[str] = []
    enqueued: List[int] = []
    app_module.TELEGRAM_EXTRACTION_MAX_RETRIES = 0

    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: (_ for _ in ()).throw(RuntimeError("boom"))

    async def _download(token, file_id):
        return b"%PDF-1.7\ntelegram", {"file_path": "documents/order.pdf"}

    async def _reply(token, chat_id, message_id, text):
        replies.append(text)

    app_module._telegram_download_file = _download
    app_module._telegram_reply = _reply
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True

    client = TestClient(app_module.app)
    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 48,
                "chat": {"id": 1},
                "document": {
                    "file_id": "file_pdf",
                    "file_name": "failed.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert enqueued == [1]
    assert calls["create_telegram_file_record"][0]["original_filename"] == "failed.pdf"
    assert calls["insert_extraction_with_rows"] == []
    assert replies == ["Order received ✅ Queued for extraction."]

    record = {
        "id": 1,
        "received_at": "2026-05-02T12:00:00+00:00",
        **calls["create_telegram_file_record"][0],
    }
    app_module.get_telegram_file = lambda file_id: record
    asyncio.run(app_module._process_telegram_queue_job(1))

    assert calls["insert_extraction_with_rows"] == []
    assert calls["update_telegram_file_record"][-1]["kwargs"]["extraction_status"] == "failed"
    assert calls["update_telegram_file_record"][-1]["kwargs"]["last_error"]
    assert replies[-1] == "Extraction failed ⚠️ Original PDF is saved in Telegram Files."


def test_telegram_file_size_limit(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    replies: List[str] = []

    async def _reply(token, chat_id, message_id, text):
        replies.append(text)

    app_module._telegram_reply = _reply
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 46,
                "chat": {"id": 1},
                "document": {
                    "file_id": "large_pdf",
                    "file_name": "large.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 5 * 1024 * 1024 + 1,
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "too_large"
    assert replies == ["File is too large. Max size is 5MB."]
    assert calls["insert_extraction_with_rows"] == []


def test_telegram_unsupported_document_replies_cleanly(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    replies: List[str] = []

    async def _reply(token, chat_id, message_id, text):
        replies.append(text)

    app_module._telegram_reply = _reply
    client = TestClient(app_module.app)
    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 47,
                "chat": {"id": 1},
                "document": {
                    "file_id": "txt",
                    "file_name": "notes.txt",
                    "mime_type": "text/plain",
                    "file_size": 100,
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unsupported"
    assert replies == ["Only PDF/image orders are supported."]
    assert calls["insert_extraction_with_rows"] == []
    assert calls["create_telegram_file_record"] == []


def test_telegram_file_view_download_and_path_traversal(monkeypatch, tmp_path):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.DATA_DIR = tmp_path
    stored = app_module._store_telegram_pdf(b"%PDF-1.7\nsaved", "customer/../order.pdf")

    app_module.get_telegram_file = lambda file_id: {
        "id": file_id,
        "original_filename": "order.pdf",
        "stored_filename": stored["stored_filename"],
        "file_path": stored["file_path"],
        "mime_type": "application/pdf",
    }
    client = TestClient(app_module.app)

    view = client.get("/telegram-files/1/view")
    download = client.get("/telegram-files/1/download")
    assert view.status_code == 200
    assert view.headers["content-type"].startswith("application/pdf")
    assert b"%PDF-1.7" in view.content
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]

    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"%PDF-1.7\noutside")
    app_module.get_telegram_file = lambda file_id: {
        "id": file_id,
        "original_filename": "outside.pdf",
        "stored_filename": "outside.pdf",
        "file_path": str(outside),
        "mime_type": "application/pdf",
    }
    blocked = client.get("/telegram-files/2/view")
    assert blocked.status_code == 404


def test_telegram_files_list_and_touch_status(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    list_calls = []
    item = {
        "id": 7,
        "original_filename": "order.pdf",
        "mime_type": "application/pdf",
        "file_size": 100,
        "extraction_status": "extracted",
        "touched": False,
        "touched_at": None,
        "view_url": "/telegram-files/7/view",
        "download_url": "/telegram-files/7/download",
    }

    def _list_telegram_files(*args, **kwargs):
        list_calls.append({"args": args, "kwargs": kwargs})
        return [item]

    app_module.list_telegram_files = _list_telegram_files
    app_module.count_untouched_telegram_files = lambda: 1
    app_module.get_telegram_file_counts = lambda: {
        "untouched_count": 1,
        "queued_count": 0,
        "processing_count": 0,
        "failed_count": 0,
    }
    client = TestClient(app_module.app)

    listed = client.get("/telegram-files")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["touched"] is False
    assert listed.json()["untouched_count"] == 1
    assert list_calls[-1]["kwargs"]["touched"] is False

    listed_all = client.get("/telegram-files?touched=all")
    assert listed_all.status_code == 200
    assert list_calls[-1]["kwargs"]["touched"] is None

    listed_touched = client.get("/telegram-files?touched=true")
    assert listed_touched.status_code == 200
    assert list_calls[-1]["kwargs"]["touched"] is True

    app_module.count_untouched_telegram_files = lambda: 0
    app_module.get_telegram_file_counts = lambda: {
        "untouched_count": 0,
        "queued_count": 0,
        "processing_count": 0,
        "failed_count": 0,
    }
    touched = client.post("/telegram-files/7/touch")
    assert touched.status_code == 200
    assert touched.json()["file"]["touched"] is True
    assert touched.json()["untouched_count"] == 0
    assert calls["touch_telegram_file_record"][0]["args"] == (7,)


def test_telegram_handling_step_endpoints_return_updated_file(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.get_telegram_file = lambda file_id: {
        "id": file_id,
        "linked_order_id": 9,
        "extraction_status": "extracted",
        "touched": False,
    }
    client = TestClient(app_module.app)

    labels = client.post("/telegram-files/7/mark-labels-printed")
    assert labels.status_code == 200
    assert labels.json()["file"]["labels_printed"] is True
    assert calls["mark_telegram_file_labels_printed"][0]["args"] == (7,)

    opened = client.post("/telegram-files/7/mark-linked-order-opened")
    assert opened.status_code == 200
    assert opened.json()["file"]["linked_order_opened"] is True
    assert calls["mark_telegram_file_linked_order_opened"][0]["args"] == (7,)


def test_telegram_pdf_printed_endpoint_returns_updated_file(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    client = TestClient(app_module.app)

    response = client.post("/telegram-files/7/mark-pdf-printed")

    assert response.status_code == 200
    file = response.json()["file"]
    assert file["pdf_printed"] is True
    assert file["pdf_printed_at"]
    assert file["labels_printed"] is False
    assert file["linked_order_opened"] is False
    assert file["touched"] is False
    assert calls["mark_telegram_file_pdf_printed"][0]["args"] == (7,)


def test_telegram_handling_step_endpoint_rejects_unlinked_file(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.get_telegram_file = lambda file_id: {
        "id": file_id,
        "linked_order_id": None,
        "extraction_status": "received",
        "touched": False,
    }
    client = TestClient(app_module.app)

    response = client.post("/telegram-files/7/mark-labels-printed")
    assert response.status_code == 409
    assert calls["mark_telegram_file_labels_printed"] == []


def test_telegram_file_delete_endpoint_soft_deletes_and_archives_linked_draft(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.get_telegram_file = lambda file_id: {
        "id": file_id,
        "linked_order_id": 9,
        "linked_order": {"id": 9, "status": "draft"},
        "extraction_status": "extracted",
        "touched": False,
    }
    client = TestClient(app_module.app)

    response = client.delete("/telegram-files/7?also_delete_linked_order=true")
    assert response.status_code == 200
    data = response.json()
    assert data["file"]["deleted"] is True
    assert data["linked_order_deleted"] is True
    assert calls["soft_delete_telegram_file_record"][0]["args"] == (7,)
    assert calls["update_order_status"][0]["args"] == (9,)
    assert calls["update_order_status"][0]["kwargs"]["status"] == "archived"


def test_telegram_file_delete_endpoint_preserves_approved_linked_order(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.get_telegram_file = lambda file_id: {
        "id": file_id,
        "linked_order_id": 9,
        "linked_order": {"id": 9, "status": "approved"},
        "extraction_status": "extracted",
        "touched": False,
    }
    client = TestClient(app_module.app)

    response = client.delete("/telegram-files/7?also_delete_linked_order=true")
    assert response.status_code == 200
    data = response.json()
    assert data["file"]["deleted"] is True
    assert data["linked_order_deleted"] is False
    assert data["warning"] == "Linked order is not draft and was not deleted."
    assert calls["update_order_status"] == []


def test_telegram_sse_broadcast_payload_is_public(monkeypatch):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    public = app_module._public_telegram_file(
        {
            "id": 1,
            "original_filename": "order.pdf",
            "file_path": "/data/telegram-files/private.pdf",
            "stored_filename": "private.pdf",
            "telegram_file_id": "secret-ish-file-id",
            "view_url": "/telegram-files/1/view",
        }
    )
    assert public["id"] == 1
    assert public["view_url"] == "/telegram-files/1/view"
    assert "file_path" not in public
    assert "stored_filename" not in public
    assert "telegram_file_id" not in public

    async def _run():
        queue = asyncio.Queue()
        with app_module._telegram_event_clients_lock:
            app_module._telegram_event_clients.add(queue)
        app_module._telegram_event_loop = asyncio.get_running_loop()
        try:
            app_module.broadcastTelegramFileEvent("telegram_counts_updated", {"counts": {"untouched_count": 2}})
            message = await asyncio.wait_for(queue.get(), timeout=1)
        finally:
            with app_module._telegram_event_clients_lock:
                app_module._telegram_event_clients.discard(queue)
        return message

    delivered = asyncio.run(_run())
    assert delivered == {"type": "telegram_counts_updated", "counts": {"untouched_count": 2}}


def test_telegram_duplicate_webhook_reuses_existing_queue_record(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    replies: List[str] = []
    enqueued: List[int] = []

    app_module.find_telegram_file_record = lambda **kwargs: {
        "id": 55,
        "extraction_status": "queued",
    }
    app_module._telegram_download_file = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not download duplicate"))

    async def _reply(token, chat_id, message_id, text):
        replies.append(text)

    app_module._telegram_reply = _reply
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 42,
                "chat": {"id": 1},
                "document": {
                    "file_id": "file_pdf",
                    "file_name": "order.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "duplicate"
    assert response.json()["telegram_file_id"] == 55
    assert enqueued == [55]
    assert replies == ["Order received ✅ Queued for extraction."]
    assert calls["create_telegram_file_record"] == []
    assert calls["insert_extraction_with_rows"] == []


def test_telegram_exact_file_duplicate_is_preserved_but_not_queued(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    replies: List[str] = []
    enqueued: List[int] = []

    async def _download(token, file_id):
        return b"%PDF-1.7\nsame-order", {"file_path": "documents/order.pdf"}

    async def _reply(token, chat_id, message_id, text):
        replies.append(text)

    app_module._telegram_download_file = _download
    app_module._telegram_reply = _reply
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True
    sha_lookup_calls: List[Dict[str, Any]] = []

    def _find_by_sha(digest, **kwargs):
        sha_lookup_calls.append({"digest": digest, "kwargs": kwargs})
        return {"id": 12, "extraction_status": "extracted"}

    app_module.find_telegram_file_by_sha256 = _find_by_sha
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 43,
                "chat": {"id": 1},
                "document": {
                    "file_id": "file_pdf_2",
                    "file_name": "order-copy.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "duplicate"
    assert response.json()["duplicate_of_file_id"] == 12
    assert enqueued == []
    assert calls["insert_extraction_with_rows"] == []
    assert sha_lookup_calls[0]["kwargs"] == {"exclude_file_id": None}
    created = calls["create_telegram_file_record"][0]
    assert created["extraction_status"] == "duplicate"
    assert created["duplicate_status"] == "duplicate"
    assert created["duplicate_of_file_id"] == 12
    assert len(created["file_sha256"]) == 64
    assert replies == ["Duplicate detected ⚠️ This PDF was already received."]


def test_telegram_same_filename_different_hash_queues_extraction(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    enqueued: List[int] = []

    async def _download(token, file_id):
        return b"%PDF-1.7\ndifferent-content", {"file_path": "documents/order.pdf"}

    async def _reply(*args, **kwargs):
        return None

    app_module._telegram_download_file = _download
    app_module._telegram_reply = _reply
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True
    app_module.find_telegram_file_by_sha256 = lambda digest, **kwargs: None
    client = TestClient(app_module.app)

    response = client.post(
        "/webhook/telegram",
        json={
            "message": {
                "message_id": 44,
                "chat": {"id": 1},
                "document": {
                    "file_id": "file_pdf_3",
                    "file_name": "order.pdf",
                    "mime_type": "application/pdf",
                    "file_size": 1024,
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert enqueued == [1]
    created = calls["create_telegram_file_record"][0]
    assert created["original_filename"] == "order.pdf"
    assert created["extraction_status"] == "queued"
    assert created.get("duplicate_status", "unique") == "unique"
    assert len(created["file_sha256"]) == 64


def test_telegram_possible_duplicate_marks_file_after_creating_warning_order(monkeypatch):
    app_module, calls = _load_app(monkeypatch, legacy_enabled="false")
    app_module.call_llm_for_pdf_base64_visual = lambda pdf_bytes, filename: _bundle(
        [
            {
                "order_number": "R-26-2001",
                "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
                "dimension": "600x1200",
                "position": "1-1",
                "quantity": 1,
                "area": 0.72,
            }
        ]
    )
    app_module.find_possible_duplicate_order = lambda **kwargs: {
        "id": 88,
        "status": "draft",
        "order_numbers": ["R-26-2001"],
    }

    result = app_module._extract_order_file_bytes(
        raw_bytes=b"%PDF-1.7\nedited",
        filename="edited.pdf",
        content_type="application/pdf",
        source="telegram",
        source_metadata={"telegram_file_record_id": 7},
    )

    assert result["status"] == "draft"
    assert result["duplicate_status"] == "possible_duplicate"
    assert result["draft_order_id"] == 1
    assert result["duplicate_of_order_id"] == 88
    assert calls["insert_extraction_with_rows"]
    update = calls["update_telegram_file_record"][-1]
    assert update["args"] == (7,)
    assert update["kwargs"]["linked_order_id"] == 1
    assert update["kwargs"]["extraction_status"] == "extracted"
    assert update["kwargs"]["duplicate_status"] == "possible_duplicate"


def test_telegram_queue_processes_fifo_one_at_a_time(monkeypatch):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    events: List[str] = []

    async def _fake_process(file_id):
        events.append(f"start:{file_id}")
        await asyncio.sleep(0.01)
        events.append(f"end:{file_id}")
        return {"status": "extracted", "file_id": file_id}

    async def _run():
        with app_module._telegram_queue_lock:
            app_module._telegram_queue.clear()
            app_module._telegram_queued_ids.clear()
            app_module._telegram_queue_task = None
        app_module.TELEGRAM_EXTRACTION_CONCURRENCY = 1
        app_module._process_telegram_queue_job = _fake_process
        for file_id in [1, 2, 3, 4]:
            assert app_module._enqueue_telegram_extraction(file_id) is True
        task = app_module._telegram_queue_task
        assert task is not None
        await task

    asyncio.run(_run())

    assert events == ["start:1", "end:1", "start:2", "end:2", "start:3", "end:3", "start:4", "end:4"]


def test_telegram_queue_recovery_reenqueues_unfinished(monkeypatch):
    app_module, _calls = _load_app(monkeypatch, legacy_enabled="false")
    enqueued: List[int] = []

    app_module.list_unfinished_telegram_file_ids = lambda stale_processing_before: [10, 11]
    app_module._enqueue_telegram_extraction = lambda file_id: enqueued.append(int(file_id)) or True

    app_module._recover_telegram_extraction_queue()

    assert enqueued == [10, 11]


def test_approve_payload_client_name_is_passed_to_save(monkeypatch):
    app_module, _ = _load_app(monkeypatch, legacy_enabled="false")
    row = {
        "order_number": "R-26-0379",
        "type": "2 VETRI C.CALDO 28mm",
        "dimension": "522x1262",
        "position": "1-1",
        "quantity": 1,
        "area": 0.66,
    }
    calls: List[Dict[str, Any]] = []

    app_module.get_order_with_extraction = lambda order_id: {
        "id": order_id,
        "status": "draft",
        "rows": [row],
        "extraction": {"prepared_text": ""},
    }

    def _update_order_rows(order_id, rows, **kwargs):
        calls.append({"order_id": order_id, "rows": rows, "kwargs": kwargs})
        return {
            "id": order_id,
            "status": kwargs.get("status"),
            "client_name": kwargs.get("client_name"),
            "client": kwargs.get("client_name"),
            "rows": rows,
        }

    app_module.update_order_rows = _update_order_rows
    client = TestClient(app_module.app)
    response = client.post(
        "/orders/12/approve",
        json={"rows": [row], "client_name": "DEDA PALLATI VAZHDIM FAZA 3"},
    )

    assert response.status_code == 200
    assert calls[0]["kwargs"]["client_name"] == "DEDA PALLATI VAZHDIM FAZA 3"
    assert response.json()["order"]["client_name"] == "DEDA PALLATI VAZHDIM FAZA 3"


def test_approve_legacy_client_payload_is_normalized(monkeypatch):
    app_module, _ = _load_app(monkeypatch, legacy_enabled="false")
    row = {
        "order_number": "R-26-0380",
        "type": "2 VETRI C.CALDO 28mm",
        "dimension": "522x1262",
        "position": "1-1",
        "quantity": 1,
        "area": 0.66,
    }
    captured: List[str] = []

    app_module.get_order_with_extraction = lambda order_id: {
        "id": order_id,
        "status": "draft",
        "rows": [row],
        "extraction": {"prepared_text": ""},
    }

    def _update_order_rows(order_id, rows, **kwargs):
        captured.append(kwargs.get("client_name"))
        return {
            "id": order_id,
            "status": kwargs.get("status"),
            "client_name": kwargs.get("client_name"),
            "client": kwargs.get("client_name"),
            "rows": rows,
        }

    app_module.update_order_rows = _update_order_rows
    client = TestClient(app_module.app)

    response = client.post("/orders/12/approve", json={"rows": [row], "client": "Legacy Client"})
    assert response.status_code == 200
    response = client.post("/orders/12/approve", json={"rows": [row], "clientName": "Legacy Camel Client"})
    assert response.status_code == 200

    assert captured == ["Legacy Client", "Legacy Camel Client"]


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
