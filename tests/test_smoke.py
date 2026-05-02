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
        "create_telegram_file_record": [],
        "update_telegram_file_record": [],
        "touch_telegram_file_record": [],
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

    def _create_telegram_file_record(**kwargs):
        calls["create_telegram_file_record"].append(kwargs)
        return {"id": len(calls["create_telegram_file_record"]), "touched": False, "touched_at": None, **kwargs}

    def _update_telegram_file_record(*args, **kwargs):
        calls["update_telegram_file_record"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, **kwargs}

    def _touch_telegram_file_record(*args, **kwargs):
        calls["touch_telegram_file_record"].append({"args": args, "kwargs": kwargs})
        return {"id": args[0] if args else 1, "touched": True, "touched_at": "2026-05-02T12:00:00+00:00"}

    fake_db.insert_extraction_with_rows = _insert_extraction_with_rows
    fake_db.update_order_rows = _update_order_rows
    fake_db.create_telegram_file_record = _create_telegram_file_record
    fake_db.update_telegram_file_record = _update_telegram_file_record
    fake_db.touch_telegram_file_record = _touch_telegram_file_record
    fake_db.list_telegram_files = lambda *args, **kwargs: []
    fake_db.count_untouched_telegram_files = lambda *args, **kwargs: 0
    fake_db.find_telegram_file_record = lambda *args, **kwargs: None
    fake_db.list_unfinished_telegram_file_ids = lambda *args, **kwargs: []
    fake_db.get_telegram_file = lambda *args, **kwargs: None
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
    assert response.json()["status"] == "extracted"
    assert seen["file_id"] == "large"
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
    app_module.list_telegram_files = lambda *args, **kwargs: [item]
    app_module.count_untouched_telegram_files = lambda: 1
    client = TestClient(app_module.app)

    listed = client.get("/telegram-files")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["touched"] is False
    assert listed.json()["untouched_count"] == 1

    app_module.count_untouched_telegram_files = lambda: 0
    touched = client.post("/telegram-files/7/touch")
    assert touched.status_code == 200
    assert touched.json()["file"]["touched"] is True
    assert touched.json()["untouched_count"] == 0
    assert calls["touch_telegram_file_record"][0]["args"] == (7,)


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
