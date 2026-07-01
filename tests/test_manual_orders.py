from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
INDEX_HTML = ROOT_DIR / "docs" / "index.html"
APP_JS = ROOT_DIR / "docs" / "js" / "app.js"


def _load_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    sys.modules.pop("db", None)
    db_module = importlib.import_module("db")
    db_module.init_db()
    return db_module


def _manual_payload(**overrides):
    payload = {
        "client_name": "Manual Client",
        "order_number": "M-2026-001",
        "order_date": "2026-07-01",
        "notes": "Factory reference",
        "status": "draft",
        "rows": [
            {
                "position": "1",
                "glass_type": "4F",
                "width_mm": 1000,
                "height_mm": 500,
                "quantity": 3,
                "area_override_m2": None,
                "notes": "",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _pdf_row():
    return {
        "order_number": "PDF-001",
        "type": "4F",
        "dimension": "800x600",
        "position": "1",
        "quantity": 1,
        "area": 0.48,
    }


def test_manual_order_uses_separate_tables_and_stays_out_of_pdf_history(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    pdf = db.insert_extraction_with_rows(
        source="pdf",
        rows=[_pdf_row()],
        raw_input="pdf",
        prepared_text="pdf",
        llm_output_json="{}",
        model_used="test",
        hash_value="pdf-order",
        confidence=1.0,
        client_name="PDF Client",
    )

    manual = db.create_manual_order(_manual_payload())
    history = db.get_orders(year="all")

    assert manual["source"] == "manual"
    assert [item["id"] for item in history] == [pdf["order_id"]]
    assert history[0]["source"] == "pdf"

    with sqlite3.connect(tmp_path / "orders.db") as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        manual_count = conn.execute("SELECT COUNT(*) FROM manual_orders").fetchone()[0]

    assert {"manual_orders", "manual_order_rows"}.issubset(tables)
    assert order_count == 1
    assert manual_count == 1


def test_manual_area_calculation_override_and_totals(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    payload = _manual_payload(
        rows=[
            {
                "position": "1",
                "glass_type": "4F",
                "width_mm": 1200,
                "height_mm": 800,
                "quantity": 2,
                "area_override_m2": None,
            },
            {
                "position": "2",
                "glass_type": "33.1F",
                "width_mm": 1000,
                "height_mm": 1000,
                "quantity": 1,
                "area_override_m2": 1.125,
            },
        ]
    )

    order = db.create_manual_order(payload)

    assert order["rows"][0]["calculated_area_m2"] == 1.92
    assert order["rows"][0]["final_area_m2"] == 1.92
    assert order["rows"][1]["calculated_area_m2"] == 1.0
    assert order["rows"][1]["final_area_m2"] == 1.125
    assert order["total_quantity"] == 3
    assert order["total_area_m2"] == 3.045


def test_manual_edit_delete_and_processing_do_not_touch_pdf_order(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    pdf = db.insert_extraction_with_rows(
        source="pdf",
        rows=[_pdf_row()],
        raw_input="pdf",
        prepared_text="pdf",
        llm_output_json="{}",
        model_used="test",
        hash_value="protected-pdf",
        confidence=1.0,
        client_name="PDF Client",
    )
    db.update_order_status(pdf["order_id"], status="approved")
    manual = db.create_manual_order(_manual_payload(status="approved"))

    updated = db.update_manual_order(
        manual["id"],
        _manual_payload(
            status="approved",
            client_name="Changed Manual Client",
            rows=[
                {
                    "position": "A",
                    "glass_type": "4F",
                    "width_mm": 900,
                    "height_mm": 700,
                    "quantity": 2,
                }
            ],
        ),
    )
    processing = db.send_manual_order_to_processing(manual["id"])
    pdf_after = db.get_order_with_extraction(pdf["order_id"])

    assert updated["client_name"] == "Changed Manual Client"
    assert processing["source"] == "manual"
    assert processing["status"] == "processing"
    assert processing["rows"][0]["dimension"] == "900x700"
    assert processing["rows"][0]["final_area_m2"] == 1.26
    assert pdf_after["status"] == "approved"
    assert pdf_after["client_name"] == "PDF Client"
    assert db.delete_manual_order(manual["id"]) is True
    assert db.get_manual_order(manual["id"]) is None
    assert db.get_order_with_extraction(pdf["order_id"]) is not None
    with sqlite3.connect(tmp_path / "orders.db") as conn:
        manual_row_count = conn.execute(
            "SELECT COUNT(*) FROM manual_order_rows WHERE manual_order_id = ?",
            (manual["id"],),
        ).fetchone()[0]
    assert manual_row_count == 0


def test_manual_duplicate_number_is_a_warning_not_pdf_comparison(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    first = db.create_manual_order(_manual_payload())
    second = db.create_manual_order(_manual_payload(client_name="Second Client"))

    assert first["duplicate_warning"] is False
    assert second["duplicate_warning"] is True
    assert len(db.list_manual_orders()) == 2


def test_manual_order_validation_rejects_empty_and_invalid_rows(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="At least one"):
        db.create_manual_order(_manual_payload(rows=[]))
    with pytest.raises(ValueError, match="greater than zero"):
        db.create_manual_order(
            _manual_payload(
                rows=[
                    {
                        "glass_type": "4F",
                        "width_mm": 0,
                        "height_mm": 500,
                        "quantity": 1,
                    }
                ]
            )
        )
    with pytest.raises(ValueError, match="Glass type"):
        db.create_manual_order(
            _manual_payload(
                rows=[
                    {
                        "glass_type": " ",
                        "width_mm": 500,
                        "height_mm": 500,
                        "quantity": 1,
                    }
                ]
            )
        )


def test_manual_orders_frontend_exposes_isolated_factory_workflow():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

    assert 'data-tab="manual"' in html
    assert 'id="tabManualOrders"' in html
    assert 'id="manualOrderRows"' in html
    assert "function manualCalculatedArea" in js
    assert "width * height * quantity / 1_000_000" in js
    assert 'source: "manual"' in js
    assert 'data-manual-action="processing"' in js
    assert 'data-manual-action="labels"' in js
    assert 'data-manual-action="invoice"' in js
    assert "manualInvoicePricingIssues" in js
