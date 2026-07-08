from __future__ import annotations

from copy import deepcopy
import importlib
import sqlite3
import sys
from pathlib import Path

import fitz
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
INDEX_HTML = ROOT_DIR / "docs" / "index.html"
APP_JS = ROOT_DIR / "docs" / "js" / "app.js"
APP_PY = ROOT_DIR / "backend" / "app.py"


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


def test_saved_manual_glass_types_remain_available_for_autocomplete(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    first = db.create_manual_order(_manual_payload())
    db.create_manual_order(
        _manual_payload(
            order_number="M-2026-002",
            rows=[
                {
                    "position": "1",
                    "glass_type": "Tr+12+Tr",
                    "width_mm": 600,
                    "height_mm": 900,
                    "quantity": 1,
                }
            ],
        )
    )
    db.create_manual_order(
        _manual_payload(
            order_number="M-2026-003",
            rows=[
                {
                    "position": "1",
                    "glass_type": "  tr+12+tr  ",
                    "width_mm": 500,
                    "height_mm": 500,
                    "quantity": 1,
                }
            ],
        )
    )

    assert db.list_manual_glass_types()[:2] == ["tr+12+tr", "4F"]
    assert db.list_manual_glass_types(query="12") == ["tr+12+tr"]

    db.delete_manual_order(first["id"])
    assert "4F" in db.list_manual_glass_types()


def test_saved_manual_clients_and_date_based_order_number(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    first = db.create_manual_order(
        _manual_payload(
            client_name="Qamili",
            order_number="05072026",
            order_date="2026-07-05",
        )
    )
    db.create_manual_order(
        _manual_payload(
            client_name="  qamili  ",
            order_number="05072026-02",
            order_date="2026-07-05",
        )
    )

    assert db.list_manual_clients() == ["qamili"]
    assert db.list_manual_clients(query="mil") == ["qamili"]
    assert db.next_manual_order_number("2026-07-05") == "05072026-03"
    assert db.next_manual_order_number("2026-07-06") == "06072026"

    db.delete_manual_order(first["id"])
    assert db.list_manual_clients() == ["qamili"]


def test_manual_print_settings_are_persisted_separately(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    settings = {
        "label_font_family": "Courier",
        "label_client_size": 14,
        "processing_dimension_unit": "mm",
        "processing_print_layout": "a4_landscape_2up",
        "processing_show_cut_guide": False,
    }

    assert db.get_manual_print_settings() == {}
    assert db.save_manual_print_settings(settings) == settings
    assert db.get_manual_print_settings() == settings


def test_manual_orders_frontend_exposes_isolated_factory_workflow():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    app_py = APP_PY.read_text(encoding="utf-8")

    assert 'data-tab="manual"' in html
    assert 'id="tabManualOrders"' in html
    assert 'id="manualOrderRows"' in html
    assert 'id="manualGlassTypeOptions"' in html
    assert 'id="manualClientOptions"' in html
    assert 'id="manualDimensionUnit"' in html
    assert "Show MANUAL" not in html
    assert 'id="manualWidthUnitLabel"' in js
    assert 'id="manualHeightUnitLabel"' in js
    assert 'id="manualPrintSettingsForm"' in html
    assert 'id="manualPrintSettingsOpen"' in html
    assert 'id="manualPrintSettingsModal"' in html
    assert 'id="manualPrintSettingsClose"' in html
    assert "function openManualPrintSettings" in js
    assert "function closeManualPrintSettings" in js
    assert 'class="card manual-print-settings-card"' not in html
    assert 'data-manual-print-setting="label_font_family"' in html
    assert 'data-manual-print-setting="processing_font_family"' in html
    assert 'data-manual-print-setting="processing_dimension_unit"' in html
    assert 'data-manual-print-setting="processing_print_layout"' in html
    assert 'data-manual-print-setting="processing_show_cut_guide"' in html
    assert 'data-manual-print-setting="processing_client_bold"' in html
    assert "A4 landscape — 2 copies" in html
    assert "function syncManualProcessingPrintLayout" in js
    assert "function manualCalculatedArea" in js
    assert "width * height * quantity / 1_000_000" in js
    assert 'source: "manual"' in js
    assert 'data-manual-action="processing-choice"' in js
    assert 'id="manualProcessingLayoutModal"' in html
    assert 'data-manual-processing-download-layout="slip"' in html
    assert 'data-manual-processing-download-layout="a4_landscape_2up"' in html
    assert "Portrait — 1 copy" in html
    assert "A4 landscape — 2 copies" in html
    assert "function openManualProcessingLayoutChoice" in js
    assert 'processing-sheet.pdf?layout=${encodeURIComponent(processingLayout)}' in js
    assert "processing_layout=layout" in app_py
    assert 'data-manual-action="labels"' in js
    assert 'data-manual-action="invoice"' in js
    assert "function downloadManualOrderDocument" in js
    assert '"labels.pdf"' in js
    assert 'activateTab("processing")' not in js[js.index('async function handleManualOrderAction'):js.index('function ensureManualOrdersReady')]
    assert 'activateTab("labels")' not in js[js.index('async function handleManualOrderAction'):js.index('function ensureManualOrdersReady')]
    assert "manualInvoicePricingIssues" in js
    assert 'list="manualGlassTypeOptions"' in js
    assert 'manualApi("/manual-orders/glass-types?limit=250")' in js
    assert 'manualApi("/manual-orders/clients?limit=250")' in js
    assert "/manual-orders/next-number?" in js
    assert "function fillNextManualOrderNumber" in js
    assert "function manualDimensionInputValue" in js
    assert "function manualDimensionValueInMm" in js
    assert 'localStorage.setItem("manual_dimension_unit"' in js
    assert 'manualApi("/manual-orders/print-settings")' in js
    assert "function collectManualPrintSettings" in js
    assert "function saveManualPrintSettings" in js
    assert 'id="manualOrderFormat"' in html
    assert "Client Positions + Red Index" in html
    assert 'id="manualOrderAddSection"' in html
    assert 'id="manualAutoNumberFrom"' in html
    assert 'id="manualOrderRenumber"' in html
    assert 'id="manualRedIndexPrint"' in html
    assert 'id="manualRedIndexSendProcessing"' in html
    assert 'id="manualRedIndexLabels"' in html
    assert "function renumberManualRows" in js
    assert "function duplicateManualIndexNumbers" in js
    assert 'manual_format: manualOrdersState.manualFormat' in js


def test_manual_order_rows_support_spreadsheet_keyboard_entry():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

    assert "Start typing a saved glass type" in html
    assert "Tab across" in html
    assert "function incrementManualPosition" in js
    assert "function nextManualRowDefaults" in js
    assert "function appendManualRow" in js
    assert 'event.key === "ArrowDown" || event.key === "Enter"' in js
    assert 'event.key === "ArrowUp"' in js
    assert 'event.key === "Tab"' in js
    assert 'const nextField = event.key === "ArrowDown" ? "width_mm" : field;' in js
    assert 'glass_type: previous?.glass_type || ""' in js
    assert 'newManualRow({ position: "1" })' in js
    assert 'readonly tabindex="-1"' in js


def test_manual_processing_sheet_matches_compact_workshop_format():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    documents = importlib.import_module("manual_documents")
    order = _manual_payload(
        status="approved",
        rows=[
            {
                "position": "1",
                "glass_type": "Tr+12+Tr",
                "width_mm": 615,
                "height_mm": 1252,
                "quantity": 2,
                "notes": "",
            },
            {
                "position": "2",
                "glass_type": "Tr+12+Tr",
                "width_mm": 615,
                "height_mm": 1030,
                "quantity": 2,
                "notes": "",
            },
        ],
    )

    pdf_bytes = documents.build_manual_processing_pdf(order)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    assert len(pdf) == 1
    assert pdf[0].rect.width == pytest.approx(100 * 72 / 25.4, abs=0.2)
    assert pdf[0].rect.height == pytest.approx(210 * 72 / 25.4, abs=0.2)
    text = pdf[0].get_text()
    assert "MANUAL PROCESSING" in text
    assert "Tr+12+Tr" in text
    assert "61.5 x 125.2" in text
    assert "61.5 x 103" in text
    assert "x 2" in text
    client_spans = [
        span
        for block in pdf[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"] == "Manual Client"
    ]
    assert len(client_spans) == 1
    assert "Bold" in client_spans[0]["font"]
    assert client_spans[0]["size"] >= 14


def test_manual_processing_sheet_keeps_multiple_glass_types_on_one_page():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    documents = importlib.import_module("manual_documents")
    order = _manual_payload(
        status="processing",
        rows=[
            {
                "position": "1",
                "glass_type": "TR+12+TR",
                "width_mm": 1200,
                "height_mm": 980,
                "quantity": 2,
                "notes": "",
            },
            {
                "position": "2",
                "glass_type": "TR+12+TR",
                "width_mm": 1200,
                "height_mm": 1000,
                "quantity": 1,
                "notes": "",
            },
            {
                "position": "3",
                "glass_type": "TR+12+SATINE",
                "width_mm": 1300,
                "height_mm": 1000,
                "quantity": 1,
                "notes": "",
            },
        ],
    )

    pdf_bytes = documents.build_manual_processing_pdf(order)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    assert len(pdf) == 1
    text = pdf[0].get_text()
    assert "TR+12+TR" in text
    assert "TR+12+SATINE" in text
    assert "3.552 m²" in text
    assert "1.300 m²" in text
    assert "GLASS TYPE" not in text
    assert "3)" in text
    assert "130 x 100" in text


def test_manual_processing_a4_landscape_renders_two_visual_copies_without_mutation():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    documents = importlib.import_module("manual_documents")
    order = _manual_payload(status="processing")
    original_order = deepcopy(order)
    settings = documents.normalize_manual_print_settings(
        {
            "processing_print_layout": "a4_landscape_2up",
            "processing_show_cut_guide": True,
            "processing_page_width_mm": 110,
            "processing_page_height_mm": 180,
        }
    )

    pdf_bytes = documents.build_manual_processing_pdf(order, settings)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    assert len(pdf) == 1
    page = pdf[0]
    assert page.rect.width == pytest.approx(297 * 72 / 25.4, abs=0.2)
    assert page.rect.height == pytest.approx(210 * 72 / 25.4, abs=0.2)
    assert page.get_text().count("M-2026-001") == 2
    order_spans = [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"] == "M-2026-001"
    ]
    assert len(order_spans) == 2
    assert order_spans[1]["bbox"][0] - order_spans[0]["bbox"][0] == pytest.approx(
        108 * 72 / 25.4,
        abs=1,
    )
    assert order == original_order


def test_red_index_manual_order_preserves_metadata_and_processing_fields(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    payload = _manual_payload(
        status="approved",
        manual_format="client_positions_red_index",
        rows=[
            {
                "section": "Villa 1",
                "client_position": "K1-",
                "index_number": 12,
                "glass_type": "4F",
                "width_mm": 1695,
                "height_mm": 2330,
                "quantity": 1,
                "area_override_m2": None,
                "notes": "",
            },
            {
                "section": "Villa 1",
                "client_position": "",
                "index_number": 12,
                "glass_type": "4F",
                "width_mm": 965,
                "height_mm": 2230,
                "quantity": 1,
                "area_override_m2": None,
                "notes": "Second piece",
            },
        ],
    )

    saved = db.create_manual_order(payload)

    assert saved["manual_format"] == "client_positions_red_index"
    assert saved["duplicate_index_numbers"] == [12]
    assert saved["rows"][0]["section"] == "Villa 1"
    assert saved["rows"][0]["client_position"] == "K1-"
    assert saved["rows"][0]["index_number"] == 12
    assert saved["rows"][0]["position"] == "K1- 12"
    assert saved["rows"][1]["position"] == "12"
    assert db.get_orders(year="all") == []

    processing = db.send_manual_order_to_processing(saved["id"])

    assert processing["manual_format"] == "client_positions_red_index"
    assert processing["rows"][0]["client_position"] == "K1-"
    assert processing["rows"][0]["index_number"] == 12
    assert processing["rows"][0]["section"] == "Villa 1"
    assert processing["rows"][0]["position"] == "K1- 12"
    assert processing["rows"][0]["manual_format"] == "client_positions_red_index"


def test_standard_manual_orders_load_with_backward_compatible_defaults(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    saved = db.create_manual_order(_manual_payload())

    assert saved["manual_format"] == "standard"
    assert saved["rows"][0]["section"] == ""
    assert saved["rows"][0]["client_position"] == ""
    assert saved["rows"][0]["index_number"] is None


def test_manual_schema_upgrade_adds_red_index_fields_to_existing_database(tmp_path, monkeypatch):
    database_path = tmp_path / "orders.db"
    with sqlite3.connect(database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE manual_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_name VARCHAR(255) NOT NULL,
                order_number VARCHAR(120) NOT NULL,
                order_date VARCHAR(10) NOT NULL,
                notes TEXT,
                status VARCHAR(20),
                source VARCHAR(20) NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE manual_order_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manual_order_id INTEGER NOT NULL,
                position VARCHAR(80),
                glass_type VARCHAR(255) NOT NULL,
                width_mm FLOAT NOT NULL,
                height_mm FLOAT NOT NULL,
                quantity INTEGER NOT NULL,
                calculated_area_m2 FLOAT NOT NULL,
                area_override_m2 FLOAT,
                final_area_m2 FLOAT NOT NULL,
                notes TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            INSERT INTO manual_orders (
                client_name, order_number, order_date, notes, status, source,
                created_at, updated_at
            ) VALUES (
                'Legacy Client', 'LEGACY-1', '2026-07-01', '', 'draft', 'manual',
                '2026-07-01 10:00:00', '2026-07-01 10:00:00'
            );
            INSERT INTO manual_order_rows (
                manual_order_id, position, glass_type, width_mm, height_mm,
                quantity, calculated_area_m2, area_override_m2, final_area_m2,
                notes, created_at, updated_at
            ) VALUES (
                1, '1', '4F', 1000, 500, 1, 0.5, NULL, 0.5, '',
                '2026-07-01 10:00:00', '2026-07-01 10:00:00'
            );
            """
        )

    db = _load_db(tmp_path, monkeypatch)
    loaded = db.get_manual_order(1)

    assert loaded["manual_format"] == "standard"
    assert loaded["rows"][0]["section"] == ""
    assert loaded["rows"][0]["client_position"] == ""
    assert loaded["rows"][0]["index_number"] is None


def test_red_index_processing_and_labels_render_black_client_position_and_red_index():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    documents = importlib.import_module("manual_documents")
    order = _manual_payload(
        status="approved",
        manual_format="client_positions_red_index",
        rows=[
            {
                "position": "K1- 12",
                "section": "Villa 3",
                "client_position": "K1-",
                "index_number": 12,
                "glass_type": "4F + 16 + LowE",
                "width_mm": 1695,
                "height_mm": 2330,
                "quantity": 1,
                "notes": "",
            }
        ],
    )

    processing = fitz.open(
        stream=documents.build_manual_processing_pdf(order),
        filetype="pdf",
    )
    processing_text = processing[0].get_text()
    assert "Villa 3" in processing_text
    assert "POS" in processing_text
    assert "INDEX" in processing_text
    assert "K1-" in processing_text
    assert "12" in processing_text
    assert "169.5 x 233" in processing_text
    processing_spans = [
        span
        for block in processing[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    processing_client = next(span for span in processing_spans if "K1-" in span["text"])
    processing_index = next(span for span in processing_spans if span["text"] == "12")
    assert processing_client["color"] != processing_index["color"]
    assert (processing_index["color"] >> 16) & 0xFF > 200
    assert (processing_index["color"] >> 8) & 0xFF < 80

    labels = fitz.open(
        stream=documents.build_manual_labels_pdf(order),
        filetype="pdf",
    )
    assert len(labels) == 1
    label_text = labels[0].get_text()
    assert "POS K1-" in label_text
    assert "#12" in label_text
    assert "Villa 3" in label_text
    assert "MANUAL" not in label_text
    label_spans = [
        span
        for block in labels[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    label_client = next(span for span in label_spans if "K1-" in span["text"])
    label_index = next(span for span in label_spans if span["text"] == "#12")
    assert label_client["color"] != label_index["color"]
    assert (label_index["color"] >> 16) & 0xFF > 200
    assert (label_index["color"] >> 8) & 0xFF < 80
    assert label_index["size"] > label_client["size"]
    assert label_index["bbox"][0] > labels[0].rect.width * 0.80
    assert label_index["bbox"][3] > labels[0].rect.height * 0.82


def test_manual_labels_are_dedicated_100x40_quantity_labels():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    documents = importlib.import_module("manual_documents")
    order = _manual_payload(
        status="approved",
        rows=[
            {
                "position": "7",
                "glass_type": "4F",
                "width_mm": 1000,
                "height_mm": 500,
                "quantity": 2,
                "notes": "",
            }
        ],
    )

    pdf_bytes = documents.build_manual_labels_pdf(order)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    assert len(pdf) == 2
    for page in pdf:
        assert page.rect.width == pytest.approx(100 * 72 / 25.4, abs=0.2)
        assert page.rect.height == pytest.approx(40 * 72 / 25.4, abs=0.2)
        text = page.get_text()
        assert "M-2026-001" in text
        assert "Manual Client" in text
        assert "1000 x 500 mm" in text
        assert "4F" in text
        assert "POS 7" in text
        assert "1/2" not in text
        assert "2/2" not in text
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span["text"] == "Manual Client"
        ]
        assert len(spans) == 1
        client_span = spans[0]
        assert "Bold" in client_span["font"]
        assert client_span["size"] >= 11
        client_center = (client_span["bbox"][0] + client_span["bbox"][2]) / 2
        assert client_center == pytest.approx(page.rect.width / 2, abs=2)


def test_manual_document_settings_change_fonts_visibility_and_processing_layout():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    documents = importlib.import_module("manual_documents")
    order = _manual_payload(
        status="approved",
        rows=[
            {
                "position": "1",
                "glass_type": "Tr+12+Tr",
                "width_mm": 615,
                "height_mm": 1252,
                "quantity": 1,
                "notes": "Priority",
            }
        ],
    )
    settings = documents.normalize_manual_print_settings(
        {
            "label_font_family": "Courier",
            "label_client_size": 15,
            "label_show_date": False,
            "label_show_manual_marker": False,
            "processing_font_family": "Times",
            "processing_page_width_mm": 110,
            "processing_page_height_mm": 180,
            "processing_dimension_unit": "mm",
            "processing_row_size": 14,
            "processing_show_notes": False,
        }
    )

    label_pdf = fitz.open(
        stream=documents.build_manual_labels_pdf(order, settings),
        filetype="pdf",
    )
    label_text = label_pdf[0].get_text()
    assert "01.07.2026" not in label_text
    assert "MANUAL" not in label_text
    label_fonts = {
        span["font"]
        for block in label_pdf[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    }
    assert any("Courier" in font for font in label_fonts)

    processing_pdf = fitz.open(
        stream=documents.build_manual_processing_pdf(order, settings),
        filetype="pdf",
    )
    assert processing_pdf[0].rect.width == pytest.approx(110 * 72 / 25.4, abs=0.2)
    assert processing_pdf[0].rect.height == pytest.approx(180 * 72 / 25.4, abs=0.2)
    processing_text = processing_pdf[0].get_text()
    assert "615 x 1252" in processing_text
    assert "DIMENSIONS (MM)" in processing_text
    assert "Priority" not in processing_text
    processing_fonts = {
        span["font"]
        for block in processing_pdf[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    }
    assert any("Times" in font for font in processing_fonts)
