from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_INDEX = ROOT_DIR / "frontend" / "index.html"


def _load_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    sys.modules.pop("db", None)
    db_module = importlib.import_module("db")
    db_module.init_db()
    return db_module


def _row():
    return {
        "order_number": "R-26-0379",
        "type": "2 VETRI C.CALDO 28mm",
        "dimension": "522x1262",
        "position": "1-1",
        "quantity": 1,
        "area": 0.66,
    }


def test_saved_order_history_record_includes_client_name(tmp_path, monkeypatch):
    db = _load_db(tmp_path, monkeypatch)
    insert_result = db.insert_extraction_with_rows(
        source="pdf",
        rows=[_row()],
        raw_input="raw pdf",
        prepared_text="prepared",
        llm_output_json="{}",
        model_used="test-model",
        hash_value="client-name-hash",
        confidence=0.9,
        client_name="DEDA PALLATI VAZHDIM FAZA 3",
    )

    db.update_order_rows(
        insert_result["order_id"],
        [_row()],
        status="approved",
        client_name="DEDA PALLATI VAZHDIM FAZA 3",
    )

    order = db.get_order_with_extraction(insert_result["order_id"])
    history = db.get_orders(year="all")

    assert order["client_name"] == "DEDA PALLATI VAZHDIM FAZA 3"
    assert order["clientName"] == "DEDA PALLATI VAZHDIM FAZA 3"
    assert order["client"] == "DEDA PALLATI VAZHDIM FAZA 3"
    assert history[0]["client_name"] == "DEDA PALLATI VAZHDIM FAZA 3"


def test_history_ui_prefers_canonical_client_name():
    html = FRONTEND_INDEX.read_text(encoding="utf-8")

    assert "function getClientName(record" in html
    assert "record.client_name" in html
    assert "record.clientName" in html
    assert "record.client," in html
    assert "const client = getClientName(order);" in html


def test_client_name_schema_migration_is_non_destructive(tmp_path, monkeypatch):
    db_path = tmp_path / "orders.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                updated_at TEXT,
                source TEXT,
                client_hint TEXT,
                order_numbers TEXT,
                units_total INTEGER DEFAULT 0,
                area_total REAL DEFAULT 0,
                hash TEXT UNIQUE,
                status TEXT DEFAULT 'draft'
            )
            """
        )
        conn.execute(
            "INSERT INTO orders (created_at, updated_at, source, client_hint, order_numbers, hash) VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-04-26T00:00:00+00:00", "2026-04-26T00:00:00+00:00", "pdf", "Legacy Client", "[]", "legacy-hash"),
        )

    _load_db(tmp_path, monkeypatch)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(orders)").fetchall()}
        row = conn.execute("SELECT client_hint, client_name FROM orders WHERE hash = ?", ("legacy-hash",)).fetchone()

    assert "client_name" in columns
    assert row == ("Legacy Client", None)
