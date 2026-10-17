"""Verify the existing server store retains the new invoice document and price snapshot."""
import importlib
import sys
from pathlib import Path


def test_invoice_document_items_and_finalized_snapshot_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv('DB_DIR', str(tmp_path))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'backend'))
    sys.modules.pop('db', None)
    db = importlib.import_module('db')
    db.init_db()
    from services.platform_repository import save_invoices, load_invoices
    job = {
        'id': 'inv-round-trip', 'source': 'invoice', 'status': 'finalized',
        'invoiceNumber': 'INV-2026-001', 'client': 'Client',
        'document': {'businessName': 'Factory', 'clientName': 'Client', 'issueDate': '2026-10-17', 'dueDate': '2026-11-16', 'notes': 'Bank transfer'},
        'items': [{'description': 'Glass', 'quantity': 3, 'rate': 1250.50}],
        'finalizedSnapshot': {'subtotal': 3751.50, 'discountValue': 375.15, 'vatValue': 675.27, 'total': 4051.62, 'lines': [{'type': 'Glass', 'lineTotal': 3751.50}]},
    }
    save_invoices(db, {'jobs': [job]})
    assert load_invoices(db, tmp_path / 'unused-legacy.json')['jobs'] == [job]
