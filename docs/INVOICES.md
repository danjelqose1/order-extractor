# Invoices

Open **Invoices** in the sidebar, then choose **New invoice**. Start a blank invoice with custom items or select an approved, in-production, or completed order. Manual orders use the same price catalog and area calculations as the existing Manual Orders invoice action.

The editor supports business and client contact details, tax IDs, invoice number, issue and due dates, line items, discounts, VAT, and payment notes. Business details carry forward from the most recent invoice. Select **Save invoice** to save edits to the existing backend invoice store. Unsaved changes are indicated, and failed saves can be retried without discarding the draft.

**Finalized** locks document fields and stores the exact price breakdown. Later catalog changes do not reprice it. Reopen as **Draft** to make corrections. Older finalized records without a stored price breakdown retain their recorded total; reopening and reviewing them is required before generating a new PDF. They are never silently recalculated with current prices.

The register supports client, invoice-number, order-number, and status filters; pagination; PDF download; copying to an independent draft; and draft deletion. Copies of order invoices bill quantities in square metres, matching the source pricing. A4 PDFs include wrapped descriptions, repeated column headers, page numbers, payment notes, and a draft label until finalized. The built-in PDF font supports Latin text, including Albanian characters; unsupported characters produce an explicit export error while preserving the saved invoice.

The existing `/api/invoices` persistence API and `/api/prices` catalog are reused. No migration or additional service is needed. Browser saves are serialized; the existing backend's whole-register API does not provide cross-device concurrent-edit conflict detection.

Validation:

- `node tests/test_invoice_workspace_browser.cjs` (system Chromium, or set `CHROMIUM_PATH`)
- `.venv/bin/python -m pytest tests/test_invoice_persistence.py tests/test_invoice_ai.py tests/test_frontend_navigation.py tests/test_frontend_security.py tests/test_frontend_theme.py tests/test_manual_orders.py -q`

The browser test uses an in-memory API and never writes production data. Screenshots and sample PDFs are written to `/tmp/invoice-workspace-qa` by default.
