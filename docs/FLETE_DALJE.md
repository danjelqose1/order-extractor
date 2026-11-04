# Fletë-Dalje

Native standalone tab in Order Extractor. Records are stored in the existing backend SQLite database (`DB_DIR/orders.db`) in `flete_dalje_documents`. `init_db()` creates the table at startup. There are no foreign keys or reads/writes to orders, invoices, customers, or production. The API follows the existing invoice API's access pattern (no separate user authentication); private MCP authentication does not apply to browser modules.

New clients start at document number 1. Suggestions use the highest saved number for the normalized recipient plus one; manual number edits remain available. Database uniqueness is per recipient and number, so different clients can each have a sheet numbered 1. Existing saved document numbers are preserved. Record IDs remain stable when numbers change. Conditional updates/deletes reject stale versions. Import merges missing records in a transaction, skips identical records, and rejects conflicts without overwriting anything. Backups contain saved records only. Browser storage is not used for document records. A backend deployment is needed for the new API before using it with the hosted frontend.

The module now follows the supplied HTML reference (preserved unchanged as `tests/fixtures/flete_dalje_reference.html` for regression comparisons). Navy/teal styling, field order, editable rows, status pill, totals panel, compact archive and print signature areas follow that design. New sheets start with one row; users add more with “Shto rresht”. Returning to one row preserves the first row, confirming removal of populated extra rows. Reopening saved sheets preserves all saved lines without adding blank rows. Saving requires a recipient and at least one populated line; incomplete named lines prompt for confirmation.

Arithmetic matches the reference: parse comma decimals, remove dot grouping when both separators occur, apply JavaScript `Math.round(quantity * price)` to each line, then sum the whole-lek results. For example, `3.782 × 100 = 378` and `0.312 × 12.5 = 4`, totaling 382 lekë. Zero line values remain visually blank. The server mirrors rounding ties toward positive infinity. Previous saved records recalculate using this rule on read.

Imports accept native backups, the original HTML's `{schema:1, records:[...]}` format, and original raw record arrays. Exported JSON includes both native documents with stable IDs and the original-compatible `records` representation. Import merges identical records by ID or the recipient/number pair, rejects conflicting content atomically, and never silently overwrites existing records. CSV follows the reference's semicolon-separated columns, includes notes, and guards text against spreadsheet formula interpretation.

Printing uses an escaped, static document outside the application shell. A5 landscape with the reference's 7 mm margins, gray table headings, Albanian field labels, boxed whole-lek total, MARRËSI and Nënshkrimi areas, and optional notes. Eight rows use 8 mm heights to reliably fit one page within the printable area. Longer documents paginate naturally with repeated table headings. Browser Save as PDF is supported.


## Checks

```sh
.venv/bin/python -m pytest -q tests/test_flete_dalje.py tests/test_frontend_navigation.py tests/test_invoice_persistence.py tests/test_frontend_security.py tests/test_frontend_theme.py
node --check docs/js/flete-dalje.js
node --check docs/js/app.js
git diff --check
```

Browser integration checks use a disposable database and real delivery API. Start this server in one terminal, then run the browser test in another:

```sh
DB_DIR=/tmp/flete-dalje-qa .venv/bin/python -m uvicorn flete_dalje_qa_server:app --app-dir tests --port 5056
node tests/test_flete_dalje_browser.cjs
```

Use a fresh DB_DIR for each browser run. Set CHROMIUM_PATH if Chromium is not at `/usr/bin/chromium`. The test checks CRUD, archive search, CSV, JSON backup/restore, mobile overflow, navigation, and writes `/tmp/flete-dalje-qa.pdf`. It mocks unrelated frontend API calls. No frontend compilation or lint command is configured in this plain JavaScript application. Repository-wide pytest collection has two `test_smoke.py` filenames; `pytest tests` avoids that collision.

## Verification result

- Focused backend/navigation/security/theme checks after reference adaptation: 30 passed.
- Before reference adaptation, `pytest tests`: 739 passed, 4 failed. All four failures reproduced against an untouched HEAD snapshot: obsolete `frontend/index.html` path, removed Production Copilot, removed workspace queue search, and unavailable Agents SDK engine. Existing order and production tests otherwise passed.
- Real-API Chromium test passed, including comparisons against the actual HTML reference, original backup compatibility, single-row defaults and first-row preservation, mobile width and CSV/JSON downloads.
- Generated PDF: one page, 594.96 × 420 points (A5 landscape), with document contents and no platform navigation.
- JavaScript syntax and `git diff --check` passed. No production deployment performed.

## Kartelat e klientëve dhe pagesat

Use the “Kartelat e klientëve” tab inside Fletë-Dalje. Each card contains every saved delivery document for that recipient, dated payment history, a running balance, and totals at the bottom. Names are grouped by Unicode normalization, case and whitespace; different names remain separate. This is exclusively a delivery-module ledger, with no connection to order/customer databases. Changing a document's recipient moves its debit to that name; payments stay under the recipient recorded when they were made. Deleted documents no longer contribute to totals. Cards with payment history remain available even without documents.

Payments have a date, positive ALL amount (up to two decimal places), method (cash, bank, other), and optional note/reference. Balances are computed server-side in integer cents: delivery totals minus active payments. Overpayments require confirmation and appear as advances/credit. Cancelling a payment requires confirmation, preserves the original record, and removes its deduction. Versions reject repeated/stale cancellation requests. Client-generated immutable payment IDs make retrying an unchanged submission idempotent; reusing an ID with different details is rejected.

The existing SQLite database now contains a separate `flete_dalje_payments` table, created automatically by `init_db()` on backend startup. All client totals are derived from saved delivery documents and payment records. Payment history, including cancelled records and creation timestamps, is included in native JSON backup/restore in the same transaction as documents. Old backups without payments remain valid and do not remove current payments. The original standalone HTML cannot preserve payments if a backup is re-exported through that old app; use the integrated module for complete backups.

Client cards can be printed or saved as PDF. The editor keeps its one-row default and saved document behavior. Browser checks cover payment entry with comma decimals, cancellation, credit balance, complete backup export, card printing, and mobile layout. Focused tests also cover cross-client isolation, document renaming, idempotent retries, invalid payments, and transactional backup conflicts.

Verification for client ledgers: 56 focused regression checks passed, then 21 delivery/payment tests passed including the complete restore case. Browser checks passed for both editor and client cards; the client PDF was verified as one A5 landscape page. Full `pytest tests` run: 775 passed, the same four previously baseline-confirmed failures remained (obsolete frontend path, retired Copilot and queue controls, unavailable Agents SDK engine).

Individual payment receipts: use “Ruaj dhe printo” when recording a payment, or “Printo mandatpagesën” on any saved payment row. Receipts contain the immutable payment reference, original payer name, date, method, paid amount, optional note/reference, and payer/recipient signature areas. They use the existing A5 landscape print/PDF surface. Cancelled payments print a prominent “E ANULUAR — NUK VLEN SI PAGESË” banner. Reprinting refreshes saved data before rendering and never changes the payment or balance. Browser checks cover normal receipts, cancelled receipts, Save and Print, and unchanged payment counts.

## Latest workflow and numbering

Saving a new or edited sheet now opens a blank, single-row sheet with today’s local date. A confirmation identifies the saved sheet and client. Selecting or typing an existing recipient suggests that client’s next number, with a datalist of existing client names. Manual numbering is still supported. Editing a saved sheet keeps its number until explicitly changed. Each archive entry has its own Print/PDF button; immediately after saving, the main print action can reprint the last saved sheet while the editor is empty.

Client balance labels now read “Totali”, with red for an amount still owed and green for settled or prepaid accounts. Text also states “Për t’u paguar”, “E shlyer” or “Parapagim / tepricë”, so color is not the only indicator.

On backend startup, `flete_dalje_schema.migrate_client_numbering` transactionally replaces the old global-number uniqueness with a `(client_key, number)` constraint. IDs, versions, contents and existing numbers are copied unchanged; payment and order tables are untouched. A failed migration rolls back, retaining the original table. Repeated startup is harmless. Backups use client-scoped duplicate checks and deterministic legacy import IDs for recipient/number pairs. Use native module backups for complete restoration: the old standalone HTML still assumes globally unique numbers and cannot represent multiple clients with the same number safely.

Verification: 60 focused checks passed, including scoped numbering, legacy data preservation and migration rollback. Browser checks passed for fresh sheets after save, same-client number suggestions, red/green totals, archive reopening, payments, receipts and printing.

Full regression run after these changes: 779 passed; the same four baseline failures remain. Saved numbering was not rewritten during migration, and original IDs/versions/payment history were verified intact.
