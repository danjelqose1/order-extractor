# Fletë-Dalje

Native standalone tab in Order Extractor. Records are stored in the existing backend SQLite database (`DB_DIR/orders.db`) in `flete_dalje_documents`. `init_db()` creates the table at startup. There are no foreign keys or reads/writes to orders, invoices, customers, or production. The API follows the existing invoice API's access pattern (no separate user authentication); private MCP authentication does not apply to browser modules.

Numbers start at 49 and are suggested from the highest archived number, with 48 as the floor. Numbers are unique database integers. Record IDs remain stable when numbers change. Conditional updates/deletes reject stale versions. Import merges missing records in a transaction, skips identical records, and rejects conflicts without overwriting anything. Backups contain saved records only. Browser storage is not used for document records. A backend deployment is needed for the new API before using it with the hosted frontend.

The module now follows the supplied HTML reference (preserved unchanged as `tests/fixtures/flete_dalje_reference.html` for regression comparisons). Navy/teal styling, field order, editable rows, status pill, totals panel, compact archive and print signature areas follow that design. New sheets start with one row; users add more with “Shto rresht”. Returning to one row preserves the first row, confirming removal of populated extra rows. Reopening saved sheets preserves all saved lines without adding blank rows. Saving requires a recipient and at least one populated line; incomplete named lines prompt for confirmation.

Arithmetic matches the reference: parse comma decimals, remove dot grouping when both separators occur, apply JavaScript `Math.round(quantity * price)` to each line, then sum the whole-lek results. For example, `3.782 × 100 = 378` and `0.312 × 12.5 = 4`, totaling 382 lekë. Zero line values remain visually blank. The server mirrors rounding ties toward positive infinity. Previous saved records recalculate using this rule on read.

Imports accept native backups, the original HTML's `{schema:1, records:[...]}` format, and original raw record arrays. Exported JSON includes both native documents with stable IDs and the original-compatible `records` representation. Import merges identical records by ID or document number, rejects conflicting content atomically, and never silently overwrites existing records. CSV follows the reference's semicolon-separated columns, includes notes, and guards text against spreadsheet formula interpretation.

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
