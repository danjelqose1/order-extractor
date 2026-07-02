# Behavioral Audit Report

Audit date: 2026-07-02  
Project: Order Extractor factory platform  
Audit type: Behavioral functionality and data-safety audit  
Reference order: `XHAMA-EDI SHPETIM SULA.pdf`

## 1. Executive summary

### Overall health: Broken

The platform contains several solid building blocks, but it is not safe for unsupervised daily factory use in its current state.

The highest-risk modules are:

1. Extraction and validation
2. Invoice generation
3. History deletion and duplicate handling
4. Processing lifecycle and production-file durability
5. Dashboard failure recovery

The supplied PDF was visually verified as:

- Client: `EDI SHPETIM SULA`
- Order: `R-25-0401`
- Rows: 8
- Pieces: 14
- Total area: 7.060 m2

Three live extractions of the identical PDF produced materially different results:

- Run 1: 14 pieces, 3.860 m2
- Run 2: 14 pieces, 7.060 m2
- Run 3: 14 pieces, 3.860 m2

The order number, client, row count, positions, dimensions, and quantities were extracted correctly. The area interpretation was not deterministic. The model alternated between the PDF's per-piece area column and quantity-total area column.

The first run also reported a false PDF total of `25 units / 0.950 m2`. The real PDF total is `14 units / 7.060 m2`. The false total came from parsing the order-number year fragment and model confidence as totals.

One shared validator was directly confirmed to change a valid quantity of 8 to 3. That validator is called from extraction, order detail, approval, and workspace validation paths.

Invoices are functionally unavailable in the deployed `docs/` frontend. Invoice JavaScript exists, but there is no Invoice navigation item, no `tabInvoices` panel, and no History Invoice controls. A manual order can create a hidden invoice job, but the user cannot open, review, edit, or export it.

Approved orders can be permanently deleted through the API and the History UI. A live isolated test deleted an approved order with HTTP 200.

After this report was written, three narrow safety fixes were applied and verified on the same date: high quantities are now warned about without mutation, declared totals are parsed only from source-derived text and support the supplied PDF's split layout, and only draft PDF orders can be hard-deleted. These fixes reduce immediate risk but do not change the overall health rating because the remaining production blockers are substantial.

### Factory-use decision

Not safe for daily factory use.

The system may be used for supervised testing only if operators independently verify:

- every quantity;
- every quantity-total area;
- every declared PDF total;
- every status transition;
- every production and label PDF;
- every invoice calculation.

Approval must not be treated as sufficient protection until the critical and high findings below are fixed.

## Audit method and scope

The audit used:

- visual rendering of the supplied PDF;
- text-layer extraction and independent total verification;
- source review of backend, frontend, persistence, validation, and export paths;
- an isolated SQLite database at `/tmp/order-extractor-audit-data`;
- a local backend and local frontend;
- three live OpenAI-backed extractions of the supplied PDF;
- live History, Processing, Labels, Analytics, Manual Orders, and mobile checks;
- focused API probes for duplicate handling, deletion, and manual/PDF collisions;
- automated tests and syntax checks.

Existing production history was not used or modified.

## 2. Module-by-module findings

### Dashboard

#### DASH-01

- Module: Dashboard
- Severity: Critical
- Current behavior: When PDF or text extraction fails, `handlePdfExtraction` and `handleTextExtraction` show an error but leave the previous successful extraction rows, draft ID, and approval state in memory. The source preview can already point at the newly failed file.
- Expected behavior: Starting a new extraction must disable approval and isolate or clear the previous result. Failure must never leave old rows associated visually with the new source file.
- Why it matters in factory use: An operator can upload order B, receive a failure, still see order A's rows, and approve order A while looking at order B's preview.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Add an extraction request token and a `resetExtractionForNewSource()` step that clears draft/result state, disables approval, and shows a failure-only state before the request starts.

#### DASH-02

- Module: Dashboard
- Severity: High
- Current behavior: Re-uploading the same PDF while its latest version is still draft silently overwrites that draft. The response has no duplicate status or duplicate warning.
- Expected behavior: Exact hash duplicates must be shown explicitly. The user should choose to reopen the existing draft or create a separate re-extraction version.
- Why it matters in factory use: A rerun can replace a previously reviewed draft and its raw structured extraction without the operator noticing.
- Likely files involved: `backend/db.py`, `backend/app.py`, `docs/js/app.js`
- Recommended minimal fix: Return `duplicate_status: exact_draft_duplicate` and do not overwrite until the frontend sends an explicit re-extract confirmation.

#### DASH-03

- Module: Dashboard
- Severity: Medium
- Current behavior: Dashboard uploads have no explicit file-size limit before `await file.read()`.
- Expected behavior: Reject oversized files before loading them fully into memory and show a clear limit.
- Why it matters in factory use: A very large or malformed PDF can exhaust memory or stall the service.
- Likely files involved: `backend/app.py`, `docs/js/app.js`
- Recommended minimal fix: Add a configurable dashboard upload limit and validate `UploadFile.size` when available plus the final byte length.

#### DASH-04

- Module: Dashboard
- Severity: Medium
- Current behavior: Source PDF preview depends on externally loaded PDF.js. A preview failure is logged only to the console.
- Expected behavior: Extraction may continue, but the review pane must show that source preview is unavailable.
- Why it matters in factory use: Reviewers can assume they are comparing against the source when the source actually failed to render.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Surface PDF preview load errors in `extractSourceHighlight` or the preview empty state.

### Extraction logic

#### EXT-01

- Module: Extraction logic
- Severity: Critical
- Post-audit status: Fixed and regression-tested on 2026-07-02.
- Audited behavior: `validate_rows` changed quantities greater than 6 to at most 3 when the pane area was below 0.4 m2. A focused probe changed quantity 8 to quantity 3.
- Expected behavior: Validation must never alter a quantity based on a heuristic. It may flag the row for review.
- Why it matters in factory use: This silently under-produces glass and corrupts totals, labels, invoices, and analytics.
- Likely files involved: `backend/validators.py`
- Recommended minimal fix: Remove the assignment and emit `warning: unusually_high_quantity` only.

#### EXT-02

- Module: Extraction logic
- Severity: Critical
- Current behavior: The single `area` field has ambiguous semantics. On the supplied PDF, it alternated between per-piece area and quantity-total area across identical extractions. The first and third runs stored 3.860 m2 instead of 7.060 m2.
- Expected behavior: Preserve both the raw per-piece area and raw line-total area when both columns exist. Use an explicit `area_basis` and a deterministic total field.
- Why it matters in factory use: Area drives material planning, analytics, invoices, client totals, and production summaries.
- Likely files involved: `backend/schema.py`, `backend/prompts.py`, `backend/area_dimension_validator.py`, `backend/app.py`, `backend/db.py`, `docs/js/app.js`
- Recommended minimal fix: Before any broad refactor, make the extraction contract explicit about which PDF area column maps to which field. A proper fix requires additive raw/calculated fields and a migration.

#### EXT-03

- Module: Extraction logic
- Severity: High
- Post-audit status: Fixed and regression-tested on 2026-07-02.
- Audited behavior: Declared-total parsing accepted model output text as if it were source PDF text. On the supplied order, it interpreted `R-25-0401`, confidence `0.95`, and a warning containing the word "total" as `25 units / 0.950 m2`.
- Expected behavior: Declared totals must come only from source text or deterministic layout extraction.
- Why it matters in factory use: The review banner gives a confident but false comparison and can push operators toward incorrect manual edits.
- Likely files involved: `backend/app.py`, `backend/utils_text.py`
- Recommended minimal fix: Never parse `bundle.output_text` for PDF totals. Support split text-layer totals such as `Totale`, `14`, `7,060`.

#### EXT-04

- Module: Extraction logic
- Severity: High
- Current behavior: Missing quantity can become 1 before validation because the Pydantic row schema supplies a default. The validator may no longer know that the source value was absent.
- Expected behavior: Preserve `raw_quantity: null`, flag the row, and require review before approval.
- Why it matters in factory use: Defaulting missing quantity to 1 can under-produce without a visible missing-field warning.
- Likely files involved: `backend/schema.py`, `backend/validators.py`, `backend/app.py`
- Recommended minimal fix: Remove the schema default for extraction input or retain field-presence metadata before model validation.

#### EXT-05

- Module: Extraction logic
- Severity: High
- Current behavior: Rows with missing dimension, blank type, blank position, or zero area can reach approval with warnings. Approval does not block unresolved critical fields.
- Expected behavior: Approval must return 409/422 while any required production field is unresolved.
- Why it matters in factory use: An approved order can be unusable or dangerous on the production floor.
- Likely files involved: `backend/app.py`, `backend/validators.py`, `docs/js/app.js`
- Recommended minimal fix: Add a deterministic `approval_blockers` check after validation and before `update_order_rows`.

#### EXT-06

- Module: Extraction logic
- Severity: Medium
- Current behavior: `SUSPICIOUS_DIMENSION_RE` flags any 2-3 digit height, including the valid supplied dimensions `472x761`.
- Expected behavior: Only dimensions meeting a defensible truncation rule should be flagged.
- Why it matters in factory use: False alarms train operators to ignore real warnings and waste review time.
- Likely files involved: `backend/dimension_repair.py`
- Recommended minimal fix: Remove the regex-only rule and use the existing numeric threshold plus source-column evidence.

#### EXT-07

- Module: Extraction logic
- Severity: Medium
- Current behavior: AI extraction is the primary row parser even for a text-layer KELI document. Deterministic logic is applied mostly after AI extraction.
- Expected behavior: Deterministic vendor parsing should establish order number, client, row columns, quantity, and both area columns before AI repair.
- Why it matters in factory use: Identical documents currently produce materially different totals.
- Likely files involved: `backend/llm.py`, `backend/prompts.py`, `backend/extraction_normalizer.py`, new vendor parser module
- Recommended minimal fix: Add a KELI parser as a first-pass source and use AI only for unresolved cells.

### History

#### HIST-01

- Module: History
- Severity: Critical
- Post-audit status: Fixed for PDF orders and regression-tested on 2026-07-02. Manual-order deletion remains a separate finding.
- Audited behavior: Approved, in-production, completed, and archived orders could be permanently deleted. The API performed no status check, and the UI enabled Delete for every status.
- Expected behavior: Only drafts may be hard-deleted. Production statuses must be archived or deleted through a separately authorized retention workflow.
- Why it matters in factory use: Production records can disappear permanently, including rows, extraction source, and status history.
- Likely files involved: `backend/app.py`, `backend/db.py`, `docs/js/app.js`
- Recommended minimal fix: Block deletion for every non-draft status and tell the user to archive instead.

#### HIST-02

- Module: History
- Severity: High
- Current behavior: Re-extracting an approved order creates a new draft version and preserves the approved row set. This protection works. However, re-extracting the newest draft overwrites that draft's rows and extraction JSON.
- Expected behavior: Every re-extraction should create an immutable extraction attempt or require explicit replacement.
- Why it matters in factory use: Raw extraction evidence and reviewed draft changes can be lost.
- Likely files involved: `backend/db.py`
- Recommended minimal fix: Store each extraction attempt as a new version, including draft attempts, or add an immutable extraction-attempt table.

#### HIST-03

- Module: History
- Severity: Medium
- Current behavior: Version and parent-order information are returned by the API but not shown in the History list. Two rows with the same client and order number appear nearly identical.
- Expected behavior: Show a compact version badge and protected-parent relationship without redesigning the list.
- Why it matters in factory use: Operators can approve the wrong duplicate version.
- Likely files involved: `docs/js/app.js`, `docs/index.html`
- Recommended minimal fix: Add `v1`, `v2`, and "re-extraction of #ID" metadata to the existing status/action area.

#### HIST-04

- Module: History
- Severity: Medium
- Current behavior: Manual row corrections persisted after approval and remained intact after the same PDF was re-extracted into a new draft. Raw PDF bytes also remained available. This behavior passed.
- Expected behavior: Keep this protection.
- Why it matters in factory use: Approved corrections are production data.
- Likely files involved: `backend/db.py`
- Recommended minimal fix: Add regression tests; do not change the current approved-version protection.

#### HIST-05

- Module: History
- Severity: Low
- Current behavior: `has_more` is true whenever a page contains exactly `limit` rows, even if no next row exists.
- Expected behavior: Fetch `limit + 1` or return a count.
- Why it matters in factory use: The user can reach an empty extra page.
- Likely files involved: `backend/app.py`, `backend/db.py`
- Recommended minimal fix: Query one extra record and trim the response.

### Processing

#### PROC-01

- Module: Processing
- Severity: High
- Current behavior: Sending a PDF order from History to Processing adds it only to browser memory. The order status remains `approved`; it does not become `in_production`.
- Expected behavior: A successful send or successful production-file generation must create a clear status event.
- Why it matters in factory use: Dashboard, work queue, and History disagree with the operator's production action.
- Likely files involved: `docs/js/app.js`, `backend/app.py`, `backend/workspace_service.py`
- Recommended minimal fix: After production files are successfully generated, call the existing status endpoint to transition `approved -> in_production`.

#### PROC-02

- Module: Processing
- Severity: High
- Current behavior: The normal Processing cart, grouped rows, header overrides, label jobs, and generated blob URLs exist only in browser memory. Reloading loses the workspace. The frontend Workspace path also generates blob URLs and records only in-memory metadata.
- Expected behavior: Production batches and files must be durable and reopenable.
- Why it matters in factory use: A refresh, browser crash, or workstation change loses the active production job and its file history.
- Likely files involved: `docs/js/app.js`, `backend/workspace_service.py`, `backend/db.py`
- Recommended minimal fix: Reactivate a backend batch endpoint that stores `ProcessingBatch` and `ProductionFile` records. The current backend generation block is unreachable after an early `frontend_workflow_required` return.

#### PROC-03

- Module: Processing
- Severity: Medium
- Current behavior: "Clear Sheet" immediately discards the session workspace without confirmation.
- Expected behavior: Confirm when rows, header edits, or generated labels exist.
- Why it matters in factory use: An accidental click loses shift work.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Add a native confirmation containing the number of orders and whether label jobs exist.

#### PROC-04

- Module: Processing
- Severity: Low
- Current behavior: On the corrected supplied order, area stayed 7.060 m2 while Danko rounding changed dimensions and displayed the original values. Grouping reduced eight source rows to four dimension lines with quantities 2, 4, 4, and 4. This behavior passed.
- Expected behavior: Keep calculated processing dimensions separate from source dimensions.
- Why it matters in factory use: Original dimensions remain traceable.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Add regression tests for area preservation and `__original` fields.

### Labels

#### LABEL-01

- Module: Labels
- Severity: High
- Current behavior: Labels can be generated from draft orders. The History action and order-detail Print Labels control do not require approval. Telegram enables Print Labels for any linked non-duplicate draft.
- Expected behavior: Production labels must require approved/in-production/completed status, or be visibly watermarked "DRAFT".
- Why it matters in factory use: Unreviewed dimensions and quantities can reach production.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Apply `canProcessingStatus` or a dedicated `canLabelStatus` check to every label entry point.

#### LABEL-02

- Module: Labels
- Severity: Medium
- Current behavior: Label jobs snapshot Processing rows. If Processing later changes, an existing job is not marked stale. "Add from Processing" skips the same source key instead of refreshing it.
- Expected behavior: Processing changes must invalidate or version dependent label jobs.
- Why it matters in factory use: A label can show old dimensions or positions after regrouping or header edits.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Store a processing-state hash on each label job and show a stale badge when it no longer matches.

#### LABEL-03

- Module: Labels
- Severity: Medium
- Current behavior: PDF generation depends on unpinned external `pdf-lib` from jsDelivr.
- Expected behavior: Factory label generation must work with the deployed application bundle without public CDN access.
- Why it matters in factory use: Network or CDN failure can stop label production.
- Likely files involved: `docs/js/app.js`, frontend asset bundle
- Recommended minimal fix: Vendor a pinned `pdf-lib` build and load it locally.

#### LABEL-04

- Module: Labels
- Severity: Low
- Current behavior: The supplied processed order created 14 label items, including more than nine labels. Quantity expansion produced 14 one-label rows and preserved processing positions. This behavior passed.
- Expected behavior: Keep this behavior.
- Why it matters in factory use: No labels were silently truncated at 9.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Add a generated-PDF page-count test for 14 and 100 labels.

### Invoices

#### INV-01

- Module: Invoices
- Severity: Critical
- Current behavior: The deployed `docs/index.html` contains no Invoice navigation button, no `tabInvoices`, no invoice detail UI, and no History Invoice buttons. The JavaScript references elements that do not exist. Manual Invoice creates a server job but leaves the user on Manual Orders.
- Expected behavior: Invoice jobs must be reachable, reviewable, editable, and exportable through the existing visual style.
- Why it matters in factory use: The module cannot be used operationally, and hidden invoice jobs accumulate without review.
- Likely files involved: `docs/index.html`, `docs/js/app.js`, `docs/css/styles.css`
- Recommended minimal fix: Restore the existing Invoice panel markup and navigation contracts expected by `app.js`. Do not redesign the module.

#### INV-02

- Module: Invoices
- Severity: High
- Current behavior: PDF rows store quantity-total area after correction, but `buildInvoiceLinesFromRaw` multiplies every row area by quantity. A total-area row with quantity 2 is doubled again.
- Expected behavior: Multiply only per-piece area. Use line-total area directly when `area_basis == total_for_quantity`.
- Why it matters in factory use: Invoice area and amount can be almost doubled.
- Likely files involved: `docs/js/app.js`, extraction area contract
- Recommended minimal fix: Carry `area_basis`, `area_per_piece`, and `area_total` into invoice rows and calculate from explicit semantics.

#### INV-03

- Module: Invoices
- Severity: High
- Current behavior: Composition parsing assumes `+` separators. The supplied type `2 vetri 33.1F 12 caldo 33.1 LowE` is treated as one pane token. Spacer thickness is missed and the code can fall back to AI matching.
- Expected behavior: Deterministically parse both plus-separated and whitespace-separated glass/spacer formats.
- Why it matters in factory use: Glass and thermal spacer prices can be omitted or assigned to the wrong component.
- Likely files involved: `docs/js/app.js`, `backend/invoice_ai.py`
- Recommended minimal fix: Introduce deterministic tokenization tests for common KELI descriptions before invoking AI.

#### INV-04

- Module: Invoices
- Severity: High
- Current behavior: AI fallback may change normalized display type and resolve pricing components. Invoice totals are therefore not fully deterministic.
- Expected behavior: Unknown composition must block finalization and request an explicit mapped price/type from the operator.
- Why it matters in factory use: Recalculating the same invoice can depend on an external model response.
- Likely files involved: `docs/js/app.js`, `backend/invoice_ai.py`
- Recommended minimal fix: Restrict AI to suggestions. Require user acceptance before storing a type mapping or recalculating a final invoice.

#### INV-05

- Module: Invoices
- Severity: Medium
- Current behavior: Invoice jobs and prices are stored as whole JSON files with no schema validation, locking, revision, or atomic replace.
- Expected behavior: Concurrent saves must not overwrite each other or corrupt the file.
- Why it matters in factory use: Two workstations can lose invoice jobs or price changes.
- Likely files involved: `backend/app.py`
- Recommended minimal fix: Store invoice jobs in SQLite, or at minimum use a lock, temporary file, atomic rename, and revision check.

### Analysis

#### ANALYSIS-01

- Module: Analysis
- Severity: Medium
- Current behavior: Monthly revenue is not implemented. The server summary contains order, unit, area, client, type, dimension, and daily metrics only.
- Expected behavior: If revenue is presented as a required KPI, compute it from finalized invoices with an explicit period.
- Why it matters in factory use: Area and invoice revenue are not interchangeable.
- Likely files involved: `backend/analytics_summary.py`, `docs/index.html`, `docs/js/app.js`
- Recommended minimal fix: Add revenue only after invoice finalization and persistence are repaired.

#### ANALYSIS-02

- Module: Analysis
- Severity: Medium
- Current behavior: Latest-orders pagination is not part of Analysis. History has page controls, but Analysis does not.
- Expected behavior: Either add the requested list or explicitly remove it from the module requirements.
- Why it matters in factory use: Operators cannot inspect the source orders behind a metric from the Analysis page.
- Likely files involved: `backend/analytics_summary.py`, `docs/index.html`, `docs/js/app.js`
- Recommended minimal fix: Add a small paginated source-order list using existing History data contracts.

#### ANALYSIS-03

- Module: Analysis
- Severity: Low
- Current behavior: Analysis correctly used only approved/in-production/completed PDF orders. Draft and manual-order records were excluded. The corrected supplied order showed 1 order, 14 units, and 7.060 m2.
- Expected behavior: Keep the status filter and make manual-order inclusion an explicit product decision.
- Why it matters in factory use: Draft extraction noise does not pollute KPIs.
- Likely files involved: `backend/analytics_summary.py`, `backend/db.py`
- Recommended minimal fix: Add explicit tests for archived, deleted, draft, and manual sources.

### Scan Studio

#### SCAN-01

- Module: Scan Studio
- Severity: Medium
- Current behavior: Perspective correction and filtering run per pixel on the browser main thread, up to 2800 pixels on the longest side and across all pages during export.
- Expected behavior: Large multi-page scans should not freeze the UI.
- Why it matters in factory use: The workstation can become unresponsive during scan cleanup or export.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Move expensive transforms to a Web Worker or process pages incrementally with progress and cancellation.

#### SCAN-02

- Module: Scan Studio
- Severity: Medium
- Current behavior: PDF export depends on the same external unpinned `pdf-lib`.
- Expected behavior: Export must work offline from the deployed bundle.
- Why it matters in factory use: A public CDN outage blocks scan PDF output.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Vendor the dependency locally.

#### SCAN-03

- Module: Scan Studio
- Severity: Low
- Current behavior: Scan Studio explicitly accepts JPG, PNG, and WebP only. PDFs dropped globally are routed to Dashboard extraction, not treated as Scan Studio extraction. The image workflow includes crop, rotation, perspective, grayscale, document filter, B/W, brightness, contrast, print, image export, and PDF export.
- Expected behavior: This is acceptable if image-only input is intentional. If PDF preview is required, add it explicitly.
- Why it matters in factory use: The current UI text is clear and Scan Studio does not silently act as an extraction engine for images.
- Likely files involved: `docs/index.html`, `docs/js/app.js`
- Recommended minimal fix: No change unless PDF page import is a confirmed requirement.

### Telegram Inbox

#### TELE-01

- Module: Telegram Inbox
- Severity: High
- Current behavior: Print Labels is enabled for a linked draft order. No status gate is applied before `handlePrint`.
- Expected behavior: Telegram-linked orders must follow the same approval requirement as Dashboard/History.
- Why it matters in factory use: Incoming unreviewed Telegram extraction can go directly to labels.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Disable label printing until the linked order is approved or later.

#### TELE-02

- Module: Telegram Inbox
- Severity: Medium
- Current behavior: `printTelegramOriginalPdf` marks `pdf_printed` even if the print trigger throws, the browser blocks the popup, or the user cancels the print dialog.
- Expected behavior: Track "print opened/requested" separately from "printed", or only mark after a supported print completion signal.
- Why it matters in factory use: Operators may believe the source PDF was printed when it was not.
- Likely files involved: `docs/js/app.js`, `backend/db.py`
- Recommended minimal fix: Rename the current state to `print_requested` or defer marking until the print window successfully opens.

#### TELE-03

- Module: Telegram Inbox
- Severity: Low
- Current behavior: Durable intake, FIFO queueing, exact SHA-256 duplicate detection, separate possible-duplicate detection, original-file preservation, retry handling, startup recovery, and soft-delete behavior are implemented and heavily tested.
- Expected behavior: Keep these protections.
- Why it matters in factory use: Telegram intake is one of the safer parts of the platform.
- Likely files involved: `backend/app.py`, `backend/db.py`, `docs/js/app.js`
- Recommended minimal fix: Add the same exact-hash behavior to Dashboard uploads.

### Manual Orders

#### MANUAL-01

- Module: Manual Orders
- Severity: High
- Current behavior: Manual and PDF orders use separate tables and sections, which is good. However, manual duplicate checking only checks `manual_orders`. A manual order with PDF order number `R-25-0401` was accepted with `duplicate_warning: false`.
- Expected behavior: Manual order-number entry must also check PDF History and clearly warn or require a source-qualified identifier.
- Why it matters in factory use: Processing and labels can contain two unrelated orders with the same visible order number.
- Likely files involved: `backend/db.py`, `backend/app.py`, `docs/js/app.js`
- Recommended minimal fix: Extend the check-number endpoint to return `manual_match` and `pdf_history_match` separately.

#### MANUAL-02

- Module: Manual Orders
- Severity: High
- Current behavior: Processing can contain a PDF order and a manual order with the same visible order number. The cart IDs remain technically separate, but the Mother Sheet order set collapses the visible order label.
- Expected behavior: Preserve source-qualified identity in combined workflows while keeping the printed business order number.
- Why it matters in factory use: Operators cannot reliably distinguish the two jobs in summaries and labels.
- Likely files involved: `docs/js/app.js`
- Recommended minimal fix: Use internal keys such as `pdf:1` and `manual:1` while rendering `R-25-0401 [PDF]` and `R-25-0401 [Manual]` when a collision exists.

#### MANUAL-03

- Module: Manual Orders
- Severity: Medium
- Current behavior: Manual orders have separate storage, explicit source, quantity-aware calculated area, optional area override, Processing support, Labels support, and their own statuses. These behaviors passed.
- Expected behavior: Keep the isolation.
- Why it matters in factory use: Manual orders do not overwrite PDF History rows.
- Likely files involved: `backend/db.py`, `backend/app.py`, `docs/index.html`, `docs/js/app.js`
- Recommended minimal fix: Add cross-source collision warnings and status-event history without merging the tables.

#### MANUAL-04

- Module: Manual Orders
- Severity: Medium
- Current behavior: Manual orders in processing or finished states can be hard-deleted.
- Expected behavior: Restrict hard deletion to drafts/cancelled orders or require a protected retention workflow.
- Why it matters in factory use: Manually entered production records can disappear.
- Likely files involved: `backend/db.py`, `backend/app.py`, `docs/js/app.js`
- Recommended minimal fix: Add the same non-draft deletion guard recommended for PDF orders.

### Shared platform and test infrastructure

#### SHARED-01

- Module: Shared platform
- Severity: High
- Current behavior: Destructive and mutating endpoints are unauthenticated unless deployment-specific controls exist outside this repository. `APP_KEY` protects extraction only, and the frontend does not send it.
- Expected behavior: Production history, status, prices, invoices, files, and delete operations require authenticated authorization.
- Why it matters in factory use: A public deployment can be modified or erased by an unauthorized caller.
- Likely files involved: `backend/app.py`, frontend request wrapper, deployment configuration
- Recommended minimal fix: Add one consistent authenticated middleware and role checks. Do not add separate ad hoc keys per route.

#### SHARED-02

- Module: Test infrastructure
- Severity: Medium
- Current behavior: `pytest -q` fails during collection because both `backend/test_smoke.py` and `tests/test_smoke.py` use the same module name. Several tests still reference a removed `frontend/index.html`. One Agents SDK test fails when the optional dependency is unavailable.
- Focused result: The relevant suite passed 215 tests with 2 explicitly excluded environment/path-dependent tests and `tests/test_pdf_editor_static.py` excluded because it targets the removed frontend path.
- Expected behavior: The default test command must collect and run reliably in CI.
- Why it matters in factory use: Broken CI hides regressions in data-safety logic.
- Likely files involved: `backend/test_smoke.py`, `tests/test_smoke.py`, `tests/test_pdf_editor_static.py`, `tests/test_client_name_persistence.py`, test configuration
- Recommended minimal fix: Keep one smoke suite or use package-safe names, update paths to `docs/`, and mark optional SDK tests with a dependency condition.

## 3. Data safety findings

### Raw extracted/PDF data overwrite

- Raw PDF bytes are preserved in `Extraction.raw_input` after successful extraction.
- Raw model JSON and base64-extraction values are preserved in `Extraction.llm_output_json`.
- Approved re-extraction creates a new draft version and preserves the approved order.
- Re-extracting the latest draft overwrites its `OrderRow` records and `Extraction` JSON. The prior extraction attempt is lost.
- Dashboard extraction failure does not preserve the failed PDF on the server.

### Approved History overwrite or deletion

- Approved rows were not overwritten by same-hash re-extraction. This passed.
- Approved orders can still be permanently deleted. This failed.
- No role or authorization layer protects approved History endpoints in the repository.

### Manual overrides

- Corrections approved on version 1 remained intact when the same PDF created version 2. This passed.
- Draft edits exist only in browser memory until approval. A reload loses them.
- A same-hash draft re-extraction can overwrite draft extraction state.

### Calculated values replacing originals

- The high-quantity heuristic replaces original quantity with a smaller value.
- The normalized `OrderRow` table stores only one dimension, quantity, and area value. Raw fields remain only inside extraction JSON.
- Processing keeps raw dimensions in browser-only `__original` fields and adds rounded dimensions separately. This passed.
- Invoice calculation does not carry enough area semantics and can multiply a quantity-total area again.

### PDF and manual order mixing

- Database storage is correctly separated.
- Cross-source order-number collision detection is missing.
- Combined Processing can visually collapse PDF and manual orders sharing the same business number.
- Analysis currently excludes manual orders entirely.

## 4. Broken or suspicious UI behavior

- Invoice module has no reachable page or navigation.
- Manual Invoice reports success but provides no way to open the generated job.
- History and Telegram can generate labels from drafts.
- "Fix all areas" does not fix anything; it only tells the user to fix rows individually.
- Failed Dashboard extraction can leave stale rows and approval enabled.
- Same-PDF draft overwrite is not surfaced as a duplicate.
- Version numbers are hidden in History.
- Normal Processing reports "Added to Processing" while History remains Approved.
- Clear Sheet has no confirmation.
- Label jobs are not marked stale after Processing changes.
- Telegram "PDF printed" can be set after a failed or cancelled print.
- Source PDF preview failures are console-only.
- Mobile checks at a 375 px viewport passed for Overview and History: no page-level horizontal overflow, controls remained discoverable, and navigation collapsed correctly.

## 5. Recommended fixes

### Must fix before production

1. Remove quantity mutation from `validate_rows`. Completed 2026-07-02.
2. Define and preserve per-piece area versus line-total area.
3. Fix declared-total parsing and stop parsing model output as source data. Completed 2026-07-02.
4. Reset/lock Dashboard result state when a new extraction begins or fails.
5. Block approval while critical fields are unresolved.
6. Block hard deletion of approved/production/completed/archived PDF orders. Completed 2026-07-02.
7. Restore a reachable Invoice UI, then fix invoice area semantics.
8. Require approval before every label-generation path.
9. Make production batches/files durable and transition status only after successful generation.
10. Add cross-source order-number collision warnings.

### Should fix soon

1. Make all re-extraction attempts immutable.
2. Show version/parent metadata in History.
3. Vendor PDF.js and pdf-lib locally.
4. Invalidate stale label jobs.
5. Protect manual non-draft deletion.
6. Correct false truncation warnings.
7. Add upload size limits.
8. Add authenticated access to all mutating endpoints.
9. Make invoice persistence transactional.
10. Move Scan Studio heavy transforms off the main thread.

### Nice to improve later

1. Add Analysis revenue after invoice finalization is reliable.
2. Add Analysis source-order pagination.
3. Improve History pagination's `has_more` calculation.
4. Add explicit Scan Studio PDF page import if required.
5. Add a source-qualified display when PDF and manual order numbers collide.

## 6. Practical test plan

### Dashboard extraction

- [ ] Upload a valid text-layer PDF.
- [ ] Upload a scanned PDF.
- [ ] Drop a PDF.
- [ ] Drop an image and confirm Scan Studio routing.
- [ ] Drop plain text and confirm text extraction.
- [ ] Upload an empty, malformed, oversized, and unsupported file.
- [ ] Confirm loading state disables Approve.
- [ ] Confirm failure clears or quarantines old rows.
- [ ] Confirm exact duplicate draft requires a choice.
- [ ] Confirm exact duplicate approved order preserves production data.
- [ ] Repeat the same PDF three times and compare deterministic output.

### History safety

- [ ] Save a draft and reopen it.
- [ ] Edit every production field, approve, and reopen.
- [ ] Re-extract the source PDF and verify approved rows are unchanged.
- [ ] Verify version metadata is visible.
- [ ] Attempt hard delete for draft, approved, production, completed, and archived.
- [ ] Test search by order and client.
- [ ] Test status, date, year, and approved-only filters.
- [ ] Verify summaries equal row totals.

### Processing

- [ ] Send one approved order.
- [ ] Confirm status changes only after files are generated.
- [ ] Apply and undo rounding.
- [ ] Group and ungroup dimensions.
- [ ] Process two orders with merge off.
- [ ] Process two orders with merge on.
- [ ] Verify raw dimensions and PDF areas never change.
- [ ] Reload and reopen the batch.
- [ ] Clear a non-empty workspace and verify confirmation.
- [ ] Render and inspect the production PDF.

### Labels

- [ ] Generate from approved History.
- [ ] Confirm draft labels are blocked.
- [ ] Generate from grouped Processing.
- [ ] Verify one label per piece.
- [ ] Verify more than 9 positions and more than 100 labels.
- [ ] Check order, position, dimension, type, and processing index.
- [ ] Change Processing after label creation and verify stale state.
- [ ] Render the PDF and count pages.

### Invoices

- [ ] Verify the Invoice page is reachable.
- [ ] Add one approved PDF order.
- [ ] Add multiple orders.
- [ ] Group same types per intended rules.
- [ ] Test a quantity-total area and ensure it is not multiplied twice.
- [ ] Test triple glazing with two spacers.
- [ ] Test `caldo`, `c.caldo`, `termico`, normal, and case variants.
- [ ] Test plus-separated and whitespace-separated compositions.
- [ ] Verify unknown types block finalization.
- [ ] Edit a type and confirm original order rows are unchanged.
- [ ] Reload and confirm jobs/prices persist.
- [ ] Render and inspect invoice PDF totals.

### Analysis

- [ ] Verify draft, archived, and deleted policy.
- [ ] Verify approved, in-production, and completed totals.
- [ ] Compare units and area against direct row totals.
- [ ] Test all-time and custom dates.
- [ ] Test client/type filters.
- [ ] Verify daily chart and top dimensions.
- [ ] Add revenue only from finalized invoices.
- [ ] Verify manual-order inclusion policy explicitly.

### Telegram Inbox

- [ ] Send multiple PDFs quickly and confirm FIFO behavior.
- [ ] Send the same update twice.
- [ ] Send the same file under another filename.
- [ ] Send same filename with different bytes.
- [ ] Force download timeout, 429, permanent 4xx, and extraction failure.
- [ ] Restart during queued and processing states.
- [ ] Verify original file remains after extraction failure.
- [ ] Confirm draft label printing is blocked.
- [ ] Confirm printed/requested state cannot report false success.
- [ ] Delete a file linked to draft and approved orders.

### Manual Orders

- [ ] Create, edit, duplicate, approve, process, label, and invoice.
- [ ] Test calculated and overridden area.
- [ ] Verify quantity greater than 6 is preserved.
- [ ] Enter a PDF History order number and require a collision warning.
- [ ] Confirm manual rows never appear in PDF History.
- [ ] Confirm internal IDs remain distinct in combined Processing.
- [ ] Protect processing/finished records from hard deletion.

## 7. Safe implementation plan

| Patch | Layer | Likely files | Must not break | Verification |
| --- | --- | --- | --- | --- |
| Stop high-quantity mutation | Shared backend validation | `backend/validators.py`, tests | Existing warning rendering and quantity parsing | Unit test quantity 8 remains 8; extraction/approval totals remain unchanged |
| Fix declared totals | Backend deterministic parsing | `backend/utils_text.py`, `backend/app.py`, tests | Existing single-line `Totale 344 192,000` parsing | Supplied text layer returns 14/7.060; model JSON returns no total |
| Protect approved deletion | Backend and History frontend | `backend/app.py`, `docs/js/app.js`, tests | Draft deletion and archive flow | Draft delete 200; approved delete 409; archive still works |
| Reset failed extraction state | Frontend | `docs/js/app.js` | Retry, activity center, source preview | Successful A, failing B, confirm A cannot be approved under B |
| Gate labels by status | Frontend/shared rule | `docs/js/app.js`, tests | Approved History, Processing, manual approved labels | Draft buttons disabled; backend-fetched draft rejected |
| Show version metadata | Frontend | `docs/js/app.js` | Existing History layout and mobile cards | Two same-number versions remain distinguishable at 375 px |
| Area model migration | Backend/database/shared frontend | `backend/db.py`, `backend/schema.py`, extraction and invoice modules | Existing approved rows and raw JSON | Migration backfill, sample PDF, quantity 1 and >1, invoice reconciliation |
| Restore Invoice UI | Frontend | `docs/index.html`, `docs/css/styles.css`, `docs/js/app.js` | Existing sidebar style and History workflows | Reachability, multi-order, pricing, PDF export, mobile |
| Durable production batches | Backend/database/frontend | `backend/workspace_service.py`, `backend/app.py`, `backend/db.py`, `docs/js/app.js` | Existing Processing output format and manual workspace isolation | Reload/reopen, file downloads, status events, duplicate batch prevention |
| Cross-source collision check | Backend/frontend | `backend/db.py`, `backend/app.py`, `docs/js/app.js` | Separate manual/PDF tables | Same-number manual creation warns without overwriting |

No broad refactor is recommended. Each patch should be independently reversible and covered by a focused test before the next patch.

## Copy-pastable follow-up prompts for larger fixes

### Follow-up prompt: area data model and migration

```text
Implement the area-safety fix from BEHAVIORAL_AUDIT_REPORT.md without redesigning the UI.

Goals:
- Preserve raw PDF per-piece area and raw PDF line-total area separately.
- Preserve raw quantity and raw dimensions separately from normalized/calculated fields.
- Add an explicit area_basis.
- Make the canonical production area deterministic and quantity-aware.
- Never overwrite existing raw extraction JSON.
- Migrate existing rows additively and keep approved rows unchanged unless a deterministic backfill is provably safe.
- Update extraction, History, Processing, Labels, Invoices, and Analysis to use explicit area semantics.

Before editing:
- Propose the additive schema and migration.
- List how quantity=1, quantity>1, missing area, and ambiguous two-area-column rows behave.

Required tests:
- The supplied XHAMA PDF must produce 14 pieces and 7.060 m2 on repeated runs.
- Invoice area must remain 7.060 m2, not 13.460 m2.
- Re-extraction must preserve approved corrections and raw source fields.
```

### Follow-up prompt: restore and repair Invoices

```text
Restore the existing Invoice module behavior described in BEHAVIORAL_AUDIT_REPORT.md.

Constraints:
- Preserve the current dark factory UI and sidebar patterns.
- Restore the DOM contracts already expected by docs/js/app.js instead of redesigning.
- Add History and Manual Order entry points only for approved/production/completed orders.
- Make all pricing deterministic; AI may suggest mappings but must not silently affect totals.
- Support whitespace-separated and plus-separated KELI compositions, triple glazing, and thermal/normal spacers.
- Use explicit per-piece versus line-total area fields.
- Do not mutate original order rows.
- Persist jobs transactionally and support multi-order invoices.

Required tests:
- Reachable Invoice page.
- Supplied order total area is 7.060 m2.
- Triple glazing prices both spacer layers.
- Unknown types block finalization.
- Multi-order PDF totals reconcile exactly.
```

### Follow-up prompt: durable Processing and production files

```text
Implement durable Processing batches using the existing ProcessingBatch and ProductionFile models.

Constraints:
- Preserve the current Processing UI and output format.
- Keep raw dimensions and raw area immutable.
- Store processed/rounded/grouped fields separately.
- Generate files through an explicit backend batch endpoint.
- Change order status to in_production only after both production and label files are saved successfully.
- Make reprocessing explicit and versioned.
- Keep multi-order boundaries unless mergeAcrossOrders is explicitly true.
- Preserve the isolated Workspace behavior.

Required tests:
- Reload and reopen a batch.
- Failed generation leaves order Approved and records a failed batch.
- Successful generation records status history and durable downloads.
- Clearing a browser workspace does not delete the saved batch.
```

### Follow-up prompt: authentication and retention

```text
Add production-safe authentication and retention controls described in BEHAVIORAL_AUDIT_REPORT.md.

Constraints:
- Do not introduce separate ad hoc keys per route.
- Protect all mutating order, status, manual-order, Telegram, invoice, pricing, and file endpoints.
- Allow hard delete only for drafts by default.
- Use archive for production records.
- Record actor and reason for status changes and protected deletes.
- Keep existing local development usable through an explicit development mode.

Required tests:
- Anonymous mutation is rejected.
- Authorized read-only users cannot delete or approve.
- Draft delete succeeds for authorized operators.
- Approved delete is rejected and archive succeeds.
```

## Post-audit tiny fixes applied

The following small, reversible protections were implemented only after the report findings were recorded:

- `backend/validators.py`: quantities above 6 on small panes now retain the source value and emit `warning: unusually_high_quantity`.
- `backend/utils_text.py` and `backend/app.py`: declared totals no longer inspect model-output prose; the supplied split text layer now resolves to 14 pieces and 7.060 m2.
- `backend/app.py` and `docs/js/app.js`: the API returns HTTP 409 for non-draft PDF-order deletion, and the History Delete control is disabled for non-drafts with an archive instruction.
- Regression coverage was added for all three protections. The relevant suite passed 215 tests, and the approved-order deletion guard was confirmed against the isolated live database.

## Final assessment

Strong existing behavior:

- Correct sample client and order-number extraction.
- Correct row count, positions, dimensions, and quantities on the sample.
- Approved re-extraction protection.
- Raw PDF preservation after successful extraction.
- Correct Processing rounding traceability.
- Correct grouped quantities and preserved area after manual correction.
- Fourteen-label generation without a nine-label cutoff.
- Correct approved-order Analysis totals after area correction.
- Good mobile behavior on Overview and History at 375 px.
- Robust Telegram intake and exact-file duplicate foundations.
- Correct separation of manual and PDF storage.

Remaining production blockers:

- Non-deterministic area semantics.
- Stale Dashboard data after failure.
- Missing Invoice UI and unsafe invoice area logic.
- Labels from unapproved drafts.
- Non-durable production workspaces/files and missing PDF status transition.
- Cross-source order-number collisions.
- Manual non-draft orders can still be hard-deleted.
