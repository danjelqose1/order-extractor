# Perfect Cut Bridge — frontend V1

Under Production → Perfect Cut Bridge, choose **Add from Processing**, select a glass/type section and prepared rows, add them, identify the cutting material/pass, confirm, and download `job.csv`.

## Behavior and boundaries

- Reads `appState.processing.preview.groups[].lines`, the canonical Mother Sheet output. No extra rounding, dimension swapping, grouping, quantity expansion, PDF parsing or label expansion occurs.
- `convertOrderToProcessingEntry` and `buildOriginRowPayload` retain provenance only. Existing Processing/Labels transformations are unchanged. Identity includes source kind, order ID and original row ID (source index fallback); grouped lines retain every origin.
- Each Bridge row is a deep copy. Prepared-value/provenance fingerprints detect changed or missing rows, including changed grouping. Export rechecks the current sheet. **Review / refresh** explicitly replaces the job with reviewed selections; nothing refreshes silently.
- Adding duplicate or overlapping origins fails atomically. Selecting another source section is disabled while adding to an existing job; refresh explicitly replaces it. Equal dimensions with different source identities are legitimate.
- Asynchronous order fetches and isolated Workspace document generation block imports/exports. Synchronous Processing transformations cannot interleave with capture.
- Drafts live only in this page session, matching the current Processing/Labels session convention. Reloading or closing the page clears the draft. No storage keys, backend changes, migrations, production statuses or AutoHotkey changes.
- Quantities must be integers 1–999 and dimensions integers 1–10000 mm. Original quantity syntax is also retained to prevent the existing numeric conversion/grouping from hiding malformed quantities. Every invalid selected position is reported; no row is silently skipped.
- Known nonrectangular geometry, shape markers and any nonempty source/order notes or special requirements block export. This is intentionally conservative, even for harmless notes. Checks depend on metadata available in Processing; no original drawing inspection is performed. The operator also confirms rectangular pieces without special requirements and the intended material/pass.
- Declared/source order areas are displayed separately, without rescaling for selections. The cutting area uses only validated prepared quantities and dimensions.
- CSV is exactly three columns: `quantity,width,height`, comma-delimited, CRLF, UTF-8 without BOM, with a final CRLF. The preview and export use the same validated rows. Browser saving does not guarantee overwrite.

## Files

| File | Change |
| --- | --- |
| `docs/index.html` | Production navigation, Bridge panel, accessible native selection dialog and instructions |
| `docs/css/styles.css` | Compact responsive Bridge layout using existing theme variables |
| `docs/js/app.js` | Tab registration and scoped asynchronous Processing busy guard |
| `docs/js/platform-workflows.js` | Source identity, version and safety metadata carried through existing origin payloads |
| `docs/js/perfect-cut-bridge.js` | Adapter, validation, immutable import, duplicate/change detection, selection/review UI and CSV download |
| `tests/fixtures/perfect_cut_order.json` | Requested R-26-0826 dimensions/quantities and source total; synthetic per-row source areas for regression testing |
| `tests/test_perfect_cut_bridge.cjs` | 14 data-contract, isolation, canonical transformation, safety and async-guard tests |
| `tests/test_perfect_cut_bridge_browser.cjs` | Isolated Chromium/WebKit browser acceptance checks; no production service access |
| `docs/PERFECT_CUT_BRIDGE.md` | Implementation and validation notes |

## Verification

Passed locally:

- `node --test tests/test_perfect_cut_bridge.cjs` — 14 tests.
- `python -m pytest -q tests/test_frontend_navigation.py tests/test_frontend_security.py tests/test_frontend_theme.py tests/test_pdf_editor_static.py tests/test_order_detail_review_ui.py tests/test_work_queue_activity_ui.py` — 49 tests.
- `python -m pytest -q tests/test_manual_orders.py tests/test_mcp_platform.py -k 'processing or label or danko'` — 16 tests, including real Processing/Labels document generation. Existing dependency deprecation warnings only.
- `node tests/test_perfect_cut_bridge_browser.cjs` with an external Playwright installation on `NODE_PATH` — Chromium and WebKit passed: empty state, subsets/order selection, rapid duplicate clicks, byte-exact download, stale picker, explicit refresh, busy guards, removal/confirmed clear isolation, unsupported shapes, multiple orders, section locking and 390px mobile layout. The script starts and stops its own static fixture server and mocks all external requests.
- Native Safari: local navigation, empty picker, disabled export, and dark appearance visually checked. Populated flows were tested in Chromium and WebKit.
- JavaScript syntax checks and `git diff --check` passed.

The fixture exports **8 rows, 22 pieces, 10.0013 m² calculated cutting area**, preserving **10.080 m² source area** separately. Double glazing does not double quantities.

No Windows XP, AutoHotkey, Perfect Cut or cutting-machine integration was tested. No deployment was performed.
