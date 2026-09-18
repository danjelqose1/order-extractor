# Perfect Cut Bridge — frontend V1

Under Production → Perfect Cut Bridge, click **Add from Processing**, then download `job.csv`. The button copies the entire current prepared sheet across glass types, in its displayed order. Each click explicitly replaces the Bridge snapshot, so repeated clicks cannot duplicate rows. **Review / refresh** offers optional row/order selection for a partial export. No material/pass input or confirmation is required.

## Add from Manual Orders

**Add from Manual Orders** opens a searchable, paginated list of saved orders. Choose one or more approved/processing orders, then review all rows or deselect rows for a partial export. As with Processing, committing the import replaces the current Bridge draft. Draft, finished and cancelled orders are ineligible, matching the existing Manual Orders production action.

The dedicated adapter copies saved `width_mm`, `height_mm` and `quantity` exactly, in each order's saved row order. It does not round, regroup, swap dimensions, filter glass types, expand quantities, or mutate the Processing cart. Repeated imports cannot duplicate rows. Source positions, IDs, status, update timestamp and declared area are retained separately.

Only the existing GET list/detail endpoints are used. Details are re-fetched before committing the reviewed selection and again before download. Changes, deletion, revoked approval or a fetch failure block export without silently updating the snapshot. **Review / refresh Manual Orders** explicitly reviews fresh values. **Open Manual Orders** opens the source module for corrections. No order statuses or stored orders are changed.

Manual-specific validation uses the same numeric/geometry/notes checks as Processing. Only saved values are imported; unsaved edits in the Manual Orders form are not included.

## Behavior and boundaries

- Reads `appState.processing.preview.groups[].lines`, the canonical Mother Sheet output. No extra rounding, dimension swapping, grouping, quantity expansion, PDF parsing or label expansion occurs.
- `convertOrderToProcessingEntry` and `buildOriginRowPayload` retain provenance only. Existing Processing/Labels transformations are unchanged. Identity includes source kind, order ID and original row ID (source index fallback); grouped lines retain every origin.
- Each Bridge row is a deep copy. Prepared-value/provenance fingerprints detect changed or missing rows, including changed grouping. Export rechecks the current sheet. **Add from Processing** explicitly replaces the job with the whole prepared sheet; **Review / refresh** replaces it with reviewed selections. Nothing refreshes silently.
- Import traverses all preview groups and their lines in existing Processing order. It does not filter by glass type, globally sort dimensions or merge additional rows. Duplicate origins within a candidate fail atomically; equal dimensions with different source identities remain legitimate.
- Asynchronous order fetches and isolated Workspace document generation block imports/exports. Synchronous Processing transformations cannot interleave with capture.
- Drafts live only in this page session, matching the current Processing/Labels session convention. Reloading or closing the page clears the draft. No storage keys, backend changes, migrations, production statuses or AutoHotkey changes.
- Quantities must be integers 1–999 and dimensions integers 1–10000 mm. Original quantity syntax is also retained to prevent the existing numeric conversion/grouping from hiding malformed quantities. Every invalid selected position is reported; no row is silently skipped.
- Known nonrectangular geometry, shape markers and any nonempty source/order notes or special requirements block export. This is intentionally conservative, even for harmless notes. Checks depend on metadata available in Processing; no original drawing inspection is performed. Material selection belongs in Perfect Cut; glass descriptions are retained as internal provenance only.
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
| `tests/fixtures/perfect_cut_processing_order.json` | R-26-0830 screenshot regression: five ungrouped rows become four grouped rows across two glass types, still five pieces |
| `tests/fixtures/perfect_cut_order.json` | Requested R-26-0826 dimensions/quantities and source total; synthetic per-row source areas for regression testing |
| `tests/test_perfect_cut_bridge.cjs` | 18 data-contract, isolation, canonical transformation, safety and async-guard tests |
| `tests/test_perfect_cut_bridge_browser.cjs` | Isolated Chromium/WebKit browser acceptance checks; no production service access |
| `docs/PERFECT_CUT_BRIDGE.md` | Implementation and validation notes |

## Verification

Passed locally:

- `node --test tests/test_perfect_cut_bridge.cjs` — 18 tests.
- `python -m pytest -q tests/test_frontend_navigation.py tests/test_frontend_security.py tests/test_frontend_theme.py tests/test_pdf_editor_static.py tests/test_order_detail_review_ui.py tests/test_work_queue_activity_ui.py` — 49 tests.
- `python -m pytest -q tests/test_manual_orders.py tests/test_mcp_platform.py -k 'processing or label or danko'` — 16 tests, including real Processing/Labels document generation. Existing dependency deprecation warnings only.
- `node tests/test_perfect_cut_bridge_browser.cjs` with an external Playwright installation on `NODE_PATH` — Chromium and WebKit passed: empty state, subsets/order selection, rapid duplicate clicks, byte-exact download, stale picker, explicit refresh, busy guards, removal/confirmed clear isolation, unsupported shapes, multiple orders, whole-sheet imports across glass types, the screenshot order before/after grouping, and 390px mobile layout. The script starts and stops its own static fixture server and mocks all external requests.
- Native Safari: local navigation, empty picker, disabled export, and dark appearance visually checked. Populated flows were tested in Chromium and WebKit.
- JavaScript syntax checks and `git diff --check` passed.

The fixture exports **8 rows, 22 pieces, 10.0013 m² calculated cutting area**, preserving **10.080 m² source area** separately. Double glazing does not double quantities.

No Windows XP, AutoHotkey, Perfect Cut or cutting-machine integration was tested. No deployment was performed.

Latest simplification checks: 15 Bridge tests and 40 frontend navigation/theme/static regression tests passed. The R-26-0830 grouped CSV is:

```csv
quantity,width,height
1,738,1835
2,815,1903
1,1268,168
1,433,848
```

Manual Orders extension verification: 18 Bridge contract tests and 68 frontend/Manual Orders regression tests passed. Chromium and WebKit browser checks passed for search, empty/error states, pagination with retained selections, disabled draft selection, multi-order import, partial row selection, exact CSV bytes, unchanged Processing/Labels, changed-source and approval rechecks, switching back to Processing, and mobile layout. Fixture: `tests/fixtures/perfect_cut_manual_order.json`. No live manual records were modified.
