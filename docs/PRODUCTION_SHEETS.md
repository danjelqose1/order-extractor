# Production sheets in Processing

After adding orders to Processing, choose **Prepare production sheet**. The preview contains two complete production copies. Print with the printer's Copies set to **1**, or Save PDF.

Automatic first tries two independent halves on landscape A4 at the selected readable text size. If that does not fit, it measures portrait and landscape layouts with one, two or three columns **inside each copy**. It prefers fewer pages, then fewer columns, then portrait when the other choices are equal. Columns are balanced without adding pages. The second set repeats the complete first set, with identical numbering. Manual controls can override the decision; a forced layout that cannot fit is rejected.

The browser uses the prepared `appState.processing.preview` text, glass/order headings, rows and provenance. The new renderer changes presentation only. It does not round, group, merge, recalculate areas, save orders, change statuses or change existing Workspace/Manual Orders exports. A source change invalidates the preview and any AI proposal. Refresh uses the current source and resets formatting. Formatting drafts survive closing/reopening the dialog for the same source within the current tab; they are not stored in the database.

## AI review

**Ask AI to choose or edit** works with a blank request or a specific formatting/note request. The browser sends JPEG images from the actual PDF: the first, middle and last pages of the first copy, with duplicate page numbers removed. The API receives the full prepared text and the sampled page numbers. The UI states how many pages were reviewed; this is not an assertion that every page was visually reviewed.

The server uses its existing `OPENAI_API_KEY`, `get_client()` and Responses API. Default model: `gpt-6.1-sol`, medium reasoning. `PRODUCTION_SHEET_MODEL` can override the model independently of extraction. No additional Render environment variable is required. Requests use a 65-second timeout, no automatic retry, `store=False` and a maximum of 6,000 output tokens.

Structured output is limited to formatting settings, a requested production note, an explanation and warnings. There are no row-edit fields or order-writing tools. The renderer validates the proposal's real fit before the UI offers Apply. Printing is disabled while a proposal is pending; Discard restores the current sheet. AI failure leaves a previously prepared sheet printable.

## Implementation and rollout

- `backend/production_sheets.py`: bounded request models, font measurement, wrapping, continuation context, balanced columns, PDF generation and visual AI proposals.
- `backend/assets/fonts/`: embedded DejaVu Sans regular/bold with their original license notices. Unsupported characters are rejected rather than silently substituted.
- `docs/js/production-sheet.js`: source snapshot, manual controls, PDF.js preview, visual AI requests, proposal review, PDF download and a browser print window.
- `POST /api/production-sheets/preview`: returns a PDF and layout/count metadata without calling AI.
- `POST /api/production-sheets/ai`: returns a validated proposal and its rendered PDF. Both routes respect the existing optional `APP_KEY` guard.

Both the frontend and Render backend must receive this change before factory use. The browser remains independent of Word and ChatGPT installation. Automatic formatting does not need AI; PDF preparation still needs the platform backend. Preview, AI images and browser printing use glyph outlines from the generated PDF to avoid browser FontFace caching between different PDF subsets. The saved PDF retains vector text; the browser print window contains page images at 180 dpi.

## Validation

Run the Python suite with the existing backend test environment:

```sh
python -m pytest tests/test_production_sheets.py tests/test_smoke.py tests/test_invoice_ai.py tests/test_manual_orders.py tests/test_frontend_theme.py tests/test_frontend_security.py -q
node --test tests/test_production_sheet_source.cjs tests/test_perfect_cut_bridge.cjs tests/test_manual_dimension_groups.cjs
```

The browser suite uses an isolated local backend, the real PDF renderer and fixture AI responses. It never contacts deployed services or production data. Set `NODE_PATH` to the available Playwright/PDF.js packages and `PRODUCTION_TEST_PYTHON` to a Python environment with backend dependencies, then run `node tests/test_production_sheet_browser.cjs`. `PRODUCTION_PDFJS_DIR` may point to a directory containing `build/pdf.mjs` and `build/pdf.worker.mjs`; production currently uses PDF.js 4.10.38. QA artifacts default to `/tmp/production-sheet-browser-qa`.

Browser coverage includes source preservation, short and long jobs, Print, Save PDF, zoom, AI page-image submission, Apply/Discard, unavailable AI, stale source, reset, and responsive light/dark themes. Live OpenAI responses, deployed integration and the physical factory printer require verification after rollout.
