# Production sheets in Processing

After adding orders to Processing, choose **Prepare production sheet**. The preview contains two complete production copies. Print with the printer's Copies set to **1**, or Save PDF.

**Let AI do it for you** is the one-click alternative in Processing. It captures the exact prepared text, builds an initial readable PDF, sends that text and actual page images to the existing AI endpoint, applies the validated AI layout, and starts downloading the finished PDF. The preview stays open with Print, Save PDF, the chosen layout explanation and formatting controls. No Apply step is required in this explicitly automatic flow. Closing the dialog or changing Processing cancels the run and prevents downloading an outdated result. If AI fails, the basic sheet remains available to print or save, and no automatic download occurs.

Automatic first tries two independent halves on landscape A4 at the selected readable text size. If that does not fit, it measures portrait and landscape layouts with one, two or three columns **inside each copy**. It prefers fewer pages, then fewer columns, then portrait when the other choices are equal. Columns are balanced without adding pages. The second set repeats the complete first set, with identical numbering. Manual controls can override the decision; a forced layout that cannot fit is rejected.

The browser uses the prepared `appState.processing.preview` text, glass/order headings, rows and provenance. The new renderer changes presentation only. It does not round, group, merge, recalculate areas, save orders, change statuses or change existing Workspace/Manual Orders exports. A source change invalidates the preview and any AI proposal. Refresh uses the current source and resets formatting. Formatting drafts survive closing/reopening the dialog for the same source within the current tab; they are not stored in the database.

## AI review

**Ask AI to choose or edit** works with a blank request or a specific formatting/note request. The browser sends JPEG images from the actual PDF: the first, middle and last pages of the first copy, with duplicate page numbers removed. The API receives the full prepared text and the sampled page numbers. The UI states how many pages were reviewed; this is not an assertion that every page was visually reviewed.

The server uses its existing `OPENAI_API_KEY`, `get_client()` and Responses API. Default model: `gpt-6.1-sol`, medium reasoning. `PRODUCTION_SHEET_MODEL` can override the model independently of extraction. No additional Render environment variable is required. Requests allow 15 minutes for an AI response, with separate 15-second connect/pool and 30-second upload timeouts, no automatic retry, `store=False` and a maximum of 6,000 output tokens. The browser allows 17 minutes, shows elapsed progress every 15 seconds and lets the user close to cancel. AI timeouts return a specific recoverable message; logs include only error type, timeout phase and duration. The AI proposes settings promptly and leaves exact fit calculations to the renderer.

Structured output is limited to formatting settings, a requested production note, an explanation and warnings. There are no row-edit fields or order-writing tools. The renderer validates the proposal's real fit before the UI offers Apply. Printing is disabled while a proposal is pending; Discard restores the current sheet. AI failure leaves a previously prepared sheet printable.

Glass headings have separate spacing controls before and after the entire heading. The AI can add space after a wrapped glass type without changing the gap above it or the dimension rows.

## Spoken conversation

**Fol me AI · Talk to AI** starts a GPT-Live 1 WebRTC conversation after browser microphone permission. It initially greets in Albanian, then follows the user's language, including switches. The microphone/audio transport goes to OpenAI; transcripts stay in tab memory. Sessions use `store=False`; the application does not save audio or transcripts in its database. This does not override OpenAI's API data-retention policies.

The Render server creates the connection using its existing key; only the negotiated SDP and an opaque, session-scoped ownership token reach the browser. Session creation requires an allowed frontend Origin and the existing optional APP_KEY guard. There is a process-local limit of four sessions and thirty connection attempts per hour per client address. These are abuse/cost bounds, not a new identity system. No new credentials or database are required.

GPT-Live uses client delegation. Its delegation events contain an ID, not task text: the browser retains exact user/assistant transcript fragments and sends the conversation plus current sheet state to `gpt-6.1-sol` with medium reasoning. A strict action schema permits only propose, apply, discard, save_pdf or clarify. Formatting requests go through the existing PDF-image AI review. A spoken edit shows a proposal; explicit approval applies the displayed proposal. Mouse controls and spoken actions share the same source/settings checks. Changes while routing a request invalidate its result. A spoken PDF request highlights Save PDF; the user taps that button because Safari requires a click to download. Spoken requests never write order data or operate printers.

The conversation disconnects after **one minute of inactivity**. Actual microphone energy (above steady background noise), assistant playback, user interaction and pending sheet work reset the timer. Transcript gaps alone do not signal silence. There is also a fifteen-minute session cap when no work or playback is pending; Start conversation reconnects. Muting does not stop connected-minute billing. End, dialog close, source changes, page leave, transport errors and permission failures release microphone tracks and close the connection. The data channel requests graceful close; an owned server hangup provides cleanup, including late session answers after cancellation. Finalization waits up to ten seconds before releasing local resources; audio is silenced immediately. Server hangup is a fallback rather than racing the final close event. Sessions are not automatically reconnected.

Browser support, OpenAI GPT-Live project access and network/media connectivity are required. Voice failure leaves the current printable sheet and text controls available. Verify Albanian speech quality on the factory laptop and its actual microphone; synthetic transport tests do not establish recognition quality in workshop noise.

## Implementation and rollout

- `backend/production_sheets.py`: bounded request models, font measurement, wrapping, continuation context, balanced columns, PDF generation and visual AI proposals.
- `backend/assets/fonts/`: embedded DejaVu Sans regular/bold with their original license notices. Unsupported characters are rejected rather than silently substituted.
- `docs/js/production-sheet.js`: source snapshot, manual controls, PDF.js preview, visual AI requests, proposal review, PDF download and a browser print window.
- `backend/production_sheet_voice.py` and `docs/js/production-sheet-voice.js`: GPT-Live session ownership, multilingual presentation intent routing, WebRTC, transcript display and inactivity cleanup.
- `POST /api/production-sheets/voice/session`, `/voice/turn`, `/voice/close`: connection, presentation-only request routing and owned server hangup.
- `POST /api/production-sheets/preview`: returns a PDF and layout/count metadata without calling AI.
- `POST /api/production-sheets/ai`: returns a validated proposal and its rendered PDF. Both routes respect the existing optional `APP_KEY` guard.

Both the frontend and Render backend must receive this change before factory use. The browser remains independent of Word and ChatGPT installation. Automatic formatting does not need AI; PDF preparation still needs the platform backend. Preview, AI images and browser printing use glyph outlines from the generated PDF to avoid browser FontFace caching between different PDF subsets. The saved PDF retains vector text; the browser print window contains page images at 180 dpi.

## Validation

Run the Python suite with the existing backend test environment:

```sh
python -m pytest tests/test_production_sheet_voice.py tests/test_production_sheets.py tests/test_smoke.py tests/test_invoice_ai.py tests/test_manual_orders.py tests/test_frontend_theme.py tests/test_frontend_security.py -q
node --test tests/test_production_sheet_source.cjs tests/test_perfect_cut_bridge.cjs tests/test_manual_dimension_groups.cjs
```

The browser suite uses an isolated local backend, the real PDF renderer and fixture AI responses. It never contacts deployed services or production data. Set `NODE_PATH` to the available Playwright/PDF.js packages and `PRODUCTION_TEST_PYTHON` to a Python environment with backend dependencies, then run `node tests/test_production_sheet_browser.cjs`. `PRODUCTION_PDFJS_DIR` may point to a directory containing `build/pdf.mjs` and `build/pdf.worker.mjs`; production currently uses PDF.js 4.10.38. QA artifacts default to `/tmp/production-sheet-browser-qa`.

Browser coverage includes source preservation, short and long jobs, Print, Save PDF, zoom, AI page-image submission, Apply/Discard, unavailable AI, stale source, reset, and responsive light/dark themes. Live OpenAI responses, deployed integration and the physical factory printer require verification after rollout.

Voice fixtures in `tests/production_sheet_voice_browser.cjs` run in the same Chromium/WebKit suite. They verify delegation IDs, Albanian/Italian requests, spoken review/PDF preparation, duplicate suppression, playback/work-aware idle disconnection, permission denial, source changes and late-connection cancellation using the real sheet UI and PDF renderer. They do not contact the live voice API.
