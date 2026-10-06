# Factory Agent · Beta

**Retired from the platform on 2026-10-06 at the owner's request.** Factory Agent,
Beta, and the Production dashboard have been removed from navigation and the
page. Processing, Perfect Cut Bridge, Labels, saved orders, and saved documents
remain available. Factory Agent HTTP routes are no longer mounted, regardless
of `ENABLE_FACTORY_AGENT`. Its startup hook only cancels/deletes unfinished
hosted resources found in the existing journal; it never creates sessions or
sends task input. Existing journal and factory data are retained.

The implementation notes below describe the archived feature, not a current
enablement path. Setting the old flag will not restore it. The server's existing
OpenAI key is still used by other platform features and must be retained.

This is a separate section in the existing platform. FastAPI calls the Agents
API directly, and OpenAI runs the browser. It has no ChatGPT shortcut, Dot, manual
handoff, local browser, or Render-hosted VM. Existing Beta, extraction, pricing,
production, MCP, database, and document workflows retain their behavior.
The existing extraction client now checks for a missing key when first used,
rather than aborting the entire app at import time; extraction still fails
without credentials. This permits health and setup-required pages to work.

The section has two explicitly selected workspaces:

* **Factory workspace · inspect and prepare** answers general questions, searches
  and reads saved factory orders, summarizes saved data, inspects existing
  processing snapshots, and prepares manual-draft proposals for review.
* **FACTORY-AGENT-TEST-001 · isolated fixture** runs the original synthetic order
  test, including the hosted browser. Its application function cannot access
  production records; its browser reaches only the allowed public static host.

Both workspaces enforce read-only factory access. A proposal is saved only in
the separate agent journal. **Accept plan records review; it does not apply an
order change.** Edits, order approvals, processing, invoices, printing, and
machinery remain in existing operator workflows. The hosted browser has no
production access: saved factory data reaches the agent only through the
backend's restricted inspection tools.

## Enable and configure

`ENABLE_FACTORY_AGENT` is disabled by default. Enable it on an authorized
deployment after reviewing the setup below. The adapter uses the existing
`httpx` and SQLAlchemy dependencies and the existing durable database location;
no dependency upgrade, new service, database migration, or local browser is
required for the Beta.

Server configuration (never frontend configuration or remote sandbox variables):

| Variable | Required / default |
| --- | --- |
| `ENABLE_FACTORY_AGENT` | `false`; set exactly `true` to reveal/enable the section |
| `APP_KEY` | Reused when already configured; always takes precedence for Beta access. Preserve its existing value because it also affects legacy routes |
| `FACTORY_AGENT_ACCESS_KEY` | Required only when `APP_KEY` is absent; separate random secret of 32–4096 printable non-space characters, never equal to `OPENAI_API_KEY`. Protects only the Beta |
| `OPENAI_API_KEY` | Required existing server key with Agents API project access |
| `OPENAI_PROJECT_ID` | Optional existing project ID; sent as `OpenAI-Project` |
| `FACTORY_AGENT_MODEL` | `gpt-6.1-sol`; independent of extraction. Listed in the Agents dashboard and chosen for lower cost |
| `FACTORY_AGENT_RUNTIME_SECONDS` | 180; clamped to 30–600 seconds |
| `FACTORY_AGENT_MAX_CONCURRENCY` | 1; clamped to 1–2 |
| `FACTORY_AGENT_STATE_DIR` | Optional; defaults to `DB_DIR/factory-agent` |
| `FRONTEND_ORIGINS` | Existing exact frontend origins; preserve current values |

The API key needs **`api.agents.read`, `api.agents.write`, and
`api.responses.write`**. Project/model access and adequate API/container quota
must also be available. The Platform dashboard being visible does not establish
these permissions for the server's key. Never paste API keys into the chat UI.

The Beta defaults to GPT-6.1 Sol. At the standard short-context rates verified
on 2026-10-06, Sol costs $2 input / $10 output per million tokens, compared with
Astra's $10 / $50: 80% lower token rates. Hosted environment charges are separate.
[Official pricing](https://developers.openai.com/api/docs/pricing) and
[Sol capabilities](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
Changing the model affects new sessions only; it does not restart old tasks.

Reuse the existing backend environment and credentials. If credentials are only
configured on Render, leave them there; local checks show setup required until
an authorized local server key is supplied through a secure environment. Do not
download or embed Render secrets to make the local test pass.

If the deployed app has no `APP_KEY`, add `FACTORY_AGENT_ACCESS_KEY` in Render's
Environment settings; leave `APP_KEY` unchanged. Enabling a new global `APP_KEY`
would also enforce authentication on older factory routes whose callers may not
send it. The dedicated Beta key uses the same existing `X-App-Key` mechanism
without changing those routes. Generate and store this separate operator secret
through your password manager or trusted secret-management workflow. Reuse the
existing Render `OPENAI_API_KEY` for provider requests; do not reveal or copy it.

For an existing local backend environment:

```sh
python scripts/check_factory_agent.py
# After configuring credentials/flag, an optional billed hosted smoke test:
python scripts/check_factory_agent.py --live
```

For the normal UI, start the backend and existing `docs/` frontend as usual.
Enable the backend flag, reload the frontend, select **Factory Agent · Beta**,
and unlock using the existing `APP_KEY`, or `FACTORY_AGENT_ACCESS_KEY` when no
global key is configured. Select **Factory workspace · inspect and prepare**
for a factory question or general conversation, or select the labelled fixture
for the original browser test. Keys are held only in the tab's memory; Lock
clears them. The application access key is not the OpenAI API key.

After a session finishes and its remote workspace has been deleted, optionally
select **Continue conversation from the selected session** before sending a
follow-up. Continuation is explicit and uses a new bounded hosted session; it
carries at most six previous user/assistant messages, with each retained
assistant answer capped at 12,000 characters. Only retained sessions owned by
the same access key and using the same workspace can supply context. Unchecked
means an independent task. Historical context is labelled as historical; the
agent must read current tools again before asserting current factory facts.
Reconnecting to observe a task never starts a continuation or submits input.

The backend has no individual web-user identity today. This release reuses its
existing shared-key mechanism, makes it mandatory for every Beta control/data
endpoint, and scopes records to a server-derived key fingerprint. Operators who
share the access key share its sessions. It does not claim per-person
isolation. The existing MCP OAuth integration remains unchanged. Rotating
the active access key removes access to old-key records; the backend still recovers and cleans
their remote resources on startup. A future user-auth rollout can replace this
principal boundary without putting credentials in the sandbox.

## Automatic setup and duplicate prevention

No dashboard-created agent, reusable environment template, vault, or manual
agent ID is needed. `POST /v1/agents/sessions` supplies inline agent settings and
an OpenAI-hosted environment. OpenAI provisions it through the supported API.
The workspace uses the existing `FACTORY_AGENT_MODEL` choice (Sol by default)
and a minimal hosted desktop with outbound networking disabled. It receives
the full `backend/factory_agent_assets/WORKSPACE_SKILL.md` as agent instructions.
No new agent ID, environment ID, template, vault, or environment variable is
required for the broader inspect/prepare scope.

Fixture sessions receive only the reviewed synthetic `order.json` and `SKILL.md`
as base64 inline files, with the complete workflow included directly in agent
instructions. They do not receive or run a fixture server. Provisioning sends
**no `setup_commands`**. Once the environment becomes `connected`, the agent
visits the application-owned static fixture:
[Factory Agent test order](https://danjelqose1.github.io/order-extractor/factory-agent-fixture/).
Its source is `docs/factory-agent-fixture/index.html`, published with the
existing GitHub Pages frontend. Deploy that page with the frontend before
expecting the browser test to pass. No local listener, setup command, extra
service, or login is required.

The separate control journal reserves the local request ID before creating a
remote resource. Repeating an identical request returns the same record; reusing
the ID with different inputs is rejected. Session creation itself is never
blindly retried: the documented API does not specify create idempotency. An
unknown create outcome is reconciled by the unique local ID in session metadata.
If reconciliation cannot prove a single resource, the slot remains blocked and
the UI reports cleanup required. No second remote session is created.

The remote ID is persisted before submitting input. Input has a stable
`Idempotency-Key`; this implementation never automatically resubmits it after
unknown acknowledgement or restart. It fetches saved turns and items from that
session. An unknown input may therefore time out rather than risk duplicate work.
Reconnect, reload, and GET requests never submit a task.

## Read-only boundary and browser approvals

These controls are enforced independently of the prompt:

* The real factory workspace retains `network: {"access":"disabled"}` for
  browser **and** code. Only authenticated backend inspection tools can return
  saved factory data. All browser-origin requests are denied in this mode.
* Fixture mode uses `network: {"access":"restricted",
  "allowed_domains":["danjelqose1.github.io"]}`. The Render production API and
  every other hostname are excluded for all HTTP methods. The browser reads
  the public static synthetic fixture with GET; GitHub Pages has no application
  routes that can edit factory records. The fixture page contains no customer
  data, JavaScript, forms, links, external resources, or backend integration.
* The fixture network policy is **host-wide, not path-scoped**. Other public
  pages on `danjelqose1.github.io` may be reachable. The agent instructions limit
  navigation to the fixture path, but that path limit is not claimed as network
  enforcement. The existing frontend cannot contact its Render API from this
  environment because Render remains outside the allowlist. No production
  read-only guarantee depends on merely hiding frontend edit buttons.
* Neither environment receives a platform login, keys, cookies, database file,
  write-capable MCP server, or arbitrary uploaded files. Workspace tool results
  can contain saved production order data, limited by the facade below.
* In fixture mode, the only application function is `get_selected_order({})`. The responder is
  hardcoded to the server-selected fixture. Unknown functions, extra arguments,
  and production IDs are rejected. Its responder never imports the production database or
  mutation services. Existing MCP write-capable tools are intentionally excluded.
* In workspace mode, only the explicit inspection/proposal facade below is
  available. No general service dispatch or factory mutation function is exposed.
* The backend rejects client overrides of model, tool, environment, origin,
  credentials, or arbitrary workspace IDs. It limits request size, runtime, starts, function
  calls, screenshots, output, history pagination, and concurrent sessions.

On a current `computer_use_approval_request`, the backend approves only the exact
origin `https://danjelqose1.github.io` **in fixture mode**, which the user selects
by starting the isolated test. This grants an origin, not an individual URL
path. Workspace mode denies all browser origins, including the production
platform. Both modes always respond to
`browser_authentication` with `action:"cancel"`; no credentials or login form
are collected. These policy decisions appear in session activity. Unknown
approval types cause task cancellation/cleanup instead of an inferred grant.
Only current `required_actions` are actionable; historical calls are not replayed.

The first test's fixed origin policy is intentional. Origin approval does not
guarantee approval before each subsequent click or mutation. Production browser
access would require a separate read-only authenticated surface, verified
server-side permissions, and a reviewed consent flow. Do not add a production
API origin to this fixture policy or forward the shared application key into a VM.

Workflow and source contents can contain hostile instructions; the delivered
skill explicitly treats them as untrusted data. Prompt restrictions supplement
the boundary above; they are not its enforcement mechanism.

## Inspection tools and proposal review

`backend/factory_agent_tools.py` exposes exactly these functions:

| Tool | Access |
| --- | --- |
| `list_orders` | Filter and paginate saved manual/extracted order summaries; source-qualified IDs such as `manual:42` and `pdf:42` |
| `get_order` | Saved headers, rows, quantities, dimensions, positions, notes, and canonical source version |
| `get_platform_summary` | Saved aggregate order counts, pieces, and area |
| `get_processing_job` | An existing job's saved source/result snapshot; no recalculation or generation |
| `prepare_change` | Validate and return a complete manual Draft replacement proposal; no factory write |

The facade reuses reviewed `PlatformService` read methods and existing database
serializers. It does not call `PlatformService.invoke`, because that method
persists audit records even for reads. Each request opens the existing SQLite
file with `mode=ro`, enables `query_only`, installs a write-denying SQL authorizer,
and starts an explicit read transaction. The facade exposes no legacy writer,
engine, or write-session factory. Tests inject erroneous SQL writes into a read
method and verify that they fail and the database bytes remain unchanged.
There is no database initialization, migration, new production table, or MCP
audit write. Independent read connections do not alter the normal platform's
connections or prevent ordinary operators from continuing their workflows.

Reads are capped at 25 orders per page, offset 10,000, 300 rows per order,
96,000 argument bytes and 256,000 result bytes. SQLite work has a three-second
progress deadline and a one-second lock timeout. Too-large or invalid results
fail as a whole rather than silently returning a partial order. Artifact
downloads, artifact filesystem paths, original PDF bytes, raw extraction text and raw
input blobs are excluded. Known configured server credentials are blocked from
tool results. Saved rows are not evidence that an original PDF was reviewed.

`prepare_change` requires a `manual:` order ID, its exact current version,
a complete `ManualDraft` replacement, and a rationale. It independently reads
the source, verifies Draft status/version, and reuses existing semantic
validation, including required/unique red indices and duplicate order-number
checks. It rejects an unchanged replacement. A successful proposal includes
the current editable snapshot, proposed replacement, field-level before/after
changes, source version, title, and rationale. Its initial state is `pending`.

The control plane stores a proposal before acknowledging the tool call, assigns
its ID from the session and call identity, and permits at most five per task.
A lost acknowledgement cannot create a second proposal. After task completion
and confirmed remote cleanup, **Accept plan** rereads the order and rejects a
stale source version; **Dismiss** records rejection. Decisions are owned by the
authenticated session and are immutable except for idempotent repeats.
Neither decision invokes an order writer: `applied` stays `false`. Applying any
chosen edit remains a separate action in the existing Manual Orders UI, with
another check of current values. Proposals for other consequential workflows
are explanatory plans only; no executable operation is implied.

## Activity, Stop, reconnect, and cleanup

The backend polls persisted session state, turns, and items, including complete
bounded pagination. Browser activity comes from actual `computer_use_call`
titles/statuses; screenshots come only from available `computer_screenshot`
data URLs. No screenshot is fabricated when OpenAI provides none. Saved assistant
messages supply the result. Intermediate stream deltas may not appear because
this release uses saved-state polling rather than a replayable event stream.

The root turn's `completed`, `failed`, or `cancelled` status establishes outcome;
an idle session, a closed connection, an accepted message, or a completed browser
operation does not. A completed turn with no result explicitly warns that order
reading was not verified. A report must also distinguish browser failure from
a successful function read.

**Stop sends `agent.session.input.cancel` to the remote session.** It is not an
AbortController pretending to stop work. The UI remains stopping until a remote
terminal outcome or confirmed deletion. Cancellation failure remains visible.
The wall-clock watchdog also cancels/deletes remotely when runtime expires,
including while polling is slow. API failure can prevent confirmation; cleanup
required retains the concurrency slot instead of falsely promising termination.

Results/screenshots/proposals are copied into the private local journal before automatic
session deletion. OpenAI confirms session deletion, then performs hosted
environment cleanup asynchronously. Delete conflicts receive up to three bounded
attempts. Retry Stop/Close workspace for unresolved cleanup; use the remote ID in
the Agents dashboard if permissions or provider availability prevent recovery.
Never remove the journal to bypass an unresolved remote task.

Failed setup/session diagnostics are saved before deletion: the session's
documented `error` text plus one best-effort reconnect to its failure-event
stream (5 seconds, 64 KiB, 16 events maximum). Only failure fields are retained;
server keys are redacted and display text is bounded. A diagnostic fetch failure
does not prevent cleanup or cause another task submission.

Closing a browser tab does not cancel work. Server restart reloads the journal,
recovers original remote IDs/turns, honors the original deadline, and never
submits an input twice. Shutdown requests cancellation and attempts cleanup;
hard process termination cannot guarantee that cleanup has already completed.
Clean up active sessions before disabling the feature or rotating/removing the
OpenAI key.

## Storage and deployment isolation

`DB_DIR/factory-agent/sessions.sqlite3` is a separate journal, not a migration of
`orders.db`. Screenshots, results, conversation context, and proposals are
private, not logs. Responses use
`Cache-Control:no-store`; API errors omit upstream bodies. Local result data is
scrubbed (including context and proposals) after 24 hours, when more than 20
cleaned results are retained, or when
the operator closes a cleaned workspace. Session lists return compact summaries;
only selected-session detail includes a screenshot. Small
request-ID tombstones remain to prevent duplicates. The journal caps at 1,000
records; archive it only after all remote sessions are confirmed deleted. There
is a 12-start/hour limit and at most two configured concurrent tasks.

Run one FastAPI worker per journal (the repository's documented Render command
already uses `--workers 1`). An OS lock fails closed for another worker. Retain
the existing durable `DB_DIR`, Render start/build commands and dependency pins.
Do not run a browser/VM, install Playwright, create another service, or change
production storage for this feature. No new port is opened on Render.

## First test and acceptance evidence

Selected fixture: `FACTORY-AGENT-TEST-001`; client:
`TEST CLIENT — Factory Agent fixture`. Expected source report:

| Index | Position | Glass type | Width × height (mm) | Quantity | Ambiguity |
| --- | --- | --- | --- | --- | --- |
| `0007` | A-01 | 4 CLEAR + 16 ARGON + 4 LOW-E | 1200 × 850 | 2 | Position repeats in another row |
| `0008` | A-01 | 6 TEMPERED CLEAR | 800,5 × 2100 | 1 | Preserve comma decimal and separate index |
| `0009` | B-02 | 44.2 LAMINATED CLEAR | 975/995 × missing | 1 | Two width alternatives; missing height |
| `0010` | missing | missing | 650 × 450 | 2? | Missing position/glass and uncertain quantity |

This table is the **deterministic fixture expectation**. Actual hosted model
reports have matched all four rows and ambiguities using the fixture function.
The latest public-page session also completed its page-read and screenshot
actions and reported agreement between browser and function. The conservative
browser/report assertions passed and remote deletion was confirmed. However,
no screenshot image was available in the saved application result; live image
delivery/display remains unverified. The UI does not invent an image.
No customer PDF is associated with this fixture.

Tests live in `tests/test_factory_agent_*.py` and
`tests/test_factory_agent_browser.cjs`. They cover the provider wire contract,
read-only server/tool boundary, auth, owner isolation, origins, input limits,
durable lifecycle/recovery, real cancellation requests, cleanup, feature-off,
setup-required, UI injection defense and responsive Chromium/WebKit behavior.
Mocks are isolated test infrastructure; the application has no simulated-agent
or fake-success mode. Local tests do not prove hosted account access.

Workspace extension tests additionally cover explicit conversation ownership,
scope, bounded context and idempotency; proposal persistence before tool
acknowledgement; stale-source rejection; review without application; and denial
of production browser origins. `tests/test_factory_agent_tools.py` has **27
passing isolated SQLite tests**, including canonical manual/PDF source versions,
proposal validation, complete unchanged database snapshots, malformed requests,
credential/result bounds and attempted SQL writes through a buggy read method.

### Local validation — 2026-10-06

* **324 Python tests passed** after the Sol/diagnostics follow-up, across the Factory Agent tests, existing app smoke,
  frontend navigation/theme/security, dashboard, production sheet learning/rendering, and production voice regression
  tests. Existing FastAPI/ReportLab deprecation warnings remain.
* Chromium and WebKit passed feature-off, missing-key setup, authentication,
  dropped-submission recovery, cancellation failure/success, cleanup, reload,
  XSS/screenshot URL checks, and light/dark layout checks at 1440, 1280, 1024 and
  390 pixels. These browser tests use isolated response fixtures.
* Existing FastAPI routing was exercised with its normal authorization path,
  the actual HTTP provider adapter, a mocked OpenAI transport, and the existing
  fake factory DB. The selected fixture's entire report survived the round trip,
  repeated submission recovered the same request, remote cleanup was confirmed,
  and no factory mutation function was called.
* A separate process exercised the full real application startup/shutdown with
  a temporary SQLite database and no credentials: health worked, Factory Agent
  reported setup required, and extraction still refused to create an API client.
* JavaScript syntax and Git whitespace checks passed. A clean
  [setup-required UI screenshot](../output/factory-agent/setup-required.png) was
  visually inspected; it contains no simulated task success.
* Local preflight found both application access keys and `OPENAI_API_KEY` absent,
  and the flag disabled.
  **No live hosted session was created during that local checkpoint**. Account/model access, actual browser
  reachability, model report quality and physical environment cleanup were
  unverified at that checkpoint. No production records were read or changed, and
  no push or deployment was performed at this local validation checkpoint.

Reproduce the Python check with the repository's installed requirements:

```sh
python -m pytest -q tests/test_factory_agent_*.py tests/test_living_dashboard.py tests/test_frontend_theme.py tests/test_frontend_navigation.py tests/test_frontend_security.py tests/test_smoke.py tests/test_production_sheet_voice.py tests/test_production_sheet_learning.py tests/test_production_sheets.py
node tests/test_factory_agent_browser.cjs
```

The browser test needs a local development Playwright installation and its
Chromium/WebKit browsers (or the Codex bundled runtime). This is test tooling;
do not add it to the Render runtime. This change leaves all production dependency
pins and deployment commands unchanged.

## First deployed account check — 2026-10-06

The inspect/prepare follow-up before the public fixture revision passed **359 Python tests**, including read-only
SQLite enforcement, proposals, conversation ownership, stale-source review,
the previous fixture startup and existing platform regressions. Chromium and WebKit passed
the broader workspace, review, continuation, XSS and responsive layout checks.
Saved `command_execution` status and bounded redacted output are now visible in
activity. A finished fixture turn warns if completed browser/read-tool evidence
is missing; a completed model turn alone is not fixture acceptance.

The existing Render key successfully created an actual hosted Agents API session
with Astra. The environment stayed pending, then the session failed before any
input or model turn. Remote deletion was confirmed; no report/browser success
was claimed. The dashboard could not load its failure details after cleanup.
That finding prompted the bounded diagnostic retention above. Sol is the new
cost-conscious default and is listed in the account's Agents model selector;
an actual successful browser run is still required to establish acceptance.

Subsequent bounded account probes using the existing Render API key established
that a minimal hosted desktop with disabled networking reaches `connected`, and
that the same environment with the fixture files but **no setup commands** also
reaches `connected`. Both probe sessions were deleted and deletion was confirmed.
The earlier fixture setup commands failed during provisioning. Moving startup
into the first agent turn was an intermediate diagnostic revision, now replaced
by the static public fixture above.

The next actual hosted session, local journal ID
`f37b2734-dffb-4216-a34e-709738765421`, returned the correct client, all four rows,
every requested source field, and the fixture ambiguities through the read-only
function. Its shell health check also succeeded. However, hosted-browser URL
policy blocked the loopback URL even after origin approval; multiple computer
actions failed and no browser screenshot was available. **This is successful
function-based order reporting, not browser acceptance.**

That observed browser restriction prompted the static GitHub Pages fixture and
single-host restricted network policy. All **245 focused Factory Agent tests**
passed after this change, including denial of the production browser origin and
the regression where a successful browser connection masked a failed page visit.
No browser success is inferred from an environment reaching `connected`.

The actual factory-workspace session with local journal ID
`a9e0aceb-73c3-46d7-8d17-719aa5539d72` subsequently **completed**, with remote
cleanup `deleted` and no session error. Its `get_platform_summary`,
`list_orders(limit=1)`, and `get_order` calls completed. The report covered the
selected manual order's 15 rows, 27 pieces, and saved area of 18.189 m²; it
distinguished saved millimetre dimensions from centimetre display units and
flagged missing fields. It explicitly said the original PDF had not been
inspected. No client identity is reproduced in this public documentation.

That result verifies live read-only factory tools and a resulting model report.
It does not establish a browser visit or proposal review: no proposals were
prepared and no factory mutations were performed in that session. Live proposal
preparation/review remains unverified against a live account; isolated tests
cover proposal preparation, review, stale-source rejection and no application.

The public fixture revision `de1451a` deployed successfully on Render. Before
starting the live check, its published HTML returned HTTP 200 and matched the
reviewed local fixture byte for byte. Session
`d576cbbe-b3e2-4d33-a75f-1cbeea0f9d91` completed and its remote session was
deleted. `get_selected_order` and the browser page-read action completed; the
model reported agreement on all four rows and flagged every expected ambiguity.
The source-value and ambiguity assertions passed. **Screenshot capture timed
out**, however, and no screenshot was returned. Because a browser action failed,
the conservative acceptance assertion remained false and the UI displayed a
verification warning. This is not full browser/screenshot acceptance.

Explicit live continuation was also verified: session
`343e4919-fb71-4889-aa35-933b0bd196a2` continued the saved workspace conversation,
called `get_order` again rather than relying on historical counts, and returned
the same 15 rows and 27 pieces with the current saved source version. It
completed without error and remote deletion was confirmed. It prepared no
proposal and performed no factory mutation.

Live Stop was verified with session
`8bb4d793-b7d9-4bd9-a18c-02ee7ef1d371`: input was accepted first, then the
application's Stop endpoint was called. OpenAI accepted cancellation, the saved
turn became `cancelled`, and remote deletion was confirmed without an error.
This exercises task cancellation, not merely disconnection from displayed events.

A final bounded fixture run, `81f82f40-7408-48ca-a950-234bacb030c5`, completed
all browser actions without a failure, including its screenshot-labelled action.
The actual report matched all source values and flagged the ambiguities.
`verify_report` returned true for source values, browser activity, the fixture
read function, ambiguities and overall acceptance. The result had no error and
remote cleanup was `deleted`. The model said a screenshot was captured and
inspected, but **the saved application result contained no screenshot image**.
Consequently this verifies the browser/read workflow, not live image delivery
to the UI. The no-screenshot state remains accurate. The supported optional
`computer_screenshot` output is rendered only when OpenAI returns a valid image
within the configured bound.

Final warning/copy refinements passed all **245 focused Python tests** and the
existing Chromium/WebKit suite. The wider **359-test regression** checkpoint
above covered the inspect/prepare implementation before these wording changes.
No new credentials, account grants or environment variables were needed for
these successful live checks. Refresh the deployed frontend and unlock with the
same application access key to load the expanded workspace UI.

## Official contract sources (verified 2026-10-06)

* [Agents API quickstart](https://developers.openai.com/api/docs/guides/agents-api/quickstart)
* [Computer use and approval handling](https://developers.openai.com/api/docs/guides/agents-api/tools/computer-use)
* [OpenAI-hosted environments and network policy](https://developers.openai.com/api/docs/guides/agents-api/environments/openai-hosted)
* [Session input, cancellation, and idempotency](https://developers.openai.com/api/docs/guides/agents-api/sessions)
* [Saved events, items, turns, and reconnect](https://developers.openai.com/api/docs/guides/agents-api/sessions/events)
* [Function result handling](https://developers.openai.com/api/docs/guides/agents-api/tools/functions)
* [Saved failure diagnostics](https://developers.openai.com/api/docs/guides/agents-api/errors)
