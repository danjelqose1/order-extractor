# Factory Agent · Beta

This is a separate section in the existing platform. FastAPI calls the Agents
API directly, and OpenAI runs the browser. It has no ChatGPT shortcut, Dot, manual
handoff, local browser, or Render-hosted VM. Existing Beta, extraction, pricing,
production, MCP, database, and document workflows retain their behavior.
The existing extraction client now checks for a missing key when first used,
rather than aborting the entire app at import time; extraction still fails
without credentials. This permits health and setup-required pages to work.

The first release deliberately supports **one isolated synthetic order only**.
The ordinary platform exposes some unauthenticated/mutating routes, so giving
the hosted browser production access cannot safely meet a read-only guarantee.
The remote environment has outbound access disabled and only a private fixture
page on its own loopback interface. No live order is selectable in this Beta.

## Enable and configure

Keep `ENABLE_FACTORY_AGENT=false` in production until this code is reviewed and
an authorized deployment is performed. There is no deployment or dependency
upgrade required as part of local implementation; the adapter uses the already
pinned `httpx` package and documented HTTP contracts.

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
global key is configured. Select the labelled fixture and
send the read-only task. Keys are held only in the tab's memory; Lock clears
them. The application access key is not the OpenAI API key.

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
Each task receives the reviewed fixture server, source JSON, and real workflow
`SKILL.md` as base64 inline files. The workflow text is also included directly
in agent instructions, so repository-only instructions are never mistaken for
instructions delivered to the remote agent.

Setup commands run **inside OpenAI's environment**: start the fixture listener
at `http://127.0.0.1:8765/order`, then verify its health. FastAPI waits for the
environment's `connected` state before submitting the task. The account smoke
test must still verify hosted loopback reachability; a failed environment stays
a visible failure, without relaxing outbound access or using production instead.

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

* The hosted environment has `network: {"access":"disabled"}` for browser **and**
  code. It receives no platform origin, keys, login cookies, production records,
  MCP write server, or arbitrary uploaded files.
* The fixture HTTP server binds loopback, loads immutable response bytes at
  startup, checks the Host, and serves only `/order`, `/order.json`, `/healthz`.
  GET/HEAD read; mutation methods return 405. Unknown/traversal/query paths fail.
  The page has no forms/scripts/external resources and escapes source content.
* The only application function is `get_selected_order({})`. The responder is
  hardcoded to the server-selected fixture. Unknown functions, extra arguments,
  and production IDs are rejected. It never imports the production database or
  mutation services. Existing MCP write-capable tools are intentionally excluded.
* The backend rejects client overrides of model, tool, environment, origin,
  credentials, or order IDs. It limits request size, runtime, starts, function
  calls, screenshots, output, history pagination, and concurrent sessions.

On a current `computer_use_approval_request`, the backend approves only the exact
origin `http://127.0.0.1:8765`, which the user selects by starting the isolated
fixture test. It denies every other origin. It always responds to
`browser_authentication` with `action:"cancel"`; no credentials or login form
are collected. These policy decisions appear in session activity. Unknown
approval types cause task cancellation/cleanup instead of an inferred grant.
Only current `required_actions` are actionable; historical calls are not replayed.

The first test's fixed origin policy is intentional. Origin approval does not
guarantee approval before each subsequent click or mutation. Production browser
access would require a separate read-only authenticated surface, verified
server-side permissions, and a reviewed consent flow. Do not add a production
origin to this fixture policy or forward the shared application key into a VM.

Workflow and source contents can contain hostile instructions; the delivered
skill explicitly treats them as untrusted data. Prompt restrictions supplement
the boundary above; they are not its enforcement mechanism.

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

Results/screenshots are copied into the private local journal before automatic
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
`orders.db`. Screenshots and results are private, not logs. Responses use
`Cache-Control:no-store`; API errors omit upstream bodies. Local result data is
scrubbed after 24 hours, when more than 20 cleaned results are retained, or when
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

This table is the **deterministic fixture expectation**, not claimed output from
a live hosted agent. A successful account smoke test must retrieve the actual
agent report, browser activity and available screenshot, compare every source
field, flag ambiguities, then confirm remote cancellation/deletion when needed.
No customer PDF is associated with this fixture.

Tests live in `tests/test_factory_agent_*.py` and
`tests/test_factory_agent_browser.cjs`. They cover the provider wire contract,
read-only server/tool boundary, auth, owner isolation, origins, input limits,
durable lifecycle/recovery, real cancellation requests, cleanup, feature-off,
setup-required, UI injection defense and responsive Chromium/WebKit behavior.
Mocks are isolated test infrastructure; the application has no simulated-agent
or fake-success mode. Local tests do not prove hosted account access.

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
  **No live hosted session was created**. Account/model access, hosted browser
  loopback reachability, actual model report quality and physical environment
  cleanup remain unverified. No production records were read or changed, and
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

The existing Render key successfully created an actual hosted Agents API session
with Astra. The environment stayed pending, then the session failed before any
input or model turn. Remote deletion was confirmed; no report/browser success
was claimed. The dashboard could not load its failure details after cleanup.
That finding prompted the bounded diagnostic retention above. Sol is the new
cost-conscious default and is listed in the account's Agents model selector;
an actual successful browser run is still required to establish acceptance.

## Official contract sources (verified 2026-10-06)

* [Agents API quickstart](https://developers.openai.com/api/docs/guides/agents-api/quickstart)
* [Computer use and approval handling](https://developers.openai.com/api/docs/guides/agents-api/tools/computer-use)
* [OpenAI-hosted environments and network policy](https://developers.openai.com/api/docs/guides/agents-api/environments/openai-hosted)
* [Session input, cancellation, and idempotency](https://developers.openai.com/api/docs/guides/agents-api/sessions)
* [Saved events, items, turns, and reconnect](https://developers.openai.com/api/docs/guides/agents-api/sessions/events)
* [Function result handling](https://developers.openai.com/api/docs/guides/agents-api/tools/functions)
* [Saved failure diagnostics](https://developers.openai.com/api/docs/guides/agents-api/errors)
