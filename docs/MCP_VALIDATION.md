# MCP validation record

Validated locally on 2026-09-01 against repository base `2a08f3c`.
The assistant made no Order Extractor production API requests, deployments,
production-secret changes, or ChatGPT connections. The supplied Auth0 tenant's public OIDC
discovery was read. All workflow records and artifacts were synthetic and stored
under temporary local DB directories. Existing `.env` loading was disabled.

## Completed levels

| Level | Evidence |
|---|---|
| 0: architecture/contracts | Inspected repository and official OpenAI documentation; 18 focused tools, typed contracts, explicit annotations, separate stores and reused generators |
| 1: static/tests | Python compilation, JavaScript syntax checks, dependency consistency and tests below |
| 2: local HTTP | Started actual Uvicorn on `127.0.0.1:5057`; called `/mcp` with a local test bearer; executed both manual modes through approval, processing, rounding, grouping and downloads |
| 3: Inspector | MCP Inspector **2.4.0 CLI** initialized the real HTTP connection, discovered **18 tools**, and called `platform_health`; the documented temporary-config CLI form was executed successfully |
| OAuth | Real RSA-signed tokens exercised the FastAPI boundary; the official MCP OAuth client completed discovery, a simulated authorization-code/PKCE exchange and a tool call against a local fake issuer |

Inspector returned backend, database, storage, MCP and workflow runtime `ready`.
`durable_storage_configured` was false because this was a local test directory,
not a verified Render persistent disk. Inspector's browser UI and authenticated
ChatGPT linking were not tested. OAuth resource-server support is implemented;
the remaining Auth0 dashboard and hosted-linking steps are in
[MCP_OAUTH.md](MCP_OAUTH.md).

## Runtime and checks

Python **3.12**, repository requirements installed exactly, MCP **1.26.0**,
FastAPI **0.111.0**, Starlette **0.37.2**, AnyIO **4.6.2**, PyJWT **2.13.0**,
PDFLib **1.17.1**.
Shared workflows ran under local Node **22.13.0**. Inspector used the bundled
Node **24** runtime because Inspector 2.4 requires a newer Node release than
22.13. Production instructions require a supported Node 22 release (22.19+ for
Inspector). Render Shell subsequently confirmed Python **3.13.4**; the
repository `.python-version` and `backend/runtime.txt` now pin that live version.

A separate Python **3.13.4** environment installed the requirements except the
existing `PyMuPDF==1.24.9`, whose native source build failed on this Mac in bundled
zlib code. Its requirement was not changed. Full application installation and
legacy PDF extraction on Python 3.13.4 have therefore not been reproduced locally;
the deployment's Linux build remains a required check.

With the actual MCP transport mounted into a minimal FastAPI host, the Python
3.13.4 check passed against temporary SQLite storage: a synthetic signed OAuth
token was accepted, an absent token received 401, all 18 tools were discovered,
the Node workflow runtime was ready, and draft creation, approval, processing,
Danko rounding, and processing PDF generation completed. A token restricted to
read permission was denied a write. This check did not import the legacy app or
contact Auth0, Render, or any production database. The full maintained tests
listed below were run previously on Python 3.12.

Executed successfully:

```bash
python -m compileall -q backend/mcp_server.py backend/mcp_contracts.py \
  backend/mcp_auth.py backend/mcp_oauth_check.py \
  backend/services/platform_repository.py backend/services/platform_service.py \
  backend/services/workflow_engine.py backend/services/manual_document_worker.py \
  backend/manual_documents.py backend/db.py backend/workspace_service.py
node --check docs/js/app.js
node --check docs/js/platform-workflows.js
node --check backend/workflow_runtime/worker.cjs
python -m pip check
git diff --check
```

No full type checker or separate linter was run. Syntax checks are not described
as type/lint validation. No website browser visual regression run was performed.

## Test results

Commands used the isolated `/tmp/order-extractor-mcp-venv/bin/python` with
`ORDER_EXTRACTOR_LOAD_DOTENV=false`.

| Run | Result |
|---|---|
| `-m pytest -q tests --ignore=tests/test_smoke.py --tb=short` | **380 passed, 2 failed** |
| `-m pytest -q tests/test_smoke.py --tb=short` (separate process) | **45 passed** |
| New `tests/test_mcp_platform.py` coverage, included above | **31 passed** |
| New `tests/test_mcp_oauth.py` coverage, included above | **36 passed** |
| Separate legacy `backend/test_smoke.py` | **1 passed, 44 failed**; same result on untouched HEAD |

The two broad-suite failures were also reproduced in an untouched `git archive
HEAD` checkout using the same environment:

* `test_history_ui_prefers_canonical_client_name`: refers to the absent legacy
  `frontend/index.html`, while the maintained website is in `docs/`.
* `test_agents_sdk_engine_available_and_tracing_not_disabled`: local
  `backend/agents` import resolution shadows the installed Agents SDK.

The separate legacy smoke failures come from its stale fake database module
lacking `reopen_order_for_correction`. The maintained `tests/test_smoke.py` suite
passes. These baseline issues were not changed as part of MCP implementation.

MCP tests cover initialization/discovery/output schema, authorization and denied
origins, both manual formats, exact row/field errors, repeated positions, duplicate
red indexes/numbers, high quantities, existing area rounding, raw values/overrides,
concurrent idempotent create retries, versioned full replacement, all protected
manual statuses, confirmation/read-only mode, real PDF-order approval/processing,
every Danko terminal digit and fractional dimensions, grouping identities,
real sheet/label generators, authenticated persisted downloads, invoice pricing
and shared REST visibility, legacy invoice migration, rollback, source changes,
rate/label limits, audit attribution, stable IDs after deletion, and safe legacy
artifact access. A rendered PDF text-bounds test covers the manual text-fitting fix.

OAuth tests cover public resource metadata, transport challenges, all 18 tool
security descriptors, correct and forged RSA signatures, algorithm/key-URL
rejection, issuer/audience/expiry/not-before checks, malformed or long-lived
tokens, approved users and clients, role/scope intersection, read-only enforcement,
actor-isolated retries, artifact permissions, key rotation and refresh throttling,
provider outages, bounded non-redirecting public fetches, and invalid configuration.
The official MCP client's simulated login verifies resource binding, S256 PKCE,
state, predefined-client token authentication and a successful authorized tool call.

The live Auth0 public OIDC document returned the configured issuer, JWKS URL,
authorization-code flow and S256 support. A later public readiness check also
verified RSA signing keys and enabled issuer identification. Public metadata cannot verify API
permissions, role assignments, exact callbacks or real user token claims.

## PDF and restart checks

Generated four actual PDFs through the live endpoint: standard Mother Sheet,
standard labels, Client Positions + Red Index sheet, and its labels. Rendered
their first pages with Poppler `pdftoppm` and visually inspected all four.
The label test suite also checks 100 × 40 mm page dimensions and one page per
piece. Standard labels include packaged KELI and CE images; the established
manual red-index layout retains its red index, position and section behavior.

Visual QA exposed a pre-existing manual font-fitting error. The shared helper
now draws at the fitted size while restoring surrounding canvas state. Regenerated
and visually inspected the corrected long-header label: the headers are legible
and do not overlap. No other layout redesign was introduced.

Stopped the Uvicorn process and started it again against the same test database.
Downloaded all four saved artifacts and verified their SHA-256 values were
unchanged. This demonstrates application restart persistence; Render durability
still depends on mounting the existing DB_DIR on its persistent disk.

## Render preparation checks

Dashboard screenshots confirmed the current backend-only source root and Python
build/start commands, On Commit auto-deploy, and a 1 GB disk at `/var/data`.
The revised deployment instructions use the full repository root so `docs/`
remains available, install the worker's locked dependencies, and retain `backend`
as the application's working directory. `.node-version` selects Node 22.
The Environment screenshot confirms `DB_DIR=/var/data`, matching the persistent
disk mount. The configured SQLite path is `/var/data/orders.db`. This confirms
the storage configuration. Render Shell confirms the deployed Python version is
**3.13.4**. The operator subsequently ran the backup command and supplied
`Backup OK: /var/data/backup-before-mcp-20260901T201015943971Z` at 20:10 UTC on
2026-09-01. The assistant has not opened or downloaded the production database.

Ran the proposed `app:app` entrypoint from `backend` with temporary storage and a
local test bearer: `/healthz` returned 200, unauthenticated `/mcp` returned 401,
18 tools were discovered, and `platform_health.workflow_runtime` was ready.
The server was stopped afterward. No live Render settings or records were changed.

The copyable Render Shell backup command in MCP_INTEGRATION.md was tested with
synthetic SQLite and JSON data. It preserved database rows and configuration,
used separate destinations on repeated runs, and refused a missing database.
The operator's successful production backup is recorded above; the assistant
did not execute that production command.

The operator's Render Environment screenshot at 20:16 UTC shows the successful
save notification and the new MCP variable names. Their values remain masked,
so the screenshot does not independently verify each value or the selected save
mode. Auto-Deploy Off has not yet been verified. GitHub `main` was checked after
this save and still points to `2a08f3c`.

## Deployment follow-up

Use [MCP_INTEGRATION.md](MCP_INTEGRATION.md) for exact startup, Inspector, Codex,
ChatGPT/tunnel and Render steps, all tool annotations and the file manifest.
Before production use, finish [MCP_OAUTH.md](MCP_OAUTH.md), retain the confirmed
database backup (refresh it if records change), approve deployment, configure the OAuth user/client restrictions,
and test read-only access through a real ChatGPT login. Preserve and account for
the invoice JSON-to-database migration when planning a rollback. SQLite generation transactions serialize
writes and limits are per process; scale through a queue/shared limits if needed.
