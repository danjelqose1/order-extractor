# Private Order Extractor MCP

This is a **tool-only Python MCP server** at `/mcp`, mounted into the existing
FastAPI application with the official `mcp==1.26.0` SDK and stateless Streamable
HTTP. There is no ChatGPT widget, second web service, or separate order store.
It is disabled until explicitly configured.

## Architecture and reuse

```text
ChatGPT -> Auth0 login and consent -> user access token
Authenticated ChatGPT / Codex / MCP client
  -> FastAPI /mcp (OAuth or private bearer, origin/body limits, per-request authorization)
  -> typed tool -> PlatformService -> shared db.py functions
  -> existing SQLite orders / manual_orders, plus job/audit/artifact metadata
  -> existing workflow functions -> persistent artifact -> protected download
```

The repository inspection found no applicable AGENTS.md or checked-in Render
configuration. `APP_KEY` only guarded selected extraction endpoints; it was not
suitable platform-wide authentication. The maintained UI is `docs/`, and the
backend entry point is `backend/app.py` (or `app:app` from `backend/`).

Processing's authoritative Danko rule, one-millimetre grouping, Mother Sheet PDF,
label PDF, manual adapter, and invoice pricing were in `docs/js/app.js`. The old
Python workspace generator is unreachable after its `frontend_workflow_required`
return. **It is not re-enabled.** Those JavaScript functions now live once in
`docs/js/platform-workflows.js`, loaded by the website before `app.js`. A bounded
Node **worker process**, invoked by Python with JSON on stdin, runs that same
file. This is why a JavaScript runtime is needed; it has no listener or database
access and receives no application credentials. PDFLib is pinned to 1.17.1.

Client Positions + Red Index documents reuse the existing
`build_manual_processing_pdf` / `build_manual_labels_pdf` ReportLab layouts and
saved manual print settings. Red indexes and sections remain distinct when
grouping; all origin positions and quantities remain available. Standard Mother
Sheets/labels reuse the existing PDFLib layouts and packaged logo/CE images.
Browser-only custom logo overrides are not server configuration. The red-index
layout retains its existing manual label behavior, which does not add logos.
Visual QA also fixed `_draw_fitted_text` in the existing manual generator: it
now applies the font size it calculated, preventing long label headers from
overlapping. The paper sizes and layout settings stay the same.

Orders remain in `orders`/`order_rows` and `manual_orders`/`manual_order_rows`.
MCP create/update/delete/processing call the same `db.py` functions as REST.
PDF approval uses the existing workspace validation and status-transition
function; proposed automatic corrections require review instead of silently
altering rows. Manual area and override rounding use `_manual_area` and
`_replace_manual_order_rows`, unchanged.

`atomic_workflow()` reuses one SQLAlchemy session for a complete mutation and
starts `BEGIN IMMEDIATE` before checks. Writes, idempotency records, generated
bytes and success audit entries commit together; failures roll back. Version
tokens hash the complete saved record, including timestamps/rows, so legacy UI
edits invalidate MCP versions without changing the meaning of PDF extraction
`version`. Job versions are monotonically increasing integers. ID reservation
tables prevent a deleted order ID being assigned to a different future order.

New metadata: `workflow_jobs`, `workflow_artifacts`, `mcp_operations`, `mcp_audit`,
`app_documents`, `manual_order_ids`, `extracted_order_ids`; one nullable
`manual_orders.raw_values_json` column. Job snapshots are immutable processing
inputs, not another order store. Artifact bytes live in SQLite, so they share
the durability of the existing `DB_DIR/orders.db`.

Invoice drafts reuse the existing JavaScript pricing engine with `allowAi:false`,
the real server price configuration and factory aliases. No OpenAI API call is
made by MCP. Unresolved prices remain flagged. There is no invoice finalization,
sending, printing, or invoice PDF tool. Draft JSON is a persistent artifact.
`/api/invoices` and MCP now share transactional `app_documents` storage: on the
first write, existing `invoices.json` jobs are retained in the database; the
original file stays untouched as a migration source. Thereafter the database
is authoritative. Do not roll back to code that only reads the old JSON file
without first exporting the current invoice document.

## Tools and permissions

All tools require authentication. `ORDER_EXTRACTOR_MCP_READ_ONLY=true` blocks
all mutations at runtime. There are no arbitrary SQL, URL, code or action tools.
All `openWorldHint` values are **false**. `R`, `D`, and `I` below are
`readOnlyHint`, `destructiveHint`, and `idempotentHint`.

| Tool | R | D | I | Preconditions / effect |
|---|---|---|---|---|
| `platform_health` | true | false | true | Safe readiness flags; durability flag is operator-supplied |
| `list_orders` | true | false | true | Both stores; filters and pagination |
| `get_order` | true | false | true | Full normalized order and opaque version |
| `list_manual_orders` | true | false | true | Manual store only |
| `get_manual_order` | true | false | true | Full manual fields, raw values and overrides |
| `get_processing_job` | true | false | true | Original/rounded rows, groups and job version |
| `get_platform_summary` | true | false | true | Counts, pieces and area; excludes cancelled/archived |
| `list_order_artifacts` | true | false | true | New artifacts and available existing workspace PDFs |
| `create_manual_order_draft` | false | false | true | Idempotency key; creates Draft only |
| `update_manual_order_draft` | false | true | false | Draft only; expected version; full replacement |
| `approve_order` | false | true | false | Expected version, validation, `confirmed:true` |
| `send_order_to_processing` | false | true | true | Approved only; expected version, key, confirmation |
| `apply_danko_rounding` | false | false | true | Job version; preserves original dimensions/areas |
| `group_processing_dimensions` | false | false | true | Job version; preserves origin identities |
| `generate_processing_sheet` | false | false | true | Job version, key, unchanged source order |
| `generate_labels` | false | false | true | Processing job, version, key, quantity cap |
| `create_invoice_draft` | false | false | true | Order version/key; only a draft, missing prices flagged |
| `delete_manual_order_draft` | false | true | false | Draft only; version and confirmation |

Read tools only add operational audit entries; they do not mutate business data.
For idempotent creates/generation, reuse the same key **and identical arguments**
on retries. A reused key with different input returns `IDEMPOTENCY_CONFLICT`.
The original successful result is returned even if the order later changes or
is deleted; fetch the order again to see its current state. Rounding/grouping
set flags, never toggle them; a stale version still returns `VERSION_CONFLICT`.
There is no automatic eviction of idempotency records.

## Contracts and examples

IDs are `manual:42`, `pdf:42`, `job:<32 hex characters>`, and
`artifact:<32 hex characters>`. `pdf:` identifies the existing extracted-order
table, including text and Telegram sources; `source` retains the original
source label. Dates filter manual `order_date` or extracted creation date using
the platform's timezone rules. Year defaults to `all`. List results are compact;
use get tools for rows. Status values retain each store's established vocabulary.
Existing workspace PDFs use `legacy:<file ID>:<SHA-256>` references. They are read
through the existing production-file catalogue, confined to
`DB_DIR/production-files`, and downloaded through the same protected endpoint.
Missing files, paths outside that directory and PDFs over 32 MB are excluded.
A changed legacy file cannot be downloaded under an old content reference.

Create a standard draft:

```json
{
  "client_name": "Example Client",
  "order_number": "MCP-EXAMPLE-001",
  "order_date": "2026-09-01",
  "mode": "standard",
  "reference_notes": "Workshop reference",
  "dimension_unit": "mm",
  "idempotency_key": "example-create-001",
  "rows": [{"position":"A", "width_mm":1001, "height_mm":604,
            "quantity":2, "glass_type":"4F", "row_notes":""}]
}
```

For `client_positions_red_index`, use rows such as:

```json
{"section":"Kitchen", "client_position":"A", "red_index":1,
 "width_mm":1001, "height_mm":604, "quantity":2,
 "glass_type":"4F", "row_notes":"Keep position", "area_override_m2":1.25}
```

Repeated client positions are valid; red indexes must be positive and unique.
Dimensions must be positive finite numbers and quantities positive integers.
Quantities above 100 create a warning; values are never clamped. Requests are
bounded to 1,000 rows and 2 MB. `dimension_unit` records the original entry/display
unit; **fields named `_mm` always contain millimetres**, including when that unit
is `cm`. Raw text and overrides are retained alongside canonical platform values.
The existing three-decimal area rounding stays authoritative.

All tool responses have this envelope (the payload differs per tool):

```json
{"ok":true, "request_id":"<request UUID>", "data":{
  "order_id":"manual:42", "order_number":"MCP-EXAMPLE-001",
  "status":"draft", "version":"<opaque 64-character token>",
  "row_count":1, "piece_count":2, "calculated_area_m2":1.209,
  "total_area_m2":1.209, "warnings":[]
}, "error":null}
```

This is an abbreviated example; `get_order` and create include full normalized
fields. Exact JSON schemas are advertised by `tools/list` and defined in
`backend/mcp_contracts.py`.

Editing requires `{"order_id":"manual:42","expected_version":"...",
"replacement":{...complete draft fields...}}`. Omitted optional fields reset to
their defaults. This is **replacement**, not a patch.

Approval requires `{"order_id":"manual:42","expected_version":"...",
"confirmed":true}`. Processing adds `"idempotency_key":"process-42-001"`.
Generate a document with `{"processing_job_id":"job:...",
"expected_version":3,"idempotency_key":"sheet-job-001"}`. The response's
`artifacts[]` entries contain ID, kind, SHA-256, size, creation time, job version,
and an authenticated `download_path`. Download with an Authorization header;
these are not anonymous share links and carry no URL credentials.

Error results set MCP `isError:true` and `ok:false`, with a stable code,
sanitized message, `issues[]` (exact row/field), and `retryable`. Codes include
`VALIDATION_ERROR`, `NOT_FOUND`, `DUPLICATE_ORDER_NUMBER`, `ORDER_PROTECTED`,
`VERSION_CONFLICT`, `ORDER_CHANGED`, `IDEMPOTENCY_CONFLICT`, `FORBIDDEN`,
`RATE_LIMITED`, `BUSY`, `GENERATION_LIMIT`, `GENERATION_TIMEOUT`,
`GENERATION_FAILED`, `DATABASE_UNAVAILABLE`, and `INTERNAL_ERROR`.

## Authentication and safety boundary

Choose one explicit `ORDER_EXTRACTOR_MCP_AUTH_MODE`:

* **`oauth` for ChatGPT:** Auth0 handles login, PKCE, consent and token issuance.
  The backend validates RS256 signatures against the configured issuer's public
  keys, issuer, audience, expiry, approved user and optional approved client IDs.
  Access requires both the requested scopes and assigned RBAC permissions:
  `orders:read` for reads, and both `orders:read` and `orders:write` for writes.
  A stable hash of issuer and user ID identifies each actor in audit records.
  Protected-resource discovery and per-tool OAuth metadata are implemented.
  Follow [MCP_OAUTH.md](MCP_OAUTH.md) for the exact Auth0 and ChatGPT setup.
* **`bearer` for private local clients:** set an independent, random
  `ORDER_EXTRACTOR_MCP_TOKEN` of at least 32 characters. Constant-time comparison
  authenticates one factory operator, logged as `private-operator`, source `mcp`.
  Rotate by replacing the environment secret and restarting. Do not reuse
  APP_KEY or an OpenAI API key. This is the default mode for existing clients.

Modes are exclusive: the private token is never a fallback in OAuth mode.
Every MCP request is authenticated, including initialization, tool discovery,
GET/POST/DELETE and artifact downloads. Only public OAuth resource metadata is
anonymous. Disabled or misconfigured authentication returns 503; missing/invalid
credentials return 401. Tokens are never accepted as query parameters. There is
no unauthenticated development bypass. Never put a token in source, metadata,
URLs or normal logs. The backend does not need the Auth0 Client Secret.

The client must request user confirmation for approval, production transition
and deletion; the backend also requires `confirmed:true`. That boolean records
the client's assertion, not cryptographic proof of a human click. Backend
status/version/validation rules remain mandatory regardless of annotations.

Audit entries contain actor/source, tool, order reference where available,
timestamp, generated request ID and outcome; they contain no request bodies,
client names, dimensions, credentials or tool results. Failed writes are audited
after rollback. Generation has a hard subprocess timeout (30 seconds by default,
maximum 60), 256 MB JavaScript heap limit, two worker slots, and 12 expensive
calls/minute. Label jobs above 2,000 pieces are rejected whole, never truncated.
The authenticated transport limit is 120 requests/minute. Limits are per process;
use one Uvicorn worker or add shared ingress limits when scaling.

The existing website/API authorization behavior is preserved. **This change
secures `/mcp`, not every historical REST endpoint.** Restrict existing ingress
appropriately before exposing private production data to new audiences. Legacy
manual REST edits/deletes retain their prior rules; MCP cannot perform those
actions on approved orders, and a source change invalidates an existing job.
Artifacts are retained independently without destructive cascades. Generating
at a later job version creates a new artifact; old versions remain identifiable.

## Local startup (Bash)

Requires Python 3.11+ and Node 22 (22.19+ if using current Inspector 2.4).
These commands use fresh test storage and disable loading the existing `.env`.

```bash
cd /Users/danjel/Documents/order-extractor-clean
python3.11 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt pytest
npm ci --prefix backend/workflow_runtime --ignore-scripts
export DB_DIR="$(mktemp -d /tmp/order-mcp-local.XXXXXX)"
export ORDER_EXTRACTOR_LOAD_DOTENV=false
export ORDER_EXTRACTOR_MCP_ENABLED=true
export ORDER_EXTRACTOR_MCP_AUTH_MODE=bearer
read -rsp 'Private local MCP token (32+ characters): ' ORDER_EXTRACTOR_MCP_TOKEN
export ORDER_EXTRACTOR_MCP_TOKEN
export OPENAI_API_KEY=local-test-placeholder
export OPENAI_AGENTS_DISABLE_TRACING=1
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 5057 --no-access-log
```

The placeholder only satisfies the legacy extraction module's import-time key
check. MCP workflows do not use it. Do not invoke extraction/AI routes with this
test configuration. No production credentials need to be copied locally.

Environment details are in `.env.example`. Keep `DB_DIR` explicit: both
`orders.db` and legacy invoice/price configuration are resolved there. The
workflow worker reads the packaged shared JS and logos from `docs/`.

## MCP Inspector

In another terminal:

```bash
npx @modelcontextprotocol/inspector@2.4.0
```

Choose **Streamable HTTP**, enter `http://127.0.0.1:5057/mcp`, and provide the
Authorization header as `Bearer <your private local token>` in Inspector's
authentication field. Use its local proxy. Set
`ORDER_EXTRACTOR_MCP_ALLOWED_ORIGINS` only if an authorized client forwards an Origin
header. The allowlist validates origins; it does not enable direct cross-origin
browser CORS/preflight. Use Inspector's proxy rather than its direct browser mode.
Do not widen the website's CORS configuration.

For automated CLI testing, create a temporary, permission-600 Inspector config
outside the repository from your environment; this keeps tokens out of argv:

```bash
export MCP_INSPECTOR_CONFIG="$(mktemp /tmp/order-mcp-inspector.XXXXXX)"
.venv/bin/python - <<'PY'
import json, os
path = os.environ['MCP_INSPECTOR_CONFIG']
os.chmod(path, 0o600)
with open(path, 'w') as output:
    json.dump({'mcpServers': {'order-extractor-local': {
        'type': 'streamable-http', 'url': 'http://127.0.0.1:5057/mcp',
        'headers': {'Authorization': 'Bearer ' + os.environ['ORDER_EXTRACTOR_MCP_TOKEN']}
    }}}, output)
PY
npx @modelcontextprotocol/inspector@2.4.0 --cli \
  --config "$MCP_INSPECTOR_CONFIG" --server order-extractor-local --method tools/list
npx @modelcontextprotocol/inspector@2.4.0 --cli \
  --config "$MCP_INSPECTOR_CONFIG" --server order-extractor-local \
  --method tools/call --tool-name platform_health
rm "$MCP_INSPECTOR_CONFIG"
```

The second terminal must receive the same token through its environment/hidden
prompt. Test create/approve/process tools only against your fresh local DB_DIR.

## Codex connection

After supplying the token to the environment of the Codex host, add:

```toml
[mcp_servers.order_extractor_private]
url = "http://127.0.0.1:5057/mcp"
bearer_token_env_var = "ORDER_EXTRACTOR_MCP_TOKEN"
startup_timeout_sec = 90
tool_timeout_sec = 90
default_tools_approval_mode = "writes"
```

Use the approved HTTPS URL for a future remote deployment. This configuration
was prepared, not installed into the user's Codex settings. Bearer HTTP support
and these configuration fields are described in the
[official Codex MCP guide](https://developers.openai.com/codex/mcp).

## ChatGPT Developer Mode and secure-tunnel testing

**OAuth is implemented:** use `ORDER_EXTRACTOR_MCP_AUTH_MODE=oauth` and complete
[MCP_OAUTH.md](MCP_OAUTH.md). Auth0 application/API creation was confirmed in the
user's screenshots; permissions, roles, tenant settings, exact callback and
hosted linking still need to be verified. ChatGPT's documented connection options
are OAuth, no authentication, and mixed OAuth/no-auth; a static bearer token is
not a direct Developer Mode auth option. Choose OAuth for this server. See
[ChatGPT Developer Mode](https://developers.openai.com/api/docs/guides/developer-mode)
and [OpenAI authentication guidance](https://developers.openai.com/plugins/build/auth).

The backend supplies protected-resource discovery, WWW-Authenticate challenges,
token verification and per-tool scope enforcement. Auth0 supplies authorization
metadata, PKCE and managed login. Register ChatGPT's exact displayed callback
with the predefined Auth0 client; no custom authorization server is needed.

After configuring OAuth, the current documented steps are:

1. Start the local backend against test storage and verify it with Inspector.
2. Use **Secure MCP Tunnel** to keep the endpoint private. Obtain a tunnel ID
   from Platform tunnel settings, associate the target ChatGPT workspace, and
   run `tunnel-client` on the host that can reach the local backend.
3. Configure its HTTP target as `http://127.0.0.1:5057/mcp`; configure app-level
   authentication separately. Verify `tunnel-client doctor --profile <profile>
   --explain`, then keep `tunnel-client run --profile <profile>` running. Tunnel
   setup requires its own authorized runtime credential and organization roles;
   none are created here. A tunnel alone does not satisfy this server's OAuth
   authentication requirement. Its public resource URL must match the Auth0 API
   identifier and backend resource setting exactly.
4. Alternatively forward an **OAuth-protected staging endpoint** over an HTTPS
   tunnel, e.g. `ngrok http 5057`; use its HTTPS `/mcp` URL. Avoid tunnelling the
   legacy unauthenticated REST surface to an unrestricted public audience.
5. In ChatGPT, open **Settings → Security and login → Developer mode**.
6. Open **Plugins**, select **+**, give the app a name and description. Under
   **Connection**, choose **Tunnel** and select the associated tunnel (or enter
   the tunnel ID), or enter the authenticated public HTTPS MCP URL.
7. Complete OAuth/account linking, create the connection and review all 18 tools.
8. Start a new conversation, choose Developer mode from the tools/plus menu,
   select this app, and test reads followed by explicitly confirmed writes on
   test records. Check retries, stale versions and out-of-scope requests.

Account/workspace availability is controlled by OpenAI and workspace policy.
These steps follow [Connect and test your plugin](https://developers.openai.com/plugins/deploy/connect-chatgpt)
and [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels).
No ChatGPT connection, tunnel, production endpoint or production secret was
modified during implementation.

After changing tool names, schemas, annotations, descriptions or auth: restart
the server, open the connection under ChatGPT Plugins, choose **Refresh**, check
the new metadata, then start a new conversation and repeat affected cases.

## Render deployment preparation — not executed

The existing service remains Python/FastAPI. Dashboard screenshots supplied on
2026-09-01 show service `srv-d3fdbt2li9vc73f2rke0` on branch `main`, live commit
`2a08f3c`, Root Directory `backend`, Build Command `pip install -r requirements.txt`,
Start Command `uvicorn app:app --host 0.0.0.0 --port $PORT`, and Auto-Deploy
**On Commit**. The service has a 1 GB persistent disk mounted at `/var/data`.
The Environment screenshot also confirms `DB_DIR=/var/data`, matching that mount;
`backend/db.py` therefore resolves the database to `/var/data/orders.db`.
Render Shell confirms Python **3.13.4**. The operator ran the backup command below
and supplied its successful output on 2026-09-01 at 20:10 UTC:
`Backup OK: /var/data/backup-before-mcp-20260901T201015943971Z`.

**Required layout change:** leave Render's Root Directory **empty** for the new
deployment. Render excludes files outside a configured root from builds and
runtime, so keeping `backend` would remove the shared `docs/` workflows and logos.
The new start command still changes into `backend` before launching the app,
preserving the existing working directory and relative data paths. See
[Render root directories](https://render.com/docs/monorepo-support#setting-a-root-directory).

Prepare these settings before the coordinated deployment. Because On Commit is
enabled, pushing to `main` currently triggers a deploy. Disable Auto-Deploy while
staging the change, and use **Save only** when staging environment variables.
Root/build/start edits can also initiate deployment: apply the final commands
only after the new commit is available and the database backup/path are verified.

1. The operator's backup completed at the path recorded above, using SQLite's
   backup API and its integrity check, plus any existing price/invoice JSON.
   Keep it on the persistent disk. If further factory edits occur before a
   later deployment, take a fresh backup. Preserve the existing Render secrets.
2. Preserve the **same existing persistent disk** and `DB_DIR=/var/data`;
   both were confirmed in the dashboard. The deployment can set
   `ORDER_EXTRACTOR_MCP_DURABLE_STORAGE=true` for this verified mapping.
   Restart the service and verify test artifacts survive before using production
   writes.
3. Keep the currently deployed Python **3.13.4**. With Root Directory empty, set
   this Build Command:

   ```bash
   python -m pip install -r backend/requirements.txt && npm ci --prefix backend/workflow_runtime --ignore-scripts && node --check docs/js/platform-workflows.js
   ```

   Render's native Python runtime includes Node and npm during builds and runtime.
   The repository `.node-version` selects Node 22; if `NODE_VERSION` is set in
   Render, set it to `22` as well because it takes precedence. Verify the deployed
   Node version and workflow health before enabling writes. See
   [native runtime tools](https://render.com/docs/native-runtimes#tools-and-utilities)
   and [Node version selection](https://render.com/docs/node-version).
4. Set this Start Command, retaining the application's `backend` working directory:

   ```bash
   cd backend && python -m uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1
   ```

   `/healthz` remains the Render health check. Keep `DB_DIR` pointing to the
   existing data directory on the same persistent disk; do not create a new
   empty data directory as part of changing the source root. The repository's
   `.python-version` pins **3.13.4**, matching the verified live runtime.
   If setting `PYTHON_VERSION` in Render, use **3.13.4** because that variable
   takes precedence. `backend/runtime.txt` is aligned for consistency; the
   documented version selectors are `PYTHON_VERSION` and `.python-version`.
   See [Render Python versions](https://render.com/docs/python-version).
5. For ChatGPT, set the OAuth issuer, resource URL, approved user IDs and client
   ID from [MCP_OAUTH.md](MCP_OAUTH.md) in Render's Environment UI; choose
   `ORDER_EXTRACTOR_MCP_AUTH_MODE=oauth`, initially set
   `ORDER_EXTRACTOR_MCP_READ_ONLY=true`, and enable with
   `ORDER_EXTRACTOR_MCP_ENABLED=true`. The Auth0 Client Secret goes directly into
   ChatGPT's OAuth configuration, not the backend. Leave existing extraction
   credentials/CORS unchanged. The endpoint will be
   `https://order-extractor-kdih.onrender.com/mcp` after an approved deployment.
6. Run the OAuth readiness check and inspect metadata, missing/invalid-token
   rejection, initialization, tool discovery and read tools. Complete a real
   ChatGPT login using an explicitly approved Auth0 user. Then
   deliberately enable writes and verify with a designated test record before
   factory use. An always-on instance avoids cold-start connection timeouts.
   Keep client startup/tool timeouts near 90 seconds; retry the same idempotency
   key after a cold start or interrupted response, not a new key.
7. Publish the two frontend JS files and updated `docs/index.html` together in the
   normal GitHub Pages process. Shared definitions must load before `app.js`.
8. To disable the integration, set `ORDER_EXTRACTOR_MCP_ENABLED=false`; ordinary
   APIs still run. Preserve the new database tables. Full code rollback after
   invoice migration requires exporting the current invoice document to the old
   JSON format first; never restore a stale JSON file over current invoices.

### Backup before deployment

Run this in the existing Render Shell while operators are not changing orders,
prices, or invoices. It creates a new dated directory on the persistent disk,
uses SQLite's online backup API, checks the copied database, and copies existing
price/invoice JSON configuration. It reads the database through a read-only
connection and refuses to create a backup when the source database is missing.
No application import or credentials are needed.

```bash
python - <<'PY'
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import time

root = Path(os.environ["DB_DIR"]).resolve()
source = root / "orders.db"
if not source.is_file():
    raise SystemExit("STOP: existing orders.db was not found")
backup = root / ("backup-before-mcp-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
backup.mkdir(mode=0o700)
deadline = time.monotonic() + 60

def progress(status, remaining, total):
    if time.monotonic() > deadline:
        raise TimeoutError("Backup timed out; deployment should wait")

with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
    with closing(sqlite3.connect(backup / "orders.db")) as dst:
        src.backup(dst, pages=128, progress=progress)
        if dst.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("Backup failed the integrity check")
for name in ("price-config.json", "invoices.json"):
    path = root / name
    if path.is_file():
        contents = path.read_bytes()
        json.loads(contents)
        (backup / name).write_bytes(contents)
print("Backup OK:", backup)
PY
```

Continue only after `Backup OK:` is printed. Keep the original data directory
and its files in place. This command was checked against synthetic data, including
repeat runs with separate destinations and refusal when the source is absent.
The operator ran this command in Render Shell and supplied the successful output
recorded above. The assistant has not downloaded or inspected the production data.

## Validation and troubleshooting

Run:

```bash
node --check docs/js/app.js
node --check docs/js/platform-workflows.js
node --check backend/workflow_runtime/worker.cjs
ORDER_EXTRACTOR_LOAD_DOTENV=false .venv/bin/python -m pytest -q tests/test_mcp_platform.py
ORDER_EXTRACTOR_LOAD_DOTENV=false .venv/bin/python -m pytest -q tests/test_mcp_oauth.py
ORDER_EXTRACTOR_LOAD_DOTENV=false .venv/bin/python -m pytest -q tests --ignore=tests/test_smoke.py
ORDER_EXTRACTOR_LOAD_DOTENV=false .venv/bin/python -m pytest -q tests/test_smoke.py
```

The separate smoke run avoids a filename collision with `backend/test_smoke.py`.
See `MCP_VALIDATION.md` for the actual results, known baseline failures and
levels completed. All tests use temporary data; no production calls are needed.

* 503 at `/mcp`: enabled flag must be exactly `true`; check the chosen auth mode's
  configuration. `AUTH_PROVIDER_UNAVAILABLE` means the public signing keys could
  not be refreshed; retry after the indicated delay.
* 401: supply a valid bearer header. In OAuth mode check issuer/audience, RS256,
  token lifetime and RBAC claim settings. URL credentials are never recognized.
* 403: check approved user/client IDs, read/write permissions, read-only mode and
  the MCP-specific browser-origin allowlist. See the OAuth troubleshooting guide.
* `GENERATION_FAILED`: install Node and run `npm ci` in `backend/workflow_runtime`;
  verify packaged JS/logos, existing manual print settings and input text.
* `DATABASE_UNAVAILABLE`: SQLite writers serialize; retry after the current bounded
  job, with the same key. A generation transaction may hold the write lock for
  its timeout. Larger installations should move work to a durable queue.
* `VERSION_CONFLICT`/`ORDER_CHANGED`: fetch current state and review it before
  proceeding; never substitute a guessed version or silently overwrite.
* `GENERATION_LIMIT`: reduce the requested workload through an explicitly reviewed
  operational plan or raise the server cap; the stored quantity is not reduced.
* Lost HTTP response: repeat the exact keyed call to recover its original result.
* Missing historical artifacts: files formerly downloaded only in a browser were
  never persisted; the integration cannot recover them. Regenerate through a
  reviewed processing job. Existing ready ProductionFile PDFs are included only
  when their bytes remain in the configured production-files directory.

Future tools must add a specific typed contract, a service method using shared
business logic, accurate annotations and backend authorization, then tests for
invalid inputs, stale versions, retries and rollback. Never register arbitrary
SQL, filesystem paths, URLs, commands or generic `execute_action` tools. Refresh
client metadata after adding tools. See the official
[MCP server](https://developers.openai.com/plugins/build/mcp-server) and
[tool planning](https://developers.openai.com/plugins/plan/tools) guides.

## File manifest

Created:

* `.env.example`: private connection settings and safe defaults.
* `.node-version`: select the worker's supported Node 22 major on Render.
* `.python-version`, `backend/runtime.txt`: retain the verified Render Python 3.13.4.
* `backend/mcp_contracts.py`: strict input models and structured output schemas.
* `backend/mcp_auth.py`: exclusive bearer/OAuth modes, bounded key cache, JWT
  verification, private user/client restrictions and scope enforcement.
* `backend/mcp_oauth_check.py`: read-only public provider/configuration checks.
* `backend/mcp_server.py`: official SDK transport, catalogue, authentication,
  limits, protected downloads and sanitized errors.
* `backend/services/platform_service.py`: guarded workflow orchestration.
* `backend/services/platform_repository.py`: shared invoice persistence and
  transactional job, artifact, audit and idempotency metadata.
* `backend/services/workflow_engine.py`: bounded invocation of shared JavaScript.
* `backend/services/manual_document_worker.py`: bounded existing ReportLab layouts.
* `backend/workflow_runtime/worker.cjs`, `package.json`, `package-lock.json`:
  pinned PDFLib worker runtime.
* `docs/js/platform-workflows.js`: extracted authoritative website functions.
* `tests/test_mcp_platform.py`: protocol, security and full workflow integration tests.
* `tests/test_mcp_oauth.py`: signed-token, scope, rotation and SDK PKCE tests.
* `docs/MCP_INTEGRATION.md`, `docs/MCP_OAUTH.md`, `docs/MCP_VALIDATION.md`: setup,
  Auth0 configuration, deployment and evidence.

Modified:

* `backend/app.py`: opt-in MCP mount, shared invoice repository, isolated-test
  dotenv switch. Existing endpoint shapes and website CORS remain unchanged.
* `backend/db.py`: additive metadata, raw-value preservation, transaction reuse,
  stable ID reservations and manual status transition service.
* `backend/workspace_service.py`: read existing files for one extracted order.
* `backend/manual_documents.py`: apply measured text fitting when drawing.
* `backend/requirements.txt`: official MCP SDK, compatible SSE dependency and
  explicit PyJWT cryptography dependency.
* `docs/js/app.js`, `docs/index.html`: use the shared workflow file and cache version.
* `tests/test_manual_orders.py`, `tests/test_frontend_navigation.py`: follow the
  shared file; verify rendered label header bounds.
* `.gitignore`: exclude local environments, secrets, caches and installed packages.
