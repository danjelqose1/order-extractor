# AI Decisions

The feature is disabled by default (`AI_DECISIONS_ENABLED=false`). Reuse the existing `OPENAI_API_KEY` in Render; never put it in GitHub Pages or browser storage. No new dependencies. The pinned OpenAI SDK uses its official low-level `post('/decisions', …)` transport until a compatible SDK exposes `client.decisions.create`. Both send the documented native `input_json` and predicate/choice/score questions to https://api.openai.com/v1/decisions with `gpt-6-luna`.

New extractions (paste, Scan Studio and Telegram PDF/image paths) and manual order creation receive advisory evaluations when enabled. The AI Decisions navigation becomes visible from `/api/features`. Existing orders can be evaluated by ID. The overview and Inbox show indicators for evaluated orders. Evaluation never changes order values, Inbox handling state, production status, printing or cutting instructions.

The local matcher preserves component notation and thickness, compares complete row multisets, retains repeated pieces and position/index relationships, and uses configured client aliases. Exact identity requires source hash, client, order identity and all piece fields. Formatting, alternate component order, modified quantities, spacer or coating differences and 0/+3/-7 mm dimensions only identify candidates for judgment. Model output never establishes exact identity. Unknown glass classification, extraction warnings, API refusal/failure, and uncertain candidates require review. Invalid dimensions or quantities and absent glass compositions block readiness. Readiness is a recommendation, separate from existing statuses.

Independent questions share one request. Routing has a separate request using earlier decisions. Predicate probability and choice confidence are distinct documented fields; both gate duplicate recommendations. The dashboard exposes compared fields and differences rather than inventing model-generated explanations. Raw typed answers and configuration are retained for inspection.

Configuration:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AI_DECISIONS_ENABLED` | `false` | Enables evaluation, endpoints and frontend navigation |
| `AI_DECISIONS_CONFIDENCE` | `0.98` | Conservative, uncalibrated confidence threshold |
| `AI_DECISIONS_MAX_CANDIDATES` | `5` | Maximum candidates sent, capped at 10 |
| `AI_DECISIONS_DIMENSION_OFFSETS` | `0,3,-7` | Signed incoming-minus-existing mm offsets on both axes |
| `AI_DECISIONS_CLIENT_ALIASES` | `{}` | JSON object mapping aliases to canonical client names |
| `AI_DECISIONS_TIMEOUT_SECONDS` | `20` | SDK per-attempt timeout; two bounded SDK retries |
| `AI_DECISIONS_MIN_INTERVAL_SECONDS` | `1` | Per-process API admission interval; excess falls back to review |

Recommendations and append-only overrides live in additive `ai_decisions` and `ai_decision_events` tables in the existing SQLite database. Tables are created only when enabled and used. Idempotency fingerprints cover normalized evidence, candidate records, warnings and configuration. A unique constraint prevents duplicate persisted evaluations; concurrent processes may still make redundant upstream requests. No undocumented API idempotency behavior is assumed. Review actions record advisory disposition and destination; they never physically move, merge or delete orders. Undo adds a reset event, preserving the audit. Freshness checks reject review acceptance against changed orders. Existing APP_KEY protection applies to all new endpoints; existing deployments without APP_KEY retain their current trust boundary. Overrides identify the shared-key application session, not an individual user identity (the app has no individual login identity).

Validation and rollout: run `python -m pytest tests/test_ai_decisions.py -q` followed by the full pytest suite and JS syntax checks. Local tests use synthetic orders, a temporary database, and mocked API responses; real orders are untouched. Before enabling on Render, verify endpoint/model access with its existing key, label representative historical duplicate/repeat/modified examples, measure false-positive and missed-duplicate rates, and calibrate thresholds. Automatic movement is deliberately left off until that calibration. Re-evaluate after edits; existing manual edits remain available regardless of API health.

Limits: retrieval is bounded to 100 recent records per client/alias/number and 20 identical source hashes locally. It covers saved incoming drafts, but cannot compare unsaved documents awaiting extraction. Rows retain original notation; semantic component equivalence is left to the model. Visual verification is recommended for scans; the engine sends structured evidence, not original images. When APP_KEY is configured, enter the backend access key in the dashboard password field; it is held only in memory and sent as X-App-Key. This is never the OpenAI key. Live Render/API access and browser layout require deployment validation.
