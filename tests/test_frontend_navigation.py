from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "docs" / "index.html"
APP_JS = ROOT / "docs" / "js" / "app.js"


def test_navigation_is_grouped_around_factory_workflows():
    html = INDEX_HTML.read_text(encoding="utf-8")

    for label in ("Overview", "Orders", "Production", "Documents", "Analytics", "Settings"):
        assert f"<span>{label}</span>" in html
    assert 'data-nav-parent="production"' in html
    assert 'data-nav-parent="documents"' in html
    assert 'data-tab="awa"' not in html


def test_approved_orders_can_be_safely_reopened_for_correction():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'id="historyReopen"' in html
    assert "Reopen for correction" in html
    assert "must be approved again before production" in html
    assert 'normalizeHistoryStatusValue(status) === "approved"' in js
    assert "reopenApprovedOrderForCorrection" in js
    assert "/orders/${orderId}/reopen" in js
    assert "The current approved copy will be saved" in js
    assert 'selectOrderDetailView("items")' in js


def test_beta_shadow_module_is_registered_without_exposing_legacy_awa():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'data-tab="beta"' in html
    assert '<span>Beta</span><span class="sidebar-beta-badge" aria-hidden="true">Beta</span>' in html
    assert 'id="tabBeta"' in html
    assert 'id="betaRunShadow"' in html
    assert 'Shadow Mode' in html
    assert 'beta: document.getElementById("tabBeta")' in js
    assert 'title: "Beta"' in js
    assert 'name === "beta"' in js
    assert "loadBetaOverview();" in js
    assert 'data-tab="awa"' not in html
    assert 'id="workspaceOpenAwa"' not in html
    assert 'id="workspaceOpenBeta">Beta operator</button>' in html
    assert 'activateTab("beta")' in js


def test_beta_frontend_has_approval_recording_but_no_execution_control():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))
    beta_html = html[html.index('id="tabBeta"'):html.index('id="tabTelegram"')]
    beta_js = js[js.index("function betaEntryMetadata"):js.index("async function refreshWorkspaceAfterAction")]

    assert 'id="betaApprovePlan">Record Approval</button>' in beta_html
    assert 'id="betaRejectPlan">Record Rejection</button>' in beta_html
    assert "Run Production" not in beta_html
    assert "Execute Plan" not in beta_html
    assert "/api/workspace/confirm-action" not in beta_js
    assert "/orders/" not in beta_js
    assert "/api/beta/sessions/shadow" in beta_js
    assert "No production action was executed." in beta_js
    assert 'approved_by: "operator"' not in beta_js


def test_beta_memory_tabs_have_keyboard_and_panel_relationships():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'id="betaMemoryTabRules"' in html
    assert 'aria-controls="betaMemoryPanelRules"' in html
    assert 'role="tabpanel" aria-labelledby="betaMemoryTabRules"' in html
    assert '["ArrowLeft", "ArrowRight", "Home", "End"]' in js
    assert 'activateBetaMemoryTab(target.dataset.betaMemoryTab, { focus: true })' in js


def test_beta_teach_mode_has_persistent_recorder_comparison_and_review_boundary():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    for element_id in (
        "betaStartTeaching",
        "betaTeachingBar",
        "betaTeachingPause",
        "betaTeachingFinish",
        "betaTeachingCompare",
        "betaOrderComparison",
        "betaDecisionReasonModal",
        "betaTeachingReview",
        "betaTeachingAcceptAll",
        "betaTeachingRejectWorkflow",
    ):
        assert f'id="{element_id}"' in html

    assert "/api/beta/teaching/start" in js
    assert "/api/beta/teaching/${encodeURIComponent(sessionId)}/events" in js
    assert "/api/beta/teaching/${encodeURIComponent(sessionId)}/compare" in js
    assert "force_vision: options.forceVision !== false" in js
    assert 'recordBetaTeachingEvent("approval_succeeded"' in js
    assert 'recordBetaTeachingEvent("decision_reason"' in js
    assert "records mouse" not in js.lower()
    comparison = js[js.index("async function compareCurrentOrderForTeaching"):js.index("async function finishBetaTeaching")]
    assert comparison.count("betaState.comparedOrderIds.add(key)") == 2
    assert "Use Compare PDF to retry" in comparison
    assert 'String(order.status || "").toLowerCase() === "approved"' in comparison
    assert "openBetaDecisionReason(order);" in comparison


def test_beta_assisted_operator_reviews_then_requires_exact_human_confirmation():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    for element_id in (
        "betaOperatorCommand",
        "betaOperatorLimit",
        "betaRunAssistedReview",
        "betaOperatorResults",
        "betaOperatorApprovalBar",
        "betaApproveSafeOrders",
        "betaDeclineSafeOrders",
    ):
        assert f'id="{element_id}"' in html

    assert "/api/beta/operator/review/start" in js
    assert "/api/beta/operator/review/${encodeURIComponent(session.id)}/approve" in js
    assert "order_ids: orderIds, confirmed: true" in js
    assert "The server will recheck every order first" in js
    assert "data-beta-operator-select" in js
    assert "safe_to_approve" in js


def test_production_control_tower_plans_before_processing_and_uses_one_copilot():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))
    workspace_html = html[html.index('id="tabWorkspace"'):html.index('id="tabAwa"')]

    for element_id in (
        "workspaceAttention",
        "workspaceRecommendations",
        "workspaceBatchPlan",
        "workspaceProcessSelected",
        "workspaceChatLog",
        "workspaceCommandForm",
    ):
        assert f'id="{element_id}"' in workspace_html
    assert "Production Copilot" in workspace_html
    assert "Plan selected batch" in workspace_html
    assert "Nothing runs until you confirm it." in workspace_html
    assert "Smart Chat" not in workspace_html
    assert "/api/workspace/batch-plan" in js
    assert "function renderWorkspaceAttention" in js
    assert "function renderWorkspaceBatchPlan" in js
    assert "Confirm &amp; create production files" in js
    assert "mutated_production_data" not in workspace_html
    assert "No raw, approved, or history data changed" in js


def test_teach_mode_never_calls_a_beta_production_execution_endpoint():
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))
    teaching_js = js[js.index("function betaTeachingIsActive"):js.index("async function loadBetaSession")]

    assert "/execute" not in teaching_js
    assert "/api/workspace/confirm-action" not in teaching_js
    assert "approveDraft(" not in teaching_js
    assert "processWorkspace" not in teaching_js


def test_teach_mode_records_full_platform_context_without_credentials_or_request_bodies():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert "Full platform context" in html
    assert "GPT-5.6 Terra" in html
    assert "installBetaFullContextRecorder();" in js
    assert 'document.addEventListener("click"' in js
    assert 'document.addEventListener("input"' in js
    assert 'document.addEventListener("change"' in js
    assert 'document.addEventListener("submit"' in js
    assert 'document.addEventListener("drop"' in js
    assert 'document.addEventListener("keydown"' in js
    assert 'window.fetch = async (input, init = {}) =>' in js
    assert '"action_result"' in js
    assert '"action_error"' in js
    assert "context_before" in js
    assert "context_after" in js
    assert "visible_warnings" in js
    assert "selected_order" in js
    assert "BETA_TEACHING_SENSITIVE_FIELD_RE" in js
    recorder = js[js.index("function installBetaFullContextRecorder"):js.index("async function controlBetaTeaching")]
    assert "init.body" not in recorder
    assert "request.body" not in recorder


def test_manual_invoice_workspace_stays_inside_manual_orders():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'data-tab="invoices"' not in html
    assert 'id="tabInvoices"' not in html
    for element_id in (
        "manualInvoiceModal",
        "manualInvoiceClose",
        "invoiceDetailCard",
        "invoiceLinesWrap",
        "invoiceGeneratePdf",
        "invoiceGlassPrices",
        "invoiceSpacerPrices",
        "invoiceSavePrices",
        "invoicePrompt",
        "invoiceAddOrderModal",
        "typeCorrectionsModal",
    ):
        assert f'id="{element_id}"' in html

    manual_action = js[js.index("async function handleManualOrderAction"):js.index("function ensureManualOrdersReady")]
    assert 'if (action === "invoice")' in manual_action
    assert "openManualInvoiceModal()" in manual_action
    assert 'activateTab("invoices")' not in manual_action
    assert "await createInvoiceJobFromOrder(shared, { allowAi: false })" in manual_action
    assert "await Promise.race([" in manual_action
    assert "manualInvoicePricingIssues(shared)" not in manual_action

    new_job_branch = js[js.index("async function addInvoiceJobFromOrder"):js.index("async function createInvoiceJobFromOrder")]
    assert new_job_branch.index("appState.invoices.jobs.unshift(job)") < new_job_branch.index(
        "await recalcInvoiceJob(job, { allowPrompt: true, allowAi })",
        new_job_branch.index("}else{"),
    )
    assert 'kind: "spacer",\n          thickness: th,\n          spacerKind: spacerMode' in js
    assert "async function fetchInvoiceEndpoint" in js
    assert "const { allowPrompt = false, allowAi = true } = options" in js
    assert 'if (!composition.panes.length && String(group.displayType || "").trim())' in js
    assert 'panes: [String(group.displayType).trim()]' in js
    assert 'numberSignature(targetCompact) === numberSignature(entry.compact)' in js


def test_overview_gates_the_new_order_workspace():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'id="overviewDashboard"' in html
    assert 'id="overviewNewOrder"' in html
    assert 'id="newOrderWorkspace" class="new-order-workspace" hidden' in html
    assert "function setNewOrderWorkspaceOpen" in js
    assert "function loadOverview" in js


def test_overview_quick_upload_reuses_the_order_extraction_workflow():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'id="overviewDropZone"' in html
    assert 'id="overviewUploadOrder"' in html
    assert 'id="overviewPdfInput"' in html
    assert 'setNewOrderWorkspaceOpen(true, { focus: false, instant: true })' in js
    assert "await handlePdfExtraction(file)" in js


def test_history_only_enables_hard_delete_for_drafts():
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    assert 'normalizedStatus === "draft"' in js
    assert "Only draft orders can be deleted; archive this order instead." in js


def test_frontend_treats_legacy_timezone_less_backend_timestamps_as_utc():
    js = (APP_JS.with_name("platform-workflows.js").read_text(encoding="utf-8") + "\n" + APP_JS.read_text(encoding="utf-8"))

    parser = js[js.index("function parsePlatformDate"):js.index("function activityTimeLabel")]
    formatter_start = js.index("function formatDate")
    formatter = js[formatter_start:js.index("function formatArea", formatter_start)]
    assert "isDateOnly" in parser
    assert "isIsoDateTime && !hasTimezone" in parser
    assert 'normalized = `${text}Z`' in parser
    assert "parsePlatformDate(value)" in formatter
    assert "platformTimestamp(item.created_at)" in js
