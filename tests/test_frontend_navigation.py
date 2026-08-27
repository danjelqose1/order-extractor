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


def test_beta_shadow_module_is_registered_without_exposing_legacy_awa():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

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
    js = APP_JS.read_text(encoding="utf-8")
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
    js = APP_JS.read_text(encoding="utf-8")

    assert 'id="betaMemoryTabRules"' in html
    assert 'aria-controls="betaMemoryPanelRules"' in html
    assert 'role="tabpanel" aria-labelledby="betaMemoryTabRules"' in html
    assert '["ArrowLeft", "ArrowRight", "Home", "End"]' in js
    assert 'activateBetaMemoryTab(target.dataset.betaMemoryTab, { focus: true })' in js


def test_manual_invoice_workspace_stays_inside_manual_orders():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

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
    js = APP_JS.read_text(encoding="utf-8")

    assert 'id="overviewDashboard"' in html
    assert 'id="overviewNewOrder"' in html
    assert 'id="newOrderWorkspace" class="new-order-workspace" hidden' in html
    assert "function setNewOrderWorkspaceOpen" in js
    assert "function loadOverview" in js


def test_overview_quick_upload_reuses_the_order_extraction_workflow():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

    assert 'id="overviewDropZone"' in html
    assert 'id="overviewUploadOrder"' in html
    assert 'id="overviewPdfInput"' in html
    assert 'setNewOrderWorkspaceOpen(true, { focus: false, instant: true })' in js
    assert "await handlePdfExtraction(file)" in js


def test_history_only_enables_hard_delete_for_drafts():
    js = APP_JS.read_text(encoding="utf-8")

    assert 'normalizedStatus === "draft"' in js
    assert "Only draft orders can be deleted; archive this order instead." in js
