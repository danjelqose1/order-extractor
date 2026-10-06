from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "docs" / "index.html"
APP_JS = ROOT / "docs" / "js" / "app.js"


def test_navigation_is_grouped_around_factory_workflows():
    html = INDEX_HTML.read_text(encoding="utf-8")

    for label in ("Overview", "Orders", "Processing", "Perfect Cut Bridge", "Labels", "Documents", "Analytics", "Settings"):
        assert f"<span>{label}</span>" in html
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


def test_retired_sections_and_their_global_controls_are_removed():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for tab in ("workspace", "beta", "factoryagent", "awa"):
        assert f'data-tab="{tab}"' not in html
        assert f'data-overview-route="{tab}"' not in html
    for element_id in (
        "tabWorkspace", "tabBeta", "tabFactoryAgent", "factoryAgentNav",
        "betaTeachingBar", "betaDecisionReasonModal", "betaOrderComparison",
        "workspaceOpenBeta", "overviewOpenBeta",
    ):
        assert f'id="{element_id}"' not in html
    assert "./js/factory-agent.js" not in html
    assert "./css/factory-agent.css" not in html


def test_processing_and_production_documents_remain_available():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for element_id in (
        "tabProcessing", "tabPerfectCut", "tabLabels", "processingAddLabels",
        "productionSheetOpen", "productionSheetDialog", "productionSheetSave",
        "manualInvoiceModal", "invoiceGeneratePdf",
    ):
        assert f'id="{element_id}"' in html
    for script in ("platform-workflows.js", "production-sheet.js", "production-sheet-voice.js", "perfect-cut-bridge.js"):
        assert f'./js/{script}' in html


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
