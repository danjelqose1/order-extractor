from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "docs" / "js" / "app.js").read_text(encoding="utf-8")
LAYOUT_JS_PATH = ROOT / "docs" / "js" / "living-dashboard.js"
LAYOUT_JS = LAYOUT_JS_PATH.read_text(encoding="utf-8")
CSS = (ROOT / "docs" / "css" / "styles.css").read_text(encoding="utf-8")
BACKEND = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")


DEFAULTS = [
    {"id": "alpha", "x": 0, "y": 0, "w": 6, "h": 3, "collapsed": False},
    {"id": "beta", "x": 6, "y": 0, "w": 6, "h": 3, "collapsed": False},
]
SPECS = [
    {"id": "alpha", "title": "Alpha", "minW": 3, "maxW": 12, "minH": 2, "maxH": 8},
    {"id": "beta", "title": "Beta", "minW": 3, "maxW": 12, "minH": 2, "maxH": 8},
]


def _node(expression: str) -> dict:
    script = f"""
const dashboard = require({json.dumps(str(LAYOUT_JS_PATH))});
const defaults = {json.dumps(DEFAULTS)};
const specs = {json.dumps(SPECS)};
const result = ({expression});
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_feature_flag_defaults_off_and_old_dashboard_stays_available():
    assert 'ENABLE_LIVING_DASHBOARD = os.getenv("ENABLE_LIVING_DASHBOARD", "false")' in BACKEND
    assert '@app.get("/api/features")' in BACKEND
    assert 'return {"living_dashboard": ENABLE_LIVING_DASHBOARD}' in BACKEND
    assert 'id="overviewLayoutToolbar" class="overview-layout-toolbar" hidden' in HTML
    assert "if (features?.living_dashboard === true) initializeLivingDashboard();" in APP_JS
    assert "keeping the stable dashboard" in APP_JS


def test_enabled_dashboard_has_stable_widgets_and_edit_mode_controls():
    for widget_id in (
        "today-overview",
        "order-counts",
        "needs-attention",
        "recent-orders",
    ):
        assert f'data-dashboard-widget="{widget_id}"' in HTML
    assert 'id="overviewEditLayout"' in HTML
    assert 'id="overviewDoneLayout" hidden' in HTML
    assert 'id="overviewResetLayout" hidden' in HTML
    assert "root.classList.toggle(\"living-dashboard-editing\", editMode)" in LAYOUT_JS
    assert 'if (!editMode || !isDesktopEditable() || event.button !== 0) return;' in LAYOUT_JS
    assert ".overview-dashboard.living-dashboard-editing .living-dashboard-resize-handle" in CSS


def test_valid_layout_loads_without_changing_positions():
    saved = {
        "version": 1,
        "widgets": [
            {"id": "alpha", "x": 6, "y": 4, "w": 6, "h": 3, "collapsed": False},
            {"id": "beta", "x": 0, "y": 1, "w": 6, "h": 4, "collapsed": True},
        ],
    }
    result = _node(
        f"dashboard.validateAndMergeLayout({json.dumps(saved)}, defaults, specs, 12)"
    )
    assert result["valid"] is True
    assert result["layout"] == saved["widgets"]


def test_invalid_or_overlapping_layout_falls_back_to_defaults():
    corrupt = {
        "version": 1,
        "widgets": [
            {"id": "alpha", "x": 0, "y": 0, "w": 6, "h": 3, "collapsed": False},
            {"id": "beta", "x": 3, "y": 0, "w": 6, "h": 3, "collapsed": False},
        ],
    }
    result = _node(
        f"dashboard.validateAndMergeLayout({json.dumps(corrupt)}, defaults, specs, 12)"
    )
    assert result["valid"] is False
    assert result["layout"] == DEFAULTS
    assert "overlapping" in result["reason"]


def test_new_widget_merges_without_moving_valid_saved_widgets():
    saved = {
        "version": 1,
        "widgets": [
            {"id": "alpha", "x": 3, "y": 5, "w": 6, "h": 3, "collapsed": False}
        ],
    }
    result = _node(
        f"dashboard.validateAndMergeLayout({json.dumps(saved)}, defaults, specs, 12)"
    )
    assert result["valid"] is True
    alpha = next(item for item in result["layout"] if item["id"] == "alpha")
    beta = next(item for item in result["layout"] if item["id"] == "beta")
    assert alpha == saved["widgets"][0]
    assert beta["id"] == "beta"
    assert not (alpha["x"] < beta["x"] + beta["w"] and alpha["x"] + alpha["w"] > beta["x"]
                and alpha["y"] < beta["y"] + (1 if beta["collapsed"] else beta["h"])
                and alpha["y"] + alpha["h"] > beta["y"])


def test_drag_resize_reset_and_collapse_all_persist_only_layout_state():
    assert LAYOUT_JS.count("persistLayout();") >= 3
    assert "function endPointer(event)" in LAYOUT_JS
    assert 'activePointer.type === "drag"' in LAYOUT_JS
    assert 'activePointer.type === "resize"' not in LAYOUT_JS  # resize is the guarded else branch
    assert "function resetLayout()" in LAYOUT_JS
    assert "layout = cloneLayout(defaults);" in LAYOUT_JS
    assert "function toggleCollapsed(id)" in LAYOUT_JS
    assert "collapsed: item.collapsed === true" in LAYOUT_JS
    assert 'storageKey: LIVING_DASHBOARD_STORAGE_KEY' in APP_JS
    assert '"extractallorder.dashboard.layout.v1"' in APP_JS
    for unsafe_fragment in (
        "/approve",
        "/status",
        "/extract",
        "/orders",
        "/telegram-files",
        "DELETE",
        "POST",
        "PUT",
    ):
        assert unsafe_fragment not in LAYOUT_JS


def test_mobile_is_fixed_single_column_and_editing_is_desktop_only():
    mobile = CSS.split("@media (max-width:1024px)", 1)[1]
    assert "display:flex;" in mobile
    assert "flex-direction:column;" in mobile
    assert "left:auto !important;" in mobile
    assert "width:100% !important;" in mobile
    assert "height:auto !important;" in mobile
    assert "display:none !important;" in mobile
    assert 'window.matchMedia("(min-width: 1025px)").matches' in LAYOUT_JS


def test_widget_failures_are_isolated_and_retryable():
    assert "function renderOverviewWidgetSafely(widgetId, renderWidget)" in APP_JS
    assert "livingDashboardState.controller?.showWidgetError(widgetId)" in APP_JS
    assert 'error.innerHTML = `<strong>${escapeText(spec.title)}</strong>' in LAYOUT_JS
    assert 'options.onRetry?.(spec.id)' in LAYOUT_JS
    for widget_id in (
        "today-overview",
        "order-counts",
        "needs-attention",
        "recent-orders",
    ):
        assert f'renderOverviewWidgetSafely("{widget_id}"' in APP_JS


def test_polling_is_visibility_aware_deduplicated_and_cleaned_up():
    assert "const OVERVIEW_POLL_INTERVAL_MS = 45000;" in APP_JS
    assert "if (!overviewDashboard || overviewLoading) return;" in APP_JS
    assert 'document.addEventListener("visibilitychange", handleOverviewVisibilityChange)' in APP_JS
    assert 'document.removeEventListener("visibilitychange", handleOverviewVisibilityChange)' in APP_JS
    assert "function stopOverviewPolling()" in APP_JS
    assert "window.clearTimeout(livingDashboardState.pollingTimer)" in APP_JS
    assert 'window.addEventListener("beforeunload", destroyLivingDashboard, { once: true })' in APP_JS


def test_existing_overview_actions_remain_bound():
    assert 'document.getElementById("overviewNewOrder")?.addEventListener("click"' in APP_JS
    assert 'document.getElementById("overviewViewOrders")?.addEventListener("click"' in APP_JS
    assert 'document.getElementById("overviewRefresh")?.addEventListener("click"' in APP_JS
    assert 'event.target.closest("[data-overview-route]")' in APP_JS
    assert 'event.target.closest("[data-overview-order-id]")' in APP_JS


def test_dashboard_scripts_parse_without_a_frontend_build_step():
    for script in (LAYOUT_JS_PATH, ROOT / "docs" / "js" / "app.js"):
        subprocess.run(["node", "--check", str(script)], cwd=ROOT, check=True)
