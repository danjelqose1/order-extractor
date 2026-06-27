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


def test_overview_gates_the_new_order_workspace():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

    assert 'id="overviewDashboard"' in html
    assert 'id="overviewNewOrder"' in html
    assert 'id="newOrderWorkspace" class="new-order-workspace" hidden' in html
    assert "function setNewOrderWorkspaceOpen" in js
    assert "function loadOverview" in js
