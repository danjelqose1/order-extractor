from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "docs" / "js" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "docs" / "css" / "styles.css").read_text(encoding="utf-8")


def test_order_detail_is_a_dedicated_lifecycle_page():
    assert 'id="tabOrderDetail"' in HTML
    assert 'id="orderLifecycle"' in HTML
    for view in ("summary", "items", "files", "activity"):
        assert f'data-order-detail-view="{view}"' in HTML
        assert f'data-order-detail-panel="{view}"' in HTML
    assert 'id="orderDetailSourceFrame"' in HTML
    assert 'orderdetail: document.getElementById("tabOrderDetail")' in JS
    assert 'activateTab("orderdetail")' in JS
    assert "function renderOrderDetailEnhancements(order)" in JS
    assert ".order-lifecycle-step" in CSS


def test_extraction_review_keeps_source_and_issues_beside_rows():
    assert 'class="extraction-review-layout"' in HTML
    assert 'id="extractPdfPreview"' in HTML
    assert 'id="extractReviewIssues"' in HTML
    assert 'id="extractPrevIssue"' in HTML
    assert 'id="extractNextIssue"' in HTML
    assert "function collectExtractReviewIssues()" in JS
    assert "function setActiveExtractReviewIssue(" in JS
    assert "showExtractPdfPreview(file)" in JS
    assert ".extraction-review-layout" in CSS
    assert "#tableWrap tr.review-row-active td" in CSS
