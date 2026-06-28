from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "docs" / "js" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "docs" / "css" / "styles.css").read_text(encoding="utf-8")


def test_orders_has_actionable_work_queue_and_batch_controls():
    assert 'id="orderWorkQueueTitle"' in HTML
    for queue in (
        "new",
        "needs_review",
        "ready_approval",
        "ready_production",
        "failed",
        "completed",
    ):
        assert f'data-history-queue="{queue}"' in HTML
    assert 'id="historyBatchApprove"' in HTML
    assert 'id="historyBatchProduction"' in HTML
    assert 'id="historyBatchArchive"' in HTML
    assert "async function loadHistoryWorkQueue(" in JS
    assert "function updateHistorySelectionUI()" in JS
    assert ".order-queue-grid" in CSS
    assert ".history-batch-toolbar" in CSS


def test_global_activity_center_tracks_long_running_work():
    assert 'id="activityCenterToggle"' in HTML
    assert 'id="activityCenter"' in HTML
    assert 'id="activityCenterList"' in HTML
    assert 'data-activity-filter="running"' in HTML
    assert 'data-activity-filter="failed"' in HTML
    assert "function startBackgroundActivity(" in JS
    assert "function completeBackgroundActivity(" in JS
    assert "function failBackgroundActivity(" in JS
    assert 'startBackgroundActivity(forceOcr ? "OCR extraction" : "PDF extraction"' in JS
    assert 'startBackgroundActivity("Label generation"' in JS
    assert 'startBackgroundActivity("Production PDF"' in JS
    assert ".activity-center" in CSS
    assert ".activity-progress" in CSS
