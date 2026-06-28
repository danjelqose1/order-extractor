from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from analytics_summary import build_analysis_summary


def _order(order_id, created_at, client, rows):
    return {
        "id": order_id,
        "created_at": created_at,
        "client": client,
        "order_numbers": [f"R-{order_id}"],
        "rows": rows,
    }


def test_summary_sorts_top_metrics_and_uses_quantity_for_area_fallback():
    orders = [
        _order(
            1,
            "2026-06-10T08:00:00Z",
            "Client A",
            [{"type": "Clear", "dimension": "1000x1000", "quantity": 1, "area": 1.0}],
        ),
        _order(
            2,
            "2026-06-11T08:00:00Z",
            "Client B",
            [{"type": "LowE", "dimension": "500x1000", "quantity": 18, "area": 0}],
        ),
    ]

    summary = build_analysis_summary(
        orders,
        start_date="2026-06-10",
        end_date="2026-06-11",
        compare_previous=False,
    )

    assert summary["kpis"]["orders"] == 2
    assert summary["kpis"]["units"] == 19
    assert summary["kpis"]["area_m2"] == 10
    assert summary["topClients"][0]["client"] == "Client B"
    assert summary["topGlassTypes"][0]["type"] == "LowE"
    assert summary["topDimensions"][0]["dimension"] == "500 × 1000"
    concentration = next(
        signal
        for signal in summary["insights"]
        if signal["type"] == "client_concentration"
    )
    assert concentration["meta"]["client"] == "Client B"


def test_summary_builds_previous_period_and_zero_filled_daily_series():
    orders = [
        _order(
            1,
            "2026-05-31T08:00:00Z",
            "Client A",
            [{"type": "Clear", "dimension": "1000x1000", "quantity": 1, "area": 1.0}],
        ),
        _order(
            2,
            "2026-06-02T08:00:00Z",
            "Client A",
            [{"type": "Clear", "dimension": "1000x1000", "quantity": 2, "area": 2.0}],
        ),
    ]

    summary = build_analysis_summary(
        orders,
        start_date="2026-06-01",
        end_date="2026-06-03",
        compare_previous=True,
    )

    assert summary["previousPeriod"] == {
        "start": "2026-05-29",
        "end": "2026-05-31",
    }
    assert [item["orders"] for item in summary["current"]["daily"]] == [0, 1, 0]
    assert summary["current"]["totals"]["avg_orders_per_day"] == 1 / 3
    assert summary["previous"]["totals"]["orders"] == 1


def test_all_time_scope_disables_period_comparison():
    summary = build_analysis_summary(
        [
            _order(
                1,
                "2026-06-10T08:00:00Z",
                "Client A",
                [{"type": "Clear", "dimension": "1000x1000", "quantity": 1, "area": 1.0}],
            )
        ],
        compare_previous=True,
        all_time=True,
    )

    assert summary["period"] == {"start": "", "end": ""}
    assert summary["previous"] is None
    assert summary["filters"]["compareToPrevious"] is False
    assert summary["filters"]["allTime"] is True


def test_frontend_uses_server_summary_and_has_actionable_controls():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "docs" / "js" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "docs" / "css" / "styles.css").read_text(encoding="utf-8")

    assert 'id="analysisRangePresets"' in html
    assert 'data-analysis-range="90"' in html
    assert 'id="analysisExport"' in html
    assert 'id="analysisRefresh">Apply filters' in html
    assert 'fetch(API_BASE + "/analysis/summary?"' in js
    assert "function renderAnalysisLoadingState()" in js
    assert "function exportAnalysisSnapshot()" in js
    assert ".analysis-chart-loading" in css
