from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from backend.agents.skills.extraction_diagnostics import (
    diagnose_extraction_row_issue,
    ocr_fallback_row_repair,
)


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "docs/js/app.js").read_text()


def _function(name):
    return re.search(rf"^function {name}\([^\n]*\)[\s\S]*?^}}", JS, re.M).group()


def _node(body):
    names = [
        "getRowDiagnostics", "isCriticalFieldMissing", "repairKeyForField",
        "rowHasOcrRepair", "prefillRecoveredDimension", "rowWarningsAfterDimensionPrefill",
        "clearOcrRepairForField", "refreshRowRepairWarning", "criticalFieldLabel",
        "normalizeRepairValue", "applyOcrFallbackResultToRow", "buildHistoryRowsPayload",
        "applyExtractionResult", "renderOrderDetail",
    ]
    constants = JS[JS.index("const OCR_REPAIR_CONFIDENCE_MIN"):JS.index("function criticalFieldLabel")]
    harness = '''
const appState = { extract: {}, historyDetail: {} };
const historyState = {};
const analysisState = { orderRowCache: new Map() };
const extractSourceMeta = null, historyRawTextEl = null;
function updateExtractUI() {}
function updateHistoryDetailUI() {}
function renderOrderDetailEnhancements() {}
function renderHistoryStatusTimeline() {}
function getClientName() { return "Test"; }
function showSavedToast() {}
'''
    script = constants + "\n" + "\n".join(_function(name) for name in names) + harness + body
    result = subprocess.run(["node", "-e", script], cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def _row(**changes):
    return {
        "order_number": "R-26-0781", "position": "1-1", "type": "LOWE",
        "dimension": "", "quantity": 1, "area": 0.73,
        "dimension_repaired": "632x1157", "repair_confidence": 0.84,
        "raw_base64_value": {"dimension": ""}, "raw_ocr_value": {"dimension": "632x1157"},
        "diagnostics": {"severity": "error", "issues": [
            {"code": "MISSING_DIMENSION", "severity": "error"},
            {"code": "POSITION_WARNING", "severity": "warning"},
        ]},
        **changes,
    }


@pytest.mark.parametrize("position,dimension", [
    ("1-1", "632x1157"), ("2-1", "637x1098"), ("3-1", "632x1140"),
    ("4-1", "827x1150"), ("9-1", "682x1077"),
])
def test_recovered_order_dimensions_reach_both_review_editors_without_changing_source(position, dimension):
    row = _row(position=position, dimension_repaired=dimension)
    result = _node(f'''
const source = {{status: "draft", rows: [{json.dumps(row)}], draft_order_id: 656,
    row_warnings: {{0: ["warning: missing_required_field:dimension", "position review"]}}}};
const before = JSON.stringify(source);
applyExtractionResult(source);
historyState.selectedOrder = source;
renderOrderDetail();
process.stdout.write(JSON.stringify({{sourceUnchanged: before === JSON.stringify(source),
    extract: appState.extract, history: appState.historyDetail,
    approvalPayload: buildHistoryRowsPayload(appState.historyDetail.rows)}}));
''')
    assert result["sourceUnchanged"]
    for bucket in (result["extract"], result["history"]):
        assert bucket["originalRows"][0]["dimension"] == ""
        edited = bucket["rows"][0]
        assert edited["dimension"] == dimension
        assert edited["dimension_prefilled"] is True
        assert edited["raw_base64_value"]["dimension"] == ""
        assert [issue["code"] for issue in edited["diagnostics"]["issues"]] == [
            "POSITION_WARNING", "RECOVERED_DIMENSION_REVIEW",
        ]
        assert bucket["rowWarnings"][edited["_rid"]] == ["position review"]
    assert result["approvalPayload"][0]["dimension"] == dimension
    assert "dimension_prefilled" not in result["approvalPayload"][0]
    assert result["extract"]["status"] == "draft"


@pytest.mark.parametrize("status,changes", [
    ("approved", {}), ("in_production", {}), ("completed", {}), ("", {}),
    ("draft", {"dimension": "700x1100"}),
    ("draft", {"repair_confidence": 0.79}), ("draft", {"repair_confidence": None}),
    ("draft", {"repair_confidence": 2}), ("draft", {"repair_warnings": {"dimension": "Ambiguous"}}),
    ("draft", {"dimension_repaired": "632x1157 or 637x1098"}),
    ("draft", {"dimension_repaired": "632x0000"}), ("draft", {"dimension_repaired": "632"}),
])
def test_prefill_preserves_protected_orders_existing_values_and_uncertain_repairs(status, changes):
    row = _row(**changes)
    result = _node(f'''
const row = {json.dumps(row)};
const before = JSON.stringify(row);
const filled = prefillRecoveredDimension(row, {json.dumps(status)});
process.stdout.write(JSON.stringify({{filled, unchanged: before === JSON.stringify(row)}}));
''')
    assert result == {"filled": False, "unchanged": True}


def test_reviewed_draft_normalizes_separators_without_swapping_axes():
    result = _node(f'''
const row = {json.dumps(_row(dimension_repaired="1157 × 632"))};
prefillRecoveredDimension(row, "reviewed");
process.stdout.write(JSON.stringify(row));
''')
    assert result["dimension"] == "1157x632"


def test_ocr_recheck_prefills_and_manual_edit_clears_recovered_marker():
    result = _node(f'''
const row = {json.dumps(_row(dimension_repaired=None))};
applyOcrFallbackResultToRow(row, "dimension", {{success: true, suggested_value: "637x1098", confidence: 0.88}}, "draft");
const recovered = row.dimension;
row.dimension = "640x1100";
clearOcrRepairForField(row, "dimension");
process.stdout.write(JSON.stringify({{recovered, row}}));
''')
    assert result["recovered"] == "637x1098"
    assert result["row"]["dimension"] == "640x1100"
    assert "dimension_repaired" not in result["row"]
    assert "dimension_prefilled" not in result["row"]


@pytest.mark.parametrize("reading,expected", [
    ("632x1157", "632x1157"), ("632x1157\n632 x 1157", "632x1157"),
    ("632x1157\n637x1098", None), ("NO_VALUE (possibly 632x1157)", None),
    ("NO_VALUE", None),
])
def test_page_repair_rejects_competing_dimension_readings(reading, expected):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "R-26-0781 1-1 LOWE 1 0.730")
    pdf = doc.tobytes()
    doc.close()
    row = _row()
    result = ocr_fallback_row_repair(
        row, diagnose_extraction_row_issue(row), target_field="dimension",
        pdf_bytes=pdf, row_index=0, order_context={"order_rows": [row]},
        openai_vision_repair_fn=lambda **kwargs: {"text": reading, "confidence": 0.9},
    )
    assert result["success"] is (expected is not None)
    assert result["suggested_value"] == expected
    assert result["safe_to_auto_apply"] is False

