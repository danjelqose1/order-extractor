from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from backend.agents.repair_orchestrator import repair_suspicious_row
from backend.agents.skills.extraction_diagnostics import diagnose_extraction_row_issue
from backend.agents.skills.family_pattern import (
    FAMILY_MISMATCH_CODE,
    analyze_dimension_family,
    attach_family_pattern_diagnostic,
)


def test_family_pattern_suggests_near_original_dimension():
    row = {
        "row_id": "row-1",
        "position": "1-1",
        "type": "LOWE",
        "dimension": "273x792",
        "quantity": 1,
        "area": 0.216,
    }

    result = analyze_dimension_family(
        row,
        order_context={
            "original_order_rows": [
                {
                    "row_id": "row-1",
                    "position": "1-1",
                    "type": "LOWE",
                    "dimension": "278x796",
                    "quantity": 1,
                    "area": 0.221,
                }
            ]
        },
    )

    assert result["success"] is True
    assert result["suggested_value"] == "278x796"
    assert 0.65 <= result["confidence"] <= 0.85
    assert result["safe_to_auto_apply"] is False


def test_family_pattern_diagnostic_attaches_warning_issue():
    row = {"row_id": "row-2", "position": "2-1", "type": "LOWE", "dimension": "273x792", "quantity": 1, "area": 0.216}
    diagnostics = diagnose_extraction_row_issue(row)

    result = attach_family_pattern_diagnostic(
        row,
        diagnostics,
        order_context={
            "original_order_rows": [
                {"row_id": "row-2", "position": "2-1", "type": "LOWE", "dimension": "278x796", "quantity": 1, "area": 0.221}
            ]
        },
    )

    assert result["severity"] == "warning"
    assert FAMILY_MISMATCH_CODE in [issue["code"] for issue in result["issues"]]
    assert result["requires_human_review"] is True


def test_unique_valid_dimension_does_not_warn():
    row = {"row_id": "row-3", "position": "3-1", "type": "LOWE", "dimension": "912x1344", "quantity": 1, "area": 1.226}

    result = analyze_dimension_family(
        row,
        order_rows=[
            {"position": "3-2", "type": "LOWE", "dimension": "420x650", "quantity": 1, "area": 0.273},
            {"position": "3-3", "type": "CLEAR", "dimension": "1100x780", "quantity": 1, "area": 0.858},
        ],
    )

    assert result["success"] is False
    assert result["suspicious"] is False
    assert result["suggested_value"] is None


def test_single_order_wide_near_dimension_without_context_does_not_warn():
    row = {"row_id": "row-3b", "position": "3-1", "type": "LOWE", "dimension": "273x792", "quantity": 1, "area": 0.216}

    result = analyze_dimension_family(
        row,
        order_rows=[
            {"position": "9-9", "type": "LOWE", "dimension": "278x796", "quantity": 1, "area": 0.221},
        ],
    )

    assert result["success"] is False
    assert result["suspicious"] is False


def test_multiple_weak_family_candidates_returns_no_suggestion():
    row = {"row_id": "row-4", "position": "4-1", "type": "LOWE", "dimension": "273x792", "quantity": 1, "area": 0.216}

    result = analyze_dimension_family(
        row,
        nearby_rows=[
            {"position": "4-2", "type": "LOWE", "dimension": "278x796", "quantity": 1, "area": 0.221},
            {"position": "4-3", "type": "LOWE", "dimension": "268x788", "quantity": 1, "area": 0.211},
        ],
    )

    assert result["success"] is False
    assert result["suggested_value"] is None
    assert "Multiple near dimension family candidates" in result["reasoning"]


def test_same_glass_type_support_increases_confidence():
    row = {"row_id": "row-5", "position": "5-1", "type": "LOWE", "dimension": "273x792", "quantity": 1, "area": 0.216}
    different_type = analyze_dimension_family(
        row,
        nearby_rows=[{"position": "5-2", "type": "CLEAR", "dimension": "278x796", "quantity": 1, "area": 0.221}],
    )
    same_type = analyze_dimension_family(
        row,
        nearby_rows=[{"position": "5-2", "type": "LOWE", "dimension": "278x796", "quantity": 1, "area": 0.221}],
    )

    assert same_type["confidence"] > different_type["confidence"]


def test_repair_orchestrator_returns_family_pattern_suggestion_without_mutating_inputs():
    row = {"row_id": "row-6", "position": "6-1", "type": "LOWE", "dimension": "273x792", "quantity": 1, "area": 0.216}
    diagnostics = attach_family_pattern_diagnostic(
        row,
        diagnose_extraction_row_issue(row),
        order_context={
            "original_order_rows": [
                {"row_id": "row-6", "position": "6-1", "type": "LOWE", "dimension": "278x796", "quantity": 1, "area": 0.221}
            ]
        },
    )
    original_row = deepcopy(row)
    original_diagnostics = deepcopy(diagnostics)

    result = repair_suspicious_row(
        row=row,
        diagnostics=diagnostics,
        order_context={
            "original_order_rows": [
                {"row_id": "row-6", "position": "6-1", "type": "LOWE", "dimension": "278x796", "quantity": 1, "area": 0.221}
            ]
        },
    )

    assert result["success"] is True
    assert result["method"] == "family_pattern_repair"
    assert result["suggested_value"] == "278x796"
    assert result["safe_to_auto_apply"] is False
    assert row == original_row
    assert diagnostics == original_diagnostics


def test_accept_suggestion_preserves_original_rows_source_context():
    js = Path("docs/js/app.js").read_text(encoding="utf-8")
    start = js.index("function acceptRowSuggestion")
    end = js.index("function keepOriginalSuggestion", start)
    accept_body = js[start:end]

    assert "originalRows" in js
    assert "row[field] = coerceSuggestionValue(field, suggestion.suggested_value);" in accept_body
    assert "originalRows" not in accept_body
