from __future__ import annotations

from copy import deepcopy

from backend.agents.repair_orchestrator import repair_suspicious_row
from backend.agents.skills.extraction_diagnostics import diagnose_extraction_row_issue
from backend.agents.skills.pattern_repair import suggest_pattern_repair


def test_pattern_repair_suggests_truncated_dimension_from_area_and_nearby_rows():
    row = {
        "row_id": "row-1",
        "position": "1-1",
        "type": "LOWE",
        "dimension": "738x61",
        "quantity": 1,
        "area": 0.45,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = suggest_pattern_repair(
        row,
        diagnostics=diagnostics,
        nearby_rows=[
            {"position": "1-2", "type": "LOWE", "dimension": "738x613", "quantity": 1, "area": 0.452},
            {"position": "1-3", "type": "LOWE", "dimension": "738x613", "quantity": 1, "area": 0.452},
        ],
    )

    assert result["success"] is True
    assert result["suggested_value"] == "738x613"
    assert result["confidence"] >= 0.85
    assert result["safe_to_auto_apply"] is False
    assert "Area 0.45 matches 738x613" in result["trace"]


def test_orchestrator_returns_repeated_nearby_dimension_suggestion():
    row = {
        "row_id": "row-2",
        "position": "A-4",
        "type": "VETRI CAMERA",
        "dimension": "500x100",
        "quantity": 2,
        "area": 1.0,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = repair_suspicious_row(
        row=row,
        diagnostics=diagnostics,
        nearby_rows=[
            {"position": "A-1", "type": "VETRI CAMERA", "dimension": "500x1000", "quantity": 2, "area": 1.0},
            {"position": "A-2", "type": "VETRI CAMERA", "dimension": "500x1000", "quantity": 2, "area": 1.0},
            {"position": "A-3", "type": "VETRI CAMERA", "dimension": "500x1000", "quantity": 2, "area": 1.0},
        ],
    )

    assert result["success"] is True
    assert result["target_field"] == "dimension"
    assert result["original_value"] == "500x100"
    assert result["suggested_value"] == "500x1000"
    assert result["methods_used"] == ["diagnostics_analyzer", "pattern_repair"]
    assert result["safe_to_auto_apply"] is False


def test_orchestrator_low_confidence_returns_manual_review_without_coordinates():
    row = {
        "row_id": "row-3",
        "position": "B-1",
        "type": "LOWE",
        "dimension": "738x61",
        "quantity": 1,
        "area": 0.45,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = repair_suspicious_row(
        row=row,
        diagnostics=diagnostics,
        nearby_rows=[
            {"position": "B-2", "type": "LOWE", "dimension": "800x900", "quantity": 1, "area": 0.72},
        ],
    )

    assert result["success"] is False
    assert result["suggested_value"] is None
    assert result["recommended_action"] == "MANUAL_REVIEW"
    assert "OCR fallback skipped because row_location is unavailable" in result["trace"]
    assert result["safe_to_auto_apply"] is False


def test_orchestrator_skips_ocr_fallback_on_high_confidence_pattern_repair():
    row = {
        "row_id": "row-4",
        "position": "C-1",
        "type": "LOWE",
        "dimension": "738x124",
        "quantity": 1,
        "area": 0.918,
        "row_location": {
            "page": 1,
            "bbox": {"x0": 10, "y0": 10, "x1": 120, "y1": 25},
            "source": "pdf_text_layer",
            "confidence": 0.9,
            "matched_text": "C-1 LOWE 738x999 1 0.918",
        },
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = repair_suspicious_row(
        row=row,
        diagnostics=diagnostics,
        nearby_rows=[
            {"position": "C-2", "type": "LOWE", "dimension": "738x1243", "quantity": 1, "area": 0.918},
            {"position": "C-3", "type": "LOWE", "dimension": "738x1243", "quantity": 1, "area": 0.918},
            {"position": "C-4", "type": "LOWE", "dimension": "738x1243", "quantity": 1, "area": 0.918},
        ],
    )

    assert result["success"] is True
    assert result["suggested_value"] == "738x1243"
    assert result["methods_used"] == ["diagnostics_analyzer", "pattern_repair"]
    assert "OCR fallback skipped" in result["trace"]


def test_orchestrator_trace_generation_and_no_automatic_mutation():
    row = {
        "row_id": "row-5",
        "position": "D-1",
        "type": "LOWE",
        "dimension": "738x124",
        "quantity": 1,
        "area": 0.918,
    }
    diagnostics = diagnose_extraction_row_issue(row)
    original_row = deepcopy(row)
    original_diagnostics = deepcopy(diagnostics)

    result = repair_suspicious_row(
        row=row,
        diagnostics=diagnostics,
        nearby_rows=[
            {"position": "D-2", "type": "LOWE", "dimension": "738x1243", "quantity": 1, "area": 0.918},
            {"position": "D-3", "type": "LOWE", "dimension": "738x1243", "quantity": 1, "area": 0.918},
        ],
    )

    assert row == original_row
    assert diagnostics == original_diagnostics
    assert result["safe_to_auto_apply"] is False
    assert result["trace"][0].startswith("Detected")
    assert any(step.startswith("Pattern repair confidence") for step in result["trace"])
