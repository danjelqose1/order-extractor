from __future__ import annotations

from copy import deepcopy

from backend.agents.skills.extraction_diagnostics import (
    diagnose_extraction_row_issue,
    diagnose_extraction_row_warning,
)


def test_valid_row_returns_ok():
    row = {
        "row_id": "row-1",
        "order_number": "R-26-0001",
        "client": "Client",
        "position": "1-1",
        "glass_type": "VETRI CAMERA",
        "dimension": "1200x1400",
        "quantity": 1,
        "area": 1.68,
    }

    result = diagnose_extraction_row_issue(row)

    assert result["row_id"] == "row-1"
    assert result["severity"] == "ok"
    assert result["issues"] == []
    assert result["computed"]["calculated_area"] == 1.68
    assert result["computed"]["extracted_area"] == 1.68
    assert result["computed"]["difference"] == 0.0
    assert result["requires_human_review"] is False


def test_missing_dimension_returns_error():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-2",
            "position": "1-2",
            "dimension": "",
            "quantity": 1,
            "area": 1.0,
        }
    )

    assert result["severity"] == "error"
    assert result["requires_human_review"] is True
    assert [issue["code"] for issue in result["issues"]] == ["MISSING_DIMENSION"]


def test_area_mismatch_returns_warning():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-3",
            "position": "1-3",
            "dimension": "500x1000",
            "quantity": 1,
            "area": 0.75,
        }
    )

    assert result["severity"] == "warning"
    assert result["requires_human_review"] is True
    assert "AREA_MISMATCH" in [issue["code"] for issue in result["issues"]]
    assert result["computed"]["calculated_area"] == 0.5
    assert result["computed"]["extracted_area"] == 0.75
    assert result["computed"]["difference"] == 0.25


def test_quantity_aware_area_calculation_works_correctly():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-4",
            "position": "1-4",
            "dimension": "500x1000",
            "quantity": 3,
            "area": 1.5,
        }
    )

    assert result["severity"] == "ok"
    assert result["computed"]["calculated_area"] == 1.5
    assert result["computed"]["difference"] == 0.0


def test_quantity_greater_than_one_does_not_compare_against_single_piece_area():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-5",
            "position": "1-5",
            "dimension": "500x1000",
            "quantity": 2,
            "area": 1.0,
        }
    )

    assert result["severity"] == "ok"
    assert result["issues"] == []
    assert result["computed"]["calculated_area"] == 1.0


def test_factory_total_area_472x1413_qty_2_returns_ok():
    row = {
        "row_id": "row-5a",
        "position": "1-5a",
        "dimension": "472x1413",
        "quantity": 2,
        "area": 1.34,
    }
    original = deepcopy(row)

    result = diagnose_extraction_row_issue(row)

    assert result["severity"] == "ok"
    assert result["issues"] == []
    assert result["computed"]["calculated_area"] == 1.334
    assert row == original


def test_factory_total_area_472x761_qty_2_returns_ok():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-5b",
            "position": "1-5b",
            "dimension": "472x761",
            "quantity": 2,
            "area": 0.72,
        }
    )

    assert result["severity"] == "ok"
    assert result["issues"] == []
    assert result["computed"]["calculated_area"] == 0.718


def test_factory_total_area_272x1202_qty_1_returns_ok():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-5c",
            "position": "1-5c",
            "dimension": "272x1202",
            "quantity": 1,
            "area": 0.33,
        }
    )

    assert result["severity"] == "ok"
    assert result["issues"] == []
    assert result["computed"]["calculated_area"] == 0.327


def test_single_piece_area_for_quantity_two_returns_warning():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-5d",
            "position": "1-5d",
            "dimension": "472x1413",
            "quantity": 2,
            "area": 0.67,
        }
    )

    assert result["severity"] == "warning"
    assert "AREA_MISMATCH" in [issue["code"] for issue in result["issues"]]
    assert result["computed"]["calculated_area"] == 1.334


def test_invalid_dimension_text_returns_error():
    result = diagnose_extraction_row_issue(
        {
            "row_id": "row-6",
            "position": "1-6",
            "dimension": "not a dimension",
            "quantity": 1,
            "area": 1.0,
        }
    )

    assert result["severity"] == "error"
    assert [issue["code"] for issue in result["issues"]] == ["INVALID_DIMENSION_FORMAT"]


def test_diagnostics_do_not_mutate_original_row():
    row = {
        "row_id": "row-7",
        "position": "1-7",
        "dimension": "337x102",
        "quantity": 1,
        "area": 0.35,
        "manual_override": {"dimension": "337x1039"},
    }
    original = deepcopy(row)

    diagnose_extraction_row_issue(row)

    assert row == original


def test_area_mismatch_diagnosis():
    row = {
        "row_id": "row-8",
        "position": "1-8",
        "dimension": "472x1413",
        "quantity": 2,
        "area": 0.67,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = diagnose_extraction_row_warning(row, diagnostics)

    assert result["severity"] == "warning"
    assert result["recommended_action"] == "CHECK_AREA"
    assert "single-piece area" in result["likely_cause"]
    assert result["safe_to_auto_fix"] is False
    assert result["suggested_fix"] is None


def test_missing_dimension_diagnosis():
    row = {
        "row_id": "row-9",
        "position": "1-9",
        "dimension": "",
        "quantity": 1,
        "area": 0.67,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = diagnose_extraction_row_warning(row, diagnostics)

    assert result["severity"] == "error"
    assert result["recommended_action"] == "OCR_FALLBACK_DIMENSION"
    assert "Dimension is missing" in result["summary"]
    assert result["safe_to_auto_fix"] is False


def test_valid_row_diagnosis_returns_ok_no_issue():
    row = {
        "row_id": "row-10",
        "position": "1-10",
        "dimension": "472x1413",
        "quantity": 2,
        "area": 1.34,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = diagnose_extraction_row_warning(row, diagnostics)

    assert result["severity"] == "ok"
    assert result["recommended_action"] == "MANUAL_REVIEW"
    assert "No extraction issue" in result["summary"]
    assert result["safe_to_auto_fix"] is False


def test_row_diagnosis_does_not_mutate_input():
    row = {
        "row_id": "row-11",
        "position": "1-11",
        "dimension": "337x102",
        "quantity": 1,
        "area": 0.35,
    }
    diagnostics = diagnose_extraction_row_issue(row)
    original_row = deepcopy(row)
    original_diagnostics = deepcopy(diagnostics)

    diagnose_extraction_row_warning(row, diagnostics)

    assert row == original_row
    assert diagnostics == original_diagnostics
