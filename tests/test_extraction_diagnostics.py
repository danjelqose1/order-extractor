from __future__ import annotations

from copy import deepcopy

import pytest

from backend.agents.skills.extraction_diagnostics import (
    attach_pdf_row_locations,
    diagnose_extraction_row_issue,
    diagnose_extraction_row_warning,
    extract_pdf_text_for_row_location,
    ocr_fallback_row_repair,
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


def test_ocr_fallback_pattern_suggests_truncated_dimension():
    row = {
        "row_id": "row-12",
        "position": "1-12",
        "dimension": "738x124",
        "quantity": 1,
        "area": 0.92,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = ocr_fallback_row_repair(
        row,
        diagnostics,
        target_field="dimension",
        order_context={
            "rows_before": [
                {"dimension": "738x1243", "quantity": 1, "area": 0.918},
                {"dimension": "738x1243", "quantity": 1, "area": 0.918},
            ],
            "rows_after": [],
        },
    )

    assert result["success"] is True
    assert result["target_field"] == "dimension"
    assert result["original_value"] == "738x124"
    assert result["suggested_value"] == "738x1243"
    assert result["confidence"] == 0.85
    assert result["method"] == "pattern_fallback_no_pdf_coordinates"
    assert result["safe_to_auto_apply"] is False


def test_ocr_fallback_missing_dimension_does_not_fake_ocr_without_coordinates():
    row = {
        "row_id": "row-13",
        "position": "1-13",
        "dimension": "",
        "quantity": 1,
        "area": 0.92,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = ocr_fallback_row_repair(row, diagnostics, target_field="dimension")

    assert result["success"] is False
    assert result["suggested_value"] is None
    assert result["reason"] == "PDF row coordinates are not available yet"
    assert result["recommended_next_step"] == "STORE_ROW_COORDINATES_DURING_EXTRACTION"
    assert result["safe_to_auto_apply"] is False


def test_ocr_fallback_does_not_mutate_backend_inputs():
    row = {
        "row_id": "row-14",
        "position": "1-14",
        "dimension": "738x124",
        "quantity": 1,
        "area": 0.92,
    }
    diagnostics = diagnose_extraction_row_issue(row)
    original_row = deepcopy(row)
    original_diagnostics = deepcopy(diagnostics)

    ocr_fallback_row_repair(
        row,
        diagnostics,
        target_field="dimension",
        order_context={"rows_before": [{"dimension": "738x1243", "quantity": 1, "area": 0.918}]},
    )

    assert row == original_row
    assert diagnostics == original_diagnostics


def test_ocr_fallback_valid_row_returns_no_fallback():
    row = {
        "row_id": "row-15",
        "position": "1-15",
        "dimension": "738x1243",
        "quantity": 1,
        "area": 0.918,
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = ocr_fallback_row_repair(row, diagnostics, target_field="dimension")

    assert result["success"] is False
    assert result["method"] == "diagnostics_ok_no_fallback"
    assert result["safe_to_auto_apply"] is False


def test_text_layer_row_location_suggests_truncated_dimension():
    row = {
        "row_id": "row-16",
        "position": "1-16",
        "dimension": "738x124",
        "quantity": 1,
        "area": 0.92,
        "row_location": {
            "page": 1,
            "bbox": {"x0": 10.0, "y0": 20.0, "x1": 200.0, "y1": 32.0},
            "source": "pdf_text_layer",
            "confidence": 0.88,
            "matched_text": "1-16 LOWE 738x1243 1 0.92",
        },
    }
    diagnostics = diagnose_extraction_row_issue(row)

    result = ocr_fallback_row_repair(row, diagnostics, target_field="dimension")

    assert result["success"] is True
    assert result["method"] == "pdf_text_layer_row_region"
    assert result["suggested_value"] == "738x1243"
    assert result["safe_to_auto_apply"] is False
    assert result["evidence"]["page"] == 1
    assert result["evidence"]["matched_text"] == "1-16 LOWE 738x1243 1 0.92"


def test_pdf_text_layer_location_is_attached_when_row_matches():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1-17 LOWE 738x1243 1 0.918")
    pdf_bytes = doc.tobytes()
    doc.close()
    row = {
        "row_id": "row-17",
        "position": "1-17",
        "dimension": "738x1243",
        "quantity": 1,
        "area": 0.918,
    }

    result = attach_pdf_row_locations([row], pdf_bytes)

    assert result[0]["row_location"]["page"] == 1
    assert result[0]["row_location"]["source"] == "pdf_text_layer"
    assert "738x1243" in result[0]["row_location"]["matched_text"]
    assert "row_location" not in row


def test_pdf_text_layer_location_is_null_when_no_row_match():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "unrelated PDF line")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = attach_pdf_row_locations(
        [{"position": "1-18", "dimension": "738x1243", "quantity": 1, "area": 0.918}],
        pdf_bytes,
    )

    assert result[0]["row_location"] is None


def test_pdf_text_layer_region_text_uses_row_location_bbox():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1-19 LOWE 738x1243 1 0.918")
    page.insert_text((72, 120), "unrelated 999x999 1 0.998")
    pdf_bytes = doc.tobytes()
    doc.close()
    location = {
        "page": 1,
        "bbox": {"x0": 65.0, "y0": 60.0, "x1": 260.0, "y1": 85.0},
        "source": "pdf_text_layer",
        "confidence": 0.9,
        "matched_text": "",
    }

    result = extract_pdf_text_for_row_location(pdf_bytes, location)

    assert "738x1243" in result
    assert "999x999" not in result
