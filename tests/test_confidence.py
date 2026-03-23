from __future__ import annotations

from backend.confidence import (
    annotate_rows_with_confidence,
    compute_order_confidence,
    score_row_confidence,
)


def test_clean_row_gets_high_confidence():
    row = {
        "order_number": "R-25-1001",
        "type": "2 vetri 4F+16+4F 24mm",
        "dimension": "1200x1400",
        "position": "12-1",
        "quantity": 1,
        "area": 1.68,
        "flags": [],
    }
    score, reasons = score_row_confidence(row, [])
    assert score >= 0.90
    assert reasons == []


def test_repaired_truncation_row_gets_low_confidence():
    row = {
        "order_number": "R-25-1001",
        "type": "2 vetri 4F+16+4F 24mm",
        "dimension": "337x102",
        "position": "20-3/KATI",
        "quantity": 1,
        "area": 0.350,
        "flags": ["AREA_DIMENSION_MISMATCH", "SUSPICIOUS_TRUNCATION"],
        "correctedDimension": "337x1039",
        "dimension_source": "raw_repair",
    }
    score, reasons = score_row_confidence(row, [])
    assert score < 0.75
    assert "area_dimension_mismatch" in reasons
    assert "suspicious_truncation" in reasons
    assert "dimension_repair_applied" in reasons


def test_missing_quantity_row_gets_low_confidence():
    row = {
        "order_number": "R-25-1001",
        "type": "2 vetri 4F+16+4F 24mm",
        "dimension": "",
        "position": "",
        "quantity": 1,
        "area": 0.0,
        "flags": [],
    }
    score, reasons = score_row_confidence(row, ["auto_fix: quantity_defaulted_to_one"])
    assert score < 0.75
    assert "quantity_missing_or_defaulted" in reasons
    assert "weak_position_format" in reasons
    assert "weak_dimension_format" in reasons


def test_order_confidence_decreases_when_multiple_rows_are_suspicious():
    clean_rows, clean_stats = annotate_rows_with_confidence(
        [
            {
                "order_number": "R-25-1001",
                "type": "2 vetri 4F+16+4F 24mm",
                "dimension": "1200x1400",
                "position": "12-1",
                "quantity": 1,
                "area": 1.68,
                "flags": [],
            },
            {
                "order_number": "R-25-1001",
                "type": "2 vetri 4F+16+4F 24mm",
                "dimension": "1000x1000",
                "position": "13-1",
                "quantity": 1,
                "area": 1.0,
                "flags": [],
            },
        ],
        row_warnings={},
    )
    suspicious_rows, suspicious_stats = annotate_rows_with_confidence(
        [
            {
                "order_number": "R-25-1001",
                "type": "2 vetri 4F+16+4F 24mm",
                "dimension": "337x102",
                "position": "20-3/KATI",
                "quantity": 1,
                "area": 0.350,
                "flags": ["AREA_DIMENSION_MISMATCH", "SUSPICIOUS_TRUNCATION"],
                "correctedDimension": "337x1039",
                "dimension_source": "raw_repair",
            },
            {
                "order_number": "R-25-1001",
                "type": "2 vetri 4F+16+4F 24mm",
                "dimension": "",
                "position": "",
                "quantity": 1,
                "area": 0.0,
                "flags": [],
            },
        ],
        row_warnings={1: ["auto_fix: quantity_defaulted_to_one"]},
    )

    clean_order = compute_order_confidence(
        rows=clean_rows,
        warnings=[],
        declared_units=None,
        declared_area=None,
        parsed_units=2,
        parsed_area=2.68,
        expected_declared_totals=False,
        repaired_row_count=clean_stats["repaired_row_count"],
    )
    suspicious_order = compute_order_confidence(
        rows=suspicious_rows,
        warnings=["declared_units_mismatch: declared 1, parsed 2"],
        declared_units=1,
        declared_area=0.2,
        parsed_units=2,
        parsed_area=0.35,
        expected_declared_totals=True,
        repaired_row_count=suspicious_stats["repaired_row_count"],
    )

    assert suspicious_order["order_confidence_score"] < clean_order["order_confidence_score"]
    assert suspicious_order["order_confidence_label"] in {"low", "medium"}
