from __future__ import annotations

from backend.area_dimension_validator import apply_area_dimension_validation


def test_area_dimension_mismatch_suggests_height_for_337x102():
    rows = [
        {"dimension": "337x102", "area": 0.350},
    ]
    validated = apply_area_dimension_validation(rows)
    row = validated[0]

    assert "AREA_DIMENSION_MISMATCH" in row.get("flags", [])
    assert "SUSPICIOUS_TRUNCATION" in row.get("flags", [])
    assert "SUGGESTED_DIMENSION_FROM_AREA" in row.get("flags", [])
    assert row.get("suggested_height") == 1039
    assert row.get("suggested_dimension") == "337x1039"
    assert row.get("area_final") == 0.350


def test_area_dimension_mismatch_suggests_height_for_704x230():
    rows = [
        {"dimension": "704x230", "area": 1.620},
    ]
    validated = apply_area_dimension_validation(rows)
    row = validated[0]

    assert "AREA_DIMENSION_MISMATCH" in row.get("flags", [])
    assert "SUSPICIOUS_TRUNCATION" in row.get("flags", [])
    assert "SUGGESTED_DIMENSION_FROM_AREA" in row.get("flags", [])
    assert row.get("suggested_height") == 2301
    assert row.get("suggested_dimension") == "704x2301"
    assert row.get("area_final") == 1.620


def test_area_dimension_match_has_no_flags():
    rows = [
        {"dimension": "625x2232", "area": 1.390},
    ]
    validated = apply_area_dimension_validation(rows)
    row = validated[0]

    assert row.get("flags") == []
    assert row.get("area_final") == 1.390
