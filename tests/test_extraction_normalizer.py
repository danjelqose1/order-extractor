from backend.extraction_normalizer import DIMENSION_IN_TYPE_WARNING, normalizeExtractedRow


def _row(type_value: str, dimension: str = "522x1262"):
    return {
        "order_number": "R-26-0379",
        "type": type_value,
        "dimension": dimension,
        "position": "1-1",
        "quantity": 1,
        "area": 0.66,
    }


def test_removes_spaced_dimension_matching_row_dimension():
    cleaned = normalizeExtractedRow(
        _row("2 VETRI 33.1 SATINAT +14+33.1 LOWE 522 x 1262 C.CALDO 28mm")
    )

    assert cleaned["type"] == "2 VETRI 33.1 SATINAT +14+33.1 LOWE C.CALDO 28mm"
    assert cleaned["dimension"] == "522x1262"
    assert cleaned["_normalization_warnings"] == [DIMENSION_IN_TYPE_WARNING]


def test_removes_multiplication_sign_dimension_matching_row_dimension():
    cleaned = normalizeExtractedRow(
        _row("2 VETRI 33.1 SATINAT +14+33.1 LOWE 522×1262 C.CALDO 28mm")
    )

    assert cleaned["type"] == "2 VETRI 33.1 SATINAT +14+33.1 LOWE C.CALDO 28mm"
    assert cleaned["dimension"] == "522x1262"
    assert cleaned["_normalization_warnings"] == [DIMENSION_IN_TYPE_WARNING]


def test_leaves_clean_type_unchanged():
    original = "2 VETRI 33.1 SATINAT +14+33.1 LOWE C.CALDO 28mm"
    cleaned = normalizeExtractedRow(_row(original))

    assert cleaned["type"] == original
    assert cleaned["dimension"] == "522x1262"
    assert "_normalization_warnings" not in cleaned


def test_does_not_remove_spacers_or_thickness_values():
    original = "2 VETRI 33.1 SATINAT +14+33.1 LOWE +14 C.CALDO 28mm"
    cleaned = normalizeExtractedRow(_row(original))

    assert cleaned["type"] == original
    assert "+14" in cleaned["type"]
    assert "28mm" in cleaned["type"]
    assert "_normalization_warnings" not in cleaned
