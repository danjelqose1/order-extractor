from backend.extraction_normalizer import (
    DIMENSION_IN_TYPE_WARNING,
    normalizeExtractedRow,
    normalize_order_metadata,
)


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


def test_keli_correlated_document_is_primary_order_number():
    source_text = (
        "ORDINE DI VETRO 26-0075\n"
        "DOCUMENTO CORRELATO R - 26-0488\n"
    )
    raw_payload = {
        "order_number": "26-0075",
        "rows": [
            {
                "order_number": "26-0075",
                "position": "R-26-0488/1-1",
            }
        ],
    }

    normalized = normalize_order_metadata(
        raw_payload,
        source_text,
        vendor_format="KELI",
    )

    assert normalized["order_number"] == "R-26-0488"
    assert normalized["supplier_document_number"] == "26-0075"
    assert normalized["rows"][0]["order_number"] == "R-26-0488"
    assert normalized["rows"][0]["position"] == "R-26-0488/1-1"
    assert raw_payload["order_number"] == "26-0075"
    assert "supplier_document_number" not in raw_payload


def test_keli_correlated_document_spacing_variants_are_normalized():
    for correlated_value in ("R - 26-0488", "R -26-0488", "R- 26-0488"):
        normalized = normalize_order_metadata(
            {"order_number": "26-0075", "rows": []},
            f"DOCUMENTO CORRELATO {correlated_value}",
            vendor_format="KELI",
        )

        assert normalized["order_number"] == "R-26-0488"


def test_non_keli_order_metadata_is_not_changed():
    payload = {
        "order_number": "26-0075",
        "rows": [{"order_number": "26-0075", "position": "R-26-0488/1-1"}],
    }

    normalized = normalize_order_metadata(
        payload,
        "DOCUMENTO CORRELATO R - 26-0488",
    )

    assert normalized == payload
