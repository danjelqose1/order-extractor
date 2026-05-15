from __future__ import annotations

from backend.utils_text import build_order_total_diagnostics, parse_declared_totals


def test_order_totals_match_returns_no_warning():
    result = build_order_total_diagnostics(344, 192.0, 344, 192.0)

    assert result is None


def test_order_units_mismatch_returns_warning():
    result = build_order_total_diagnostics(344, 192.0, 332, 192.0)

    assert result["severity"] == "warning"
    assert result["pdf_units"] == 344
    assert result["extracted_units"] == 332
    assert result["unit_delta"] == 12
    assert "Missing 12 units" in result["message"]


def test_order_area_mismatch_returns_warning():
    result = build_order_total_diagnostics(344, 192.0, 344, 184.9)

    assert result["severity"] == "warning"
    assert result["pdf_area"] == 192.0
    assert result["extracted_area"] == 184.9
    assert result["area_delta"] == 7.1
    assert "7.100 m²" in result["message"]


def test_missing_pdf_totals_returns_no_warning():
    result = build_order_total_diagnostics(None, None, 332, 184.9)

    assert result is None


def test_parse_declared_totals_supports_italian_comma_decimals():
    text = """
    Cliente ABC
    Totale PEZZI 344 m² 192,000
    """

    units, area = parse_declared_totals(text)

    assert units == 344
    assert area == 192.0


def test_parse_declared_totals_supports_plain_totale_row():
    units, area = parse_declared_totals("Totale 344 192,000")

    assert units == 344
    assert area == 192.0
